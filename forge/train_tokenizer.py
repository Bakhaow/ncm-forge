"""Stage 1: train the multiscale residual VQ-VAE tokenizer.

The upstream VQVAE module already implements the multiscale residual quantizer
(per-level codebooks, residual accumulation, EMA codebook updates, level-wise
down/upsampling). What upstream does not ship is the training loop, so this
script supplies it:

    loss = cross_entropy(decoder(z_hat), tokens) + vq_loss

`aux.fine_drop` in the quantizer config truncates the reconstruction at a
random level for a fraction of the steps: that is the nested-dropout surrogate
which forces the coarse levels to carry usable information on their own.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forge import config as cfgmod
from forge.optim import make_adamw
from forge import data as datamod
from forge.vendor import require


def build_model(cfg):
    require()
    from models.vqvae import VQVAE

    vq_cfg = dict(cfg.tokenizer.vqvae)
    vq_cfg.pop("checkpoint_path", None)
    model = VQVAE(**vq_cfg)
    # Upstream seeds the EMA accumulator with the random init
    # (vendor/ncm/models/quant.py:303), so the first EMA step divides that init
    # by (1-decay)*count: ||w|| ~ 1e6 and vq_loss ~ 1e3 for ~100 steps, which
    # makes argmin||z-w|| decide on the code norm instead of its direction.
    # Zeroing ema_w restores the standard EMA-VQ semantics (e_i = sum(x)/count).
    for i in range(len(model.vq.vqvae_vocab_sizes)):
        getattr(model.vq, f"ema_w_{i}").zero_()
    return model


def build_token_stream(tokenizer, docs: list[str], eos_id: int) -> torch.Tensor:
    chunks = []
    for doc in docs:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if not ids:
            continue
        chunks.append(torch.tensor(ids + [eos_id], dtype=torch.long))
    if not chunks:
        raise RuntimeError("empty corpus")
    return torch.cat(chunks)


def build_dataset(cfg, tokenizer):
    task = cfg.data.get("task", "unconditional")
    vq_len = int(cfg.tokenizer.vqvae.context_length)
    target_len = int(cfg.data.get("target_len", vq_len))
    if target_len != vq_len:
        raise ValueError(
            f"data.target_len ({target_len}) must equal tokenizer.vqvae.context_length ({vq_len})"
        )
    prefix_len = int(cfg.model.get("prefix_len", 1))
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    if task == "unconditional":
        docs = datamod.read_documents(cfg.data.corpus, cfg.data.get("format", "auto"))
        if cfg.data.get("max_docs"):
            docs = docs[: int(cfg.data.max_docs)]
        ids = build_token_stream(tokenizer, docs, eos_id)
        return datamod.build_stream_dataset(ids, target_len, prefix_len, eos_id)

    records = datamod.jsonl_records(cfg.data.corpus)
    if cfg.data.get("max_docs"):
        records = records[: int(cfg.data.max_docs)]
    if task == "fim":
        return datamod.build_fim_dataset(
            records,
            tokenizer,
            prefix_len=int(cfg.data.get("prefix_len", prefix_len)),
            suffix_len=int(cfg.data.get("suffix_len", prefix_len)),
            target_len=target_len,
            cond_order=cfg.data.get("cond_order", "suffix_first"),
        )
    if task == "conditional":
        return datamod.build_conditional_dataset(
            records,
            tokenizer,
            prefix_len=int(cfg.data.get("prefix_len", prefix_len)),
            target_len=target_len,
            source_key=cfg.data.get("source_key", "source"),
            target_key=cfg.data.get("target_key", "target"),
        )
    raise ValueError(f"unknown task: {task}")


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> dict:
    """Exact-token reconstruction accuracy and codebook utilization."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch[0].to(device)
        logits, _, per_level, global_util, _ = model(ids)
        pred = logits.argmax(dim=-1)
        correct += (pred == ids).sum().item()
        total += ids.numel()
    if was_training:
        model.train()
    return {
        "token_accuracy": correct / max(total, 1),
        "codebook_utilization": global_util,
        "per_level_utilization": per_level,
    }


def train(cfg):
    require()
    from transformers import AutoTokenizer

    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pre_id = cfg.tokenizer.pre_tokenizer.model_id
    tokenizer = AutoTokenizer.from_pretrained(pre_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(cfg).to(device)
    dataset = build_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.get("num_workers", 2)),
        drop_last=True,
    )
    print(f"[tokenizer] {len(dataset)} windows, batch={cfg.train.batch_size}, device={device}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_adamw(
        params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )
    steps = int(cfg.train.steps)
    warmup = int(cfg.train.get("warmup", max(1, steps // 100)))
    if len(dataset) < int(cfg.train.batch_size):
        raise RuntimeError(
            f"dataset has {len(dataset)} items < batch_size {cfg.train.batch_size}: "
            "with drop_last=True the loader yields nothing"
        )

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(float(prog))).item())

    use_fp16 = bool(cfg.train.get("fp16", False))
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16 and device.type == "cuda")

    out_dir = Path(cfg.train.get("tokenizer_out_dir", "out/tokenizer"))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_every = int(cfg.train.get("eval_every", 1000))
    log_every = int(cfg.train.get("log_every", 50))

    model.train()
    step = 0
    t0 = time.time()
    running = 0.0
    running_n = 0
    accum = max(1, int(cfg.train.get("grad_accum", 1)))
    micro = 0
    opt.zero_grad(set_to_none=True)
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            ids = batch[0].to(device, non_blocking=True)
            with torch.autocast(
                "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                logits, vq_loss, _, _, _ = model(ids)
                # keep the logits in the autocast dtype: forcing fp32 here
                # materialises batch x context x 151680 x 4 bytes (2.5 GB at b=64)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids.reshape(-1)
                )
                loss = ce + float(cfg.train.get("vq_weight", 1.0)) * vq_loss
            scaler.scale(loss / accum).backward()
            running += float(loss.detach())
            running_n += 1
            micro += 1
            if micro < accum:
                continue
            micro = 0
            for g in opt.param_groups:
                g["lr"] = float(cfg.train.lr) * lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.get("grad_clip", 1.0)))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                dt = time.time() - t0
                print(
                    f"[tokenizer] step {step}/{steps} loss {running / running_n:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {step / dt:.2f} it/s",
                    flush=True,
                )
                running = 0.0
                running_n = 0
            if step % eval_every == 0 or step == steps:
                metrics = evaluate(model, loader, device)
                print(f"[tokenizer] eval step {step}: {metrics}", flush=True)
                torch.save(
                    {"model": model.state_dict(), "step": step, "metrics": metrics},
                    out_dir / "tokenizer.pt",
                )
    torch.save({"model": model.state_dict(), "step": step}, out_dir / "tokenizer.pt")
    print(f"[tokenizer] saved {out_dir / 'tokenizer.pt'}")
    return out_dir / "tokenizer.pt"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = cfgmod.load(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
