"""Stage 2: train the NCM generator on top of a frozen tokenizer.

Mirrors Algorithm 1 of the paper:

    encode -> corrupt conditioning codes -> rebuild the per-level inputs
    -> one forward pass predicts every level -> per-level cross entropy

Level weights are w_k proportional to l_k^-alpha, normalised to sum to 1.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forge import config as cfgmod
from forge.train_tokenizer import build_dataset
from forge.vendor import require


def corrupt_codes(idx: torch.Tensor, levels: list[int], vocab_sizes: list[int], rate) -> torch.Tensor:
    """Replace conditioning codes with random codes from the same level codebook.

    `rate` is either a float (same rate for every level) or a list with one
    rate per level. The last level is skipped: its codes are never fed back
    into the model input, so corrupting them would be a no-op.
    """
    out = idx.clone()
    off = 0
    for k, level in enumerate(levels):
        seg = out[:, off : off + level]
        r = rate[k] if isinstance(rate, (list, tuple)) else rate
        if r > 0 and k < len(levels) - 1:
            mask = torch.rand(seg.shape, device=seg.device) < float(r)
            if mask.any():
                rnd = torch.randint(
                    0, vocab_sizes[k], seg.shape, device=seg.device, dtype=seg.dtype
                )
                seg[mask] = rnd[mask]
        off += level
    return out


def level_weights(levels: list[int], alpha: float) -> torch.Tensor:
    ls = torch.tensor([float(l) for l in levels])
    w = ls.pow(-alpha)
    return w / w.sum()


def level_ce(logits, idx, levels, vocab_sizes, lvl_first, predict_level0, alpha):
    """Per-level cross entropy over the concatenated level logits.

    logits layout (see models/next_concept.py::_run_stack):
      [level-0 head block (levels[0] positions)] when predict_level0
      [one block per level in levels[lvl_first:]]
    """
    total = logits.new_zeros(())
    stats = []
    pos = 0
    if predict_level0:
        l0 = levels[0]
        v0 = vocab_sizes[0]
        lg = logits[:, pos : pos + l0, :v0]
        tgt = idx[:, :l0]
        loss0 = F.cross_entropy(lg.reshape(-1, v0), tgt.reshape(-1))
        w0 = level_weights([l0], alpha)[0]
        total = total + w0 * loss0
        stats.append(("level0", float(loss0.detach())))
        pos += l0

    w = level_weights(levels[lvl_first:], alpha)
    off_idx = sum(levels[:lvl_first])
    for j, level in enumerate(levels[lvl_first:]):
        v = vocab_sizes[lvl_first + j]
        lg = logits[:, pos : pos + level, :v]
        tgt = idx[:, off_idx : off_idx + level]
        loss_k = F.cross_entropy(lg.reshape(-1, v), tgt.reshape(-1))
        total = total + w[j] * loss_k
        stats.append((f"level{lvl_first + j}", float(loss_k.detach())))
        pos += level
        off_idx += level
    return total, stats


def build_prefix_latents(tk, cond_ids: torch.Tensor, cfg, device):
    """Conditioning block as the upstream generate() builds it."""
    prefix_len = int(cfg.model.prefix_len)
    enc = tk.vqvae.encoder
    if bool(cfg.model.get("contextual_prefix", False)) and len(getattr(enc, "blocks", [])) > 0:
        lat = enc.forward_nopos(cond_ids)[:, :prefix_len, :]
    elif bool(cfg.model.get("raw_token_prefix", False)):
        lat = enc.wte(cond_ids)[:, :prefix_len, :]
    else:
        lat = tk.vqvae.encode_z(cond_ids)[:, :prefix_len, :]
    return lat.to(device)


def train(cfg):
    require()
    from transformers import AutoTokenizer
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    device = torch.device(cfg.train.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pre_id = cfg.tokenizer.pre_tokenizer.model_id
    tokenizer = AutoTokenizer.from_pretrained(pre_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)

    dataset = build_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.get("num_workers", 2)),
        drop_last=True,
    )
    print(f"[generator] {len(dataset)} windows, batch={cfg.train.batch_size}, device={device}")
    if len(dataset) < int(cfg.train.batch_size):
        raise RuntimeError(
            f"dataset has {len(dataset)} items < batch_size {cfg.train.batch_size}: "
            "with drop_last=True the loader yields nothing"
        )

    levels = list(tk.vqvae.vq.levels)
    vocab_sizes = list(tk.vqvae.vq.vqvae_vocab_sizes)
    lvl_first = 0 if bool(cfg.model.get("query_level0", False)) else 1
    if lvl_first == 0:
        raise NotImplementedError(
            "query_level0=True is not supported: idx_to_ncm_input() only produces "
            "levels[1:] (vendor/ncm/models/quant.py:679) while the model expects a "
            "level-0 chunk. Use query_level0: false, as in every upstream config."
        )
    predict_level0 = bool(cfg.model.get("predict_level0", False))
    head_sizes = list(cfg.model.vocab_size)
    expected = vocab_sizes[lvl_first:]
    if head_sizes != expected:
        raise ValueError(
            f"model.vocab_size {head_sizes} does not match the tokenizer codebooks "
            f"for levels[lvl_first:] {expected}"
        )
    alpha = float(cfg.train.get("loss_alpha", 0.0))
    corruption = cfg.train.get("corruption_rate", 0.0)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )
    steps = int(cfg.train.steps)
    warmup = int(cfg.train.get("warmup", max(1, steps // 100)))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(float(prog))).item())

    out_dir = Path(cfg.train.get("out_dir", "out/generator"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_every = int(cfg.train.get("log_every", 50))
    save_every = int(cfg.train.get("save_every", 2000))

    net.train()
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
            tgt_ids, cond_ids, cond_mask = batch
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            cond_ids = cond_ids.to(device, non_blocking=True)
            cond_mask = cond_mask.to(device, non_blocking=True)
            prefix_len = int(cfg.model.prefix_len)
            if cond_ids.shape[1] != prefix_len:
                raise ValueError(
                    f"conditioning block is {cond_ids.shape[1]} wide but "
                    f"model.prefix_len is {prefix_len}: data.prefix_len + data.suffix_len "
                    "must equal model.prefix_len"
                )

            with torch.no_grad():
                idx = tk.vqvae.encode(tgt_ids)
                idx_c = corrupt_codes(idx, levels, vocab_sizes, corruption)
                x_ncm = tk.vqvae.vq.idx_to_ncm_input(idx_c)
                prefix_lat = build_prefix_latents(tk, cond_ids, cfg, device)

            x = torch.cat([prefix_lat, x_ncm], dim=1)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = net(x, prefix_ids=None, prefix_pad_mask=cond_mask)
                loss, _ = level_ce(
                    logits.float(), idx, levels, vocab_sizes, lvl_first, predict_level0, alpha
                )
            (loss / accum).backward()
            running += float(loss.detach())
            running_n += 1
            micro += 1
            if micro < accum:
                continue
            micro = 0
            for g in opt.param_groups:
                g["lr"] = float(cfg.train.lr) * lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.get("grad_clip", 1.0)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                dt = time.time() - t0
                print(
                    f"[generator] step {step}/{steps} loss {running / running_n:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {step / dt:.2f} it/s "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                    flush=True,
                )
                running = 0.0
                running_n = 0
            if step % save_every == 0 or step == steps:
                torch.save(
                    {"model": net.state_dict(), "step": step, "config": cfg.model},
                    out_dir / "generator.pt",
                )
    torch.save({"model": net.state_dict(), "step": step, "config": cfg.model}, out_dir / "generator.pt")
    print(f"[generator] saved {out_dir / 'generator.pt'}")
    return out_dir / "generator.pt"


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
