"""Cross-evaluate two FIM checkpoints at several context lengths.

All arms see the SAME held-out windows; only the number of conditioning tokens
changes, so the delta measures how much the model actually uses the far context.

    python scripts/ab_context.py --records 2048 --offset 100000
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.vendor import require  # noqa: E402


def load_records(path, offset, n):
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < offset:
                continue
            if len(out) >= n:
                break
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/js_ab_64.yaml")
    ap.add_argument("--corpus", default="data/js_ab/fim.jsonl")
    ap.add_argument("--offset", type=int, default=100000)
    ap.add_argument("--records", type=int, default=2048)
    ap.add_argument("--arms", nargs="+", default=[
        "64:out/ab/gen64/generator.pt",
        "128:out/ab/gen128/generator.pt",
    ], help="eval_context:checkpoint")
    ap.add_argument("--evals", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    require()
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer
    from forge.data import build_fim_dataset
    from forge.train_generator import build_prefix_latents, level_ce, corrupt_codes

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)
    pre = AutoTokenizer.from_pretrained(cfg.tokenizer.pre_tokenizer.model_id)
    if pre.pad_token_id is None:
        pre.pad_token = pre.eos_token
    levels = list(tk.vqvae.vq.levels)
    vocab = list(tk.vqvae.vq.vqvae_vocab_sizes)
    lvl_first = 0 if bool(cfg.model.get("query_level0", False)) else 1
    predict_level0 = bool(cfg.model.get("predict_level0", False))
    alpha = float(cfg.train.get("loss_alpha", 0.0))

    records = load_records(args.corpus, args.offset, args.records)
    print(f"{len(records)} held-out windows from {args.corpus} (offset {args.offset})")

    datasets = {}
    for p in args.evals:
        full = p
        datasets[p] = build_fim_dataset(
            records, pre, prefix_len=full // 2, suffix_len=full // 2,
            target_len=int(cfg.tokenizer.vqvae.context_length),
            cond_order=cfg.data.get("cond_order", "suffix_first"),
        )

    rows = []
    for arm in args.arms:
        p_str, ckpt = arm.split(":", 1)
        for eval_p in args.evals:
            ecfg = OmegaConf.create(OmegaConf.to_container(cfg))
            ecfg.model.prefix_len = eval_p
            mcfg = ecfg.model
            net = NextConceptTransformer(vqvae_local=tk.vqvae, config=mcfg).to(device)
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = sd["model"]
            # the attention mask is a persistent buffer sized by prefix_len:
            # the current model already built the right one for eval_p
            state.pop("attn_mask_for_masking", None)
            missing, unexpected = net.load_state_dict(state, strict=False)
            assert not unexpected, unexpected
            assert set(missing) <= {"attn_mask_for_masking"}, missing
            net.eval()
            ds = datasets[eval_p]
            total, n = 0.0, 0
            per_level = {}
            with torch.no_grad():
                for start in range(0, len(ds), args.batch):
                    batch = [ds[i] for i in range(start, min(start + args.batch, len(ds)))]
                    tgt = torch.stack([b[0] for b in batch]).to(device)
                    cond = torch.stack([b[1] for b in batch]).to(device)
                    mask = torch.stack([b[2] for b in batch]).to(device)
                    idx = tk.vqvae.encode(tgt)
                    idx_c = corrupt_codes(idx, levels, vocab, 0.0)
                    x = torch.cat([build_prefix_latents(tk, cond, ecfg, device),
                                   tk.vqvae.vq.idx_to_ncm_input(idx_c)], dim=1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(x, prefix_ids=None, prefix_pad_mask=mask)
                    loss, stats = level_ce(logits.float(), idx, levels, vocab,
                                           lvl_first, predict_level0, alpha)
                    total += float(loss) * len(batch)
                    n += len(batch)
                    for name, v in stats:
                        per_level.setdefault(name, []).append(v)
            rows.append((Path(ckpt).parent.name, eval_p, total / n,
                         {k: sum(v) / len(v) for k, v in per_level.items()}))
            del net
            torch.cuda.empty_cache()

    print(f"{'arm':10s} {'ctx':>4s} {'weighted CE':>12s}  level0   level1   level7")
    for arm, ctx, ce, pl in rows:
        print(f"{arm:10s} {ctx:4d} {ce:12.4f}  "
              f"{pl.get('level0', float('nan')):7.4f}  "
              f"{pl.get('level1', float('nan')):7.4f}  "
              f"{pl.get('level7', float('nan')):7.4f}")


if __name__ == "__main__":
    main()
