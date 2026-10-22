"""Latency benchmark for a config, without training.

Weights are random: only the shape of the computation matters. This is the
instrument that decides whether the product is viable on consumer hardware.

    .venv/bin/python scripts/bench_generate.py --config configs/code_pilot.yaml

NCM decodes one level at a time; the VQ always upsamples to the full token
grid, so the number of levels is fixed by the tokenizer. The knobs are the
number of candidates (best-of-N) and the batch size.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forge import config as cfgmod
from forge.vendor import require


def build(cfg, device):
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE

    vq_cfg = dict(cfg.tokenizer.vqvae)
    vq_cfg.pop("checkpoint_path", None)
    vq = VQVAE(**vq_cfg).to(device).eval()
    net = NextConceptTransformer(vqvae_local=vq, config=cfg.model).to(device).eval()
    return vq, net


def timed(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", default="1,8", help="comma separated batch sizes")
    ap.add_argument("--candidates", default="1,8", help="comma separated best-of-N values")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gpu-hour", type=float, default=0.30, help="rental price, USD/hour")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    require()
    torch.manual_seed(0)

    levels = cfgmod.levels_of(cfg)
    P = int(cfg.model.prefix_len)

    vq, net = build(cfg, device)
    params = sum(p.numel() for p in net.parameters()) + sum(
        p.numel() for p in vq.parameters()
    )
    print(f"config   {args.config}")
    print(f"device   {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"params   {params/1e6:.1f}M")
    print(f"levels   {levels} -> {sum(levels)} codes per request")
    print(f"prefix   {P} tokens")
    print()

    batches = [int(b) for b in str(args.batch).split(",")]
    cands = [int(c) for c in str(args.candidates).split(",")]

    print(f"{'batch':>5} {'N':>3} {'p50 ms':>8} {'min ms':>8} {'ms/req':>8} "
          f"{'codes/s':>9} {'$/1M req':>10}")
    for batch in batches:
        prefix = torch.randint(0, 1000, (batch, P), device=device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for cand in cands:
            def call():
                with torch.no_grad():
                    for _ in range(cand):
                        net.generate(None, prefix, temperature=1.0, top_k=1, top_p=0.0)

            p50, best = timed(call, device, args.warmup, args.repeats)
            per_req = p50 / batch
            codes = sum(levels) * batch
            cps = codes / p50 if p50 > 0 else float("nan")
            cost = per_req * 1e6 / 3600 * args.gpu_hour
            print(f"{batch:>5} {cand:>3} {p50*1e3:>8.1f} {best*1e3:>8.1f} "
                  f"{per_req*1e3:>8.1f} {cps:>9.0f} {cost:>10.2f}")
    if torch.cuda.is_available():
        print(f"\npeak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
