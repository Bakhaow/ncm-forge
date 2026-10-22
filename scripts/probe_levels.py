"""Coarse-to-fine probe on arbitrary text, using an upstream checkpoint.

Two uses:

1. Research - does the hierarchy encode *structure* or just topic? Feed a piece
   of code and read what each level decodes to. The paper probes the hierarchy
   on prose (Appendix H); nobody has probed it on code.
2. Latency - measure the real per-candidate cost on your own GPU, which is the
   number that decides whether a local autocomplete is viable.

    python scripts/probe_levels.py --task xsum \
        --text "def parse_invoice(raw):\n    lines = raw.splitlines()\n    total = 0" \
        --timing
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.vendor import require  # noqa: E402

REPO = "hazyresearch/ncm-checkpoints"


def fetch(task: str, local_dir: Path, local: str = ""):
    """Load an upstream checkpoint + its config, from disk or from the Hub."""
    from huggingface_hub import hf_hub_download

    if local:
        d = Path(local)
        files = {name: str(d / name) for name in ("config.yaml", "tokenizer.pt", "generator.pt")}
        for name, path in files.items():
            if not Path(path).exists():
                raise SystemExit(f"missing {path}")
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("config.yaml", "tokenizer.pt", "generator.pt"):
            files[name] = hf_hub_download(
                repo_id=REPO, filename=f"{task}/{name}", local_dir=str(local_dir)
            )
    cfg = OmegaConf.load(files["config.yaml"])
    cfg.tokenizer.vqvae.checkpoint_path = files["tokenizer.pt"]
    return cfg, files["generator.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="xsum", help="upstream task folder (xsum, wmt14, sudoku/easy, ...)")
    ap.add_argument("--text", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--cache", default=".cache/upstream")
    ap.add_argument("--local", default="", help="local dir with config.yaml/tokenizer.pt/generator.pt")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timing", action="store_true", help="measure per-candidate latency")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-candidates", type=int, default=1)
    args = ap.parse_args()

    require()
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg, gen_path = fetch(args.task, Path(args.cache) / args.task.replace("/", "_"), args.local)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(gen_path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise SystemExit("provide --text or --file")

    P = int(cfg.model.prefix_len)
    pad = tk.pre_tokenizer.pad_token_id
    if pad is None:  # e.g. the sudoku tokenizer has no pad token
        pad = 0
    ids = (tk.pre_tokenizer.encode(text)[:P] + [pad] * P)[:P]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Upstream always supplies the level-0 codes explicitly when the model does
    # not predict them (vendor/ncm/models/next_concept.py:629).
    l0 = None
    if not bool(cfg.model.get("predict_level0", False)):
        vq = tk.vqvae.vq
        l0 = torch.randint(0, vq.vqvae_vocab_sizes[0], (1, vq.levels[0]), device=device)

    with torch.no_grad():
        g = net.generate(
            l0, x, temperature=args.temperature, top_k=args.top_k, top_p=0.0
        )
        levels = tk.decode_multiscales(g)[0]

    print(f"=== coarse-to-fine ({len(levels)} levels) ===")
    for i, lvl in enumerate(levels):
        print(f"[level {i}] {lvl!r}")

    if args.timing:
        times = []
        for _ in range(args.repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.n_candidates):
                    net.generate(
                        l0, x, temperature=args.temperature, top_k=args.top_k, top_p=0.0
                    )
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / args.n_candidates)
        times.sort()
        print(
            f"=== latency on {device} ===\n"
            f"candidates per call: {args.n_candidates}\n"
            f"median {statistics.median(times) * 1000:.1f} ms   "
            f"p10 {times[len(times) // 10] * 1000:.1f} ms   "
            f"min {times[0] * 1000:.1f} ms"
        )


if __name__ == "__main__":
    main()
