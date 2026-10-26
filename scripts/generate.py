"""Generate with a trained ncm-forge checkpoint.

    python scripts/generate.py --config configs/code_171m.yaml \
        --tokenizer out/tokenizer/tokenizer.pt \
        --generator out/generator/generator.pt \
        --text "def parse_invoice(raw):\n    lines = raw.splitlines()" \
        --show-levels

Prints the coarse-to-fine trajectory (one decoded string per hierarchy level),
which is the interesting artefact: level 1 holds the global shape of the
completion, the last level the final text.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.vendor import require  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--generator", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--suffix", default="", help="tokens after the gap (FIM); the model was trained with both sides known")
    ap.add_argument("--file", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--show-levels", action="store_true")
    args = ap.parse_args()

    require()
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(args.config)
    if args.tokenizer:
        cfg.tokenizer.vqvae.checkpoint_path = args.tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    if args.generator:
        sd = torch.load(args.generator, map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"], strict=True)
    net.eval()

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise SystemExit("provide --text or --file")

    # The conditioning block is what the model saw while training: two halves,
    # (suffix, prefix) or (prefix, suffix) depending on data.cond_order. A half is
    # truncated on the gap side, then right-padded. generate() does NOT apply the
    # training-time pad mask, so pass real tokens on both sides when you can.
    P = int(cfg.model.prefix_len)
    half = P // 2
    pad = tk.pre_tokenizer.pad_token_id
    if pad is None:  # e.g. the sudoku tokenizer has no pad token
        pad = 0
    before = tk.pre_tokenizer.encode(text)[-half:]
    after = tk.pre_tokenizer.encode(args.suffix)[:half] if args.suffix else []
    sides = {"prefix": before[-half:] + [pad] * (half - len(before[-half:])),
             "suffix": after + [pad] * (half - len(after))}
    order = str(cfg.data.get("cond_order", "suffix_first"))
    key = ["suffix", "prefix"] if order.startswith("suffix") else ["prefix", "suffix"]
    ids = sides[key[0]] + sides[key[1]]
    if not args.suffix:
        print("[generate] no --suffix: the suffix half is padding, which is out of "
              "distribution for this FIM model; prefer passing the text after the gap",
              file=sys.stderr)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Upstream requires either predict_level0 (the model predicts the coarsest
    # level) or an explicit level-0 code (vendor/ncm/models/next_concept.py:629).
    l0 = None
    if not bool(cfg.model.get("predict_level0", False)):
        vq = tk.vqvae.vq
        l0 = torch.randint(
            0, vq.vqvae_vocab_sizes[0], (1, vq.levels[0]), device=device
        )

    with torch.no_grad():
        g = net.generate(
            l0, x,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            cfg_scale=args.cfg_scale, cfg_null_eos=tk.pre_tokenizer.eos_token_id,
        )
        levels = tk.decode_multiscales(g)[0]

    if args.show_levels:
        for i, lvl in enumerate(levels):
            print(f"--- level {i} ---")
            print(lvl)
    else:
        print(levels[-1])


if __name__ == "__main__":
    main()
