#!/usr/bin/env bash
# End-to-end sanity check of the ncm-forge pipeline.
#   1. synthetic corpus
#   2. stage 1: train the VQ-VAE tokenizer (a few steps)
#   3. stage 2: train the generator on the frozen codes (a few steps)
#   4. generate with the trained checkpoint and print the coarse-to-fine levels
#
# It validates shapes, losses, checkpoint formats and the upstream-compatible
# loading path. It proves nothing about quality.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/smoke.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    *) CONFIG="$1"; shift;;
  esac
done
PY="${PY:-.venv/bin/python}"

echo "[smoke] python: $($PY -V 2>&1)"
echo "[smoke] config: $CONFIG"

mkdir -p data
if [ ! -f data/smoke.txt ]; then
  echo "[smoke] generating synthetic corpus"
  $PY - <<'PYEOF'
import random
random.seed(0)
words = ["def", "parse", "invoice", "total", "return", "value", "line", "items", "config", "key"]
lines = []
for i in range(400):
    n = random.randint(6, 14)
    lines.append(" ".join(random.choice(words) for _ in range(n)))
open("data/smoke.txt", "w").write("\n".join(lines) + "\n")
print("wrote data/smoke.txt")
PYEOF
fi

echo "[smoke] stage 1 - tokenizer"
$PY -m forge.cli train-tokenizer --config "$CONFIG"

echo "[smoke] stage 2 - generator"
$PY -m forge.cli train-generator --config "$CONFIG"

echo "[smoke] stage 3 - generation"
export NCM_SMOKE_CONFIG="$CONFIG"
$PY - <<'PYEOF'
import os

import torch
from omegaconf import OmegaConf
from forge.vendor import require
require()

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

cfg = OmegaConf.load(os.environ.get("NCM_SMOKE_CONFIG", "configs/smoke.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

tk = VQVAETokenizer.from_config(cfg.tokenizer)
tk.vqvae = tk.vqvae.to(device).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
sd = torch.load("out/smoke/generator/generator.pt", map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()

P = int(cfg.model.prefix_len)
pad = tk.pre_tokenizer.pad_token_id
x = torch.tensor([[pad] * P], dtype=torch.long, device=device)
with torch.no_grad():
    g = net.generate(None, x, temperature=1.0, top_k=1, top_p=0.0)
    levels = tk.decode_multiscales(g)[0]

print(f"[smoke] {len(levels)} levels decoded:")
for i, text in enumerate(levels):
    print(f"  level {i}: {text!r}")
print("[smoke] OK")
PYEOF
