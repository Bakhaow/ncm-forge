#!/usr/bin/env bash
# One-shot setup: venv + dependencies + vendored upstream repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python torch transformers omegaconf hydra-core \
  rouge_score sacrebleu regex numpy

if [ ! -d vendor/ncm ]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/HazyResearch/ncm.git vendor/ncm
fi

echo "setup done: $(.venv/bin/python -c 'import torch; print("torch", torch.__version__)')"
