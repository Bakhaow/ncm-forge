#!/usr/bin/env bash
# Wait for the running tokenizer to exit, then launch the generator.
# Deliberately tolerant of a killed tokenizer: the last checkpoint is still a
# valid warm start for the generator, so the run is never lost.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "forge.cli train-tokenizer" >/dev/null; do sleep 30; done
ckpt=out/js/tokenizer/tokenizer.pt
if [ ! -f "$ckpt" ]; then
  echo "[chain] no tokenizer checkpoint, aborting" >&2
  exit 1
fi
echo "[chain] tokenizer process gone; checkpoint $(date -r "$ckpt" -Is)" >&2
exec .venv/bin/python -u -m forge.cli train-generator --config configs/js_final.yaml
