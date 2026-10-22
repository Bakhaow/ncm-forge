#!/usr/bin/env bash
# Unattended pilot: tokenizer (~4 h) then generator (~8 h) on the RTX 4070.
# Survives the terminal: nohup + setsid, logs to logs/pilot.log.
#
#   bash scripts/run_pilot.sh          # train, then evaluate
#   bash scripts/run_pilot.sh --force  # relaunch even if out/pilot exists
#
# Stop with: kill $(cat logs/pilot.pid)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/code_pilot.yaml}
LOG=logs/pilot.log
mkdir -p logs

if [ -d out/pilot ] && [ "${1:-}" != "--force" ]; then
  echo "out/pilot exists; pass --force to overwrite" >&2
  exit 1
fi

python=${PYTHON:-.venv/bin/python}
if ! "$python" -c "import torch" 2>/dev/null; then
  echo "no torch in $python" >&2
  exit 1
fi

start=$(date +%s)
{
  echo "=== $(date -Is) tokenizer ==="
  "$python" -m forge.cli train-tokenizer --config "$CONFIG"
  echo "=== $(date -Is) generator ==="
  "$python" -m forge.cli train-generator --config "$CONFIG"
  echo "=== $(date -Is) probe ==="
  "$python" scripts/generate.py --config "$CONFIG" \
      --tokenizer out/pilot/tokenizer/tokenizer.pt \
      --generator out/pilot/generator/generator.pt \
      --text "def parse_invoice(payload):" --top-k 20 --show-levels || true
  echo "=== $(date -Is) done in $(( ($(date +%s) - start) / 60 )) min ==="
} >>"$LOG" 2>&1 &
echo $! > logs/pilot.pid
echo "pilot started, pid $(cat logs/pilot.pid), log $LOG"
