#!/usr/bin/env bash
# Wait for a checkpoint to be safely on disk, then stop the training.
#
# The generator logs "step N" before saving at N % save_every == 0, so we wait
# for the *next* log line (N + log_every) as proof the save completed.
#
# Anchor the marker on the stage's own prefix. A bare "step 2050/10000" also
# matches the tokenizer stage when both share the log file, and that mistake
# kills the run immediately (it happened once). Pass the checkpoint as the
# fourth argument and the script refuses to kill if it is stale.
#
#   bash scripts/stop_after_checkpoint.sh 312401 "[generator] step 2050/" \
#        logs/pilot.log out/pilot/generator/generator.pt
set -euo pipefail
cd "$(dirname "$0")/.."

pid=${1:?training pid}
marker=${2:?log line that proves the checkpoint is written}
log=${3:-logs/pilot.log}
ckpt=${4:-}
deadline=$(( $(date +%s) + 4 * 3600 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
  if grep -qF -- "$marker" "$log" 2>/dev/null; then
    if [ -n "$ckpt" ]; then
      if [ ! -f "$ckpt" ]; then
        echo "=== $(date -Is) marker seen but $ckpt missing, not killing ===" >>"$log"
        exit 1
      fi
      age=$(( $(date +%s) - $(stat -c %Y "$ckpt") ))
      if [ "$age" -gt 900 ]; then
        echo "=== $(date -Is) $ckpt is ${age}s old, not killing (wrong marker?) ===" >>"$log"
        exit 1
      fi
    fi
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 3
      kill -KILL "$pid" 2>/dev/null || true
    fi
    pkill -P "$pid" 2>/dev/null || true
    echo "=== $(date -Is) stopped pid $pid after '$marker' ===" >>"$log"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "=== $(date -Is) pid $pid already gone, nothing to stop ===" >>"$log"
    exit 0
  fi
  sleep 20
done
echo "=== $(date -Is) timeout waiting for '$marker', training left running ===" >>"$log"
exit 1
