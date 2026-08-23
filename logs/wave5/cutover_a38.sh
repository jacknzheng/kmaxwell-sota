#!/usr/bin/env bash
# After ws50 scores, kill the leftover age-35 A shots and start the a38 grid.
# The running `run.sh all` is already in memory; editing it will not change B.
set -euo pipefail
ROOT="/workspace/modded-nanogpt"
LOGDIR="$ROOT/logs/wave5"
WS50="$LOGDIR/K6_ws50.stdout"
B="$LOGDIR/run_B_a38.sh"

echo "==== cutover wait $(date -u +%F\ %T) ===="
while ! grep -qE 'exit=0 3150=[0-9]' "$WS50" 2>/dev/null; do
  sleep 3
done
echo "==== ws50 scored $(date -u +%F\ %T) 3150=$(grep -E 'step:3150/3250 val_loss:' "$WS50" | tail -1) ===="

# Parent will try tmin2 next. Kill it and any new torchrun on 29511.
parent="$(pgrep -f '/workspace/modded-nanogpt/logs/wave5/run.sh all' || true)"
if [[ -n "$parent" ]]; then
  echo "killing run.sh all pids: $parent"
  kill $parent || true
fi
sleep 1
# If tmin2 already spawned, drop it. Do not touch run_B_a38 / a36+.
while read -r pid; do
  [[ -z "$pid" ]] && continue
  cmd="$(ps -p "$pid" -o args= || true)"
  if echo "$cmd" | grep -q 'run_B_a38'; then
    continue
  fi
  if echo "$cmd" | grep -Eq 'tmin2|tmax64|run.sh all|train_gpt_kmaxwell'; then
    echo "killing leftover $pid $cmd"
    kill "$pid" || true
  fi
done < <(pgrep -f 'torchrun|train_gpt_kmaxwell|logs/wave5/run.sh all' || true)
sleep 2
# Make sure GPUs are free before B.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  used="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)"
  if [[ "${used:-0}" -eq 0 ]]; then
    break
  fi
  echo "GPUs still busy ($used), waiting"
  sleep 2
done

if tmux has-session -t persist 2>/dev/null; then
  if tmux list-windows -t persist -F '#{window_name}' | grep -Fxq wave5b; then
    echo "wave5b window already exists" >&2
    exit 1
  fi
  tmux new-window -t persist -n wave5b -c "$ROOT" -- /bin/bash "$B"
else
  tmux new-session -d -s persist -n wave5b -c "$ROOT" -- /bin/bash "$B"
fi
echo "==== started persist:wave5b $(date -u +%F\ %T) ===="
tmux list-windows -t persist
