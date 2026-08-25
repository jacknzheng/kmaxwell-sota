#!/usr/bin/env bash
# Wait for a37, kill a39+, start a38_tmin2.
set -euo pipefail
ROOT="/workspace/modded-nanogpt"
LOGDIR="$ROOT/logs/wave5"
A37="$LOGDIR/K6_a37.stdout"
B="$LOGDIR/run_B_decay.sh"

echo "==== a38_tmin2 cutover wait $(date -u +%F\ %T) ===="
while ! grep -qE 'exit=0 3150=[0-9]' "$A37" 2>/dev/null; do
  sleep 3
done
echo "==== a37 scored $(date -u +%F\ %T) ===="
grep -E 'exit=0 3150=' "$A37" | tail -1

parent="$(pgrep -f '/workspace/modded-nanogpt/logs/wave5/run_B_a38.sh' || true)"
if [[ -n "$parent" ]]; then
  echo "killing run_B_a38.sh pids: $parent"
  kill $parent || true
fi
sleep 1
while read -r pid; do
  [[ -z "$pid" ]] && continue
  cmd="$(ps -p "$pid" -o args= || true)"
  if echo "$cmd" | grep -qE 'run_B_decay|cutover_decay|a38_tmin2'; then
    continue
  fi
  if echo "$cmd" | grep -Eq 'K6_a39|K6_a40|a38_ws50|run_B_a38|train_gpt_kmaxwell|torchrun'; then
    echo "killing leftover $pid $cmd"
    kill "$pid" || true
  fi
done < <(pgrep -f 'torchrun|train_gpt_kmaxwell|run_B_a38.sh' || true)
sleep 2
for _ in 1 2 3 4 5 6 7 8 9 10; do
  used="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)"
  if [[ "${used:-0}" -eq 0 ]]; then
    break
  fi
  echo "GPUs still busy ($used), waiting"
  sleep 2
done

if tmux has-session -t persist 2>/dev/null; then
  if tmux list-windows -t persist -F '#{window_name}' | grep -Fxq wave5d; then
    echo "wave5d window already exists" >&2
    exit 1
  fi
  tmux new-window -t persist -n wave5d -c "$ROOT" -- /bin/bash "$B"
else
  tmux new-session -d -s persist -n wave5d -c "$ROOT" -- /bin/bash "$B"
fi
echo "==== started persist:wave5d a38_tmin2 $(date -u +%F\ %T) ===="
tmux list-windows -t persist
