#!/usr/bin/env bash
# After a37: one decay shot, age held at 38, window [2, 56].
set -euo pipefail
ROOT="${ROOT:-/workspace/modded-nanogpt}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
TRAIN="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave5"
mkdir -p "$LOGDIR"
cd "$ROOT"

final_loss() {
  local f="$1"
  grep -E 'step:3250/3250 val_loss:' "$f" | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p'
}

score_3150() {
  local f="$1"
  grep -E 'step:3150/3250 val_loss:' "$f" | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p'
}

run_one() {
  local name="$1"
  shift
  local tee="$LOGDIR/${name}.stdout"
  if [[ -f "$tee" ]] && grep -qE 'exit=0 3150=[0-9]' "$tee"; then
    echo "==== skip $name (already scored 3150=$(score_3150 "$tee")) ===="
    return 0
  fi
  echo "==== $(date -u +%F\ %T) $name $* ====" | tee "$tee"
  torchrun --standalone --nproc_per_node=8 --master_port=29511 -- \
    "$TRAIN" --seed 0 --start 1000 "$@" \
    2>&1 | tee -a "$tee"
  local rc=${PIPESTATUS[0]}
  echo "==== $(date -u +%F\ %T) $name exit=$rc 3150=$(score_3150 "$tee") final=$(final_loss "$tee") ====" | tee -a "$tee"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc" >&2
    return "$rc"
  fi
}

echo "==== wave5 a38_tmin2  $(date -u +%F\ %T) ====" | tee "$LOGDIR/DECISION.txt"
echo "after a37; skip a39/a40/ws50; k=6 age 38 window [2,56]" | tee -a "$LOGDIR/DECISION.txt"

run_one K6_a38_tmin2 --k 6 --tau-min 2 --tau-max 56 --weights 0.02977,0.05954,0.08932,0.11909,0.14886,0.55342

echo "ref   K6_a38  3150=3.28020 3250=3.27398  [3,56] age 38"
echo "K6_a38_tmin2  3150=$(score_3150 "$LOGDIR/K6_a38_tmin2.stdout")  3250=$(final_loss "$LOGDIR/K6_a38_tmin2.stdout")"
