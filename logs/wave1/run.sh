#!/usr/bin/env bash
# Wave 1 on master: 4-GPU sequential. Logs: this dir (tee) + trainer uuid files.
set -euo pipefail
ROOT="/workspace/modded-nanogpt"
TRAIN="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave1"
TMIN=5.666666666666666
TMAX49=48.99999999999996
WINNER_W="0.05,0.20,0.25,0.50"
WINNER_LOSS=3.27439
mkdir -p "$LOGDIR"
cd "$ROOT"

final_loss() {
  local f="$1"
  grep -E 'step:3250/3250 val_loss:' "$f" | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p'
}

run_one() {
  local name="$1"
  shift
  local tee="$LOGDIR/${name}.stdout"
  echo "==== $(date -u +%F\ %T) $name $* ====" | tee "$tee"
  # Trainer still writes logs/<uuid>.txt with source + recipe + val_loss (print0).
  # Tee captures the same console val_loss / train_time lines.
  torchrun --standalone --nproc_per_node=4 --master_port=29511 -- \
    "$TRAIN" --seed 0 --k 4 --start 1000 "$@" \
    2>&1 | tee -a "$tee"
  local rc=${PIPESTATUS[0]}
  echo "==== $(date -u +%F\ %T) $name exit=$rc final=$(final_loss "$tee") ====" | tee -a "$tee"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc" >&2
    return "$rc"
  fi
}

# A. Age scan at fixed shape (skip 25/30/33). User weights for 36/42; mid-bundle solve for 48.
run_one A36_age36 --tau-min "$TMIN" --tau-max "$TMAX49" --weights 0.041,0.163,0.204,0.592
run_one A42_age42 --tau-min "$TMIN" --tau-max "$TMAX49" --weights 0.022,0.088,0.110,0.780
run_one A48_age48 --tau-min "$TMIN" --tau-max "$TMAX49" --weights 0.00314,0.012561,0.015701,0.968597

# B. Mild tau_max stretch, winner weights frozen. 80 only if 56 or 64 still win.
run_one B56_tmax56 --tau-min "$TMIN" --tau-max 56 --weights "$WINNER_W"
run_one B64_tmax64 --tau-min "$TMIN" --tau-max 64 --weights "$WINNER_W"

b56=$(final_loss "$LOGDIR/B56_tmax56.stdout")
b64=$(final_loss "$LOGDIR/B64_tmax64.stdout")
echo "B56=$b56 B64=$b64 winner=$WINNER_LOSS"
if python3 -c "import sys; sys.exit(0 if min(float('$b56'), float('$b64')) <= $WINNER_LOSS else 1)"; then
  run_one B80_tmax80 --tau-min "$TMIN" --tau-max 80 --weights "$WINNER_W"
else
  echo "skip B80: 56/64 did not beat $WINNER_LOSS"
fi
echo "wave1 done"
