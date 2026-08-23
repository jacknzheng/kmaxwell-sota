#!/usr/bin/env bash
# Wave 3 on 1×H100: sequential 1-GPU runs. Logs: this dir (tee) + trainer uuid files.
set -euo pipefail
ROOT="/workspace/modded-nanogpt"
TRAIN="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave3"
WINNER_W="0.05,0.20,0.25,0.50"
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
  echo "==== $(date -u +%F\ %T) $name $* ====" | tee "$tee"
  torchrun --standalone --nproc_per_node=1 --master_port=29511 -- \
    "$TRAIN" --seed 0 --k 4 --start 1000 "$@" \
    2>&1 | tee -a "$tee"
  local rc=${PIPESTATUS[0]}
  echo "==== $(date -u +%F\ %T) $name exit=$rc 3150=$(score_3150 "$tee") final=$(final_loss "$tee") ====" | tee -a "$tee"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc" >&2
    return "$rc"
  fi
}

run_one DEC2 --tau-min 2 --tau-max 56 --weights "$WINNER_W"
run_one DEC4 --tau-min 4 --tau-max 56 --weights "$WINNER_W"
run_one DEC3_t49 --tau-min 3 --tau-max 49 --weights "$WINNER_W"
run_one DEC3_wfast --tau-min 3 --tau-max 56 --weights 0.10,0.20,0.25,0.45

echo "wave3 done"
for n in DEC2 DEC4 DEC3_t49 DEC3_wfast; do
  f="$LOGDIR/${n}.stdout"
  echo "$n 3150=$(score_3150 "$f") 3250=$(final_loss "$f")"
done
