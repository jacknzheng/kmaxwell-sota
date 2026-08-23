#!/usr/bin/env bash
# Wave 4 on 8xH100: sequential. Logs: this dir (tee) + trainer uuid files.
# Interiors at frozen DEC mix / age 35, plus one k=6 hail mary. See PLAN.txt.
set -euo pipefail
ROOT="/workspace/modded-nanogpt"
TRAIN="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave4"
DEC_W="0.05,0.20,0.25,0.50"
K6_W="0.03673,0.07345,0.11018,0.14690,0.18363,0.44911"
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

run_one INT_spread --k 4 --taus 3,4.5,23.8,56 --weights "$DEC_W"
run_one INT_mid    --k 4 --taus 3,11,18.6,56  --weights "$DEC_W"
run_one INT_bunch  --k 4 --taus 3,13.5,16.6,56 --weights "$DEC_W"
run_one K6_a35     --k 6 --tau-min 3 --tau-max 56 --weights "$K6_W"

echo "wave4 done"
echo "DEC ref  3150=3.28040 3250=3.27415"
for n in INT_spread INT_mid INT_bunch K6_a35; do
  f="$LOGDIR/${n}.stdout"
  echo "$n 3150=$(score_3150 "$f") 3250=$(final_loss "$f")"
done
