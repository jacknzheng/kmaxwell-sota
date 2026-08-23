#!/usr/bin/env bash
# Wave 5B — denser age grid around a38. Same family, seed 0, 8-GPU, start=1000.
# a38: 3150=3.28020  3250=3.27398  (a35 was 3.28036 / 3.27407; a32 killed at 3.28132)
# Slope is older. a40 is now legal. Skip leftover age-35 tmin2/tmax64.
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

echo "==== wave5 B around a38  $(date -u +%F\ %T) ====" | tee "$LOGDIR/DECISION.txt"
echo "age a38 3150=3.28020 3250=3.27398; skip tmin2/tmax64; denser than a36/a39" | tee -a "$LOGDIR/DECISION.txt"

# Neighbors on the same [3,56] family. a36/a37 sit between a35 and a38;
# a39/a40 walk the slope older now that a32 died and a38 beat 3.28020.
run_one K6_a36 --k 6 --tau-min 3 --tau-max 56 --weights 0.03498,0.06995,0.10493,0.13991,0.17489,0.47534
run_one K6_a37 --k 6 --tau-min 3 --tau-max 56 --weights 0.03323,0.06646,0.09969,0.13291,0.16614,0.50157
run_one K6_a39 --k 6 --tau-min 3 --tau-max 56 --weights 0.02973,0.05946,0.08919,0.11892,0.14865,0.55404
run_one K6_a40 --k 6 --tau-min 3 --tau-max 56 --weights 0.02798,0.05596,0.08395,0.11193,0.13991,0.58027

# Held-age mix at 38 (ws50-style; younger mix already lost at age 35).
run_one K6_a38_ws50 --k 6 --tau-min 3 --tau-max 56 --weights 0.02802,0.05605,0.08407,0.11209,0.21977,0.50000

# One tau follow-up at age 38. A never scored tmin2/tmax64, so use the in-between.
run_one K6_a38_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.03065,0.06130,0.09196,0.12261,0.15326,0.54022

echo "wave5 B a38 neighborhood done"
echo "ref   K6_a35 3150=3.28036 3250=3.27407"
echo "ref   K6_a38 3150=3.28020 3250=3.27398"
for n in K6_a36 K6_a37 K6_a39 K6_a40 K6_a38_ws50 K6_a38_tmin25; do
  f="$LOGDIR/${n}.stdout"
  echo "$n  3150=$(score_3150 "$f")  3250=$(final_loss "$f")"
done
