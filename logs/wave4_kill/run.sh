#!/usr/bin/env bash
# Freeze K6_a35. Stack on Muon #36 / SOAP #46 / Aurora #42 / AdamH #4.
#
#   logs/wave4_kill/run.sh probe          # 8-GPU seed-0 of #46 (do this first)
#   logs/wave4_kill/run.sh trees          # 8-GPU seed-0 Aurora, AdamH, muon
#   logs/wave4_kill/run.sh n8 soap|aurora|muon|adam
#   logs/wave4_kill/run.sh one soap 0     # one named job
set -euo pipefail
ROOT="${ROOT:-/workspace/modded-nanogpt}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
STACK="$ROOT/records/track_3_optimization/results/20260823_kmaxwell_wr_stack"
MUON="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave4_kill"
K6=(--k 6 --tau-min 3 --tau-max 56
    --weights 0.03673,0.07345,0.11018,0.14690,0.18363,0.44911 --start 1000)
mkdir -p "$LOGDIR"
cd "$ROOT"

nproc_all() { nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '; }

script_for() {
  case "$1" in
    soap)   echo "$STACK/train_gpt_cwd_kmaxwell.py" ;;
    aurora) echo "$STACK/train_gpt_aurora_kmaxwell.py" ;;
    adam)   echo "$STACK/train_gpt_adamh_kmaxwell.py" ;;
    muon)   echo "$MUON" ;;
    *) echo "unknown stack: $1" >&2; return 1 ;;
  esac
}

run_job() {
  local name="$1" nproc="$2" seed="$3"
  shift 3
  local script="$1"; shift
  local tee="$LOGDIR/${name}.stdout"
  local port=$((29511 + (seed % 50) + nproc))
  echo "==== $(date -u +%F\ %T) $name nproc=$nproc seed=$seed $* ====" | tee "$tee"
  torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" -- \
    "$script" --seed "$seed" "${K6[@]}" "$@" \
    2>&1 | tee -a "$tee"
  local rc=${PIPESTATUS[0]}
  echo "==== $(date -u +%F\ %T) $name exit=$rc ====" | tee -a "$tee"
  return "$rc"
}

mode="${1:-probe}"
shift || true

case "$mode" in
  probe)
    # Kill shot. Compare val_ema at 2690 to #46 seed-0 = 3.27795.
    STOP_STEP="${STOP_STEP:-2720}" run_job SOAP_s0 "$(nproc_all)" 0 "$(script_for soap)"
    ;;
  trees)
    # Remaining per-optimizer seed-0s, sequential 8-GPU. Skip if probe already won.
    n=$(nproc_all)
    run_job AURORA_s0 "$n" 0 "$(script_for aurora)"
    run_job ADAM_s0 "$n" 0 "$(script_for adam)"
    run_job MUON_s0 "$n" 0 "$(script_for muon)"
    ;;
  n8)
    stack="${1:?n8 needs soap|aurora|muon|adam}"
    script="$(script_for "$stack")"
    # 8 independent 1-GPU jobs. SOAP/Aurora may take ~1.5-2h on 1xH100.
    pids=()
    for seed in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES="$seed" \
        run_job "${stack}_s${seed}" 1 "$seed" "$script" &
      pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do
      wait "$pid" || rc=1
    done
    exit "$rc"
    ;;
  one)
    stack="${1:?one needs stack}"
    seed="${2:-0}"
    n="${3:-$(nproc_all)}"
    run_job "${stack}_s${seed}" "$n" "$seed" "$(script_for "$stack")"
    ;;
  *)
    echo "usage: $0 {probe|trees|n8 soap|one soap 0}" >&2
    exit 2
    ;;
esac
