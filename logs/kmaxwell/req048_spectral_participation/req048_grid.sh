#!/bin/bash
set -uo pipefail
SEEDS="$1"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"
STAT=/root/req048_grid_status.tsv; [ -f "$STAT" ]||echo -e "seed\tsfork\tbase_val\tcurv_json" > "$STAT"
for SEED in $SEEDS; do
  rm -rf eos_shared_state
  $TR records/track_3_optimization/run.py configs/req047/eos_shared_base.yaml seed=$SEED stop_after_step=1500 > /root/req048_base_s${SEED}.log 2>&1
  BV=$(grep -oE "val_loss:[0-9.]+" /root/req048_base_s${SEED}.log|tail -1)
  for SF in s060 s100 s170; do
    dd=dumps_eos_f1500_${SF}_s${SEED}; rm -rf $dd
    $TR records/track_3_optimization/run.py configs/req047/eos_f1500_${SF}.yaml seed=$SEED > /root/req048_fork_s${SEED}_${SF}.log 2>&1
    # the fork config dumps to dumps_eos_f1500_<SF>; rename per-seed to avoid clobber
    mv dumps_eos_f1500_${SF} $dd 2>/dev/null || true
    $TR records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py --dump_dir $dd --steps 2750 --out_tag req048_s${SEED}_${SF} --iters 8 > /root/req048_curv_s${SEED}_${SF}.log 2>&1
    J=$([ -f $dd/req048_s${SEED}_${SF}.json ] && echo Y || echo N)
    echo -e "${SEED}\t${SF}\t${BV}\t${J}" >> "$STAT"
    echo "seed $SEED $SF curv_json=$J"
  done
  touch /root/REQ048_SEED${SEED}_DONE
done
touch /root/REQ048_GRID_DONE
echo "GRID-DONE seeds=$SEEDS"
