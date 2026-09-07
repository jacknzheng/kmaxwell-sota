#!/bin/bash
set -uo pipefail
SEEDS="$1"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"; PY=/root/venv019/bin/python
$PY records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py --out configs/req054 >/dev/null 2>&1
STAT=/root/req054_status.tsv; echo -e "seed\tkernel\tbase_val2000\tfinal_val2750\thash16" > "$STAT"
for SEED in $SEEDS; do
  rm -rf eos_shared_state
  $TR records/track_3_optimization/run.py configs/req054/eos_shared_base.yaml seed=$SEED stop_after_step=2000 > /root/req054_base_s${SEED}.log 2>&1
  BV=$(grep -oE "val_loss:[0-9.]+" /root/req054_base_s${SEED}.log|tail -1|cut -d: -f2)
  H=$(sha256sum eos_shared_state/train_state_model_step002000.pt 2>/dev/null|cut -c1-16)
  $PY /root/make_req054_configs.py --out configs/req054 --seed $SEED > /root/req054_gen_s${SEED}.log 2>&1
  for k in kmax agema; do
    $TR records/track_3_optimization/run.py configs/req054/req054_s${SEED}_${k}.yaml > /root/req054_run_s${SEED}_${k}.log 2>&1
    FV=$(grep -oE "val_loss:[0-9.]+" /root/req054_run_s${SEED}_${k}.log|tail -1|cut -d: -f2)
    echo -e "${SEED}\t${k}\t${BV:-NA}\t${FV:-NA}\t${H:-NA}" >> "$STAT"
    echo "s$SEED $k base=$BV final=$FV"
  done
  touch /root/REQ054_SEED${SEED}_DONE
done
touch /root/REQ054_DONE; echo REQ054-DONE
