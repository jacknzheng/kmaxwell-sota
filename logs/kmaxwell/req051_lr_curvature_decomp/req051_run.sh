#!/bin/bash
set -uo pipefail
SEEDS="$1"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"
PY=/root/venv019/bin/python
PROV=/root/req051_provenance.tsv; [ -f "$PROV" ]||echo -e "seed\tbase_val2000\tstate_hash16" > "$PROV"
STAT=/root/req051_status.tsv; [ -f "$STAT" ]||echo -e "seed\tarm\tstep\tcurv_json\tact_json" > "$STAT"
for SEED in $SEEDS; do
  rm -rf eos_shared_state
  $TR records/track_3_optimization/run.py configs/req051/eos_shared_base.yaml seed=$SEED stop_after_step=2000 > /root/req051_base_s${SEED}.log 2>&1
  BV=$(grep -oE "val_loss:[0-9.]+" /root/req051_base_s${SEED}.log|tail -1|cut -d: -f2)
  H=$(sha256sum eos_shared_state/train_state_model_step002000.pt 2>/dev/null|cut -c1-16)
  echo -e "${SEED}\t${BV:-NA}\t${H:-NA}" >> "$PROV"
  $PY /root/make_req051_arms.py --out configs/req051 --seed $SEED > /root/req051_gen_s${SEED}.log 2>&1
  for arm in 0 1 2 3 4 5; do
    dd=dumps_req051_s${SEED}_arm${arm}; rm -rf $dd
    $TR records/track_3_optimization/run.py configs/req051/req051_s${SEED}_arm${arm}.yaml > /root/req051_fork_s${SEED}_a${arm}.log 2>&1
    # prune model checkpoints: keep only 2050 & 2750
    for f in $dd/model_step*.pt; do case "$f" in *002050.pt|*002750.pt) ;; *) rm -f "$f";; esac; done
    for step in 2050 2750; do
      $TR records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py \
        --dump_dir $dd --steps $step --out_tag req051_s${SEED}_arm${arm}_step${step}_curv --iters 8 --tokens 32768 \
        > /root/req051_curv_s${SEED}_a${arm}_${step}.log 2>&1
      export TORCHDYNAMO_DISABLE=1
      CUDA_VISIBLE_DEVICES=0 $PY /root/measure_activation_backward_v2.py \
        --model $dd/model_step$(printf %06d $step).pt --out $dd/req051_s${SEED}_arm${arm}_step${step}_act.json --tokens 32768 \
        > /root/req051_act_s${SEED}_a${arm}_${step}.log 2>&1
      unset TORCHDYNAMO_DISABLE
      cj=$([ -f $dd/req051_s${SEED}_arm${arm}_step${step}_curv.json ]&&echo Y||echo N)
      aj=$([ -f $dd/req051_s${SEED}_arm${arm}_step${step}_act.json ]&&echo Y||echo N)
      echo -e "${SEED}\t${arm}\t${step}\t${cj}\t${aj}" >> "$STAT"
      echo "s$SEED arm$arm step$step curv=$cj act=$aj"
    done
  done
  touch /root/REQ051_SEED${SEED}_DONE
done
touch /root/REQ051_DONE; echo "REQ051-DONE seeds=$SEEDS"
