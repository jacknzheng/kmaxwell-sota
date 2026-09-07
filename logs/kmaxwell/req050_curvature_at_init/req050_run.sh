#!/bin/bash
set -uo pipefail
SEEDS="$1"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"
PY=/root/venv019/bin/python
# build REQ-050 base config: train 0->1500, checkpoint model every 125 (covers 0,125,250,500,1000,1500)
$PY - <<PYE
import yaml
c=yaml.safe_load(open("configs/req047/eos_shared_base.yaml"))
c["stop_after_step"]=1500
pre=[h for h in c.get("pre_optimizer",[]) if h["name"]!="checkpoint_model_at_cadence"]
pre.insert(0,{"name":"checkpoint_model_at_cadence","hyperparams":{"every":125,"dump_dir":"REQ050_DD"}})
c["pre_optimizer"]=pre
open("configs/req050_base.yaml","w").write(yaml.safe_dump(c,sort_keys=False))
print("cfg built")
PYE
STAT=/root/req050_status.tsv; [ -f "$STAT" ]||echo -e "seed\tstep\trep\tjson\tsecs" > "$STAT"
STEPS="0 125 250 500 1000 1500"
for SEED in $SEEDS; do
  dd=dumps_req050_s${SEED}; rm -rf $dd
  sed "s/REQ050_DD/${dd}/" configs/req050_base.yaml > configs/req050_s${SEED}.yaml
  $TR records/track_3_optimization/run.py configs/req050_s${SEED}.yaml seed=$SEED > /root/req050_train_s${SEED}.log 2>&1
  echo "seed $SEED trained; ckpts: $(ls $dd/model_step*.pt 2>/dev/null|wc -l)"
  ls $dd/model_step*.pt 2>/dev/null | sed 's/.*model_step0*//;s/\.pt//' | tr '\n' ' '; echo
  for STEP in $STEPS; do
    for REP in 0 1 2; do
      t0=$(date +%s)
      $TR records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py \
        --dump_dir $dd --steps $STEP --out_tag req050_s${SEED}_step$(printf %06d $STEP)_rep${REP} --probe_rep $REP --iters 8 \
        > /root/req050_probe_s${SEED}_${STEP}_${REP}.log 2>&1
      secs=$(( $(date +%s)-t0 ))
      J=$([ -f $dd/req050_s${SEED}_step$(printf %06d $STEP)_rep${REP}.json ] && echo Y || echo N)
      echo -e "${SEED}\t${STEP}\t${REP}\t${J}\t${secs}" >> "$STAT"
      echo "seed$SEED step$STEP rep$REP json=$J ${secs}s"
    done
  done
  touch /root/REQ050_SEED${SEED}_DONE
done
touch /root/REQ050_DONE
echo "REQ050-DONE seeds=$SEEDS"
