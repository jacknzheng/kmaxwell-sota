#!/bin/bash
set -uo pipefail
SEEDS="$1"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache_gelu
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"; PY=/root/venv019/bin/python
# patch nonlinearity: ReLU^2 -> GELU (width stays 4x)
$PY - <<PYE
p="records/track_3_optimization/harness/model_gpt.py"; s=open(p).read()
s2=s.replace("x = x.relu().square()","x = F.gelu(x)")
open(p,"w").write(s2); print("gelu patched:", "x = F.gelu(x)" in s2)
PYE
$PY records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py --out configs/req053_gelu >/dev/null 2>&1
STAT=/root/req053_gelu_status.tsv; echo -e "seed\tarm\tcurv_json" > "$STAT"
PROV=/root/req053_gelu_prov.tsv; echo -e "seed\tbase_val2000\thash16" > "$PROV"
for SEED in $SEEDS; do
  rm -rf eos_shared_state
  $TR records/track_3_optimization/run.py configs/req053_gelu/eos_shared_base.yaml seed=$SEED stop_after_step=2000 > /root/req053_gelu_base_s${SEED}.log 2>&1
  BV=$(grep -oE "val_loss:[0-9.]+" /root/req053_gelu_base_s${SEED}.log|tail -1|cut -d: -f2)
  H=$(sha256sum eos_shared_state/train_state_model_step002000.pt 2>/dev/null|cut -c1-16)
  echo -e "${SEED}\t${BV:-NA}\t${H:-NA}" >> "$PROV"
  $PY /root/make_req051_arms.py --out configs/req053_gelu --seed $SEED > /root/req053_gelu_gen_s${SEED}.log 2>&1
  for arm in 0 1 2 3 4 5; do
    dd=dumps_req053_gelu_s${SEED}_arm${arm}; rm -rf $dd
    cfg=configs/req053_gelu/req051_s${SEED}_arm${arm}.yaml
    $PY - <<PYE2
import yaml
c=yaml.safe_load(open("$cfg"))
for h in c.get("pre_optimizer",[]):
    if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["dump_dir"]="$dd"; h["hyperparams"]["every"]=250
open("$cfg","w").write(yaml.safe_dump(c,sort_keys=False))
PYE2
    $TR records/track_3_optimization/run.py $cfg > /root/req053_gelu_fork_s${SEED}_a${arm}.log 2>&1
    for f in $dd/model_step*.pt; do case "$f" in *002750.pt) ;; *) rm -f "$f";; esac; done
    $TR records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py \
      --dump_dir $dd --steps 2750 --out_tag req053_gelu_s${SEED}_arm${arm}_curv --iters 8 --tokens 32768 \
      > /root/req053_gelu_curv_s${SEED}_a${arm}.log 2>&1
    echo -e "${SEED}\t${arm}\t$([ -f $dd/req053_gelu_s${SEED}_arm${arm}_curv.json ]&&echo Y||echo N)" >> "$STAT"
  done
done
$PY -c "p='records/track_3_optimization/harness/model_gpt.py';s=open(p).read();open(p,'w').write(s.replace('x = F.gelu(x)','x = x.relu().square()'))"
touch /root/REQ053_GELU_DONE; echo GELU-DONE
