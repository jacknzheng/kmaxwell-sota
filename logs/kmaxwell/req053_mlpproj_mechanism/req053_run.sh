#!/bin/bash
set -uo pipefail
WIDTH="$1"; SEEDS="$2"   # WIDTH in {2,8}
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache_w${WIDTH}
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"
PY=/root/venv019/bin/python
# patch MLP expansion width: hdim = 4*dim -> WIDTH*dim
$PY - <<PYE
import re
p="records/track_3_optimization/harness/model_gpt.py"; s=open(p).read()
s2=re.sub(r"hdim = \d+ \* dim", f"hdim = ${WIDTH} * dim", s)
open(p,"w").write(s2); print("width patched:", "hdim = ${WIDTH} * dim" in s2)
PYE
# regenerate eos + arm configs (sorted order reflects the new width) into configs/req053_w${WIDTH}
$PY records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py --out configs/req053_w${WIDTH} >/dev/null 2>&1
STAT=/root/req053_w${WIDTH}_status.tsv; echo -e "width\tseed\tarm\tcurv_json" > "$STAT"
PROV=/root/req053_w${WIDTH}_prov.tsv; echo -e "width\tseed\tbase_val2000\thash16" > "$PROV"
for SEED in $SEEDS; do
  rm -rf eos_shared_state
  $TR records/track_3_optimization/run.py configs/req053_w${WIDTH}/eos_shared_base.yaml seed=$SEED stop_after_step=2000 > /root/req053_w${WIDTH}_base_s${SEED}.log 2>&1
  BV=$(grep -oE "val_loss:[0-9.]+" /root/req053_w${WIDTH}_base_s${SEED}.log|tail -1|cut -d: -f2)
  H=$(sha256sum eos_shared_state/train_state_model_step002000.pt 2>/dev/null|cut -c1-16)
  echo -e "${WIDTH}\t${SEED}\t${BV:-NA}\t${H:-NA}" >> "$PROV"
  $PY /root/make_req051_arms.py --out configs/req053_w${WIDTH} --seed $SEED > /root/req053_w${WIDTH}_gen_s${SEED}.log 2>&1
  # rename generated arm configs to req053 namespace (make_req051 wrote req051_s{seed}_arm{a})
  for arm in 0 1 2 3 4 5; do
    dd=dumps_req053_w${WIDTH}_s${SEED}_arm${arm}; rm -rf $dd
    cfg=configs/req053_w${WIDTH}/req051_s${SEED}_arm${arm}.yaml
    # point its checkpoint dump_dir at $dd
    $PY - <<PYE2
import yaml
c=yaml.safe_load(open("$cfg"))
for h in c.get("pre_optimizer",[]):
    if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["dump_dir"]="$dd"; h["hyperparams"]["every"]=250
c["run_id"]="req053_w${WIDTH}_s${SEED}_arm${arm}"
open("$cfg","w").write(yaml.safe_dump(c,sort_keys=False))
PYE2
    $TR records/track_3_optimization/run.py $cfg > /root/req053_w${WIDTH}_fork_s${SEED}_a${arm}.log 2>&1
    # keep only model@2750 (every=750 from 0 -> fires 2250? no. use prune: keep 2750)
    for f in $dd/model_step*.pt; do case "$f" in *002750.pt) ;; *) rm -f "$f";; esac; done
    $TR records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py \
      --dump_dir $dd --steps 2750 --out_tag req053_w${WIDTH}_s${SEED}_arm${arm}_curv --iters 8 --tokens 32768 \
      > /root/req053_w${WIDTH}_curv_s${SEED}_a${arm}.log 2>&1
    cj=$([ -f $dd/req053_w${WIDTH}_s${SEED}_arm${arm}_curv.json ]&&echo Y||echo N)
    echo -e "${WIDTH}\t${SEED}\t${arm}\t${cj}" >> "$STAT"
    echo "w${WIDTH} s${SEED} arm${arm} curv=$cj"
  done
done
# restore model to 4x
$PY -c "import re;p='records/track_3_optimization/harness/model_gpt.py';s=open(p).read();open(p,'w').write(re.sub(r'hdim = \d+ \* dim','hdim = 4 * dim',s))"
touch /root/REQ053_W${WIDTH}_DONE; echo "REQ053-W${WIDTH}-DONE"
