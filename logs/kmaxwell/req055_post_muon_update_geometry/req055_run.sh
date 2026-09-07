#!/bin/bash
set -uo pipefail
KERNEL="$1"; SEED="${2:-0}"
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"; PY=/root/venv019/bin/python
[ -f /root/apply_req054_opt.py ] && $PY /root/apply_req054_opt.py >/dev/null 2>&1 || true
$PY records/track_3_optimization/offline_analysis/make_eos_state_dependence_configs.py --out configs/req055 >/dev/null 2>&1
rm -rf eos_shared_state
$TR records/track_3_optimization/run.py configs/req055/eos_shared_base.yaml seed=$SEED stop_after_step=2000 > /root/req055_base_s${SEED}.log 2>&1
$PY /root/make_req054_configs.py --out configs/req055 --seed $SEED >/dev/null 2>&1
cfg=configs/req055/req054_s${SEED}_${KERNEL}.yaml; dd="dumps_req055_${KERNEL}_s${SEED}"
$PY - <<PYE
import yaml
c=yaml.safe_load(open("$cfg"))
pre=[h for h in c.get("pre_optimizer",[]) if h["name"]!="checkpoint_model_at_cadence"]
pre.insert(0,{"name":"checkpoint_model_at_cadence","hyperparams":{"every":100000,"dump_dir":"$dd","dense_windows":[[2050,2051],[2250,2251],[2500,2501],[2749,2750]]}})
c["pre_optimizer"]=pre; c["run_id"]="req055_${KERNEL}_s${SEED}"
open("$cfg","w").write(yaml.safe_dump(c,sort_keys=False)); print("dense model dumps ->",c["pre_optimizer"][0]["hyperparams"]["dense_windows"])
PYE
$TR records/track_3_optimization/run.py $cfg > /root/req055_run_${KERNEL}_s${SEED}.log 2>&1
echo "arm $KERNEL model dumps: $(ls $dd/model_step*.pt 2>/dev/null|sed 's#.*model_step0*##;s#.pt##'|tr '\n' ' ')"
touch /root/REQ055_${KERNEL}_s${SEED}_DONE
