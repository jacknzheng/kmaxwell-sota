#!/bin/bash
# Idempotent box bootstrap for the kmaxwell campaign. Run BY PATH after scp
# (never heredoc-over-ssh). Expects /root/nanogpt.tgz and /root/km/ present.
set -euo pipefail
log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "python3 + venv tooling + dev headers (Triton/inductor JIT needs Python.h)"
command -v python3 >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq python3 python3-venv python3-pip; }
python3 -m venv /tmp/_venvprobe >/dev/null 2>&1 && rm -rf /tmp/_venvprobe || { apt-get update -qq; apt-get install -y -qq python3-venv; }
[ -f /usr/include/python3.12/Python.h ] || { apt-get update -qq; apt-get install -y -qq python3-dev build-essential; }

log "repo"
if [ ! -d /root/modded-nanogpt ]; then
  tar xzf /root/nanogpt.tgz -C /root
fi

log "venv (torch 2.10.0+cu128)"
[ -x /root/venv/bin/python ] || python3 -m venv /root/venv
/root/venv/bin/python -c "import torch; v=torch.__version__; assert v.startswith('2.10.0') and 'cu128' in v, v" 2>/dev/null \
  || /root/venv/bin/pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
/root/venv/bin/python -c "import numpy, huggingface_hub, tqdm" 2>/dev/null \
  || /root/venv/bin/pip install -q numpy huggingface_hub tqdm

log "anneal trainer into repo (pinned copy from /root/km)"
cp /root/km/train_gpt_kmaxwell_anneal.py \
   /root/modded-nanogpt/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/

log "fineweb10B data (20 train chunks + val)"
cd /root/modded-nanogpt && /root/venv/bin/python data/cached_fineweb10B.py 20

log "GPU count"
NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -eq 8 ] || { echo "FATAL: $NGPU GPUs, need 8"; exit 1; }

log "pin gate"
cd /root/km && python3 - <<'PYEOF'
import hashlib, json, sys
R = "/root/modded-nanogpt/records/track_3_optimization/results"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
pins   = json.load(open("/root/km/pins.json"))
pins_a = json.load(open("/root/km/pins_anneal.json"))
pins_w = json.load(open("/root/km/pins_wr.json"))
checks = [
  (f"{R}/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py",        pins["trainer_sha256"]),
  (f"{R}/20260715_bimaxwell_baseline_3210/kmaxwell_kernel.py",           pins["kernel_sha256"]),
  (f"{R}/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell_anneal.py", pins_a["anneal_trainer_sha256"]),
  (f"{R}/20260823_kmaxwell_wr_stack/train_gpt_cwd_kmaxwell.py",          pins_w["cwd_trainer_sha256"]),
  (f"{R}/20260823_kmaxwell_wr_stack/kmaxwell_kernel.py",                 pins_w["kernel_sha256"]),
  (f"{R}/20260713_bimaxwell_2635/train_gpt_bimaxwell_st1000.py",         pins_w["bm339_trainer_sha256"]),
]
bad = [(p, sha(p), want) for p, want in checks if sha(p) != want]
for p, got, want in bad: print(f"PIN MISMATCH {p}: {got} != {want}")
sys.exit(1 if bad else 0)
PYEOF

log "data sanity"
ls /root/modded-nanogpt/data/fineweb10B/fineweb_train_*.bin | wc -l
ls /root/modded-nanogpt/data/fineweb10B/fineweb_val_*.bin | wc -l

echo BOOTSTRAP-OK
