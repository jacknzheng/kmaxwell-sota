#!/bin/bash
# REQ-075 Stage 3: equal-token 4-config x 3-seed (capture) + LR search (no capture). Assumes patches+configs ready.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; cd $ROOT
export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT/records/track_3_optimization:/root/impl
DUR=/root/.cache/user_artifacts/req075
log(){ echo "[s3 $(date -u +%H:%M:%S)] $*"; }
tr(){ $V/torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py "$@"; }
runcfg(){  # $1=config path, $2=outdir tag, $3=logname
  local cfg=$1 tag=$2 lg=$3
  rm -rf $DUR/stage3_runs/$tag 2>/dev/null
  tr $cfg > /root/s3_$lg.log 2>&1 || log "$lg EXIT $?"
  local v0 vN
  v0=$(grep -oE 'step:0/[0-9]+ val_loss:[0-9.]+' /root/s3_$lg.log | head -1 | grep -oE '[0-9.]+$')
  vN=$(grep -oE 'val_loss:[0-9.]+' /root/s3_$lg.log | tail -1 | cut -d: -f2)
  log "$lg done: val $v0 -> $vN"
  grep -iE "Error:|Traceback|Root Cause|OutOfMemory" /root/s3_$lg.log | head -1
}
log "=== MAIN 3-SEED (capture) ==="
for cfg in B-mom B-nomom 16B-mom 16B-nomom; do
  for s in 0 1 2; do runcfg $DUR/configs_s3_main/${cfg}_s${s}.yaml "${cfg}_s${s}" "main_${cfg}_s${s}"; done
done
log "=== LR SEARCH (tuning seed 0, no capture) ==="
for f in $DUR/configs_s3_lr/*.yaml; do
  nm=$(basename $f .yaml); runcfg $f "lr_$nm" "lr_$nm"
done
log "=== STAGE 3 ALL DONE ==="
echo S3_RC=0 > /root/s3.rc
