#!/bin/bash
# REQ-075 Stage 2: 5-arm 256-update mechanism pilot (full). Assumes patches applied + configs generated.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; cd $ROOT
export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT/records/track_3_optimization:/root/impl
DUR=/root/.cache/user_artifacts/req075
log(){ echo "[s2 $(date -u +%H:%M:%S)] $*"; }
tr(){ $V/torchrun --standalone --nproc_per_node=8 records/track_3_optimization/run.py "$@"; }
for arm in B-mom B-nomom 16B-mom 16B-nomom 16B-nomom-halfLR; do
  log "=== ARM $arm ==="
  rm -rf $DUR/stage2/$arm
  tr $DUR/configs_s2/$arm.yaml > /root/s2_$arm.log 2>&1 || log "$arm EXIT $?"
  v0=$(grep -oE 'step:2500/[0-9]+ val_loss:[0-9.]+' /root/s2_$arm.log | head -1 | grep -oE '[0-9.]+$')
  vN=$(grep -oE 'val_loss:[0-9.]+' /root/s2_$arm.log | tail -1 | cut -d: -f2)
  log "$arm done: val 2500->end $v0 -> $vN  stages=$(for s in s1_grad s2_premom s3_postpolar s4_disp; do echo -n "$(ls $DUR/stage2/$arm/$s/*.pt 2>/dev/null|wc -l) "; done)"
  grep -iE "Error:|Traceback|Root Cause|OutOfMemory" /root/s2_$arm.log | head -2
done
log "=== STAGE 2 ALL ARMS DONE ==="
echo S2_RC=0 > /root/s2.rc
