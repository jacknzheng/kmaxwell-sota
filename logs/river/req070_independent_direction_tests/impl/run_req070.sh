#!/bin/bash
# REQ-070 loss probes: perturb REQ-069 probe states by candidate directions x scales, eval fresh loss.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req070 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT
DUR69=/root/.cache/user_artifacts/req069
OUT=/root/.cache/user_artifacts/req070; mkdir -p $OUT
log "check REQ-069 branch captures + probe states reachable"
[ -f $DUR69/br0/probes/train_state_model_step001532.pt ] || { log "PROBE STATES MISSING - abort"; ls $DUR69/br0/probes 2>/dev/null|head; exit 1; }
[ -f $DUR69/br0/grads/manifest.json ] || { log "BRANCH GRADS MISSING - abort"; exit 1; }
mkdir -p logs/river/req068_full_gradient_history/impl logs/river/req070_independent_direction_tests/impl
cp /root/impl/req068_capture.py logs/river/req068_full_gradient_history/impl/
cp /root/impl/req070_directions.py logs/river/req070_independent_direction_tests/impl/
log "run loss probe (single GPU)"
$V/torchrun --standalone --nproc_per_node=1 /root/impl/req070_loss_probe.py \
   --branch_root $DUR69 --probe_root $DUR69 --fork 1500 --offsets 32 48 63 \
   --val_data 'data/fineweb10B/fineweb_val_*.bin' --val_tokens 2097152 --out $OUT/loss_probe.json > /root/req070.out 2>&1 || true
log "=== SUMMARY ==="
grep -E 'offset|mean loss delta|WROTE|Error:|Traceback' /root/req070.out 2>/dev/null | tail -20
echo REQ070_DONE
