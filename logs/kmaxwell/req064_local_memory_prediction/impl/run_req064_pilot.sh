#!/bin/bash
# REQ-064 PILOT phase (<=4 node-hours): seed-0 base -> dump@2000; pretreatment feature probe (18 sentinels,
# FW + Lanczos budget gates); 15 pilot continuations (block-6 sentinels x2 + 3 globals), 256 updates.
# Runs torchrun from REPO ROOT (data/config globs are cwd-relative). Restore fix applied so a-treatments hold.
set -uo pipefail
V=/root/venv019/bin
ROOT=/root/kmaxwell-sota
export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req064 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT
RP=records/track_3_optimization
CFG=$RP/configs/req064
FEAT=logs/kmaxwell/req064_local_memory_prediction/raw/feat_s0
LOSS=logs/kmaxwell/req064_local_memory_prediction/raw/full064
mkdir -p $FEAT $LOSS

log "apply patches: REQ-058 + REQ-059 + REQ-063 restore fix"
$V/python /root/impl/apply_req058_opt.py 2>&1 | tail -1
$V/python /root/impl/apply_req059_opt.py 2>&1 | tail -1
$V/python /root/impl/apply_req063_restorefix.py 2>&1 | tail -1

cp /root/impl/make_req058_configs.py /root/impl/make_req064_configs.py $ROOT/
log "generate pilot configs (seed 0, 15 arms) -> $CFG"
$V/python make_req064_configs.py --out $CFG --seed 0 --pilot 2>&1 | tail -2

tr(){ $V/torchrun --standalone --nproc_per_node=8 "$@"; }

if [ -d req064_state_s0 ] && [ "$(ls req064_state_s0 2>/dev/null | wc -l)" -ge 9 ]; then
  log "=== BASE seed 0: reusing existing dump ($(ls req064_state_s0 | wc -l) files) ==="
else
  log "=== BASE seed 0: run to 2000, dump req064_state_s0 ==="
  tr $RP/run.py $CFG/base_s0.yaml > /root/req064_base_s0.log 2>&1 || true
  [ -d req064_state_s0 ] && log "BASE_DUMP_OK ($(ls req064_state_s0 | wc -l) files)" || { log "BASE_DUMP_MISSING"; tail -20 /root/req064_base_s0.log; exit 1; }
fi

log "=== FEATURE PROBE seed 0 (18 sentinels: S_i/G_i FW + Euclidean Lanczos) ==="
tr /root/impl/measure_req064_features.py --state_dir req064_state_s0 --step 2000 --pilot \
   --iters_list 20 50 --restarts 5 --lanczos_iters 8 --out_dir $FEAT > /root/req064_feat_s0.log 2>&1 || true
FJSON=$(ls $FEAT/features_*.json 2>/dev/null | head -1)
[ -n "$FJSON" ] && log "FEATURE_OK -> $FJSON" || { log "FEATURE_MISSING"; tail -25 /root/req064_feat_s0.log; }

log "=== 15 PILOT continuations (256 updates each) ==="
ARMS=$(ls $CFG/*_s0.yaml | xargs -n1 basename | grep -v '^base_' | sed 's/_s0.yaml//')
for arm in $ARMS; do
  tr $RP/run.py $CFG/${arm}_s0.yaml > /root/req064_${arm}_s0.log 2>&1 || true
  vl=$(grep -oE 'step:2256[^ ]* val_loss:[0-9.]+' /root/req064_${arm}_s0.log | tail -1)
  echo "  $arm: $vl"
  # harvest step->val_loss into raw tsv (text only)
  grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' /root/req064_${arm}_s0.log | sed -E 's#step:([0-9]+)/[0-9]+ val_loss:([0-9.]+)#\1\t\2#' > $LOSS/${arm}_s0.tsv
done

log "=== SUMMARY: step:2256 val_loss per pilot arm ==="
for arm in $ARMS; do echo "  $arm: $(grep -oE 'step:2256[^ ]* val_loss:[0-9.]+' /root/req064_${arm}_s0.log | tail -1)"; done
log "=== budget-gate hints from feature log (Lanczos convergence + FW) ==="
grep -iE "resid|lambda_i|converged" /root/req064_feat_s0.log | tail -6
echo REQ064_PILOT_DONE
