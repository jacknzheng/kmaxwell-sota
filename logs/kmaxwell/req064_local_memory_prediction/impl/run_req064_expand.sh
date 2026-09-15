#!/bin/bash
# REQ-064 EXPANSION (<=24 node-hours): 39 arms x 4 bases = 156 continuations. Dev seeds 0,1; test 7,8.
# Idempotent: skips existing base dumps, feature JSONs, and arm logs (seed-0 pilot arms are reused).
# Runs torchrun from repo root; patches (058/059/063 restore fix) re-applied idempotently.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req064x $(date -u +%H:%M:%S)] $*"; }
cd $ROOT
RP=records/track_3_optimization
FEATBASE=logs/kmaxwell/req064_local_memory_prediction/raw
LOSS=logs/kmaxwell/req064_local_memory_prediction/raw/full064
mkdir -p $LOSS
tr(){ $V/torchrun --standalone --nproc_per_node=8 "$@"; }

log "re-apply patches (idempotent)"
$V/python /root/impl/apply_req058_opt.py >/dev/null 2>&1
$V/python /root/impl/apply_req059_opt.py >/dev/null 2>&1
$V/python /root/impl/apply_req063_restorefix.py >/dev/null 2>&1
cp /root/impl/make_req058_configs.py /root/impl/make_req064_configs.py $ROOT/ 2>/dev/null

for seed in 0 1 7 8; do
  CFG=$RP/configs/req064_s${seed}
  log "=== SEED $seed ==="
  $V/python make_req064_configs.py --out $CFG --seed $seed >/dev/null 2>&1  # full 39 arms
  # base dump
  if [ -d req064_state_s${seed} ] && [ "$(ls req064_state_s${seed} 2>/dev/null|wc -l)" -ge 9 ]; then
    log "base s$seed: reuse dump"
  else
    log "base s$seed: run to 2000"
    tr $RP/run.py $CFG/base_s${seed}.yaml > /root/req064x_base_s${seed}.log 2>&1 || true
    [ -d req064_state_s${seed} ] || { log "BASE s$seed MISSING"; tail -15 /root/req064x_base_s${seed}.log; continue; }
  fi
  # feature probe (18 sentinels)
  FEAT=$FEATBASE/feat_s${seed}; mkdir -p $FEAT
  if ls $FEAT/features_*.json >/dev/null 2>&1; then
    log "features s$seed: reuse"
  else
    log "features s$seed: probe 18 sentinels"
    tr /root/impl/measure_req064_features.py --state_dir req064_state_s${seed} --step 2000 --pilot \
       --iters_list 20 50 --restarts 5 --lanczos_iters 8 --out_dir $FEAT > /root/req064x_feat_s${seed}.log 2>&1 || true
    ls $FEAT/features_*.json >/dev/null 2>&1 && log "features s$seed OK" || log "features s$seed FAILED"
  fi
  # 39 arms
  ARMS=$(ls $CFG/*_s${seed}.yaml | xargs -n1 basename | grep -v '^base_' | sed "s/_s${seed}.yaml//")
  for arm in $ARMS; do
    # reuse seed-0 pilot arm logs (block 6 + globals) already harvested
    if [ -f $LOSS/${arm}_s${seed}.tsv ] && grep -q "^2256" $LOSS/${arm}_s${seed}.tsv 2>/dev/null; then continue; fi
    # also reuse the pilot run's raw logs for seed 0 if present
    SRC=/root/req064_${arm}_s${seed}.log
    if [ "$seed" = "0" ] && [ -f "$SRC" ] && grep -q "step:2256" "$SRC" 2>/dev/null; then
      grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' "$SRC" | sed -E 's#step:([0-9]+)/[0-9]+ val_loss:([0-9.]+)#\1\t\2#' > $LOSS/${arm}_s${seed}.tsv
      continue
    fi
    tr $RP/run.py $CFG/${arm}_s${seed}.yaml > /root/req064x_${arm}_s${seed}.log 2>&1 || true
    grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' /root/req064x_${arm}_s${seed}.log | sed -E 's#step:([0-9]+)/[0-9]+ val_loss:([0-9.]+)#\1\t\2#' > $LOSS/${arm}_s${seed}.tsv
    ep=$(grep -E "^2256" $LOSS/${arm}_s${seed}.tsv | cut -f2)
    echo "  s$seed $arm: 2256=$ep"
  done
done
log "=== harvest complete: $(ls $LOSS/*.tsv 2>/dev/null|wc -l) tsv files ==="
echo REQ064_EXPAND_DONE
