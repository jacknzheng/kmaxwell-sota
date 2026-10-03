#!/bin/bash
# REQ-069: 10 data-seed branch fork-captures from the retained theta_1500 + ensemble analysis (both streams).
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req069 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT; RP=records/track_3_optimization
FORKS=/root/.cache/user_artifacts/req068_s0/forks
DUR=/root/.cache/user_artifacts/req069
log "check theta_1500 fork reachable"
[ -f $FORKS/train_state_model_step001500.pt ] || { log "FORK theta_1500 MISSING - abort"; exit 1; }

log "stage impl + apply patches"
mkdir -p logs/river/req068_full_gradient_history/impl logs/river/req069_ten_branch_ensemble/impl
cp /root/impl/req068_capture.py /root/impl/req068_gram.py logs/river/req068_full_gradient_history/impl/
cp /root/impl/req069_ensemble.py logs/river/req069_ten_branch_ensemble/impl/
$V/python /root/impl/apply_req068_capture.py 2>&1 | tail -1
$V/python /root/impl/apply_req068_displacement.py 2>&1 | tail -1
cp /root/impl/make_req068_config.py /root/impl/make_req069_branch_config.py $ROOT/

tr(){ $V/torchrun --standalone --nproc_per_node=8 $RP/run.py "$@"; }
for b in 0 1 2 3 4 5 6 7 8 9; do
  BR=$DUR/br$b
  log "=== BRANCH $b (fork 1500, data window $((1500+b*64))..$((1500+b*64+63))) ==="
  $V/python make_req069_branch_config.py --out $RP/configs/req069_br$b.yaml --branch $b --fork 1500 \
     --forks_dir $FORKS --grad_dir $BR/grads --disp_dir $BR/disp --probe_dir $BR/probes 2>&1 | tail -1
  tr $RP/configs/req069_br$b.yaml > /root/req069_br$b.log 2>&1 || true
  echo "  br$b: grad=$(ls $BR/grads/grad_chunk_*.pt 2>/dev/null|wc -l) disp=$(ls $BR/disp/grad_chunk_*.pt 2>/dev/null|wc -l) probes=$(ls $BR/probes/*.pt 2>/dev/null|wc -l)"
  grep -iE 'resumed training|Error:|Traceback|Root Cause' /root/req069_br$b.log 2>/dev/null|tail -1
done

log "=== ENSEMBLE ANALYSIS (both streams) ==="
mkdir -p $DUR/analysis
for S in grad disp; do
  $V/python /root/impl/req069_ensemble_analyze.py --branch_root $DUR --stream $S --window 1500 \
     --tokens 524288 --out $DUR/analysis/ensemble_$S.json 2>&1 | tail -2
done
echo REQ069_DONE
