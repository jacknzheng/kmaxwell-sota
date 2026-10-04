#!/bin/bash
# REQ-073: ensure 72-chunk corpus -> apply patch -> 4 arms (B/4B x momentum) -> 4-stage spectra.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
log(){ echo "[req073 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT; RP=records/track_3_optimization; DUR=/root/.cache/user_artifacts/req073; mkdir -p $DUR/configs $DUR/analysis
I68=logs/river/req068_full_gradient_history/impl; I72=logs/river/req072_optimizer_spectrograms/impl; I73=logs/river/req073_muon_batch_spectrograms/impl
mkdir -p $I68 $I72 $I73
cp /root/impl/req068_capture.py $I68/; cp /root/impl/req072_spectrogram.py $I72/
cp /root/impl/req073_stages.py /root/impl/req073_analyze.py $I73/

log "ensure >=72 fineweb train chunks (4B arm needs 6.82B tokens)"
have=$(find data/fineweb10B -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)
log "have $have train chunks"
if [ "$have" -lt 72 ]; then
  DS=data/cached_fineweb10B.py
  for a in 1 2 3; do $V/python data/cached_fineweb10B.py 72 >/root/req073_data.log 2>&1 && break; done
  log "after download: $(find data/fineweb10B -name 'fineweb_train_*.bin'|wc -l) chunks"
fi
[ "$(find data/fineweb10B -name 'fineweb_train_*.bin'|wc -l)" -ge 69 ] || { log "INSUFFICIENT CHUNKS - abort 4B"; exit 1; }

log "apply REQ-073 patch"
$V/python /root/impl/apply_req073_capture.py 2>&1 | tail -1
cp /root/impl/make_req073_configs.py $ROOT/
$V/python make_req073_configs.py --out_dir $DUR/configs --dur $DUR --muon_lr 0.025 2>&1 | tail -1

tr(){ $V/torchrun --standalone --nproc_per_node=8 $RP/run.py "$@"; }
for arm in B-mom B-nomom 4B-mom 4B-nomom; do
  log "=== ARM $arm ==="
  tr $DUR/configs/$arm.yaml > /root/req073_$arm.log 2>&1 || true
  echo "  $arm: s3_chunks=$(ls $DUR/$arm/s3_postpolar/grad_chunk_*.pt 2>/dev/null|wc -l) final_val=$(grep -oE 'val_loss:[0-9.]+' /root/req073_$arm.log|tail -1|cut -d: -f2)"
  grep -iE 'req073 capture tagged|Error:|Traceback|Root Cause' /root/req073_$arm.log 2>/dev/null|tail -2
done

log "=== STAGE SPECTRA ANALYSIS ==="
N="blocks.5.attn.q.weight blocks.5.attn.k.weight blocks.5.mlp.proj.weight"
for arm in B-mom B-nomom 4B-mom 4B-nomom; do
  bt=$([ "${arm:0:2}" = "4B" ] && echo 2097152 || echo 524288)
  log "analyze $arm (batch $bt)"
  $V/python /root/impl/req073_analyze.py --arm_dir $DUR/$arm --names $N --batch_tokens $bt --window 64 --hop 32 --out $DUR/analysis/$arm.json 2>&1 | tail -5
done
log "=== SUMMARY: period-two fraction per stage, blocks.5.attn.q, per arm ==="
for arm in B-mom B-nomom 4B-mom 4B-nomom; do
  $V/python - "$DUR/analysis/$arm.json" "$arm" <<'PY' 2>&1 | grep -v record_context
import json,sys
try:
    d=json.load(open(sys.argv[1])); n='blocks.5.attn.q.weight'
    row=' '.join(f"{s}={d['stages'][s][n]['band']['period2_frac']:.3f}" for s in ['s1_grad','s2_premom','s3_postpolar','s4_disp'] if s in d['stages'])
    print(f"  {sys.argv[2]:9s} {row}")
except Exception as e: print(f"  {sys.argv[2]}: {e}")
PY
done
echo REQ073_DONE
