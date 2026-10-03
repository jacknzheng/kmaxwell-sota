#!/bin/bash
# REQ-072: LR stability pilot -> freeze LRs -> 6 capture runs (3250 steps) -> STFT spectrogram analysis.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req072 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT; RP=records/track_3_optimization
DUR=/root/.cache/user_artifacts/req072; mkdir -p $DUR/configs $DUR/analysis
I68=logs/river/req068_full_gradient_history/impl; I72=logs/river/req072_optimizer_spectrograms/impl
mkdir -p $I68 $I72
cp /root/impl/req068_capture.py $I68/
cp /root/impl/req072_capture_adamw.py /root/impl/req072_spectrogram.py /root/impl/req072_analyze.py $I72/
log "apply REQ-072 patch (registers capture_adamw + hooks)"
$V/python /root/impl/apply_req072_capture.py 2>&1 | tail -2
cp /root/impl/make_req072_configs.py $ROOT/

tr(){ $V/torchrun --standalone --nproc_per_node=8 $RP/run.py "$@"; }
valloss(){ grep -oE 'val_loss:[0-9.]+' "$1" 2>/dev/null | tail -1 | cut -d: -f2; }

# ---- LR stability pilot (150 steps, no capture) ----
log "=== LR PILOT (150 steps, no capture) ==="
best_adamw=""; best_sgd=""; ba=99; bs=99
for lr in 0.001 0.002 0.004; do
  $V/python make_req072_configs.py --out_dir $DUR/pilot_a_$lr --dur $DUR/none --lr_adamw $lr --pilot_steps 150 >/dev/null 2>&1
  tr $DUR/pilot_a_$lr/adamw-mom.yaml > /root/pilot_a_$lr.log 2>&1 || true
  vl=$(valloss /root/pilot_a_$lr.log); echo "  adamw lr=$lr val=$vl"
  if [ -n "$vl" ] && awk "BEGIN{exit !($vl<$ba)}"; then ba=$vl; best_adamw=$lr; fi
done
for lr in 0.01 0.03 0.1; do
  $V/python make_req072_configs.py --out_dir $DUR/pilot_s_$lr --dur $DUR/none --lr_sgd $lr --pilot_steps 150 >/dev/null 2>&1
  tr $DUR/pilot_s_$lr/sgd-mom.yaml > /root/pilot_s_$lr.log 2>&1 || true
  vl=$(valloss /root/pilot_s_$lr.log); echo "  sgd lr=$lr val=$vl"
  if [ -n "$vl" ] && awk "BEGIN{exit !($vl<$bs)}"; then bs=$vl; best_sgd=$lr; fi
done
: ${best_adamw:=0.002}; : ${best_sgd:=0.03}
log "FROZEN LRs: adamw=$best_adamw (val $ba) sgd=$best_sgd (val $bs) muon=0.025"

# ---- 6 capture runs (3250 steps) ----
$V/python make_req072_configs.py --out_dir $DUR/configs --dur $DUR --lr_adamw $best_adamw --lr_sgd $best_sgd --lr_muon 0.025 >/dev/null 2>&1
for arm in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  log "=== ARM $arm (3250 steps, capture) ==="
  tr $DUR/configs/$arm.yaml > /root/req072_$arm.log 2>&1 || true
  echo "  $arm: uT_chunks=$(ls $DUR/$arm/uT/grad_chunk_*.pt 2>/dev/null|wc -l) final_val=$(valloss /root/req072_$arm.log)"
  grep -iE 'req072 capture tagged|Error:|Traceback|Root Cause' /root/req072_$arm.log 2>/dev/null|tail -2
done

# ---- spectrogram analysis ----
log "=== SPECTROGRAM ANALYSIS ==="
N="blocks.0.attn.proj.weight blocks.5.attn.proj.weight blocks.11.attn.proj.weight"
for arm in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  log "analyze $arm"
  $V/python /root/impl/req072_analyze.py --arm_dir $DUR/$arm --names $N --tokens 524288 --window 64 --hop 32 --out $DUR/analysis/$arm.json 2>&1 | tail -4
done
log "=== SUMMARY: period-two fraction of u_t (blocks.5) per arm ==="
for arm in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  $V/python - "$DUR/analysis/$arm.json" "$arm" <<'PY' 2>&1 | grep -v record_context
import json,sys
try:
    d=json.load(open(sys.argv[1])); b=d["streams"]["u_t"]["blocks.5.attn.proj.weight"]["band"]
    g=d["streams"]["g_t"]["blocks.5.attn.proj.weight"]["band"]
    print(f"  {sys.argv[2]:12s} u_t period2={b['period2_frac']:.3f} low={b['low_frac']:.3f} | g_t period2={g['period2_frac']:.3f}")
except Exception as e: print(f"  {sys.argv[2]}: {e}")
PY
done
echo REQ072_DONE
