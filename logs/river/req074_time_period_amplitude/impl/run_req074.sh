#!/bin/bash
# REQ-074: verify REQ-072/073 histories intact, then recompute TIME-RESOLVED spectrograms from them (no rerun).
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[req074 $(date -u +%H:%M:%S)] $*"; }
cd $ROOT
I68=logs/river/req068_full_gradient_history/impl; I74=logs/river/req074_time_period_amplitude/impl
mkdir -p $I68 $I74; cp /root/impl/req068_capture.py $I68/; cp /root/impl/req074_spectrogram.py /root/impl/req074_recompute.py $I74/
D72=/root/.cache/user_artifacts/req072; D73=/root/.cache/user_artifacts/req073
OUT=/root/.cache/user_artifacts/req074; mkdir -p $OUT/arrays
N72="blocks.0.attn.proj.weight blocks.5.attn.proj.weight blocks.11.attn.proj.weight"
N73="blocks.5.attn.q.weight blocks.5.attn.k.weight blocks.5.mlp.proj.weight"

log "=== VERIFY histories intact ==="
for a in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  for s in grad uT disp; do echo "  req072/$a/$s: $(ls $D72/$a/$s/manifest.json 2>/dev/null && echo OK || echo MISSING) chunks=$(ls $D72/$a/$s/grad_chunk_*.pt 2>/dev/null|wc -l)"; done
done
for a in B-mom B-nomom 4B-mom 4B-nomom; do
  for s in s1_grad s2_premom s3_postpolar s4_disp; do echo "  req073/$a/$s: chunks=$(ls $D73/$a/$s/grad_chunk_*.pt 2>/dev/null|wc -l)"; done
done

rc(){ $V/python /root/impl/req074_recompute.py "$@"; }
log "=== RECOMPUTE REQ-072 (grad mean/token; uT; muon uT falls back to disp) ==="
for a in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  rc --src_dir $D72/$a/grad --names $N72 --tokens 524288 --out $OUT/arrays/req072_${a}_grad.npz 2>&1|tail -1
  uTdir=$D72/$a/uT
  if $V/python -c "import sys;sys.path.insert(0,'$I68');from req068_capture import GradientHistoryReader as R;r=R('$uTdir');import sys as s;s.exit(0 if r.grad(r.steps()[0],'blocks.5.attn.proj.weight'.replace('5','0')) is not None else 3)" 2>/dev/null; then
    rc --src_dir $uTdir --names $N72 --tokens 1 --out $OUT/arrays/req072_${a}_uT.npz 2>&1|tail -1
  else
    log "  $a uT is None (muon sharding) -> using disp as u_t proxy (wd=0: delta=-lr*u_t)"
    rc --src_dir $D72/$a/disp --names $N72 --tokens 1 --out $OUT/arrays/req072_${a}_uT_fromdisp.npz 2>&1|tail -1
  fi
done
log "=== RECOMPUTE REQ-073 (4 stages) ==="
for a in B-mom B-nomom 4B-mom 4B-nomom; do
  bt=$([ "${a:0:2}" = "4B" ] && echo 2097152 || echo 524288)
  for st in s1_grad s2_premom s3_postpolar s4_disp; do
    tok=$([ "$st" = "s1_grad" ] && echo $bt || echo 1)
    rc --src_dir $D73/$a/$st --names $N73 --tokens $tok --out $OUT/arrays/req073_${a}_${st}.npz 2>&1|tail -1
  done
done
log "=== DONE: $(ls $OUT/arrays/*.npz 2>/dev/null|wc -l) npz arrays, total $(du -sh $OUT/arrays 2>/dev/null|cut -f1) ==="
echo REQ074_RECOMPUTE_DONE
