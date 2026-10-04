#!/bin/bash
# REQ-074 ON-BOX: verify REQ-072/073 histories + time-resolved recompute (no rerun). Self-contained in /root/impl.
set -uo pipefail
VPY=/root/venv_req074/bin/python; export PYTHONUNBUFFERED=1 PYTHONPATH=/root/impl
log(){ echo "[req074 $(date -u +%H:%M:%S)] $*"; }
D72=/root/.cache/user_artifacts/req072; D73=/root/.cache/user_artifacts/req073
OUT=/root/.cache/user_artifacts/req074; mkdir -p $OUT/arrays
N72="blocks.0.attn.proj.weight blocks.5.attn.proj.weight blocks.11.attn.proj.weight"
N73="blocks.5.attn.q.weight blocks.5.attn.k.weight blocks.5.mlp.proj.weight"
rc(){ $VPY /root/impl/req074_recompute.py "$@"; }
uT_ok(){ $VPY -c "import sys;sys.path.insert(0,'/root/impl');from req068_capture import GradientHistoryReader as R;r=R('$1');import sys as s;s.exit(0 if r.grad(r.steps()[0],'blocks.0.attn.proj.weight') is not None else 3)" 2>/dev/null; }

log "=== VERIFY REQ-072 histories ==="
for a in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  for s in grad uT disp; do echo "  req072/$a/$s: manifest=$([ -f $D72/$a/$s/manifest.json ] && echo Y || echo N) chunks=$(ls $D72/$a/$s/*_chunk_*.pt 2>/dev/null|wc -l)"; done
done
log "=== VERIFY REQ-073 histories ==="
for a in B-mom B-nomom 4B-mom 4B-nomom; do
  for s in s1_grad s2_premom s3_postpolar s4_disp; do echo "  req073/$a/$s: manifest=$([ -f $D73/$a/$s/manifest.json ] && echo Y || echo N) chunks=$(ls $D73/$a/$s/*_chunk_*.pt 2>/dev/null|wc -l)"; done
done

log "=== RECOMPUTE REQ-072 ==="
for a in adamw-mom adamw-nomom sgd-mom sgd-nomom muon-mom muon-nomom; do
  log "req072 $a grad"; rc --src_dir $D72/$a/grad --names $N72 --tokens 524288 --out $OUT/arrays/req072_${a}_grad.npz 2>&1|tail -4
  if uT_ok $D72/$a/uT; then
    log "req072 $a uT"; rc --src_dir $D72/$a/uT --names $N72 --tokens 1 --out $OUT/arrays/req072_${a}_uT.npz 2>&1|tail -4
  else
    log "req072 $a uT is None (muon sharding) -> disp proxy (wd=0: delta=-lr*u_t)"
    rc --src_dir $D72/$a/disp --names $N72 --tokens 1 --out $OUT/arrays/req072_${a}_uT_fromdisp.npz 2>&1|tail -4
  fi
done
log "=== RECOMPUTE REQ-073 ==="
for a in B-mom B-nomom 4B-mom 4B-nomom; do
  bt=$([ "${a:0:2}" = "4B" ] && echo 2097152 || echo 524288)
  for st in s1_grad s2_premom s3_postpolar s4_disp; do
    tok=$([ "$st" = "s1_grad" ] && echo $bt || echo 1)
    log "req073 $a $st (tokens=$tok)"; rc --src_dir $D73/$a/$st --names $N73 --tokens $tok --out $OUT/arrays/req073_${a}_${st}.npz 2>&1|tail -4
  done
done
log "=== DONE: $(ls $OUT/arrays/*.npz 2>/dev/null|wc -l) npz, $(du -sh $OUT/arrays 2>/dev/null|cut -f1) ==="
echo REQ074_RECOMPUTE_DONE
