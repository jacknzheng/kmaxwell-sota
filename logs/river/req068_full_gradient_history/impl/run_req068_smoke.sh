#!/bin/bash
# REQ-068 in-harness GPU smoke test (cheap, node-local): validates the capture pipeline on real HW before
# the TB-scale full run. Checks: logging ON vs OFF training parity (val_loss identical), round-trip reader
# (all indices, reconstruct, nonzero), fork dumps, + durable-storage capacity and sustained-write benchmark.
set -uo pipefail
V=/root/venv019/bin; ROOT=/root/kmaxwell-sota; export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
log(){ echo "[smoke $(date -u +%H:%M:%S)] $*"; }
cd $ROOT
RP=records/track_3_optimization

log "stage capture module + patch harness"
mkdir -p logs/river/req068_full_gradient_history/impl
cp /root/impl/req068_capture.py logs/river/req068_full_gradient_history/impl/
$V/python /root/impl/apply_req068_capture.py 2>&1 | tail -1

log "generate ON (capture) + OFF (no capture) configs, 20 steps, validate every 5"
$V/python /root/impl/make_req068_config.py --out $RP/configs/smoke_on.yaml \
   --grad_dir /root/smoke_grads --state_dir /root/smoke_forks --seed 0 --train_steps 20 --chunk_steps 5 --fork_steps 5 10 15 >/dev/null
# dense validation every 5 for a comparable signal; OFF = ON minus the two capture hooks
$V/python - <<'PY'
import yaml
on=yaml.safe_load(open("records/track_3_optimization/configs/smoke_on.yaml"))
for h in on["post_optimizer"]:
    if h["name"]=="validate_at_step_boundaries": h["hyperparams"]={"every":5,"dense_window":[0,20],"dense_every":5}
yaml.safe_dump(on, open("records/track_3_optimization/configs/smoke_on.yaml","w"), sort_keys=False)
off=yaml.safe_load(open("records/track_3_optimization/configs/smoke_on.yaml"))
off["run_id"]="req068_smoke_off"
off["pre_optimizer"]=[h for h in off["pre_optimizer"] if h["name"]!="capture_full_gradients"]
off["teardown"]=[h for h in off["teardown"] if h["name"]!="finalize_gradient_capture"]
yaml.safe_dump(off, open("records/track_3_optimization/configs/smoke_off.yaml","w"), sort_keys=False)
print("wrote smoke_on.yaml (capture) + smoke_off.yaml (no capture)")
PY

tr(){ $V/torchrun --standalone --nproc_per_node=8 $RP/run.py "$@"; }
log "run OFF (no capture)"; rm -rf /root/smoke_forks_off; tr $RP/configs/smoke_off.yaml > /root/smoke_off.log 2>&1 || true
log "run ON (capture)";    rm -rf /root/smoke_grads /root/smoke_forks; tr $RP/configs/smoke_on.yaml > /root/smoke_on.log 2>&1 || true

log "=== PARITY: val_loss lines ON vs OFF (must be identical) ==="
grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' /root/smoke_off.log | sort -u > /root/voff.txt
grep -oE 'step:[0-9]+/[0-9]+ val_loss:[0-9.]+' /root/smoke_on.log  | sort -u > /root/von.txt
if diff -q /root/von.txt /root/voff.txt >/dev/null; then echo "PARITY_PASS ($(wc -l </root/von.txt) boundaries identical)"; else echo "PARITY_FAIL"; diff /root/von.txt /root/voff.txt | head; fi

log "=== ROUND-TRIP: reader verification over captured grads ==="
$V/python - <<'PY'
import sys; sys.path.insert(0,"logs/river/req068_full_gradient_history/impl")
from req068_capture import GradientHistoryReader
r=GradientHistoryReader("/root/smoke_grads")
steps=r.steps(); v=r.verify(expected_steps=list(range(20)))
print("steps captured:",len(steps),"first/last:",steps[0],steps[-1])
print("verify:",v)
# reconstruct a known matrix + a scalar at a step; check nonzero + dtype
g=r.grad(7,"blocks.0.attn.q.weight"); import torch
print("blocks.0.attn.q grad@7:",tuple(g.shape),g.dtype,"nonzero:",bool((g!=0).any()))
sc=r.grad(7,[n for n in r.params() if "gains" in n or len(r.schema["params"][n]["shape"])==1][0])
print("a vector/scalar grad@7 present:",sc is not None)
print("ROUNDTRIP_PASS" if (v["ok"] and len(steps)==20) else "ROUNDTRIP_FAIL")
PY

log "=== FORK DUMPS present (5,10,15) ==="
ls /root/smoke_forks/ 2>/dev/null | grep -c train_state | xargs echo "fork state files:"

log "=== STORAGE: durable mounts capacity + sustained-write benchmark ==="
echo "-- df of candidate durable mounts --"
df -h /root/.cache/user_artifacts /root/.cache/team_artifacts 2>/dev/null | grep -v safe || echo "(user/team_artifacts not mounted here)"
df -h /root 2>/dev/null | tail -1
for M in /root/.cache/user_artifacts /root/.cache/team_artifacts /root; do
  [ -d "$M" ] || continue
  TD="$M/req068_wbench_$$"; mkdir -p "$TD" 2>/dev/null || { echo "  $M: mkdir FAILED (quota?)"; continue; }
  # write 2 GiB, measure throughput
  if dd if=/dev/zero of="$TD/probe.bin" bs=1M count=2048 oflag=direct 2>/tmp/dd_$$.txt || dd if=/dev/zero of="$TD/probe.bin" bs=1M count=2048 2>/tmp/dd_$$.txt; then
    echo "  $M: $(grep -oE '[0-9.]+ [GM]B/s' /tmp/dd_$$.txt | tail -1) sustained (2GiB write)"
  else echo "  $M: WRITE FAILED (quota/ENOSPC?)"; tail -1 /tmp/dd_$$.txt; fi
  rm -rf "$TD"
done
echo SMOKE_DONE
