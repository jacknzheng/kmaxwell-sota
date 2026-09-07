#!/bin/bash
# Sequential per-node runner. Args: config basenames (in configs/req056/). Applies patch, runs each, marks DONE.
set -uo pipefail
cd /root/kmaxwell-sota
export PYTHONPATH=records/track_3_optimization TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
[ -f /root/apply_req056_opt.py ] && /root/venv019/bin/python /root/apply_req056_opt.py >/dev/null 2>&1 || true
TR="/root/venv019/bin/torchrun --standalone --nproc_per_node=8"
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  rm -f /root/REQ056_${name}_DONE
  $TR records/track_3_optimization/run.py configs/req056/$cfg > /root/req056_${name}.log 2>&1
  rc=$?
  echo "$name exit=$rc $(date +%H:%M:%S)" >> /root/req056_progress.log
  touch /root/REQ056_${name}_DONE
done
