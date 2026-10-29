#!/usr/bin/env bash
# Build the Python environment and download every input into data/ (resumable; rerun after any interruption).
#   scripts/setup.sh            environment + all downloads
#   scripts/setup.sh --env-only environment only
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P"

# Production runtime: Python 3.12, torch 2.11.0+cu128 (code/solver/requirements.txt).
if [ ! -x venv/bin/python ]; then
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip
  venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
  venv/bin/pip install numpy==2.4.3 scipy==1.18.1 wandb==0.31.0 pyyaml==6.0.3
fi
venv/bin/python -c "import torch; assert torch.cuda.device_count() == 8, torch.cuda.device_count(); print('torch', torch.__version__, '8 GPUs ok')"
[ "${1:-}" = "--env-only" ] && exit 0

# 1) One-file links (saved snapshots, warm-start vectors, train shards 16-22, validation shard): ~26 GB, sha256/size checked.
python3 -I download_inputs.py --workers 8

# 2) Owner restart package (boundary_2009, MaxEnt step_01250/01750/02250, train shards 1-15): ~373 GB, block-resumable.
#    Exit code 3 means the owner's links expired: ask for a fresh downloads.jsonl.gz and drop it into access/restart-downloads/.
python3 -I access/restart-downloads/download_plan.py access/restart-downloads/downloads.jsonl.gz --dest data/restarts --workers 16

df -h "$P" | tail -1
echo "SETUP-DONE"
