#!/usr/bin/env bash
# Rebuild boundary_2009 from boundary_2001 (+8 updates, PR323 baseline, LR factor 1.0, microbatch 8).
# Deterministic; run per node. Verifies rank sizes + COMPLETE.json, then places it where run_job.sh expects.
set -euo pipefail
P=/root/blog_six; R=$P/code/replay
PARENT=$P/data/links/restart/boundary_2001
BUILD=$P/work/build_2009
DEST=$P/data/restarts/checkpoints/critroll/pr323_lr_snap_portable_20261019/continuous20/shared/restart/boundary_2009
EXPECT_SIZE=11908782730
VPY=$P/venv/bin/python
mkdir -p "$BUILD"
cd "$R"
export PYTHONPATH="$R/source:$R" NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 PYTORCH_ALLOC_CONF=expandable_segments:True

# data shards must be visible under cwd (the trainer globs data/fineweb10B/*.bin) -- mirror run_job.sh runtime()
RS=$P/data/restarts; L=$P/data/links
mkdir -p "$R/data/fineweb10B"
for n in $(seq -f %06g 1 15); do ln -sf "$RS/datasets/fineweb10B/fineweb_train_$n.bin" "$R/data/fineweb10B/"; done
for n in $(seq -f %06g 16 22); do ln -sf "$L/train/fineweb_train_$n.bin" "$R/data/fineweb10B/"; done
ln -sf "$L/val/fineweb_val_000000.bin" "$R/data/fineweb10B/"
echo "data shards linked: $(ls "$R/data/fineweb10B/"/fineweb_train_*.bin | wc -l) train + val"

# 1) build training.yaml (baseline continuation, factor 1.0, checkpoint at 2009)
"$VPY" - "$PARENT" "$BUILD" <<'PY'
import sys, yaml
from extension_config import build
parent, out = sys.argv[1], sys.argv[2]
cfg = build('.', parent, out, 2009, 1.0, 2001)   # window_start=2001 so start<=first<=last holds
cfg.update(microbatch_sequences=8, run_id='build_2009',
           output_dir=out+'/training', checkpoint_dir=out+'/restart', train_data_seed=0)
open(out+'/training.yaml','w').write(yaml.safe_dump(cfg, sort_keys=False))
print('wrote', out+'/training.yaml', 'checkpoint_every=', cfg['checkpoint_every'], 'stop_step=', cfg['stop_step'],
      'resume=', cfg['resume_checkpoint'])
PY

# 2) train 2001 -> 2009 and save the boundary checkpoint
"$P/venv/bin/torchrun" --standalone --nproc_per_node=8 continuous20_train.py "$BUILD/training.yaml"

# 3) verify
SRC="$BUILD/restart/boundary_2009"
[ -f "$SRC/COMPLETE.json" ] || { echo "BUILD FAIL: no COMPLETE.json at $SRC"; exit 1; }
"$VPY" - "$SRC" "$EXPECT_SIZE" <<'PY'
import sys, json, os
src, exp = sys.argv[1], int(sys.argv[2])
r = json.load(open(src+'/COMPLETE.json'))
assert r['boundary']==2009 and r['world_size']==8, r
for k in range(8):
    sz=os.path.getsize(f'{src}/rank{k}.pt')
    assert sz==exp, f'rank{k} size {sz} != {exp}'
print('VERIFY OK: boundary=2009 world_size=8, all 8 ranks ==', exp, 'bytes; protocol', r.get('protocol_sha256','')[:12])
PY

# 4) place where run_job.sh links restart/shared_2009
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"; cp -r "$SRC" "$DEST"
echo "BUILD_2009_DONE -> $DEST"
