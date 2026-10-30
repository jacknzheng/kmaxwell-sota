#!/usr/bin/env bash
# Run one job of the request. Every job resumes where it stopped if rerun.
#   scripts/run_job.sh saved [--only NAME]                  12 solves on downloaded snapshots (LR x2, MaxEnt 2675)
#   scripts/run_job.sh maxent_1250|maxent_1750|maxent_2250  replay from a native MaxEnt restart, then its solves
#   scripts/run_job.sh higher_zeta|lrhalf --phase 1         replay from boundary_2009 saving every candidate snapshot, then the round-1 dense solves
#   scripts/run_job.sh higher_zeta|lrhalf --phase 2 --updates U1,U2,...   more solves from the saved snapshots (printed by check_equilibrium.py)
# Results are copied to outputs/<job>/ (no snapshots, no restarts).
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-$P/venv/bin/python}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$P/work/torch_cache}"
RS="$P/data/restarts"; L="$P/data/links"; VAL="$L/val/fineweb_val_000000.bin"
JOB="$1"; shift
PHASE=""; UPDATES=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2;;
    --updates) UPDATES="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
mkdir -p "$P/work" "$P/outputs"

# Working directory for a replay job: the replay code plus links to the train/validation shards and warm vectors.
runtime() {
  local R="$1"
  mkdir -p "$R/data/fineweb10B" "$R/warm"
  cp -rn "$P/code/replay/." "$R/"
  for n in $(seq -f %06g 1 15); do ln -sf "$RS/datasets/fineweb10B/fineweb_train_$n.bin" "$R/data/fineweb10B/"; done
  for f in "$L"/train/fineweb_train_*.bin; do ln -sf "$f" "$R/data/fineweb10B/"; done  # shards 16+ (more added 2026-10-30)
  ln -sf "$VAL" "$R/data/fineweb10B/"
}
# Copy a replay's results, logs and receipts (not captures or restarts) to outputs/<job>/.
collect() {
  local R="$1" RID="$2" D="$P/outputs/$JOB/$RID"
  mkdir -p "$D"
  cp -r "$R/blog_outputs/$RID/results" "$D/" 2>/dev/null || true
  cp "$R/blog_outputs/$RID"/*.json "$R/blog_outputs/$RID"/*.log "$R/blog_outputs/$RID"/*.yaml "$D/" 2>/dev/null || true
  [ -f "$R/RESTORE_COMPLETE.json" ] && cp "$R/RESTORE_COMPLETE.json" "$D/"
  echo "collected -> $D"
}

case "$JOB" in
saved)
  grep -v '^#' "$P/jobs/saved.txt" | while read -r name arm u lineage warm; do
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    out="$P/outputs/saved/$name"
    [ -f "$out/result.json" ] && { echo "done already: $name"; continue; }
    case "$warm" in out:*) w="$P/outputs/saved/${warm#out:}";; *) w="$L/$warm";; esac
    cap="$L/saved/$arm/captures"
    mkdir -p "$out"
    "$PYTHON" "$P/code/solver/predict.py" --capture "$cap" --update "$u" > "$out/prediction.json"
    bash "$P/code/solver/launch.sh" --capture "$cap" --validation "$VAL" --output "$out" --update "$u" --warm "$w"
    "$PYTHON" "$P/scripts/make_points.py" "$arm" "$lineage" "$P/outputs/saved/points/$arm" "$out"
    for r in 0 1 2 3 4 5 6 7; do rm -f "$cap/rank$r/boundary_$(printf %06d "$u").pt"; done  # free disk
    echo "DONE $name $(date -u +%FT%TZ)"
  done
  ;;
maxent_1250|maxent_1750|maxent_2250)
  A="${JOB#maxent_}"; R="$P/work/$JOB"
  runtime "$R"
  cp "$P/jobs/$JOB.json" "$R/plan.json"
  mkdir -p "$R/input"
  ln -sfn "$RS/checkpoints/critroll/backward_maxentslop_defazio_20261005/step_0$A" "$R/input/maxentslop"
  W=$(python3 -c "import json;print(json.load(open('$R/plan.json'))['directive'].rsplit(' ',1)[1])")
  ln -sfn "$L/warm/$W" "$R/warm/$W"
  touch "$R/DIRECT_DOWNLOAD_COMPLETE"
  (cd "$R" && GAP_ROOT="$R" GAP_ARCHIVE=0 "$PYTHON" run_gaps.py)
  collect "$R" "$(python3 -c "import json;print(json.load(open('$R/plan.json'))['run_id'])")"
  ;;
higher_zeta|lrhalf|higher_zeta_ext|lrhalf_ext)
  R="$P/work/$JOB"; Q="$P/jobs/$JOB.json"
  RID=$(python3 -c "import json;print(json.load(open('$Q'))['run_id'])")
  runtime "$R"
  mkdir -p "$R/restart"
  ln -sfn "$RS/checkpoints/critroll/pr323_lr_snap_portable_20261019/continuous20/shared/restart/boundary_2009" "$R/restart/shared_2009"
  W=$(python3 -c "import json;print(json.load(open('$Q'))['warm_path'].split('/',1)[1])")
  ln -sfn "$L/warm/$W" "$R/warm/$W"
  case "$PHASE" in
    1) BLOG_TRAIN_MICROBATCH=8 "$PYTHON" "$R/run_arm.py" --runtime "$R" --queue "$Q" --phase all;;
    2) [ -n "$UPDATES" ] || { echo "phase 2 needs --updates U1,U2,... (from scripts/check_equilibrium.py)" >&2; exit 2; }
       for t in ${UPDATES//,/ }; do
         "$PYTHON" "$R/run_arm.py" --runtime "$R" --queue "$Q" --phase adaptive_one --timestamp "$t"
       done;;
    *) echo "give --phase 1 or --phase 2" >&2; exit 2;;
  esac
  collect "$R" "$RID"
  ;;
*) echo "unknown job $JOB" >&2; exit 2;;
esac
echo "JOB-DONE $JOB $PHASE $(date -u +%FT%TZ)"
