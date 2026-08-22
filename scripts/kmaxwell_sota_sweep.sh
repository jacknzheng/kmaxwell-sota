#!/usr/bin/env bash
# Wrapper around kmaxwell_sota_sweep.py. Run from repo root.
#   scripts/kmaxwell_sota_sweep.sh --stage 1
#   scripts/kmaxwell_sota_sweep.sh --stage 2 --k 4
#   scripts/kmaxwell_sota_sweep.sh --stage 0 --dry-run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/records/track_3_optimization/results/20260821_kmaxwell_sota/kmaxwell_sota_sweep.py" "$@"
