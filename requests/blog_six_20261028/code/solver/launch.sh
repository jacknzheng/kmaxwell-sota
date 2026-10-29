#!/usr/bin/env bash
# One fixed-16M zeta solve on one saved checkpoint. Production settings only.
#
#   ./launch.sh --capture CAPTURE_DIR --validation VAL_BIN --output OUT_DIR --update N [--warm WARM_DIR]
#
# CAPTURE_DIR  directory holding rank0..rank7/boundary_N.pt (and index.jsonl when available)
# VAL_BIN      fineweb_val_000000.bin, sha256 5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640
# OUT_DIR      new or partially finished solve directory (resumed automatically)
# N            checkpoint update number, e.g. 2041
# WARM_DIR     a finished OUT_DIR of a nearby update (reads current_vector_rank{0..7}.pt); omit for the first point
#
# Optional: --microbatch M (default: try 16, then 8, then 4 on CUDA OOM, like production)
#           --independent  (protocol audit only: fresh Gaussian start instead of the warm vector)
#           --skip-hash    (do not re-hash the 8 capture shards; byte sizes still checked)
# Env:      PYTHON (default: python3), NPROC (default 8; the solver requires exactly 8 ranks),
#           TORCHINDUCTOR_CACHE_DIR (default: <package>/torch_cache), DRY_RUN=1 prints the command only.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
NPROC="${NPROC:-8}"
CAPTURE=""; VALIDATION=""; OUTPUT=""; UPDATE=""; WARM=""; MICRO=""; INDEPENDENT=0; HASHFLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --capture) CAPTURE="$2"; shift 2;;
    --validation) VALIDATION="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --update) UPDATE="$2"; shift 2;;
    --warm) WARM="$2"; shift 2;;
    --microbatch) MICRO="$2"; shift 2;;
    --independent) INDEPENDENT=1; shift;;
    --skip-hash) HASHFLAG="--skip-hash"; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$CAPTURE" ] && [ -n "$VALIDATION" ] && [ -n "$OUTPUT" ] && [ -n "$UPDATE" ] || { echo "need --capture --validation --output --update" >&2; exit 2; }
CAPTURE="$(cd "$CAPTURE" && pwd)"
VALIDATION="$(cd "$(dirname "$VALIDATION")" && pwd)/$(basename "$VALIDATION")"
mkdir -p "$OUTPUT"; OUTPUT="$(cd "$OUTPUT" && pwd)"
if [ -n "$WARM" ]; then WARM="$(cd "$WARM" && pwd)"; fi

# CPU-side input verification; writes OUT_DIR/input_manifest.json (needed by the solver's resume check).
"$PYTHON" "$PKG/check_inputs.py" --capture "$CAPTURE" --validation "$VALIDATION" --output "$OUTPUT" --update "$UPDATE" $HASHFLAG

# Resume convention (same as production): a partial solve restarts from its own last Ritz vector.
RESUME=()
if [ -f "$OUTPUT/progress.json" ]; then
  complete=0
  for r in 0 1 2 3 4 5 6 7; do [ -f "$OUTPUT/current_vector_rank$r.pt" ] && complete=$((complete+1)); done
  if [ "$complete" -eq 8 ]; then WARM="$OUTPUT"; fi
  RESUME=(--resume-from "$OUTPUT")
fi
if [ -n "$WARM" ]; then
  for r in 0 1 2 3 4 5 6 7; do [ -f "$WARM/current_vector_rank$r.pt" ] || { echo "warm dir lacks current_vector_rank$r.pt: $WARM" >&2; exit 2; }; done
fi

export PYTHONPATH="$PKG/source:$PKG${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export DENSE_PREFIX_WINDOW=9 DENSE_INDEPENDENT="$INDEPENDENT"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PKG/torch_cache}"
[ "${DRY_RUN:-0}" = "1" ] || mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

run_once() {  # $1 = microbatch
  local argv=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$PKG/fullrun_solve.py"
    --capture "$CAPTURE" --validation "$VALIDATION" --output "$OUTPUT" --update "$UPDATE"
    --token-sizes 16777216 --dimensions 8,16,24,32,48,64,96,128 --microbatches "$1"
    --fixed-probes --hvp-mode compile --head-backend cached --matmul-precision high --gemm-precision bf16_fp32
    --include-weight-decay)
  if [ -n "$WARM" ]; then argv+=(--warm-start "$WARM"); fi
  argv+=("${RESUME[@]}")
  echo "launch: ${argv[*]}" | tee -a "$OUTPUT/solve.log"
  if [ "${DRY_RUN:-0}" = "1" ]; then return 0; fi
  (cd "$PKG" && "${argv[@]}" 2>&1 | tee -a "$OUTPUT/solve.log"; exit "${PIPESTATUS[0]}")
}

if [ -n "$MICRO" ]; then candidates=("$MICRO"); else candidates=(16 8 4); fi
ok=0
for mb in "${candidates[@]}"; do
  if run_once "$mb"; then ok=1; break; fi
  if ! grep -qE 'CUDA out of memory|OutOfMemoryError|No microbatch fits' "$OUTPUT/solve.log"; then
    echo "solve failed for a reason other than GPU memory; see $OUTPUT/solve.log" >&2; exit 1
  fi
  echo "microbatch $mb did not fit; retrying with a smaller one" | tee -a "$OUTPUT/solve.log"
done
[ "$ok" = 1 ] || { echo "no microbatch fit; see $OUTPUT/solve.log" >&2; exit 1; }
if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi
"$PYTHON" "$PKG/read_result.py" "$OUTPUT"
echo "next point: pass --warm $OUTPUT"
