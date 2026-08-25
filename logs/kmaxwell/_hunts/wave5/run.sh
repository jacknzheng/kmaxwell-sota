#!/usr/bin/env bash
# Wave 5 on 8xH100: sequential. Logs: this dir (tee) + trainer uuid files.
# A then (if invoked as `all`) scored Wave B. See PLAN.txt.
#
#   logs/wave5/run.sh A      # six unconfounded k=6 shots
#   logs/wave5/run.sh B      # branch from A scores (refuses if A incomplete)
#   logs/wave5/run.sh all    # A then B
set -euo pipefail
ROOT="${ROOT:-/workspace/modded-nanogpt}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
TRAIN="$ROOT/records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py"
LOGDIR="$ROOT/logs/wave5"
BEAT="3.28036"
AGE_MOVE="3.28020"
KILL="3.28050"
STACK_3150="3.27980"
STACK_3250="3.27360"
mkdir -p "$LOGDIR"
cd "$ROOT"

final_loss() {
  local f="$1"
  grep -E 'step:3250/3250 val_loss:' "$f" | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p'
}

score_3150() {
  local f="$1"
  grep -E 'step:3150/3250 val_loss:' "$f" | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p'
}

py_le() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"; }
py_lt() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <  float(sys.argv[2]) else 1)" "$1" "$2"; }

run_one() {
  local name="$1"
  shift
  local tee="$LOGDIR/${name}.stdout"
  if [[ -f "$tee" ]] && grep -qE 'exit=0 3150=[0-9]' "$tee"; then
    echo "==== skip $name (already scored 3150=$(score_3150 "$tee")) ===="
    return 0
  fi
  echo "==== $(date -u +%F\ %T) $name $* ====" | tee "$tee"
  torchrun --standalone --nproc_per_node=8 --master_port=29511 -- \
    "$TRAIN" --seed 0 --start 1000 "$@" \
    2>&1 | tee -a "$tee"
  local rc=${PIPESTATUS[0]}
  echo "==== $(date -u +%F\ %T) $name exit=$rc 3150=$(score_3150 "$tee") final=$(final_loss "$tee") ====" | tee -a "$tee"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc" >&2
    return "$rc"
  fi
}

summarize() {
  local title="$1"; shift
  echo "$title"
  echo "ref     K6_a35 3150=3.28036 3250=3.27407   DEC 3150=3.28040 3250=3.27415"
  local n f s f3250
  for n in "$@"; do
    f="$LOGDIR/${n}.stdout"
    s="$(score_3150 "$f" || true)"
    f3250="$(final_loss "$f" || true)"
    echo "$n  3150=${s:-missing}  3250=${f3250:-missing}"
  done
}

need_score() {
  local n="$1"
  local s
  s="$(score_3150 "$LOGDIR/${n}.stdout" || true)"
  if [[ -z "$s" ]]; then
    echo "missing 3150 for $n" >&2
    return 1
  fi
  echo "$s"
}

wave_A() {
  run_one K6_a32    --k 6 --tau-min 3 --tau-max 56 --weights 0.04197,0.08395,0.12592,0.16789,0.20986,0.37041
  run_one K6_a38    --k 6 --tau-min 3 --tau-max 56 --weights 0.03148,0.06296,0.09444,0.12592,0.15740,0.52781
  run_one K6_ws40   --k 6 --tau-min 3 --tau-max 56 --weights 0.03062,0.06124,0.09187,0.12249,0.29378,0.40000
  run_one K6_ws50   --k 6 --tau-min 3 --tau-max 56 --weights 0.04305,0.08611,0.12916,0.17221,0.06947,0.50000
  run_one K6_tmin2  --k 6 --tau-min 2 --tau-max 56 --weights 0.03473,0.06947,0.10420,0.13894,0.17367,0.47899
  run_one K6_tmax64 --k 6 --tau-min 3 --tau-max 64 --weights 0.04351,0.08702,0.13052,0.17403,0.21754,0.34738
  summarize "wave5 A scored" K6_a32 K6_a38 K6_ws40 K6_ws50 K6_tmin2 K6_tmax64 | tee "$LOGDIR/SCORE_A.txt"
}

# Tau follow-up window after A: better of tmin2/tmax64 if it beat 3.28036, else [2.5,56].
pick_tau_followup() {
  local s2 s64
  s2="$(need_score K6_tmin2)"
  s64="$(need_score K6_tmax64)"
  if py_le "$s2" "$BEAT" && py_le "$s2" "$s64"; then
    echo tmin2
  elif py_le "$s64" "$BEAT" && py_lt "$s64" "$s2"; then
    echo tmax64
  else
    echo tmin25
  fi
}

better_age() {
  local s32 s38
  s32="$(need_score K6_a32)"
  s38="$(need_score K6_a38)"
  if py_le "$s32" "$s38"; then echo a32; else echo a38; fi
}

wave_B_age() {
  local winner="$1" follow
  follow="$(pick_tau_followup)"
  echo "B branch=age winner=$winner tau_follow=$follow" | tee "$LOGDIR/DECISION.txt"
  if [[ "$winner" == a32 ]]; then
    run_one K6_a31 --k 6 --tau-min 3 --tau-max 56 --weights 0.04372,0.08744,0.13116,0.17489,0.21861,0.34418
    run_one K6_a34 --k 6 --tau-min 3 --tau-max 56 --weights 0.03848,0.07695,0.11543,0.15390,0.19238,0.42287
    run_one K6_a32_ws40 --k 6 --tau-min 3 --tau-max 56 --weights 0.04565,0.09130,0.13695,0.18261,0.14349,0.40000
    case "$follow" in
      tmin2)  run_one K6_a32_tmin2  --k 6 --tau-min 2   --tau-max 56 --weights 0.03970,0.07939,0.11909,0.15878,0.19848,0.40456 ;;
      tmax64) run_one K6_a32_tmax64 --k 6 --tau-min 3   --tau-max 64 --weights 0.04801,0.09602,0.14403,0.19204,0.24005,0.27986 ;;
      *)      run_one K6_a32_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.04087,0.08174,0.12261,0.16348,0.20435,0.38696 ;;
    esac
    summarize "wave5 B age-32" K6_a31 K6_a34 K6_a32_ws40 K6_a32_tmin2 K6_a32_tmax64 K6_a32_tmin25
  else
    run_one K6_a36 --k 6 --tau-min 3 --tau-max 56 --weights 0.03498,0.06995,0.10493,0.13991,0.17489,0.47534
    run_one K6_a39 --k 6 --tau-min 3 --tau-max 56 --weights 0.02973,0.05946,0.08919,0.11892,0.14865,0.55404
    run_one K6_a38_ws50 --k 6 --tau-min 3 --tau-max 56 --weights 0.02802,0.05605,0.08407,0.11209,0.21977,0.50000
    case "$follow" in
      tmin2)  run_one K6_a38_tmin2  --k 6 --tau-min 2   --tau-max 56 --weights 0.02977,0.05954,0.08932,0.11909,0.14886,0.55342 ;;
      tmax64) run_one K6_a38_tmax64 --k 6 --tau-min 3   --tau-max 64 --weights 0.03901,0.07801,0.11702,0.15603,0.19504,0.41489 ;;
      *)      run_one K6_a38_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.03065,0.06130,0.09196,0.12261,0.15326,0.54022 ;;
    esac
    summarize "wave5 B age-38" K6_a36 K6_a39 K6_a38_ws50 K6_a38_tmin2 K6_a38_tmax64 K6_a38_tmin25
  fi
}

wave_B_mix() {
  local winner="$1"
  echo "B branch=mix winner=$winner" | tee "$LOGDIR/DECISION.txt"
  if [[ "$winner" == ws40 ]]; then
    run_one K6_ws35 --k 6 --tau-min 3 --tau-max 56 --weights 0.02441,0.04881,0.07322,0.09762,0.40594,0.35000
    run_one K6_ws40_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.02745,0.05489,0.08234,0.10978,0.32555,0.40000
    run_one K6_ws40_tmax49 --k 6 --tau-min 3 --tau-max 49 --weights 0.00809,0.01617,0.02426,0.03235,0.51913,0.40000
    summarize "wave5 B mix-ws40" K6_ws35 K6_ws40_tmin25 K6_ws40_tmax49
  else
    echo "ws55 infeasible at age 35; window reuse only"
    run_one K6_ws50_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.04052,0.08104,0.12157,0.16209,0.09478,0.50000
    run_one K6_ws50_tmax49 --k 6 --tau-min 3 --tau-max 49 --weights 0.02006,0.04012,0.06018,0.08024,0.29939,0.50000
    summarize "wave5 B mix-ws50" K6_ws50_tmin25 K6_ws50_tmax49
  fi
}

wave_B_tau() {
  local winner="$1" agew
  agew="$(better_age)"
  echo "B branch=tau winner=$winner age_rerun=$agew" | tee "$LOGDIR/DECISION.txt"
  if [[ "$winner" == tmin2 ]]; then
    run_one K6_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.03576,0.07152,0.10728,0.14304,0.17880,0.46359
    if [[ "$agew" == a32 ]]; then
      run_one K6_a32_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.04087,0.08174,0.12261,0.16348,0.20435,0.38696
    else
      run_one K6_a38_tmin25 --k 6 --tau-min 2.5 --tau-max 56 --weights 0.03065,0.06130,0.09196,0.12261,0.15326,0.54022
    fi
    summarize "wave5 B tau-tmin" K6_tmin25 K6_a32_tmin25 K6_a38_tmin25
  else
    run_one K6_tmax49 --k 6 --tau-min 3 --tau-max 49 --weights 0.02859,0.05717,0.08576,0.11435,0.14293,0.57120
    if [[ "$agew" == a32 ]]; then
      run_one K6_a32_tmax49 --k 6 --tau-min 3 --tau-max 49 --weights 0.03471,0.06942,0.10414,0.13885,0.17356,0.47932
    else
      run_one K6_a38_tmax49 --k 6 --tau-min 3 --tau-max 49 --weights 0.02246,0.04492,0.06738,0.08984,0.11230,0.66309
    fi
    summarize "wave5 B tau-tmax" K6_tmax49 K6_a32_tmax49 K6_a38_tmax49
  fi
}

wave_B_embed() {
  echo "B branch=DEC-embed (nothing beat $BEAT)" | tee "$LOGDIR/DECISION.txt"
  run_one K6_DECemb0  --k 6 --tau-min 3 --tau-max 56 --weights 0.05000,0.00000,0.20000,0.25000,0.00000,0.50000
  run_one K6_DECemb01 --k 6 --tau-min 3 --tau-max 56 --weights 0.04900,0.01000,0.19600,0.24500,0.01000,0.49000
  run_one K6_DECemb02 --k 6 --tau-min 3 --tau-max 56 --weights 0.04800,0.02000,0.19200,0.24000,0.02000,0.48000
  run_one K6_DECemb04 --k 6 --tau-min 3 --tau-max 56 --weights 0.04600,0.04000,0.18400,0.23000,0.04000,0.46000
  summarize "wave5 B DEC-embed" K6_DECemb0 K6_DECemb01 K6_DECemb02 K6_DECemb04
}

wave_B() {
  local a32 a38 ws40 ws50 tmin2 tmax64
  a32="$(need_score K6_a32)"
  a38="$(need_score K6_a38)"
  ws40="$(need_score K6_ws40)"
  ws50="$(need_score K6_ws50)"
  tmin2="$(need_score K6_tmin2)"
  tmax64="$(need_score K6_tmax64)"

  python3 - "$a32" "$a38" "$ws40" "$ws50" "$tmin2" "$tmax64" "$BEAT" "$AGE_MOVE" "$LOGDIR/DECISION.txt" <<'PY'
import sys
names = ["a32", "a38", "ws40", "ws50", "tmin2", "tmax64"]
scores = [float(x) for x in sys.argv[1:7]]
beat, age_move = float(sys.argv[7]), float(sys.argv[8])
path = sys.argv[9]
axis = {
    "a32": "age", "a38": "age",
    "ws40": "mix", "ws50": "mix",
    "tmin2": "tau", "tmax64": "tau",
}
best_i = min(range(6), key=lambda i: (scores[i], i))
best_name, best = names[best_i], scores[best_i]
# Prefer a named axis that actually moved, not a 1e-5 wiggle over a real age hit.
moved = [(n, s, axis[n]) for n, s in zip(names, scores) if s <= beat]
age_hits = [(n, s) for n, s, a in moved if a == "age" and s <= age_move]
if best > beat:
    decision = "embed"
    reason = f"nothing beat {beat:.5f} (best {best_name}={best:.5f})"
elif age_hits:
    winner = min(age_hits, key=lambda t: t[1])[0]
    decision = f"age {winner}"
    reason = f"age moved ({winner}={dict(age_hits)[winner]:.5f} <= {age_move:.5f})"
elif any(a == "mix" for _, _, a in moved) and axis[best_name] == "mix":
    decision = f"mix {best_name}"
    reason = f"mix moved (best {best_name}={best:.5f})"
elif any(a == "mix" for _, _, a in moved):
    mix = min(((n, s) for n, s, a in moved if a == "mix"), key=lambda t: t[1])
    decision = f"mix {mix[0]}"
    reason = f"mix moved ({mix[0]}={mix[1]:.5f}); {best_name}={best:.5f} is not age-thresholded"
elif axis[best_name] == "tau":
    decision = f"tau {best_name}"
    reason = f"tau moved (best {best_name}={best:.5f})"
else:
    decision = "embed"
    reason = f"no clean axis (best {best_name}={best:.5f})"
open(path, "w").write(decision + "\n" + reason + "\n")
print(decision)
print(reason)
PY

  local decision
  decision="$(head -1 "$LOGDIR/DECISION.txt")"
  echo "==== Wave B decision: $decision ===="
  case "$decision" in
    "age a32") wave_B_age a32 ;;
    "age a38") wave_B_age a38 ;;
    "mix ws40") wave_B_mix ws40 ;;
    "mix ws50") wave_B_mix ws50 ;;
    "tau tmin2") wave_B_tau tmin2 ;;
    "tau tmax64") wave_B_tau tmax64 ;;
    embed) wave_B_embed ;;
    *) echo "unknown decision: $decision" >&2; return 1 ;;
  esac

  echo "stack rule: only if some 3150 <= $STACK_3150 or 3250 <= $STACK_3250; else stay off Aurora/SOAP"
}

mode="${1:-all}"
case "$mode" in
  A|a) wave_A ;;
  B|b) wave_B ;;
  all)
    wave_A
    wave_B
    echo "wave5 done"
    ;;
  *)
    echo "usage: $0 {A|B|all}" >&2
    exit 2
    ;;
esac
