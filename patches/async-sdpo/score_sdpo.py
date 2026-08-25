#!/usr/bin/env python3
"""Build the REQ-002 §3 summary.tsv from a pulled arm log + eval dump.

Usage: score_sdpo.py <benchmark> <arm> <box> <model> <arm.log> [baseline.log] [eval_dump.jsonl]

Parses the patched trainer's per-step line:
  step N | loss .. | teacher-student gap +X | adv_clip A% | ratio_clip lo L% hi H% |
    staleness S (max M) | grad .. | lr .. | tokens T | hint_drop D% | sandbox_fail F | empty_eps E
and eval lines: "EVAL launched at step N (policy V): {..dict..}" / "ZERO-SHOT BASELINE: {..dict..}".
Emits one TSV row (+ header) with the REQ-002 §3 columns. Missing values -> NA (never blank/guessed).
"""
import sys, re, ast, statistics as st

STEP_RE = re.compile(
    r"step (\d+) \| loss ([\d.eE+-]+) \| teacher-student gap ([+-][\d.eE]+) \| "
    r"adv_clip ([\d.]+)% \| ratio_clip lo ([\d.]+)% hi ([\d.]+)% \| "
    r"staleness ([\d.]+) \(max (\d+)\) \| grad [\d.eE+-]+ \| lr [\d.eE+-]+ \| "
    r"tokens (\d+) \| hint_drop ([\d.]+)% \| sandbox_fail (\d+) \| empty_eps (\d+)"
)
EVAL_RE = re.compile(r"EVAL launched at step (\d+) \(policy \d+\): (\{.*\})")
BASE_RE = re.compile(r"ZERO-SHOT BASELINE: (\{.*\})")
WANDB_RE = re.compile(r"wandb run: \S+ \((https?://\S+)\)")

def pctl(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs)-1)*p/100.0
    lo = int(k); hi = min(lo+1, len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(k-lo)

def parse_dict(s):
    try: return ast.literal_eval(s)
    except Exception: return {}

def NA(x): return "NA" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))

def main():
    bench, arm, box, model, armlog = sys.argv[1:6]
    baselog = sys.argv[6] if len(sys.argv) > 6 else None
    metric_key = "pass1" if bench == "tau2" else "judge_score"
    gaps=[]; adv=[]; rlo=[]; rhi=[]; stal=[]; maxs=0; hint=[]; sbx=0; steps=0
    evals=[]; wandb_url=None
    for line in open(armlog, errors="replace"):
        m = STEP_RE.search(line)
        if m:
            steps = int(m.group(1))
            gaps.append(float(m.group(3))); adv.append(float(m.group(4))/100)
            rlo.append(float(m.group(5))/100); rhi.append(float(m.group(6))/100)
            stal.append(float(m.group(7))); maxs=max(maxs,int(m.group(8)))
            hint.append(float(m.group(11))/100); sbx=max(sbx,int(m.group(12)))
        e = EVAL_RE.search(line)
        if e: evals.append((int(e.group(1)), parse_dict(e.group(2))))
        w = WANDB_RE.search(line)
        if w: wandb_url = w.group(1)
    base_metric = None
    if baselog:
        for line in open(baselog, errors="replace"):
            b = BASE_RE.search(line)
            if b: base_metric = parse_dict(b.group(1)).get(metric_key)
    step0 = next((d.get(metric_key) for s,d in evals if s==0), None)
    final = evals[-1][1].get(metric_key) if evals else None
    finald = evals[-1][1] if evals else {}
    delta = (final - base_metric) if (final is not None and base_metric is not None) else None
    gap_dead = (sum(1 for g in gaps if abs(g)<1e-3)/len(gaps)) if gaps else None
    cols = [
        ("benchmark",bench),("arm",arm),("seed","0"),("box",box),
        ("status","ok" if steps>=1 else "no_steps"),("steps",steps),("model",model),
        ("base_metric",base_metric),("baseline_metric",base_metric),
        ("step0_metric",step0),("final_metric",final),("delta_vs_baseline",delta),
        ("gap_mean", st.mean(gaps) if gaps else None),
        ("gap_abs_mean", st.mean([abs(g) for g in gaps]) if gaps else None),
        ("gap_p10", pctl(gaps,10)),("gap_p90", pctl(gaps,90)),("gap_dead_frac", gap_dead),
        ("adv_clip_frac", st.mean(adv) if adv else None),
        ("ratio_clip_frac_low", st.mean(rlo) if rlo else None),
        ("ratio_clip_frac_high", st.mean(rhi) if rhi else None),
        ("store_mean_staleness", st.mean(stal) if stal else None),
        ("store_max_staleness_seen", maxs),
        ("store_hint_dropped_percent", st.mean(hint) if hint else None),
        ("sandbox_fail_count", sbx),("episodes_total","NA"),("episodes_empty","NA"),
        ("wandb_url", wandb_url),("log_path", armlog),
    ]
    # per-domain / per-section extras from the final eval dict
    for k,v in sorted(finald.items()):
        if k.startswith("pass1_") or k.startswith("judge_"):
            cols.append((k, v))
    print("\t".join(c[0] for c in cols))
    print("\t".join(NA(c[1]) for c in cols))

if __name__ == "__main__":
    main()
