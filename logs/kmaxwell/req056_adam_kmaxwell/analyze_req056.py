"""REQ-056 analysis: does K-Maxwell first-moment help STANDARD Adam?
Consumes raw_logs/val_{arm}_s{seed}.tsv (columns: step  val_loss  step_avg_ms), arm in {adam,kmaxwell,agema},
seed in {0,1,2}. Reports:
  - per-seed endpoint (last val_loss) for each arm
  - per-seed paired diffs: kmaxwell-adam, agema-adam, kmaxwell-agema
  - across-seed mean +/- std of each diff, vs the 0.0005 practical-equivalence margin
  - val-vs-step curves (subsampled) per arm/seed
Verdict logic: |mean diff| < 0.0005 -> practical-equivalence; else improvement/regression; INCONCLUSIVE if the
across-seed std swamps the mean."""
import os, glob, statistics as st

RAW = os.path.join(os.path.dirname(__file__), "raw_logs")
ARMS = ["adam", "kmaxwell", "agema"]
SEEDS = [0, 1, 2]
MARGIN = 0.0005


def load(arm, seed):
    p = os.path.join(RAW, f"val_{arm}_s{seed}.tsv")
    if not os.path.exists(p):
        return None
    rows = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else float("nan")))
    return rows


def endpoint(rows):
    return rows[-1][1] if rows else float("nan")


data = {(a, s): load(a, s) for a in ARMS for s in SEEDS}

print("=== endpoint val_loss (step 3250) per arm/seed ===")
print(f"{'seed':>5} | {'adam':>9} | {'kmaxwell':>9} | {'agema':>9}")
ep = {}
for s in SEEDS:
    row = []
    for a in ARMS:
        e = endpoint(data[(a, s)]) if data[(a, s)] else float("nan")
        ep[(a, s)] = e
        row.append(f"{e:9.5f}")
    print(f"{s:>5} | " + " | ".join(row))

print("\n=== per-seed paired diffs (negative = better than baseline Adam) ===")
print(f"{'seed':>5} | {'KM - Adam':>11} | {'agema - Adam':>13} | {'KM - agema':>11}")
diffs = {"km_adam": [], "age_adam": [], "km_age": []}
for s in SEEDS:
    a, k, g = ep[("adam", s)], ep[("kmaxwell", s)], ep[("agema", s)]
    d1, d2, d3 = k - a, g - a, k - g
    for key, v in (("km_adam", d1), ("age_adam", d2), ("km_age", d3)):
        if v == v:
            diffs[key].append(v)
    print(f"{s:>5} | {d1:+11.5f} | {d2:+13.5f} | {d3:+11.5f}")


def verdict(vals, label):
    if not vals:
        return f"{label}: NO DATA"
    m = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    n = len(vals)
    tag = "practical-equivalence" if abs(m) < MARGIN else ("IMPROVEMENT" if m < 0 else "REGRESSION")
    incon = " (INCONCLUSIVE: |mean| < std)" if abs(m) < sd else ""
    return f"{label}: mean={m:+.5f} +/- {sd:.5f} (n={n})  vs margin {MARGIN}  -> {tag}{incon}"


print("\n=== across-seed verdict (margin = 0.0005 loss) ===")
print("  " + verdict(diffs["km_adam"], "K-Maxwell - Adam   "))
print("  " + verdict(diffs["age_adam"], "age-matched - Adam "))
print("  " + verdict(diffs["km_age"], "K-Maxwell - agema  "))
print("\n  Interpretation: KM-Adam tests whether the multi-timescale first moment helps standard Adam at the")
print("  frozen LR; KM-agema isolates whether MULTIPLE timescales matter beyond having the right average age.")

print("\n=== val-vs-step (seed 0, every ~500 steps) ===")
s0 = data[("adam", 0)]
if s0:
    steps = [r[0] for r in s0 if r[0] % 500 == 0 or r[0] >= 3249]
    hdr = f"{'step':>6} | " + " | ".join(f"{a:>9}" for a in ARMS)
    print(hdr)
    for stp in steps:
        cells = []
        for a in ARMS:
            rows = data[(a, 0)]
            v = next((vl for (st_, vl, _) in rows if st_ == stp), float("nan")) if rows else float("nan")
            cells.append(f"{v:9.5f}")
        print(f"{stp:>6} | " + " | ".join(cells))
