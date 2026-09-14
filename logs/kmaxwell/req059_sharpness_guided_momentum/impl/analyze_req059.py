"""REQ-059 stats: does the sharpness-guided allocation beat every control? Paired across seeds {3,4,5,6}.
Reads raw/full059/{arm}_s{seed}.tsv (step<TAB>val_loss). Primary metric = endpoint (2750); secondary = mean of
last three (2500,2625,2750). Comparisons: guided vs each control; paired mean diff, 95% CI (t, n=4), Holm
correction across the 8 comparisons. Practical win: guided mean >=0.0005 LOWER than EVERY control AND
Holm-adjusted CI excludes 0. Usage: python analyze_req059.py <raw/full059 dir>
"""
import os, sys, math, statistics
RAW = sys.argv[1] if len(sys.argv) > 1 else "raw/full059"
SEEDS = [3, 4, 5, 6]
ARMS = ["guided", "global_a05", "global_a1", "global_a2", "euclidean", "typedepth", "shuffled", "reversed", "nomom"]
CONTROLS = [a for a in ARMS if a != "guided"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided t_0.975 by dof=n-1


def load(arm, s):
    p = os.path.join(RAW, f"{arm}_s{s}.tsv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = [ln.split() for ln in open(p) if ln.strip()]
    d = {int(st): float(v) for st, v in rows}
    return d


def endpoint(d):
    return d.get(2750, d[max(d)]) if d else None


def last3(d):
    ks = [k for k in (2500, 2625, 2750) if k in d]
    return statistics.mean([d[k] for k in ks]) if ks else None


def paired(metric):
    vals = {a: {s: metric(load(a, s)) for s in SEEDS} for a in ARMS}
    print(f"  per-seed {metric.__name__}:")
    print("    seed  " + "  ".join(f"{a[:9]:>9}" for a in ARMS))
    for s in SEEDS:
        print(f"    {s:>4}  " + "  ".join(f"{(vals[a][s] if vals[a][s] is not None else float('nan')):9.5f}" for a in ARMS))
    results = []
    for c in CONTROLS:
        diffs = [vals["guided"][s] - vals[c][s] for s in SEEDS if vals["guided"][s] is not None and vals[c][s] is not None]
        if len(diffs) < 2:
            results.append((c, None, None, None, len(diffs))); continue
        m = statistics.mean(diffs); sd = statistics.stdev(diffs); n = len(diffs)
        half = T95[n-1] * sd / math.sqrt(n)
        # p-value proxy: |t| = |m|/(sd/sqrt(n)); we report CI + a normal-approx two-sided p for Holm ordering
        t = m / (sd / math.sqrt(n)) if sd > 0 else float('inf')
        p = math.erfc(abs(t) / math.sqrt(2))  # normal approx
        results.append((c, m, (m - half, m + half), p, n))
    # Holm across the comparisons
    valid = [r for r in results if r[1] is not None]
    order = sorted(valid, key=lambda r: r[3])
    k = len(order)
    holm = {}
    for i, r in enumerate(order):
        holm[r[0]] = min(1.0, r[3] * (k - i))
    print(f"\n  guided - control (negative = guided better), 95% CI, Holm-adj p (n={len(SEEDS)} seeds):")
    win = True
    for c, m, ci, p, n in results:
        if m is None:
            print(f"    vs {c:11s}: insufficient data ({n})"); win = False; continue
        hp = holm.get(c, float('nan'))
        beats = (m <= -0.0005) and (ci[1] < 0)
        win = win and beats
        print(f"    vs {c:11s}: {m:+.5f}  CI[{ci[0]:+.5f},{ci[1]:+.5f}]  Holm-p={hp:.3f}  {'BEATS' if beats else 'no'}")
    print(f"  -> PRACTICAL WIN (guided >=0.0005 better than EVERY control, CI excludes 0): {'YES' if win else 'NO'}")
    return win


print("=== ENDPOINT (step 2750) — primary ===")
paired(endpoint)
print("\n=== mean of last three (2500/2625/2750) — secondary ===")
paired(last3)
print("\nNote: 'guided beats the balanced controls (shuffled/reversed/type-depth/euclidean)' shows the ASSIGNMENT")
print("matters; 'guided beats the globals' is the stricter unconstrained-resource claim. Report both.")
