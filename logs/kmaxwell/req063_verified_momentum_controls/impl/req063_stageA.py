"""REQ-063 Stage A (CPU): repair the returned REQ-058/059 evidence. No training.
 (1) REQ-059: paired t-tests across the 4 seeds (df=3), Holm cumulative-max adjusted p, correct 95% CIs.
 (2) REQ-058: relabel fork-1500 as 60-update (files end at 1560, not 1564) -> exclude from the 64-update
     analysis; recompute retrospective feature models (raw S_i, S_i/G_i, type/depth prior) fit on seed-0,
     reported per test seed 1/2. Euclidean & actual-direction features are UNAVAILABLE for these seeds
     (documented; deferred to REQ-064).
 (3) Recovery manifest: committed vs UNVERIFIED (node-local) evidence.
Pure Python (self-contained t-CDF via regularized incomplete beta). Preserves the negative REQ-059 outcome.
"""
import os, sys, math, statistics, itertools
from collections import defaultdict

R59 = "logs/kmaxwell/req059_sharpness_guided_momentum/raw/full059"
R58 = "logs/kmaxwell/req058_layerwise_momentum_response/raw/full058"
R57 = "logs/kmaxwell/req057_layerwise_spectral_sharpness/raw/full"


# ---- self-contained stats (no scipy) ----
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0; d = 1 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < EPS: break
    return h


def betai(a, b, x):  # regularized incomplete beta I_x(a,b)
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):  # two-sided p-value = I_{df/(df+t^2)}(df/2, 1/2)
    if t == 0: return 1.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


T95 = {1: 12.706, 2: 4.303, 3: 3.182}


def paired(diffs):
    n = len(diffs); m = statistics.mean(diffs); sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = m / se if se > 0 else float('inf')
    half = T95[n - 1] * se
    return dict(n=n, mean=m, sd=sd, t=t, p=t_two_sided_p(t, n - 1), ci=(m - half, m + half))


def load_tsv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return None
    return {int(a): float(b) for a, b in (ln.split() for ln in open(path) if ln.strip())}


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x); i = 0
        while i < len(x):
            j = i
            while j+1 < len(x) and x[o[j+1]] == x[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2.0
            i = j+1
        return r
    ra, rb = rk(a), rk(b); n = len(a); ma, mb = sum(ra)/n, sum(rb)/n
    cov = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n)); va = sum((x-ma)**2 for x in ra)**.5; vb = sum((x-mb)**2 for x in rb)**.5
    return cov/(va*vb) if va > 0 and vb > 0 else float('nan')


# ================= (1) REQ-059 corrected stats =================
print("=" * 70)
print("(1) REQ-059 corrected paired stats (endpoint 2750, n=4 seeds 3-6, df=3)")
SEEDS59 = [3, 4, 5, 6]
ARMS = ["guided", "global_a05", "global_a1", "global_a2", "euclidean", "typedepth", "shuffled", "reversed", "nomom"]
ep = {a: {s: (load_tsv(f"{R59}/{a}_s{s}.tsv") or {}).get(2750) for s in SEEDS59} for a in ARMS}
controls = [a for a in ARMS if a != "guided"]
res = []
for c in controls:
    diffs = [ep["guided"][s] - ep[c][s] for s in SEEDS59 if ep["guided"][s] is not None and ep[c][s] is not None]
    if len(diffs) >= 2:
        r = paired(diffs); r["control"] = c; res.append(r)
# Holm cumulative-max
res_sorted = sorted(res, key=lambda r: r["p"]); m = len(res_sorted)
cummax = 0.0
for i, r in enumerate(res_sorted):
    cummax = max(cummax, (m - i) * r["p"]); r["holm"] = min(1.0, cummax)
print(f"  {'control':11s} {'mean':>10} {'95% CI':>22} {'t':>8} {'p(t,df3)':>10} {'Holm':>8}  verdict")
for r in sorted(res, key=lambda r: r["control"]):
    beats = r["mean"] <= -0.0005 and r["ci"][1] < 0 and r["holm"] < 0.05
    print(f"  {r['control']:11s} {r['mean']:+.5f} [{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] {r['t']:>8.2f} {r['p']:>10.2e} {r['holm']:>8.3f}  {'BEATS' if beats else 'no'}")
win = all((r['mean'] <= -0.0005 and r['ci'][1] < 0 and r['holm'] < 0.05) for r in res)
print(f"  -> PRACTICAL WIN after Holm correction: {'YES' if win else 'NO'} (negative outcome preserved: guided loses to global_a05)")

# ================= (2) REQ-058 endpoint relabel + retrospective =================
print("=" * 70)
print("(2) REQ-058 endpoint audit + retrospective feature models")
# count fork-1500 files that end at 1560 (60-update), not 1564
miss = 0; tot = 0
for f in os.listdir(R58):
    if f.endswith("_f1500.tsv"):
        tot += 1; d = load_tsv(f"{R58}/{f}")
        if d and 1564 not in d: miss += 1
print(f"  fork-1500 files ending <1564 (=60-update, EXCLUDED from 64-update analysis): {miss}/{tot}")
# clean 64-update retrospective on fork-2000 seeds; features: raw S_i, S_i/G_i, type/depth prior
def si_gi(seed, name):
    p = f"{R57}/full_s{seed}_step2000.json"
    if not os.path.exists(p): return None, None
    import json; iso = json.load(open(p))["subset0"]["isolated"].get(name)
    return (iso["S_i"], iso["G_i"]) if iso else (None, None)
TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]; BLOCKS = [0, 6, 11]
import json
devmean = {}
for blk in BLOCKS:
    for t in TYPES:
        name = f"blocks.{blk}.{t}.weight"; vs = []
        for s in (0, 1, 2):
            si, _ = si_gi(s, name)
            if si is not None: vs.append(si)
        if vs: devmean[name] = statistics.mean(vs)
def resp(seed, blk, t):  # a2-penalty at fork+64=2064
    tt = t.replace(".", ""); a1 = load_tsv(f"{R58}/a1all_s{seed}_f2000.tsv"); a2 = load_tsv(f"{R58}/a2_b{blk}_{tt}_s{seed}_f2000.tsv")
    if not a1 or not a2 or 2064 not in a1 or 2064 not in a2: return None
    return a2[2064] - a1[2064]
for feat in ("rawSi", "Si_over_Gi", "typedepth"):
    print(f"  feature={feat}: Spearman(feature, a2-penalty) fit-report per seed (64-update fork-2000 data)")
    for seed in (0, 1, 2):
        xs, ys = [], []
        for blk in BLOCKS:
            for t in TYPES:
                name = f"blocks.{blk}.{t}.weight"; r = resp(seed, blk, t)
                si, gi = si_gi(seed, name)
                if r is None or si is None: continue
                if feat == "rawSi": x = si
                elif feat == "Si_over_Gi": x = si / gi if gi else None
                else: x = devmean.get(name)
                if x is None: continue
                xs.append(x); ys.append(r)
        tag = "DEV" if seed == 0 else "TEST"
        print(f"    seed{seed} ({tag}): rho={spearman(xs, ys):+.3f} (n={len(xs)})")
print("  MISSING feature columns (deferred to REQ-064, prospective): Euclidean lambda_i, actual-update-direction curvature (not computed for REQ-058 seeds 0-2).")

# ================= (3) recovery manifest =================
print("=" * 70)
print("(3) Recovery manifest")
print("  COMMITTED (verifiable): impl generators+patches, raw selection-loss/JSON per arm, READMEs/readouts for REQ-057/058/059/060.")
print("  UNVERIFIED (node-local, lost — REQ-058/059/060 nodes qj9epjw/w7r58o3/3ypd5k3 stopped): stdout logs, resolved run configs, per-rank state/data manifests, named allocation JSONs on host.")
print("  Recovered harness SHA: 365c392d695f95dc9a4fb89095e85a6a7b5d551e + committed REQ-058/059 patches (apply_req058_opt.py / apply_req059_opt.py in the impl dirs).")
