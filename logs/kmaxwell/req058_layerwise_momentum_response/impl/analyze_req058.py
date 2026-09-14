"""REQ-058 aggregation: does layer spectral sharpness predict the memory-response?
Response(matrix,a) = selection-loss(perturbed arm)@(fork+64) - selection-loss(a1all)@(fork+64), paired within
(seed,fork). Joins each matrix's response with its REQ-057 isolated spectral sharpness S_i at the SAME
(seed, fork-step). Dev = seed 0 (forks 2000,1500); held-out prediction test = seeds 1,2 (fork 2000).
Reads raw/full058/{arm}_s{seed}_f{fork}.tsv (step<TAB>val_loss) and the REQ-057 JSONs.
Usage: python analyze_req058.py <raw/full058 dir> <req057 raw/full dir>
"""
import json, os, sys, glob, statistics, itertools
from collections import defaultdict

R58 = sys.argv[1] if len(sys.argv) > 1 else "raw/full058"
R57 = sys.argv[2] if len(sys.argv) > 2 else "../req057_layerwise_spectral_sharpness/raw/full"
TYPES = {"attnq": "attn.q", "attnk": "attn.k", "attnv": "attn.v", "attnproj": "attn.proj",
         "mlpfc": "mlp.fc", "mlpproj": "mlp.proj"}
BLOCKS = [0, 6, 11]
BASES = [(0, 2000), (0, 1500), (1, 2000), (2, 2000)]


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
    return cov/(va*vb) if va > 0 and vb > 0 else float("nan")


def final_loss(arm, s, f):
    p = os.path.join(R58, f"{arm}_s{s}_f{f}.tsv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = [ln.split() for ln in open(p) if ln.strip()]
    target = str(f + 64)
    for st, vl in rows:
        if st == target:
            return float(vl)
    return float(rows[-1][1])  # fallback: last


def sharpness(s, step, matrix):
    p = os.path.join(R57, f"full_s{s}_step{step}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    iso = d["subset0"]["isolated"].get(matrix)
    return iso["S_i"] if iso else None


print("=== global controls: selection-loss@fork+64 minus a1all (negative=better) ===")
for s, f in BASES:
    a1 = final_loss("a1all", s, f)
    if a1 is None:
        print(f"  s{s} f{f}: a1all missing"); continue
    row = []
    for ctl in ("a05all", "a2all", "nomom", "matched"):
        v = final_loss(ctl, s, f)
        row.append(f"{ctl}={'NA' if v is None else f'{v-a1:+.5f}'}")
    print(f"  s{s} f{f}: a1all={a1:.5f}  " + "  ".join(row))

print("\n=== per-matrix memory response vs REQ-057 sharpness ===")
recs = []  # (seed, fork, matrix, S_i, resp_a2, resp_a05)
for s, f in BASES:
    a1 = final_loss("a1all", s, f)
    if a1 is None:
        continue
    for blk in BLOCKS:
        for tt, sub in TYPES.items():
            matrix = f"blocks.{blk}.{sub}.weight"
            Si = sharpness(s, f, matrix)
            r2 = final_loss(f"a2_b{blk}_{tt}", s, f)
            r05 = final_loss(f"a05_b{blk}_{tt}", s, f)
            if None in (Si, r2, r05):
                continue
            recs.append((s, f, matrix, Si, r2 - a1, r05 - a1))

dev = [r for r in recs if r[0] == 0]
test = [r for r in recs if r[0] in (1, 2)]
print(f"  records: {len(recs)} total ({len(dev)} dev seed0, {len(test)} held-out seeds1,2)")


def report(rows, label):
    if len(rows) < 4:
        print(f"  {label}: too few ({len(rows)})"); return
    S = [r[3] for r in rows]; a2 = [r[4] for r in rows]; a05 = [r[5] for r in rows]
    print(f"  {label} (n={len(rows)}): "
          f"Spearman(S_i, longer-mem penalty a2) = {spearman(S, a2):+.3f} ; "
          f"Spearman(S_i, shorter-mem resp a05) = {spearman(S, a05):+.3f}")
    # mean responses by sharpness tertile
    order = sorted(rows, key=lambda r: r[3]); t = len(order)//3
    lo, hi = order[:t], order[-t:]
    print(f"    low-sharpness tertile mean a2-penalty={statistics.mean([r[4] for r in lo]):+.5f}; "
          f"high-sharpness tertile mean a2-penalty={statistics.mean([r[4] for r in hi]):+.5f}")


report(dev, "DEV seed0")
report(test, "HELD-OUT seeds1,2")
print("\n  Interpretation: a positive Spearman(S_i, a2-penalty) means sharper matrices are hurt MORE by longer")
print("  memory (i.e., sharpness predicts preferring shorter memory). The held-out value is the prediction gate")
print("  for REQ-059. Sign consistency dev vs held-out is the key check.")
