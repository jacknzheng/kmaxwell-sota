"""REQ-057 full-stage aggregation over 72 matrices x steps {1500,2000,2500} x seeds {0,1,2}.
Reads raw/full/full_s{seed}_step{step}.json (validated-cheapest budget K=20, restarts=2, 1 subset;
subset0 carries the 72x72 Q_ij + cross-layer + HVP validation). Reports:
  - per-step 72-matrix S_i (mean+/-std over seeds), G_i, Z_i
  - cross-seed reproducibility: pairwise Spearman of S_i/G_i across seeds at each step
  - joint vs diagonal coupling: c_cross/|c| at the gradient-polar direction per checkpoint (mean)
  - spectral within-type depth ordering (S_i by block) at step 2000
  - step evolution of S_joint and mean S_i
Pure Python. Usage: python analyze_full.py <raw/full dir>
"""
import json, os, sys, glob, statistics, itertools
from collections import defaultdict

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw/full"
STEPS = [1500, 2000, 2500]
SEEDS = [0, 1, 2]


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x); i = 0
        while i < len(x):
            j = i
            while j+1 < len(x) and x[order[j+1]] == x[order[i]]: j += 1
            for k in range(i, j+1): r[order[k]] = (i+j)/2.0
            i = j+1
        return r
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra)/n, sum(rb)/n
    cov = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    va = sum((ra[i]-ma)**2 for i in range(n))**0.5; vb = sum((rb[i]-mb)**2 for i in range(n))**0.5
    return cov/(va*vb) if va > 0 and vb > 0 else float("nan")


def load(seed, step):
    p = os.path.join(RAW, f"full_s{seed}_step{step}.json")
    return json.load(open(p)) if os.path.exists(p) else None


data = {(s, st): load(s, st) for s in SEEDS for st in STEPS}
present = [(s, st) for (s, st), d in data.items() if d]
names = list(data[present[0]]["subset0"]["isolated"].keys())
print(f"loaded {len(present)}/9 checkpoints; {len(names)} matrices")

# --- cross-seed reproducibility of S_i/G_i at each step ---
print("\n=== cross-seed reproducibility: pairwise Spearman of S_i/G_i (72 matrices) ===")
for st in STEPS:
    vecs = {}
    for s in SEEDS:
        d = data.get((s, st))
        if not d: continue
        iso = d["subset0"]["isolated"]
        vecs[s] = [(iso[n]["S_i"]/iso[n]["G_i"]) if (iso[n]["S_i"] is not None and iso[n]["G_i"]) else float("nan") for n in names]
    keep = [i for i in range(len(names)) if all(v[i] == v[i] for v in vecs.values())]
    corrs = [spearman([vecs[a][i] for i in keep], [vecs[b][i] for i in keep]) for a, b in itertools.combinations(sorted(vecs), 2)]
    med = statistics.median(corrs) if corrs else float("nan")
    print(f"  step {st}: median cross-seed Spearman = {med:.3f}  (pairs {[round(c,3) for c in corrs]}, n={len(keep)})")

# --- joint vs diagonal coupling per checkpoint ---
print("\n=== joint vs diagonal curvature (c_cross/|c| at gradient-polar direction) ===")
fracs = []
for (s, st) in present:
    cl = data[(s, st)]["subset0"].get("cross_layer_at_gradpolar")
    if cl and cl["c"]:
        f = cl["c_cross"]/abs(cl["c"]); fracs.append(f)
        print(f"  seed{s} step{st}: c={cl['c']:.3e} c_diag={cl['c_diag']:.3e} c_cross={cl['c_cross']:.3e}  cross/|c|={f:.1%}")
print(f"  -> mean cross/|c| = {statistics.mean(fracs):.1%} +/- {statistics.pstdev(fracs):.1%} over {len(fracs)} checkpoints")

# --- S_joint and mean S_i evolution across steps (per seed) ---
print("\n=== S_joint and mean isolated S_i by step (per seed) ===")
for st in STEPS:
    for s in SEEDS:
        d = data.get((s, st))
        if not d: continue
        iso = d["subset0"]["isolated"]
        S_is = [iso[n]["S_i"] for n in names if iso[n]["S_i"] is not None]
        print(f"  seed{s} step{st}: S_joint={d['subset0']['S_joint']:.3f}  mean S_i={statistics.mean(S_is):.3f}  "
              f"Z_joint={d['subset0']['Z_joint']}")

# --- per-matrix S_i mean+/-std over seeds at step 2000 (the reference), grouped by type/depth ---
print("\n=== S_i (mean over seeds) at step 2000, by TYPE and DEPTH (block) ===")
st = 2000
by_type = defaultdict(dict)
for n in names:
    typ = ".".join(n.split(".")[2:]); blk = int(n.split(".")[1])
    vals = [data[(s, st)]["subset0"]["isolated"][n]["S_i"] for s in SEEDS
            if data.get((s, st)) and data[(s, st)]["subset0"]["isolated"][n]["S_i"] is not None]
    if vals:
        by_type[typ][blk] = (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
for typ in sorted(by_type):
    blks = sorted(by_type[typ])
    cells = " ".join(f"b{b}:{by_type[typ][b][0]:.2f}" for b in blks)
    mono = all(by_type[typ][blks[i]][0] >= by_type[typ][blks[i+1]][0] for i in range(len(blks)-1))
    print(f"  {typ:16s} {cells}   {'(monotone decreasing w/ depth)' if mono else ''}")

# --- HVP validation summary across checkpoints ---
print("\n=== HVP validation (central-diff vs autograd, smallest eps) ===")
for (s, st) in present:
    hv = data[(s, st)]["subset0"].get("hvp_validation")
    if hv:
        small = min(hv["by_eps"].items(), key=lambda kv: float(kv[0]))
        print(f"  seed{s} step{st}: eps={small[0]} rel_err={small[1]['rel_err_vs_autograd']:.2%}")
