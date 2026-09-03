import json, glob, math, sys, os
from collections import defaultdict

BASE=sys.argv[1] if len(sys.argv)>1 else "/Users/jerryhong/.claude/jobs/e9994aef/tmp/req035_curvature"
S={"s060":0.60,"s100":1.00,"s170":1.70}
STEPS=["2250","2375","2500","2625","2750"]

def seed_curvature(seed):
    """Return {matrix: {s_val: lam_eq}} for a seed (lam_eq = mean top_eigenvalue over steps, merged ranks)."""
    out=defaultdict(dict)
    for stag,sval in S.items():
        # collect per-matrix top_eigenvalue across ranks + steps
        acc=defaultdict(list)
        for f in glob.glob(f"{BASE}/seed{seed}/req035_seed{seed}_{stag}_rank*of8.json"):
            d=json.load(open(f))
            for st in STEPS:
                mats=d.get(st,{}).get("matrices",{})
                for name,rec in mats.items():
                    le=rec.get("top_eigenvalue")
                    if le is not None and le>0 and math.isfinite(le): acc[name].append(le)
        for name,vals in acc.items():
            if vals: out[name][sval]=sum(vals)/len(vals)
    return out

def fit_logC(sd):
    """log10 lam = log10 C - k log10 s ; return (logC, k) from the s-points."""
    xs=[math.log10(s) for s in sorted(sd)]; ys=[math.log10(sd[s]) for s in sorted(sd)]
    n=len(xs)
    if n<2: return None
    mx=sum(xs)/n; my=sum(ys)/n
    den=sum((x-mx)**2 for x in xs)
    if den==0: return None
    k_slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den   # slope = -k
    logC=my - k_slope*mx     # intercept at log s = 0 (s=1) = log C
    return logC, -k_slope

seeds=[s for s in [0,1,2,3] if os.path.isdir(f"{BASE}/seed{s}")]
print(f"seeds present: {seeds}")
logC={}  # seed -> {matrix: logC}
kval={}
for s in seeds:
    cur=seed_curvature(s)
    logC[s]={}; kval[s]={}
    for m,sd in cur.items():
        r=fit_logC(sd)
        if r: logC[s][m]=r[0]; kval[s][m]=r[1]
    print(f"seed{s}: {len(logC[s])} matrices fitted, median k={sorted(kval[s].values())[len(kval[s])//2]:.2f}")

# seed replication: median |delta log C| across seed pairs (common matrices)
import itertools
common=set.intersection(*[set(logC[s]) for s in seeds]) if seeds else set()
print(f"\ncommon matrices across seeds: {len(common)}")
pair_meds=[]
for a,b in itertools.combinations(seeds,2):
    deltas=sorted(abs(logC[a][m]-logC[b][m]) for m in common)
    med=deltas[len(deltas)//2]
    rms=math.sqrt(sum(d*d for d in deltas)/len(deltas))
    pair_meds.append(med)
    print(f"seed{a} vs seed{b}: median |dlogC|={med:.4f} dex, rms={rms:.4f} dex")
if pair_meds:
    overall=sorted(pair_meds)[len(pair_meds)//2]
    print(f"\n=== READOUT: median-of-pairs |delta log C| = {overall:.4f} dex (noise floor ~0.10) ===")
    verdict=("<=0.10 => C SEED-INDEPENDENT (architecture determines it)" if overall<=0.10 else
             ">=0.20 => C is a LEARNED per-network property" if overall>=0.20 else
             "in (0.10,0.20) => partial; report seed-reproducible fraction")
    print("VERDICT:",verdict)
# type ordering: attn.v highest, attn.proj lowest — reproduces across seeds?
def type_of(m):
    for t in ["attn.q","attn.k","attn.v","attn.proj","mlp.fc","mlp.proj"]:
        if t in m: return t
    return "other"
print("\ntype-mean logC per seed (does ordering reproduce?):")
for s in seeds:
    tm=defaultdict(list)
    for m,v in logC[s].items(): tm[type_of(m)].append(v)
    order=sorted(tm, key=lambda t: sum(tm[t])/len(tm[t]), reverse=True)
    print(f"  seed{s}: "+" > ".join(order))
