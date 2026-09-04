"""REQ-046 registered regression: d log lambda / d log(clip) with matrix fixed effects,
across 3 balanced arms x 72 matrices. Plus the 3 registered checks (monotone reduced form,
per-type ratios, first stage d log g_clipped/d log clip ~ 1). Reads raw_curvature_json/ + req046_assign.json."""
import json, math, os
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
assign=json.load(open(f"{HERE}/req046_assign.json"))
A=assign["assign"]; TYPES=assign["types"]; arms=["req046_a0","req046_a1","req046_a2"]
lam={}; gcl={}
for arm in arms:
    d=json.load(open(f"{HERE}/raw_curvature_json/{arm}.json"))
    M=d[list(d.keys())[0]]["matrices"]
    for n,r in M.items():
        if n.startswith(("embed","proj")): continue
        lam[(n,arm)]=r["top_eigenvalue"]; gcl[(n,arm)]=r["clipped_gradient_block_norm"]
names=sorted(set(n for (n,a) in lam))
def fe_slope(yval, subset=None):
    ns=subset or names; num=den=0.0
    for n in ns:
        xs=[math.log(A[a][n]) for a in arms if (n,a) in yval and yval[(n,a)]>0]
        ys=[math.log(yval[(n,a)]) for a in arms if (n,a) in yval and yval[(n,a)]>0]
        if len(xs)<2: continue
        mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
        for x,y in zip(xs,ys): num+=(x-mx)*(y-my); den+=(x-mx)**2
    return num/den if den else float("nan")
out=[]
out.append(f"EXPONENT\td_log_lambda/d_log_clip\t{fe_slope(lam):+.4f}")
out.append(f"FIRST_STAGE\td_log_gclipped/d_log_clip\t{fe_slope(gcl):+.4f}\t(must be ~1)")
by=defaultdict(list)
for (n,a),v in lam.items(): by[A[a][n]].append(math.log10(v))
lv=sorted(by); means=[(l,sum(by[l])/len(by[l])) for l in lv]
out.append("MONOTONE\tmean_log10_lambda_by_clip\t"+" ".join(f"{l}:{m:+.4f}" for l,m in means))
mono=(means[0][1]<means[1][1]<means[2][1]) or (means[0][1]>means[1][1]>means[2][1])
out.append(f"MONOTONE_PASS\t{mono}")
allpos=True
for t in sorted(set(TYPES[n] for n in names)):
    s=fe_slope(lam,[n for n in names if TYPES[n]==t]); allpos=allpos and s>0
    out.append(f"TYPE\t{t}\t{s:+.4f}")
out.append(f"PERTYPE_ALL_POSITIVE\t{allpos}")
print("\n".join(out))
if __name__=="__main__":
    open(f"{HERE}/readout.tsv","w").write("\n".join(out)+"\n")
