"""REQ-047 registered checks (n=4) + per-type summary. Reads raw_json/req047_seed{0..3}.json."""
import json, math, os
from collections import defaultdict
H=os.path.dirname(os.path.abspath(__file__))
def typ(n): return ".".join(n.split(".")[2:4])
seeds={s: json.load(open(f"{H}/raw_json/req047_seed{s}.json"))["matrices"] for s in range(4)}
types=["attn.q","attn.k","attn.v","attn.proj","mlp.fc","mlp.proj"]
def tmean(s,t,field,sub=None):
    v=[]
    for n,r in seeds[s].items():
        if typ(n)==t and field in r and r[field] is not None:
            x=r[field][sub] if sub else r[field]
            if x==x: v.append(x)
    return sum(v)/len(v) if v else float("nan")
def corr(xs,ys):
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
    cov=sum((a-mx)*(b-my) for a,b in zip(xs,ys));sx=math.sqrt(sum((a-mx)**2 for a in xs));sy=math.sqrt(sum((b-my)**2 for b in ys))
    return cov/(sx*sy) if sx>0 and sy>0 else float("nan")
out=[]
# CHECK 1
p=0
for s in range(4):
    qk=(tmean(s,"attn.q","da_cos_mean")+tmean(s,"attn.k","da_cos_mean"))/2
    vp=(tmean(s,"attn.v","da_cos_mean")+tmean(s,"attn.proj","da_cos_mean"))/2
    p+= qk<vp
    out.append(f"CHECK1_seed{s}\tda_cos (q,k)={qk:+.4f} < (v,proj)={vp:+.4f}\t{qk<vp}")
out.append(f"CHECK1_RESULT\tq,k less token-coherent than v/proj in {p}/4 seeds (>=3 pass)\t{'PASS' if p>=3 else 'FAIL'}")
# CHECK 2
xs=[];ys=[]
for s in range(4):
    for n,r in seeds[s].items():
        if r.get("grad_rank1_frac")==r.get("grad_rank1_frac") and r.get("align_ratio")==r.get("align_ratio") and "grad_rank1_frac" in r and "align_ratio" in r:
            xs.append(r["align_ratio"]); ys.append(r["grad_rank1_frac"])
rc=corr(xs,ys)
out.append(f"CHECK2_RESULT\tcorr(align_ratio, grad_rank1_frac)={rc:+.3f} over {len(xs)} obs (|r|>0.5)\t{'PASS' if abs(rc)>0.5 else 'FAIL'}")
# CHECK 3
for t in types:
    xs=[];ys=[]
    for s in range(4):
        for n,r in seeds[s].items():
            if typ(n)==t and r.get("a_token_norms") and r.get("d_token_norms"):
                xs.append(r["a_token_norms"]["participation"]); ys.append(r["d_token_norms"]["participation"])
    out.append(f"CHECK3_{t}\tcorr(a_part,d_part) across depth={corr(xs,ys):+.3f}\t(neg=support, ~0=scale)")
# per-type summary of the 4 new fields (n=4 mean)
out.append("#\n# per-type means (n=4): da_cos_mean | grad_rank1_frac | a_participation | d_participation")
for t in types:
    dc=sum(tmean(s,t,"da_cos_mean") for s in range(4))/4
    r1=sum(tmean(s,t,"grad_rank1_frac") for s in range(4))/4
    ap=sum(tmean(s,t,"a_token_norms","participation") for s in range(4))/4
    dp=sum(tmean(s,t,"d_token_norms","participation") for s in range(4))/4
    out.append(f"TYPE_{t}\tda_cos={dc:+.4f}\trank1={r1:.4f}\ta_part={ap:.1f}\td_part={dp:.1f}")
print("\n".join(out))
if __name__=="__main__": open(f"{H}/readout.tsv","w").write("\n".join(out)+"\n")
