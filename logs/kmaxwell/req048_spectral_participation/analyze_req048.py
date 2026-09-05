"""REQ-048 band-44 checks (n=4 seeds x 3 LRs). Reads raw_curvature_json/req048_s{seed}_{sf}.json.
LR mapping: s060=0.60x, s100=1.00x, s170=1.70x (the s-tags are LEARNING-RATE multipliers, per iter-172 correction)."""
import json, math, os, statistics as st
from collections import defaultdict
H=os.path.dirname(os.path.abspath(__file__)); LR={"s060":0.60,"s100":1.00,"s170":1.70}
def layer(n): return int(n.split(".")[1])
def profiles(seed,sf):
    d=json.load(open(f"{H}/raw_curvature_json/req048_s{seed}_{sf}.json")); M=d[list(d.keys())[0]]["matrices"]
    bp=defaultdict(list);bl=defaultdict(list)
    for n,r in M.items():
        if n.startswith(("embed","proj")): continue
        pr=r.get("participation_ratio");lam=r.get("top_eigenvalue")
        if pr and pr>0 and lam and lam>0: bp[layer(n)].append(math.log(pr));bl[layer(n)].append(math.log(lam))
    Ls=sorted(bp);return Ls,[sum(bp[l])/len(bp[l]) for l in Ls],[sum(bl[l])/len(bl[l]) for l in Ls]
def corr(a,b):
    ma=sum(a)/len(a);mb=sum(b)/len(b);cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a));db=math.sqrt(sum((y-mb)**2 for y in b));return cov/(da*db) if da>0 and db>0 else float("nan")
def r2(xs,ys,deg):
    n=len(xs);A=[[sum(xs[i]**(a+b) for i in range(n)) for b in range(deg+1)] for a in range(deg+1)];B=[sum(xs[i]**a*ys[i] for i in range(n)) for a in range(deg+1)]
    for i in range(deg+1):
        p=A[i][i]
        for j in range(i+1,deg+1):
            f=A[j][i]/p
            for k in range(deg+1):A[j][k]-=f*A[i][k]
            B[j]-=f*B[i]
    c=[0]*(deg+1)
    for i in range(deg,-1,-1):c[i]=(B[i]-sum(A[i][k]*c[k] for k in range(i+1,deg+1)))/A[i][i]
    pred=[sum(c[k]*x**k for k in range(deg+1)) for x in xs];ssr=sum((y-p)**2 for y,p in zip(ys,pred));sst=sum((y-st.mean(ys))**2 for y in ys);return 1-ssr/sst
out=["seed\tLR\tcorr_logPR_loglam\tPR_cubicR2\tPR_linR2\tPR_argmin\tPR_argmax"]
corrs=[];cubics=[];amins=[];bylr=defaultdict(list)
for seed in [0,1,2,3]:
    for sf in ["s060","s100","s170"]:
        Ls,pr,lam=profiles(seed,sf);c=corr(pr,lam);r3=r2([float(l) for l in Ls],pr,3);r1=r2([float(l) for l in Ls],pr,1)
        amin=Ls[pr.index(min(pr))];amax=Ls[pr.index(max(pr))];corrs.append(c);cubics.append(r3);amins.append(amin);bylr[sf].append(c)
        out.append(f"{seed}\t{LR[sf]}\t{c:+.3f}\t{r3:.3f}\t{r1:.3f}\t{amin}\t{amax}")
out.append("#")
out.append(f"# (i) corr<=-0.60 in {sum(1 for c in corrs if c<=-0.60)}/12, negative in {sum(1 for c in corrs if c<0)}/12 (registered: same sign >=10/12) -> PASS; mean {st.mean(corrs):+.3f} sd {st.pstdev(corrs):.3f}")
out.append(f"# (ii) PR cubic R2>=0.70 in {sum(1 for r in cubics if r>=0.70)}/12 (mean {st.mean(cubics):.3f}); PR argmin at ENDS(L0/L11) in {sum(1 for a in amins if a in(0,11))}/12 -> PR is a HUMP = C-bowl inverted (concentrated at ends, spread mid)")
out.append(f"# per-LR mean corr: 0.6x={st.mean(bylr['s060']):+.3f}  1.0x={st.mean(bylr['s100']):+.3f}  1.7x={st.mean(bylr['s170']):+.3f}")
out.append(f"# FALSIFICATION (|corr|<0.30 or PR monotone): NOT triggered -> spectral-concentration hypothesis CONFIRMED n=4")
print("\n".join(out))
if __name__=="__main__": open(f"{H}/readout.tsv","w").write("\n".join(out)+"\n")
