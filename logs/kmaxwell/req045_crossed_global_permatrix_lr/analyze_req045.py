"""REQ-045 registered check: regress d log lambda on log(own mult) AND log(others' mean mult),
matrix fixed effects. Confirm separability |corr(own,others)|<0.9, then test the neighbour coefficient."""
import json, math, os
from collections import defaultdict
H=os.path.dirname(os.path.abspath(__file__))
draws=json.load(open(f"{H}/req045_draws.json")); names=draws["names"]; TYPES=draws["types"]; D=draws["draws"]
arms=["s07","s10","s14"]; lam={}
for a in arms:
    d=json.load(open(f"{H}/raw_curvature_json/req045_{a}.json")); M=d[list(d.keys())[0]]["matrices"]
    for n,r in M.items():
        if not n.startswith(("embed","proj")): lam[(n,a)]=r["top_eigenvalue"]
blk=[n for n in names if not n.startswith(("embed","proj"))]; mats=sorted(set(n for (n,a) in lam))
rows=[]
for n in mats:
    for a in arms:
        S=D[a]["S"]; mi=D[a]["m"][n]; oth=[D[a]["m"][m] for m in blk if m!=n]
        rows.append((n,math.log(lam[(n,a)]),math.log(S*mi),math.log(S*sum(oth)/len(oth))))
bym=defaultdict(list)
for r in rows: bym[r[0]].append(r)
dy=[];dx1=[];dx2=[]
for n,rs in bym.items():
    my=sum(r[1] for r in rs)/len(rs);m1=sum(r[2] for r in rs)/len(rs);m2=sum(r[3] for r in rs)/len(rs)
    for r in rs: dy.append(r[1]-my);dx1.append(r[2]-m1);dx2.append(r[3]-m2)
N=len(dy); dot=lambda a,b: sum(x*y for x,y in zip(a,b))
def corr(a,b):
    ma=sum(a)/len(a);mb=sum(b)/len(b);av=[x-ma for x in a];bv=[y-mb for y in b]
    return dot(av,bv)/(math.sqrt(dot(av,av))*math.sqrt(dot(bv,bv)))
c=corr(dx1,dx2);S11=dot(dx1,dx1);S22=dot(dx2,dx2);S12=dot(dx1,dx2);S1y=dot(dx1,dy);S2y=dot(dx2,dy);det=S11*S22-S12*S12
b1=(S22*S1y-S12*S2y)/det;b2=(S11*S2y-S12*S1y)/det
res=[dy[i]-b1*dx1[i]-b2*dx2[i] for i in range(N)];dof=N-len(bym)-2;sig2=dot(res,res)/dof
se1=math.sqrt(sig2*S22/det);se2=math.sqrt(sig2*S11/det)
out=[f"SEPARABILITY\tcorr_own_vs_othersmean_postFE\t{c:+.3f}\t{'PASS<0.9' if abs(c)<0.9 else 'FAIL'}",
     f"beta_own\td_log_lambda/d_log_own_mult\t{b1:+.4f}\tSE{se1:.4f}\tt={b1/se1:+.2f}",
     f"beta_neighbour\td_log_lambda/d_log_others_mean\t{b2:+.4f}\tSE{se2:.4f}\tt={b2/se2:+.2f}",
     f"NEIGHBOUR_SIGNIFICANT_t>2\t{abs(b2/se2)>2.0}"]
print("\n".join(out))
if __name__=="__main__": open(f"{H}/readout.tsv","w").write("\n".join(out)+"\n")
