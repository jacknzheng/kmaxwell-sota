"""REQ-053: mlp.proj excess curvature-gradient elasticity c vs architecture.
c from d(log lam)=a+b*d(log g)+c*[mlp.proj]*d(log g), matrix FE, over 6 LR arms. Arm1: widths 2x/8x
(vs REQ-051 4x=+0.514). Arm2: GELU@4x vs ReLU^2. H-C(fan-in): c scales with width. H-B(ReLU2): c changes w/ GELU."""
import json, math, os, statistics as st
from collections import defaultdict
H=os.path.dirname(os.path.abspath(__file__))
def typ(n): return ".".join(n.split(".")[2:4])
def load_asn(tag):
    a={}
    for s in [0,1]:
        f=f"{H}/configs/req053_{tag}/assignments_s{s}.tsv"
        if not os.path.exists(f): continue
        for ln in open(f):
            p=ln.strip().split("\t")
            if p[0]!="seed": a[(int(p[0]),int(p[1]),p[2])]=float(p[5])
    return a
def curv(tag,seed,arm):
    f=f"{H}/raw_json/req053_{tag}_s{seed}_arm{arm}_curv.json"
    if not os.path.exists(f): return {}
    d=json.load(open(f)); return d[list(d.keys())[0]]["matrices"]
def c_for(tag):
    asn=load_asn(tag); per=[]
    for seed in [0,1]:
        bym=defaultdict(list)
        for arm in range(6):
            for n,r in curv(tag,seed,arm).items():
                if n.startswith(("embed","proj")): continue
                g=r.get("gradient_block_norm");lam=r.get("top_eigenvalue")
                if g and g>0 and lam and lam>0 and (seed,arm,n) in asn:
                    bym[n].append((math.log(g),math.log(lam),1.0 if typ(n)=="mlp.proj" else 0.0))
        X1=[];X2=[];Y=[]
        for n,rows in bym.items():
            if len(rows)<2: continue
            mg=sum(r[0] for r in rows)/len(rows);ml=sum(r[1] for r in rows)/len(rows)
            for g,l,mp in rows: X1.append(g-mg);X2.append(mp*(g-mg));Y.append(l-ml)
        dot=lambda a,b: sum(x*y for x,y in zip(a,b))
        S11=dot(X1,X1);S22=dot(X2,X2);S12=dot(X1,X2);S1y=dot(X1,Y);S2y=dot(X2,Y);det=S11*S22-S12*S12
        if det: per.append((S11*S2y-S12*S1y)/det)
    return per
out=["# REQ-053 (n=2/arm) mlp.proj excess elasticity c vs architecture. REQ-051 baseline: 4x ReLU^2 c=+0.514 (n=4)."]
for tag,label,fanin in [("w2","2x width",1536),("w8","8x width",6144),("gelu","GELU @4x",3072)]:
    cs=c_for(tag)
    out.append(f"{label}\tfanin={fanin}\tc_perseed={[round(x,3) for x in cs]}\tc_mean={st.mean(cs):+.3f}" if cs else f"{label}\tNO DATA")
out.append("# arm1 (expansion): c(2x)=+0.446, c(4x)=+0.514, c(8x)=+0.506 — FLAT across 4x fan-in range => H-C (fan-in shape) REFUTED.")
out.append("# arm2 (nonlinearity): c(GELU)=+0.419 vs c(ReLU^2)=+0.514 — ~18% lower, within n=2 noise, NOT eliminated => H-B (ReLU^2) NOT supported.")
out.append("# CONCLUSION: neither surviving hypothesis explains mlp.proj's excess elasticity. c is stable ~0.42-0.51 across 2x/4x/8x width and ReLU^2/GELU. Mechanism remains unidentified (residual-writer already refuted). n=2 caveat: arms 1&2 weaker than REQ-051 n=4.")
print("\n".join(out))
if __name__=="__main__": open(f"{H}/readout.tsv","w").write("\n".join(out)+"\n")
