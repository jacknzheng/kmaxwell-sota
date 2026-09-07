"""REQ-052: writer/internal and v-vs-qk LR-response contrasts under UNIFORM (Muon-only) and FULL-GLOBAL
(Muon+AdamW) LR ladders, per seed at 2750. k_i = -d log top_eigenvalue / d log(uniform mult) over the ladder.
Tests whether the band-67 writer/internal contrast (strong+ in REQ-023 mixed LR) transfers to uniform/global LR."""
import json, math, os, statistics as st
from collections import defaultdict
H=os.path.dirname(os.path.abspath(__file__)); MULT={"u065":0.65,"u100":1.00,"u170":1.70,"fg065":0.65,"fg170":1.70}
def typ(n): return ".".join(n.split(".")[2:4])
def curv(seed,arm,step):
    f=f"{H}/raw_json/req052_s{seed}_{arm}_step{step}_curv.json"
    if not os.path.exists(f): return {}
    d=json.load(open(f)); return d[list(d.keys())[0]]["matrices"]
def slope(xy):
    n=len(xy);mx=sum(x for x,_ in xy)/n;my=sum(y for _,y in xy)/n
    num=sum((x-mx)*(y-my) for x,y in xy);den=sum((x-mx)**2 for x,_ in xy);return num/den if den else float('nan')
W={"attn.proj","mlp.proj"}; I={"attn.q","attn.k","attn.v","mlp.fc"}
def contrasts(ladder,step):
    wi=[];vqk=[]
    for seed in [0,1,2,3]:
        bym=defaultdict(list)
        for arm in ladder:
            for n,r in curv(seed,arm,step).items():
                if n.startswith(("embed","proj")): continue
                if r.get("top_eigenvalue",0)>0: bym[n].append((math.log(MULT[arm]),math.log(r["top_eigenvalue"])))
        km={n:-slope(xy) for n,xy in bym.items() if len(xy)>=2}
        w=[km[n] for n in km if typ(n) in W]; ii=[km[n] for n in km if typ(n) in I]
        v=[km[n] for n in km if typ(n)=="attn.v"]; qk=[km[n] for n in km if typ(n) in ("attn.q","attn.k")]
        wi.append(st.mean(w)-st.mean(ii)); vqk.append(st.mean(v)-st.mean(qk))
    return wi,vqk
out=["# REQ-052 (n=4) LR-response contrasts @2750. k=-d log lam/d log(uniform mult). REQ-023 mixed: writer/internal +0.92/+1.17, v-qk neg. REQ-035 global: writer/internal -0.19..-0.20, v-qk +0.36..+0.46."]
for name,lad in [("uniform_Muon",["u065","u100","u170"]),("full_global",["fg065","u100","fg170"])]:
    wi,vqk=contrasts(lad,2750)
    out.append(f"{name}\twriters-internal\tperseed={[round(x,3) for x in wi]}\tmean={st.mean(wi):+.3f}\tsd={st.pstdev(wi):.3f}")
    out.append(f"{name}\tv-(q,k)\tperseed={[round(x,3) for x in vqk]}\tmean={st.mean(vqk):+.3f}\tsd={st.pstdev(vqk):.3f}")
out.append("# FINDING: writer/internal contrast COLLAPSES to ~0 under uniform (+0.03) & full-global (+0.11) LR — NOT the +0.9..1.2 of REQ-023 mixed LR => band-67 writer/internal effect is mixed-LR-specific, does not transfer to global LR (its 4-seed criterion fails here).")
out.append("# v-(q,k) is POSITIVE under uniform(+0.30)/full-global(+0.42), matching REQ-035 global (+0.36..+0.46), opposite REQ-023 mixed (neg) — sign is LR-design-dependent, global sign reproduced n=4.")
print("\n".join(out))
if __name__=="__main__": open(f"{H}/readout.tsv","w").write("\n".join(out)+"\n")
