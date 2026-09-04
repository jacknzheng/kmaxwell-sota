"""REQ-045: crossed global x per-matrix LR (identifiable partial/total design). Run on-box @25d3208.
3 arms S_arm in {0.7,1.0,1.4}; per-matrix m_i drawn independently per arm from {0.6,0.85,1.0,1.2,1.7};
effective multiplier per matrix = S_arm * m_i. Fork@2000, stop@2750, per-matrix curvature+weight_frob@2750.
Reuses REQ-023/036 PerMatrixLrMuon machinery (make_per_matrix_lr_configs.py). Emits req045_draws.json
(own m_i, S, type per matrix per arm) for the registered regression, and pre-checks corr(own, others' mean)<0.9."""
import sys, os, re, json, yaml, random, argparse, math
sys.path.insert(0, "records/track_3_optimization")
import importlib.util
spec=importlib.util.spec_from_file_location("mplr","records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py")
mplr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mplr)
names=mplr.sorted_matrix_names(); mtype=mplr.matrix_type
FORK,STOP=2000,2750
LEVELS=[0.6,0.85,1.0,1.2,1.7]
S_ARMS=[("s07",0.7),("s10",1.0),("s14",1.4)]

def corr(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx>0 and sy>0 else float("nan")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=45)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    man=["run_id\tarm\tS_arm"]; draws={}
    own_l=[]; oth_l=[]  # for identifiability pre-check: log(own m_i) vs log(others' mean m_j), pooled over arms
    for ai,(arm,S) in enumerate(S_ARMS):
        rng=random.Random(a.seed*10+ai)               # reproducible independent draw per arm
        mi={n: rng.choice(LEVELS) for n in names}
        draws[arm]={"S":S,"m":mi}
        mult=[S*mi[n] for n in names]
        rid=f"req045_{arm}"
        cfg=mplr.common(rid,FORK,mult); cfg["stop_after_step"]=STOP
        for h in cfg.get("pre_optimizer",[]):
            if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["every"]=250   # fires at 2750 (750 missed it)
        open(f"{a.out}/{rid}.yaml","w").write(yaml.safe_dump(cfg,sort_keys=False))
        man.append(f"{rid}\t{arm}\t{S}")
        # pooled own/others' for the crossing-identifiability check (uses m_i only; S is shared, absorbed by fixed effects)
        for i,n in enumerate(names):
            others=[mi[m] for j,m in enumerate(names) if j!=i]
            own_l.append(math.log(mi[n])); oth_l.append(math.log(sum(others)/len(others)))
    c=corr(own_l,oth_l)
    open(f"{a.out}/manifest.tsv","w").write("\n".join(man)+"\n")
    json.dump({"names":names,"types":{n:mtype(n) for n in names},"draws":draws,
               "identifiability_corr_own_vs_othersmean":c}, open(f"{a.out}/req045_draws.json","w"),indent=1)
    print(f"wrote 3 arms; matrices={len(names)}; identifiability corr(own,others'mean)={c:.3f} (need |corr|<0.9)")
    if abs(c)>=0.9: print("WARN: identifiability corr >=0.9 -- design degenerate, reseed"); sys.exit(3)
if __name__=="__main__": main()
