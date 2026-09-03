"""REQ-036: 5 per-type equalized-curvature LR arms. Run on-box (needs the model for sorted_matrix_names).
Builds per-matrix lr_multipliers (matrix -> type -> arm multiplier), fork@2000, stop@2750, checkpoint+curvature@2750."""
import sys, json, yaml, re
sys.path.insert(0, "records/track_3_optimization")
import importlib.util
spec=importlib.util.spec_from_file_location("mplr","records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py")
mplr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mplr)

names = mplr.sorted_matrix_names()          # ordered blocks matrices
mtype = mplr.matrix_type                     # name -> "attn.q" etc.
FORK, STOP = 2000, 2750

A2 = {"attn.proj":0.40,"attn.k":0.88,"mlp.fc":0.91,"attn.q":1.18,"attn.v":1.25,"mlp.proj":1.56}
A5 = {"attn.q":0.568,"attn.k":0.755,"attn.proj":0.642,"attn.v":1.101,"mlp.fc":1.260,"mlp.proj":2.462}
A4 = {t: round(1.0/v,4) for t,v in A2.items()}                       # anti-rule = 1/arm2
def block_of(n): return int(re.match(r"blocks\.(\d+)\.", n).group(1))

def per_matrix(arm):
    out=[]
    for n in names:
        t=mtype(n); b=block_of(n)
        if arm=="a1_control": m=1.0
        elif arm=="a2_pertype": m=A2[t]
        elif arm=="a4_antirule": m=A4[t]
        elif arm=="a5_polar": m=A5[t]
        elif arm=="a3_endcap":
            if b in (0,11) and t=="attn.proj": m=1.20
            elif b in (0,11) and t=="mlp.proj": m=3.00
            else: m=A2[t]
        out.append(float(m))
    return out

import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
import os; os.makedirs(a.out,exist_ok=True)
man=["run_id\tarm"]
for arm in ["a1_control","a2_pertype","a3_endcap","a4_antirule","a5_polar"]:
    rid=f"req036_{arm}"
    cfg=mplr.common(rid, FORK, per_matrix(arm))
    cfg["stop_after_step"]=STOP
    # ensure a checkpoint + curvature dump at STOP: set cadence hooks to fire at 2750
    for h in cfg.get("pre_optimizer",[]):
        if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["every"]=STOP-FORK  # fires at 2750
    open(f"{a.out}/{rid}.yaml","w").write(yaml.safe_dump(cfg,sort_keys=False))
    man.append(f"{rid}\t{arm}")
open(f"{a.out}/manifest.tsv","w").write("\n".join(man)+"\n")
print(f"wrote 5 arm configs to {a.out}; matrices={len(names)}; types={sorted(set(mtype(n) for n in names))}")
