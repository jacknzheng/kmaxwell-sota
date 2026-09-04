"""REQ-046: per-matrix gradient-clip instrument. 3 arms, balanced clip {0.5,1.0,2.0}
(REQ-023 Latin square: matrix i, arm a -> LEVELS[(i+a)%3]; each matrix sees all 3 levels
across arms, each arm balanced). per_matrix_clip_muon (grad scaled pre-momentum). Fork@2000
-> stop@2750, ckpt@2750 (every=250), per-matrix curvature+clipped_gradient_block_norm.
Emits req046_a{a}_clip.json (name->clip) for the probe's --clip_json."""
import sys, os, json, yaml, argparse
sys.path.insert(0, "records/track_3_optimization")
import importlib.util
spec=importlib.util.spec_from_file_location("mplr","records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py")
mplr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mplr)
names=mplr.sorted_matrix_names(); mtype=mplr.matrix_type
FORK,STOP=2000,2750
LEVELS=[0.5,1.0,2.0]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    man=["run_id\tarm\tn_at_0.5\tn_at_1.0\tn_at_2.0"]
    assign={}  # arm -> {name: clip}
    for ai in range(3):
        clip_by_name={n: LEVELS[(i+ai)%3] for i,n in enumerate(names)}
        mult=[clip_by_name[n] for n in names]        # sorted-param order for the optimizer
        rid=f"req046_a{ai}"
        cfg=mplr.common(rid, FORK, [1.0]*len(names))  # neutral lr_multipliers placeholder
        for g in cfg["optimizer_groups"]:
            if "blocks" in g.get("pattern",""):
                g["optimizer"]="per_matrix_clip_muon"
                g["hyperparams"].pop("lr_multipliers",None)
                g["hyperparams"]["clip_multipliers"]=mult
        cfg["stop_after_step"]=STOP
        for h in cfg.get("pre_optimizer",[]):
            if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["every"]=250   # fires at 2750
        open(f"{a.out}/{rid}.yaml","w").write(yaml.safe_dump(cfg,sort_keys=False))
        json.dump(clip_by_name, open(f"{a.out}/{rid}_clip.json","w"))
        assign[rid]=clip_by_name
        cnt={l:sum(1 for v in clip_by_name.values() if v==l) for l in LEVELS}
        man.append(f"{rid}\t{ai}\t{cnt[0.5]}\t{cnt[1.0]}\t{cnt[2.0]}")
    open(f"{a.out}/manifest.tsv","w").write("\n".join(man)+"\n")
    json.dump({"names":names,"types":{n:mtype(n) for n in names},"assign":assign,"levels":LEVELS},
              open(f"{a.out}/req046_assign.json","w"),indent=1)
    print(f"wrote 3 arms; matrices={len(names)}; balanced clip levels {LEVELS}")
if __name__=="__main__": main()
