"""REQ-052: 5 matched uniform/full-global LR control arms for one seed. per_matrix_lr_muon @25d3208.
Arms: u065(all mult .65), u100(all 1.0), u170(all 1.7) [Muon-only, AdamW at reference];
      fg065(all mult .65 AND non-Muon AdamW x0.65), fg170(all 1.7 AND AdamW x1.70).
Fork@2000->2750, checkpoint every 50 (2050 & 2750)."""
import sys,os,yaml,argparse
sys.path.insert(0,"records/track_3_optimization")
import importlib.util
spec=importlib.util.spec_from_file_location("mplr","records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py")
mplr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mplr)
names=mplr.sorted_matrix_names(); FORK,STOP=2000,2750
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    ARMS=[("u065",0.65,False),("u100",1.00,False),("u170",1.70,False),("fg065",0.65,True),("fg170",1.70,True)]
    for tag,s,fullglobal in ARMS:
        cfg=mplr.common(f"req052_s{a.seed}_{tag}",FORK,[s]*len(names)); cfg["stop_after_step"]=STOP; cfg["seed"]=a.seed
        for h in cfg.get("pre_optimizer",[]):
            if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["every"]=50
        if fullglobal:  # scale non-Muon AdamW group LRs by s
            for g in cfg["optimizer_groups"]:
                if g["optimizer"]=="adamw": g["hyperparams"]["lr"]=round(g["hyperparams"]["lr"]*s,6)
        open(f"{a.out}/req052_s{a.seed}_{tag}.yaml","w").write(yaml.safe_dump(cfg,sort_keys=False))
    print(f"seed {a.seed}: wrote 5 arms (u065/u100/u170/fg065/fg170)")
if __name__=="__main__": main()
