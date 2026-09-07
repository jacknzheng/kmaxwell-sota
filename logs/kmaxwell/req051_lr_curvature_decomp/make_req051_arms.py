"""REQ-051: 6-level per-matrix LR Latin square for one seed. per_matrix_lr_muon @25d3208.
Levels {0.50,0.65,0.85,1.00,1.30,1.70}; 6 arms; mult(arm,block)=L[(block+arm+seed)%6] so:
 - each matrix gets each level exactly once across the 6 arms;
 - each arm assigns each level to exactly 12 matrices (2 of each of the 6 types);
 - the block->level mapping is rotated per seed.
Fork@2000 -> stop@2750, model checkpoint every 50 (covers 2050 & 2750; driver prunes the rest).
Emits assignments.tsv (matrix, type, block, arm -> multiplier) for the decomposition analysis."""
import sys, os, re, yaml, argparse
sys.path.insert(0, "records/track_3_optimization")
import importlib.util
spec=importlib.util.spec_from_file_location("mplr","records/track_3_optimization/offline_analysis/make_per_matrix_lr_configs.py")
mplr=importlib.util.module_from_spec(spec); spec.loader.exec_module(mplr)
names=mplr.sorted_matrix_names(); mtype=mplr.matrix_type
FORK,STOP=2000,2750
L=[0.50,0.65,0.85,1.00,1.30,1.70]
def block(n): return int(re.match(r"blocks\.(\d+)\.",n).group(1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    man=["run_id\tarm\tlevel_counts"]; assign=["seed\tarm\tmatrix\ttype\tblock\tmultiplier"]
    for arm in range(6):
        mult_by_name={n: L[(block(n)+arm+a.seed)%6] for n in names}
        mult=[mult_by_name[n] for n in names]
        rid=f"req051_s{a.seed}_arm{arm}"
        cfg=mplr.common(rid,FORK,mult); cfg["stop_after_step"]=STOP; cfg["seed"]=a.seed
        for h in cfg.get("pre_optimizer",[]):
            if h["name"]=="checkpoint_model_at_cadence": h["hyperparams"]["every"]=50   # 2050 & 2750 both multiples of 50
        open(f"{a.out}/{rid}.yaml","w").write(yaml.safe_dump(cfg,sort_keys=False))
        cnt={l:sum(1 for v in mult_by_name.values() if v==l) for l in L}
        man.append(f"{rid}\t{arm}\t"+",".join(f"{l}:{cnt[l]}" for l in L))
        for n in names: assign.append(f"{a.seed}\t{arm}\t{n}\t{mtype(n)}\t{block(n)}\t{mult_by_name[n]}")
    open(f"{a.out}/manifest_s{a.seed}.tsv","w").write("\n".join(man)+"\n")
    open(f"{a.out}/assignments_s{a.seed}.tsv","w").write("\n".join(assign)+"\n")
    # verify Latin properties
    import collections
    per_type_per_level=collections.defaultdict(lambda: collections.defaultdict(set))
    for arm in range(6):
        for n in names:
            l=L[(block(n)+arm+a.seed)%6]; per_type_per_level[(mtype(n),arm)][l].add(n)
    ok=all(len(s)==2 for d in per_type_per_level.values() for s in d.values())
    print(f"seed {a.seed}: wrote 6 arms; matrices={len(names)}; stratified(2/type/level/arm)={ok}")
if __name__=="__main__": main()
