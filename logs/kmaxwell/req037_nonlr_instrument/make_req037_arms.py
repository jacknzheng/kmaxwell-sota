"""REQ-037 arms 1-3: the batch instrument (move gradient noise, LR fixed) at fork@2000, 750 steps, per-matrix curvature.
Built from eos_f2000_s100 (the s=1.0 control fork). arm4 (per-matrix grad clip) needs a new hook -> deferred."""
import yaml, copy, argparse, os
FORK, STOP, BASE_BATCH = 2000, 2750, 524288
ARMS = {
  "a1_control":  (BASE_BATCH,   64),   # 1x
  "a2_batch05x": (BASE_BATCH//2, 32),   # 0.5x -> mbs 32 (needs eager, compile mbs<64 bug)
  "a3_batch2x":  (BASE_BATCH*2,  64),   # 2x
}
ap=argparse.ArgumentParser(); ap.add_argument("--template",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
os.makedirs(a.out,exist_ok=True)
tmpl=yaml.safe_load(open(a.template))
man=["run_id\tbatch_tokens\tmicrobatch_sequences\tskip_batches\teager"]
for arm,(bt,mbs) in ARMS.items():
    c=copy.deepcopy(tmpl)
    rid=f"req037_{arm}"; c["run_id"]=rid
    c["batch_tokens"]=bt; c["microbatch_sequences"]=mbs
    c["stop_after_step"]=STOP
    skip=FORK*BASE_BATCH//bt
    assert FORK*BASE_BATCH%bt==0, arm
    for s in c.get("setup",[]):
        if s.get("name")=="load_training_state": s["hyperparams"]["skip_batches"]=skip
    for h in c.get("pre_optimizer",[]):
        if h.get("name")=="checkpoint_model_at_cadence": h["hyperparams"]["dump_dir"]=f"dumps_{rid}"  # keep every:125 -> hits 2750
    yaml.safe_dump(c, open(f"{a.out}/{rid}.yaml","w"), sort_keys=False)
    man.append(f"{rid}\t{bt}\t{mbs}\t{skip}\t{'yes' if mbs<64 else 'no'}")
open(f"{a.out}/manifest.tsv","w").write("\n".join(man)+"\n")
print("wrote 3 arm configs (a1_control 1x, a2_batch05x 0.5x/eager, a3_batch2x 2x)")
