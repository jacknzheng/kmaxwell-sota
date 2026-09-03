"""REQ-034: K-Maxwell on the fork@2000 batch ladder (bi-Maxwell protocol). 6 arms:
5x annealed_weights_muon (1x/2x/4x/8x/16x) + 1x muon mu0 @2x control. Fork@2000, 750 steps, stop@2750, mbs=64 (no eager).
Base = eos_shared_base dump@2000 (val@2000 must ~ 3.44367). Diff kmax - mu0 @2750 vs stored mu0."""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
BASE_BATCH=524288; FORK=2000; STOP=2750
DECAYS=[0.75,0.822852439855,0.877930338626,0.917598547218,0.945180941073,0.963893920846,0.97637869689,0.984615384615]
SW=[0.005093975,0.010187949,0.015281924,0.020375898,0.025469873,0.030563847,0.035657822,0.857368713]
EW=[0.032261839,0.064523678,0.096785516,0.129047355,0.161309194,0.193571033,0.225832871,0.096668514]
FIX={"lr":0.025,"weight_decay":0.05,"mu":0.95}
# (label, batch_tokens)
LADDER={"b1x":BASE_BATCH,"b2x":2*BASE_BATCH,"b4x":4*BASE_BATCH,"b8x":8*BASE_BATCH,"b16x":16*BASE_BATCH}
def blocks(kernel):
    if kernel=="anneal":
        return {"optimizer":"annealed_weights_muon","hyperparams":{**FIX,"decays":DECAYS,"start_weights":SW,"end_weights":EW,
                "switch_step":FORK,"anneal_end_step":STOP,"warm_streams_before_switch":False}}
    if kernel=="mu0":
        return {"optimizer":"muon","hyperparams":{"lr":0.025,"weight_decay":0.05,"mu":0.0}}
def common(rid,bt,blk):
    skip=FORK*BASE_BATCH//bt; assert FORK*BASE_BATCH%bt==0
    return {"loop":"gpt_record","run_id":rid,"seed":0,"require_world_size":8,"train_steps":3250,
      "batch_tokens":bt,"microbatch_sequences":64,
      "train_data":"data/fineweb10B/fineweb_train_*.bin","val_data":"data/fineweb10B/fineweb_val_*.bin","val_tokens":10485760,
      "model":{"vocab_size":50304,"num_layers":12,"model_dim":768},
      "optimizer_groups":[
        {"pattern":r"^embed\.weight$","optimizer":"adamw","hyperparams":{"lr":0.7,"weight_decay":0.001}},
        {"pattern":r"^proj\.weight$","optimizer":"adamw","hyperparams":{"lr":0.004,"weight_decay":0.001}},
        {"pattern":r"^blocks\..*\.weight$",**blk},
        {"pattern":".*","optimizer":"adamw","hyperparams":{"lr":0.015,"weight_decay":0.001}}],
      "setup":[{"name":"open_rank_zero_log"},{"name":"load_validation_tokens"},{"name":"build_compiled_gpt"},
        {"name":"seed_then_initialize_parameters"},{"name":"assemble_grouped_optimizer"},{"name":"open_training_batches"},
        {"name":"broadcast_initial_parameters"},
        {"name":"load_training_state","hyperparams":{"state_dir":"eos_shared_state","step":FORK,"skip_batches":skip}},
        {"name":"validate_at_step_boundaries"}],
      "pre_optimizer":[{"name":"checkpoint_model_at_cadence","hyperparams":{"every":STOP,"dump_dir":f"dumps_{rid}"}},
        {"name":"cool_down_learning_rate","hyperparams":{"cooldown_frac":0.7}}],
      "post_optimizer":[{"name":"print_training_progress"},
        {"name":"validate_at_step_boundaries","hyperparams":{"every":125,"dense_window":[STOP-250,STOP],"dense_every":10}}],
      "teardown":[{"name":"mark_log_finished"}],"start_step":FORK,"stop_after_step":STOP}
ARMS=[("b1x","anneal"),("b2x","anneal"),("b4x","anneal"),("b8x","anneal"),("b16x","anneal"),("b2x","mu0")]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    man=["run_id\tbatch\tbatch_tokens\tskip\tkernel"]
    for bl,k in ARMS:
        bt=LADDER[bl]; rid=f"req034_{bl}_{k}"
        (a.out/f"{rid}.yaml").write_text(yaml.safe_dump(common(rid,bt,blocks(k)),sort_keys=False))
        man.append(f"{rid}\t{bl}\t{bt}\t{FORK*BASE_BATCH//bt}\t{k}")
    (a.out/"manifest.tsv").write_text("\n".join(man)+"\n"); print(f"wrote {len(ARMS)} configs")
if __name__=="__main__": main()
