"""REQ-054: paired K-Maxwell vs age-matched single-EMA fork configs. fork@2000->2750, switch@2000.
Also emits a checkpoint@2050,2250,2500,2749 for the REQ-055 displacement probe (every=... covers them)."""
import sys,os,yaml,argparse
sys.path.insert(0,"records/track_3_optimization")
BASE_BATCH=524288; FORK=2000; STOP=2750
DECAYS=[0.75,0.822852439855,0.877930338626,0.917598547218,0.945180941073,0.963893920846,0.97637869689,0.984615384615]
SW=[0.005093975,0.010187949,0.015281924,0.020375898,0.025469873,0.030563847,0.035657822,0.857368713]
EW=[0.032261839,0.064523678,0.096785516,0.129047355,0.161309194,0.193571033,0.225832871,0.096668514]
FIX={"lr":0.025,"weight_decay":0.05,"mu":0.95}
def blocks(kernel):
    if kernel=="kmax": return {"optimizer":"annealed_weights_muon","hyperparams":{**FIX,"decays":DECAYS,"start_weights":SW,"end_weights":EW,"switch_step":FORK,"anneal_end_step":STOP,"warm_streams_before_switch":False}}
    if kernel=="agema": return {"optimizer":"age_matched_ema_muon","hyperparams":{**FIX,"decays":DECAYS,"start_weights":SW,"end_weights":EW,"switch_step":FORK,"anneal_end_step":STOP}}
def common(rid,blk,seed):
    skip=FORK  # 1x batch: skip 2000 batches
    return {"loop":"gpt_record","run_id":rid,"seed":seed,"require_world_size":8,"train_steps":3250,
      "batch_tokens":BASE_BATCH,"microbatch_sequences":64,
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
      "pre_optimizer":[{"name":"checkpoint_model_at_cadence","hyperparams":{"every":250,"dump_dir":f"dumps_{rid}"}},
        {"name":"cool_down_learning_rate","hyperparams":{"cooldown_frac":0.7}}],
      "post_optimizer":[{"name":"print_training_progress"},
        {"name":"validate_at_step_boundaries","hyperparams":{"every":125,"dense_window":[STOP-250,STOP],"dense_every":10}}],
      "teardown":[{"name":"mark_log_finished"}],"start_step":FORK,"stop_after_step":STOP}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    for k in ("kmax","agema"):
        rid=f"req054_s{a.seed}_{k}"
        open(f"{a.out}/{rid}.yaml","w").write(yaml.safe_dump(common(rid,blocks(k),a.seed),sort_keys=False))
    print(f"seed {a.seed}: wrote kmax + agema configs")
if __name__=="__main__": main()
