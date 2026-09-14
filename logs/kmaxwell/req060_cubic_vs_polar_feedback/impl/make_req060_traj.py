"""REQ-060 trajectory configs: a=0.5/1/2 continuations from base@2000 -> 2064, full-state dumps at 2032 & 2064."""
import argparse, yaml
from pathlib import Path
DECAYS=[0.75,0.822852439855,0.877930338626,0.917598547218,0.945180941073,0.963893920846,0.97637869689,0.984615384615]
SW=[0.005093975,0.010187949,0.015281924,0.020375898,0.025469873,0.030563847,0.035657822,0.857368713]
EW=[0.032261839,0.064523678,0.096785516,0.129047355,0.161309194,0.193571033,0.225832871,0.096668514]
FORK=2000
def hp(extra=None):
    h={"lr":0.025,"weight_decay":0.05,"mu":0.95,"decays":DECAYS,"start_weights":SW,"end_weights":EW,"switch_step":FORK,"anneal_end_step":FORK+750}
    if extra:h.update(extra)
    return h
def groups(b):
    return [{"pattern":r"^embed\.weight$","optimizer":"adamw","hyperparams":{"lr":0.7,"weight_decay":0.001}},
            {"pattern":r"^proj\.weight$","optimizer":"adamw","hyperparams":{"lr":0.004,"weight_decay":0.001}},b,
            {"pattern":".*","optimizer":"adamw","hyperparams":{"lr":0.015,"weight_decay":0.001}}]
def cfg(seed,arm):
    if arm=="a1": blocks={"pattern":r"^blocks\..*\.weight$","optimizer":"annealed_weights_muon","hyperparams":hp()}
    else:
        a=0.5 if arm=="a05" else 2.0
        blocks={"pattern":r"^blocks\..*\.weight$","optimizer":"perturbed_annealed_weights_muon","hyperparams":hp({"perturb_a":a,"perturb_all":True})}
    c={"loop":"gpt_record","run_id":f"req060_{arm}_s{seed}","seed":seed,"require_world_size":8,"train_steps":3250,
       "batch_tokens":524288,"microbatch_sequences":64,"train_data":"data/fineweb10B/fineweb_train_*.bin",
       "val_data":"data/fineweb10B/fineweb_val_*.bin","val_tokens":10485760,
       "model":{"vocab_size":50304,"num_layers":12,"model_dim":768},"optimizer_groups":groups(blocks),
       "setup":[{"name":"open_rank_zero_log"},{"name":"load_validation_tokens"},{"name":"build_compiled_gpt"},
                {"name":"seed_then_initialize_parameters"},{"name":"assemble_grouped_optimizer"},{"name":"open_training_batches"},
                {"name":"broadcast_initial_parameters"},
                {"name":"load_training_state","hyperparams":{"state_dir":f"req058_state_s{seed}_f2000","step":FORK,"skip_batches":FORK}},
                {"name":"validate_at_step_boundaries"}],
       "start_step":FORK,"stop_after_step":2064,
       "pre_optimizer":[{"name":"dump_training_state_at_steps","hyperparams":{"steps":[2032,2064],"dump_dir":f"req060_traj_{arm}_s{seed}"}},
                        {"name":"cool_down_learning_rate","hyperparams":{"cooldown_frac":0.7}}],
       "post_optimizer":[{"name":"print_training_progress"}],"teardown":[{"name":"mark_log_finished"}]}
    return c
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--seed",type=int,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for arm in ("a05","a1","a2"): (a.out/f"{arm}_s{a.seed}.yaml").write_text(yaml.safe_dump(cfg(a.seed,arm),sort_keys=False))
    print(f"seed{a.seed}: 3 trajectory configs")
