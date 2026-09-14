"""REQ-059 arm config generator: base (K-Maxwell, dump@2000) + 9 arm continuations (2000->2750, 750 updates),
validation on the 10485760-tok final set at 2000/2125/2250/2375/2500/2625/2750.
Arms: global_a05/global_a1/global_a2/guided/euclidean/typedepth/shuffled/reversed (allocated optimizer +
tag_req059_allocation) + nomom (muon mu=0). Usage: make_req059_configs.py --out DIR --seed S --alloc ALLOC.json
"""
import argparse, yaml
from pathlib import Path
DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218, 0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
SW = [0.005093975, 0.010187949, 0.015281924, 0.020375898, 0.025469873, 0.030563847, 0.035657822, 0.857368713]
EW = [0.032261839, 0.064523678, 0.096785516, 0.129047355, 0.161309194, 0.193571033, 0.225832871, 0.096668514]
FORK, END, TRAIN_STEPS = 2000, 2750, 3250
VAL_STEPS = [2000, 2125, 2250, 2375, 2500, 2625, 2750]
ALLOC_ARMS = ["global_a05", "global_a1", "global_a2", "guided", "euclidean", "typedepth", "shuffled", "reversed"]


def base_hp(extra=None):
    hp = {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95, "decays": DECAYS, "start_weights": SW,
          "end_weights": EW, "switch_step": FORK, "anneal_end_step": END}
    if extra: hp.update(extra)
    return hp


def groups(blocks):
    return [{"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            blocks,
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.001}}]


def common(run_id, seed):
    return {"loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
            "train_steps": TRAIN_STEPS, "batch_tokens": 524288, "microbatch_sequences": 64,
            "train_data": "data/fineweb10B/fineweb_train_*.bin",
            "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
            "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
            "setup": [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                      {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                      {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                      {"name": "broadcast_initial_parameters"}, {"name": "validate_at_step_boundaries"}],
            "post_optimizer": [{"name": "print_training_progress"}], "teardown": [{"name": "mark_log_finished"}]}


def base_config(seed):
    c = common(f"req059_base_s{seed}", seed)
    c["optimizer_groups"] = groups({"pattern": r"^blocks\..*\.weight$", "optimizer": "annealed_weights_muon", "hyperparams": base_hp()})
    c.update(stop_after_step=FORK)
    c["pre_optimizer"] = [{"name": "dump_training_state_at_steps", "hyperparams": {"steps": [FORK], "dump_dir": f"req059_state_s{seed}"}},
                          {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    return c


def arm_config(seed, arm, alloc_file):
    c = common(f"req059_{arm}_s{seed}", seed)
    if arm == "nomom":
        blocks = {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.0}}
    else:
        blocks = {"pattern": r"^blocks\..*\.weight$", "optimizer": "allocated_annealed_weights_muon",
                  "hyperparams": base_hp({"a_values": [0.5, 2.0]})}
    c["optimizer_groups"] = groups(blocks)
    c.update(start_step=FORK, stop_after_step=END)
    c["setup"].insert(-1, {"name": "load_training_state", "hyperparams": {"state_dir": f"req059_state_s{seed}", "step": FORK, "skip_batches": FORK}})
    if arm in ALLOC_ARMS:
        c["setup"].insert(-1, {"name": "tag_req059_allocation", "hyperparams": {"alloc_file": alloc_file, "arm": arm}})
    c["setup"][-1] = {"name": "validate_at_step_boundaries", "hyperparams": {"dense_window": [FORK, END], "dense_every": 125}}
    c["pre_optimizer"] = [{"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    c["post_optimizer"] = [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"dense_window": [FORK, END], "dense_every": 125}}]
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--alloc", required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"base_s{a.seed}.yaml").write_text(yaml.safe_dump(base_config(a.seed), sort_keys=False))
    arms = ALLOC_ARMS + ["nomom"]
    for arm in arms:
        (a.out / f"{arm}_s{a.seed}.yaml").write_text(yaml.safe_dump(arm_config(a.seed, arm, a.alloc), sort_keys=False))
    print(f"seed{a.seed}: base + {len(arms)} arm configs -> {a.out}")


if __name__ == "__main__":
    main()
