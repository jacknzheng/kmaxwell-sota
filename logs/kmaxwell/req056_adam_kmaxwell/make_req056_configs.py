"""REQ-056 config generator: K-Maxwell first moment inside standard Adam.
Emits, under --out:
  pilot_lr{tag}.yaml         baseline ordinary Adam (no shadow), separate seed, 0->3250, one per LR in {1e-4,3e-4,1e-3}
  base_s{seed}.yaml          ordinary-Adam base (numerator_mode=adam) with shadow K/age buffers accumulating,
                             0->1000, dumps full training state at 1000 into state_s{seed}
  arm_{adam,kmaxwell,agema}_s{seed}.yaml   fork@1000 -> 3249, loads state_s{seed}, blocks numerator_mode per arm
All arms share: beta1=0.9, beta2=0.999, eps=1e-8, zero weight decay; K-Maxwell intervention on the 72
blocks.*.weight matrices only; ordinary Adam (via kmaxwell_adam adam-mode, no shadow) on embed/proj/rest.
Bias correction: K streams 1-decay^t; age EMA variable-decay mass 1-prod(beta(t)). Age schedule 58->26 over
[1000,3250] from the REQ-054 decays/weights. --dump_seed: which seed's arms also dump consecutive model pairs
+ training state at 1000/2050/3248 for the geometry probe + offline telemetry."""
import argparse, yaml
from pathlib import Path

# REQ-054 constants (verified: scheduled age A = 58.0 at start-weights, 26.0 at end-weights)
DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
          0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
START_W = [0.005093975, 0.010187949, 0.015281924, 0.020375898,
           0.025469873, 0.030563847, 0.035657822, 0.857368713]
END_W = [0.032261839, 0.064523678, 0.096785516, 0.129047355,
         0.161309194, 0.193571033, 0.225832871, 0.096668514]
SWITCH, ANNEAL_END, ENDPOINT = 1000, 3250, 3250
PILOT_SEED = 100
LRS = [("1em4", 1e-4), ("3em4", 3e-4), ("1em3", 1e-3)]
SEEDS = [0, 1, 2]
PROBE_STEPS_DENSE = [[1000, 1001], [2050, 2051], [3248, 3249]]  # x_3250 unavailable (range ends 3249)
PROBE_STATE_STEPS = [1000, 2050, 3248]


def adam_group(pattern, lr, mode="adam", shadow=False):
    hp = {"lr": lr, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0,
          "numerator_mode": mode, "accumulate_shadow": shadow}
    if shadow or mode in ("kmaxwell", "age_matched"):
        hp.update(decays=DECAYS, start_weights=START_W, end_weights=END_W,
                  switch_step=SWITCH, anneal_end_step=ANNEAL_END)
    return {"pattern": pattern, "optimizer": "kmaxwell_adam", "hyperparams": hp}


def groups(lr, blocks_mode, blocks_shadow):
    # embed/proj/rest: ordinary Adam (adam-mode kmaxwell_adam, no shadow). blocks: the intervention group.
    return [
        adam_group(r"^embed\.weight$", lr),
        adam_group(r"^proj\.weight$", lr),
        adam_group(r"^blocks\..*\.weight$", lr, mode=blocks_mode, shadow=blocks_shadow),
        adam_group(r".*", lr),
    ]


def common(run_id, seed, lr, blocks_mode, blocks_shadow):
    return {
        "loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
        "train_steps": ENDPOINT, "batch_tokens": 524288, "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": groups(lr, blocks_mode, blocks_shadow),
        "setup": [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"}, {"name": "validate_at_step_boundaries"}],
        "post_optimizer": [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125}}],
        "teardown": [{"name": "mark_log_finished"}],
    }


def pilot(tag, lr):
    c = common(f"req056_pilot_{tag}", PILOT_SEED, lr, "adam", False)
    c.update(stop_after_step=3249)
    c["pre_optimizer"] = [{"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    return c


def base(seed, lr):
    c = common(f"req056_base_s{seed}", seed, lr, "adam", True)  # shadow ON in base
    c.update(stop_after_step=1000)
    c["pre_optimizer"] = [
        {"name": "dump_training_state_at_steps", "hyperparams": {"steps": [1000], "dump_dir": f"state_s{seed}"}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    return c


def arm(seed, lr, arm_name, mode, do_dump):
    c = common(f"req056_{arm_name}_s{seed}", seed, lr, mode, True)
    c.update(start_step=SWITCH, stop_after_step=3249)
    c["setup"].insert(-1, {"name": "load_training_state",
                           "hyperparams": {"state_dir": f"state_s{seed}", "step": SWITCH, "skip_batches": SWITCH}})
    pre = [{"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}]
    if do_dump:
        dd = f"dumps_req056_{arm_name}_s{seed}"
        pre.insert(0, {"name": "checkpoint_model_at_cadence",
                       "hyperparams": {"every": 100000, "dump_dir": dd, "dense_windows": PROBE_STEPS_DENSE}})
        pre.insert(1, {"name": "dump_training_state_at_steps",
                       "hyperparams": {"steps": PROBE_STATE_STEPS, "dump_dir": f"optstate_{arm_name}_s{seed}"}})
    c["pre_optimizer"] = pre
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=None, help="frozen LR for base+arms (from pilot); if None only pilots emitted")
    ap.add_argument("--dump_seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for tag, lr in LRS:
        (a.out / f"pilot_lr{tag}.yaml").write_text(yaml.safe_dump(pilot(tag, lr), sort_keys=False))
    if a.lr is not None:
        for s in SEEDS:
            (a.out / f"base_s{s}.yaml").write_text(yaml.safe_dump(base(s, a.lr), sort_keys=False))
            for arm_name, mode in (("adam", "adam"), ("kmaxwell", "kmaxwell"), ("agema", "age_matched")):
                (a.out / f"{arm_name}_s{s}.yaml").write_text(
                    yaml.safe_dump(arm(s, a.lr, arm_name, mode, do_dump=(s == a.dump_seed)), sort_keys=False))
    print(f"wrote configs to {a.out} (lr={a.lr})")


if __name__ == "__main__":
    main()
