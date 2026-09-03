"""REQ-033: does the ANNEALED K-Maxwell kernel survive a change of batch size?

12 arms = batch{0.25x,0.5x,2x} x kernel{A single-EMA, B bimaxwell, C annealed
shipped, D annealed batch-rescaled}, seed 0, 2250-step continuations from a shared
step-1000 state. The 1x row is deliberately omitted (existing n=8 full-run context).

Shared base: the Track-3 bimaxwell base with switch_step=1000 runs as PLAIN MUON
through step 1000, so dumping training state at step 1000 (a pre_optimizer hook,
before the switch fires) yields a single-momentum state with NO K-buffers. Arms
B/C/D lazy-init their streams from that momentum at switch_step=1000 exactly as
PR #357 does; arm A continues as muon. Base val@1000 is recorded (val every 125).

Every arm resumes eos_shared_state/step 1000 and token-aligns the data skip
(1000*524288 // batch_tokens) so all arms share the same ~0.524B-token position.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

BASE_BATCH = 524288          # 1x batch_tokens
FORK, STOP, TRAIN_STEPS = 1000, 3250, 3250
DENSE_WINDOW = [3000, 3250]  # dense val every 10 in the tail

# --- weight lists: scale-invariant, identical for C and every D (only decays vary)
START_WEIGHTS = [0.005093975, 0.010187949, 0.015281924, 0.020375898,
                 0.025469873, 0.030563847, 0.035657822, 0.857368713]   # mean age 58*s
END_WEIGHTS   = [0.032261839, 0.064523678, 0.096785516, 0.129047355,
                 0.161309194, 0.193571033, 0.225832871, 0.096668514]   # mean age 26*s

# --- decays: C shipped (all batches); D per-batch (ages x (1x/batch))
C_DECAYS = [0.75, 0.822852439855, 0.877930338626, 0.917598547218,
            0.945180941073, 0.963893920846, 0.97637869689, 0.984615384615]
D_DECAYS = {
    "b0.25x": [0.923076923077, 0.948927596166, 0.966407077116, 0.978042648561,
               0.985707613901, 0.99072224263, 0.993988168757, 0.996108949416],  # x4
    "b0.5x":  [0.857142857143, 0.902818485868, 0.934997769159, 0.957028830199,
               0.971818015605, 0.981615056307, 0.988048189779, 0.992248062016],  # x2
    "b2x":    [0.6, 0.699022338163, 0.782420529533, 0.84774327017,
               0.896059786816, 0.930304280845, 0.953847574219, 0.969696969697],  # x0.5
}

# batch label -> (batch_tokens, microbatch_sequences)
BATCHES = {
    "b0.25x": (131072, 16),
    "b0.5x":  (262144, 32),
    "b2x":    (1048576, 64),
}

FIXED = {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95}

def blocks_group(klabel: str, blabel: str) -> dict:
    if klabel == "A":
        return {"optimizer": "muon", "hyperparams": dict(FIXED)}
    if klabel == "B":
        return {"optimizer": "bimaxwell_muon",
                "hyperparams": {**FIXED, "fast_decay": 0.85, "slow_decay": 0.98,
                                "fast_weight": 0.4385, "switch_step": 1000}}
    if klabel in ("C", "D"):
        decays = C_DECAYS if klabel == "C" else D_DECAYS[blabel]
        return {"optimizer": "annealed_weights_muon",
                "hyperparams": {**FIXED, "decays": decays,
                                "start_weights": START_WEIGHTS, "end_weights": END_WEIGHTS,
                                "switch_step": 1000, "anneal_end_step": 3250,
                                "warm_streams_before_switch": False}}
    raise ValueError(klabel)

def common(run_id: str, batch_tokens: int, microbatch_sequences: int) -> dict:
    return {
        "loop": "gpt_record", "run_id": run_id, "seed": 0, "require_world_size": 8,
        "train_steps": TRAIN_STEPS, "batch_tokens": batch_tokens,
        "microbatch_sequences": microbatch_sequences,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": r"^blocks\..*\.weight$"},  # filled by caller
            {"pattern": ".*", "optimizer": "adamw",
             "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
            {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"},
            {"name": "validate_at_step_boundaries"},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries",
             "hyperparams": {"every": 125, "dense_window": DENSE_WINDOW, "dense_every": 10}},
        ],
        "teardown": [{"name": "mark_log_finished"}],
    }

def base_config() -> dict:
    # 1x, plain-Muon-through-1000 (bimaxwell switch_step=1000), dump state @1000.
    cfg = common("eos_shared_base", BASE_BATCH, 64)
    cfg["optimizer_groups"][2] = {"pattern": r"^blocks\..*\.weight$",
                                  **blocks_group("B", "b1x_base")}
    cfg.update(stop_after_step=FORK)
    cfg["pre_optimizer"] = [
        {"name": "dump_training_state_at_steps",
         "hyperparams": {"steps": [FORK], "dump_dir": "eos_shared_state"}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
    ]
    return cfg

def arm_config(blabel: str, klabel: str) -> dict:
    batch_tokens, mbs = BATCHES[blabel]
    run_id = f"req033_{blabel}_{klabel}_s0"
    skip = FORK * BASE_BATCH // batch_tokens
    assert FORK * BASE_BATCH % batch_tokens == 0, f"skip not integer for {blabel}"
    cfg = common(run_id, batch_tokens, mbs)
    cfg["optimizer_groups"][2] = {"pattern": r"^blocks\..*\.weight$",
                                  **blocks_group(klabel, blabel)}
    cfg["setup"].insert(-1, {
        "name": "load_training_state",
        "hyperparams": {"state_dir": "eos_shared_state", "step": FORK, "skip_batches": skip}})
    cfg["pre_optimizer"] = [
        {"name": "checkpoint_model_at_cadence",
         "hyperparams": {"every": STOP, "dump_dir": f"dumps_{run_id}"}},
        {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
    ]
    cfg.update(start_step=FORK, stop_after_step=STOP)
    return cfg

# grid order = the request's explicit 1..12 (batch-major: 0.25x, 0.5x, 2x)
ARMS = [(b, k) for b in ("b0.25x", "b0.5x", "b2x") for k in ("A", "B", "C", "D")]

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eos_shared_base.yaml").write_text(
        yaml.safe_dump(base_config(), sort_keys=False))
    man = ["run_id\tbatch_label\tbatch_tokens\tmicrobatch_sequences\tskip_batches\tkernel\toptimizer"]
    man.append("eos_shared_base\tb1x_base\t524288\t64\t0\tbase(muon-thru-1000)\tbimaxwell_muon")
    for blabel, klabel in ARMS:
        bt, mbs = BATCHES[blabel]
        cfg = arm_config(blabel, klabel)
        rid = f"req033_{blabel}_{klabel}_s0"
        (args.out / f"{rid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        opt = cfg["optimizer_groups"][2]["optimizer"]
        man.append(f"{rid}\t{blabel}\t{bt}\t{mbs}\t{FORK*BASE_BATCH//bt}\t{klabel}\t{opt}")
    (args.out / "manifest.tsv").write_text("\n".join(man) + "\n")
    print(f"wrote base + {len(ARMS)} arm configs to {args.out}")

if __name__ == "__main__":
    main()
