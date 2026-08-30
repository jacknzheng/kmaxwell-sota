"""Generate the shared-state EoS pinning-law replication configs.

One base run follows the ordinary Track-3 schedule and writes complete training
states at steps 1500 and 2000. Every intervention arm resumes one of those exact
states and applies its fixed learning-rate multiplier beginning with the fork
update. This makes shared state a serialization invariant instead of relying on
separate GPU runs to reproduce a chaotic trajectory bit for bit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROGRAMS = (
    # fork, stop-after step, curvature checkpoints, (label, multiplier)
    (1500, 2750, [2250, 2375, 2500, 2625, 2750],
     [("s060", 0.60), ("s077", 0.77), ("s100", 1.00),
      ("s100dup", 1.00), ("s130", 1.30), ("s170", 1.70)]),
    (2000, 3249, [2750, 2875, 3000, 3125, 3249],
     [("s060", 0.60), ("s100", 1.00), ("s170", 1.70)]),
)


def common(run_id: str) -> dict:
    return {
        "loop": "gpt_record",
        "run_id": run_id,
        "seed": 0,
        "require_world_size": 8,
        # Keep the planned duration at the Track-3 value so the shared
        # pre-fork cooldown is identical in every program.
        "train_steps": 3250,
        "batch_tokens": 524288,
        "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin",
        "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "bimaxwell_muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                             "fast_decay": 0.85, "slow_decay": 0.98,
                             "fast_weight": 0.4385, "switch_step": 1000}},
            {"pattern": ".*", "optimizer": "adamw",
             "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"},
            {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"},
            {"name": "seed_then_initialize_parameters"},
            {"name": "assemble_grouped_optimizer"},
            {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"},
            {"name": "validate_at_step_boundaries"},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125}},
        ],
        "teardown": [{"name": "mark_log_finished"}],
    }


def base_config() -> dict:
    config = common("eos_shared_base")
    config.update(stop_after_step=2000)
    config["pre_optimizer"] = [
            {"name": "dump_training_state_at_steps",
             "hyperparams": {"steps": [1500, 2000],
                             "dump_dir": "eos_shared_state"}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ]
    return config


def config_for(fork: int, stop: int, label: str, multiplier: float) -> dict:
    run_id = f"eos_f{fork}_{label}"
    config = common(run_id)
    config.update(start_step=fork, stop_after_step=stop)
    # load_training_state must follow optimizer construction, data opening, and
    # parameter broadcast. It restores model + every optimizer shard and then
    # advances the data stream to the fork batch.
    config["setup"].insert(-1, {
        "name": "load_training_state",
        "hyperparams": {"state_dir": "eos_shared_state", "step": fork,
                        "skip_batches": fork},
    })
    config["pre_optimizer"] = [
            {"name": "checkpoint_model_at_cadence",
             "hyperparams": {"every": 125, "dump_dir": f"dumps_{run_id}"}},
            {"name": "cool_down_learning_rate",
             "hyperparams": {"cooldown_frac": 0.7,
                             "fixed_eta_after_step": fork,
                             "fixed_eta_after": multiplier}},
        ]
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = base_config()
    (args.out / "eos_shared_base.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False))
    manifest = ["run_id\tfork_step\tmultiplier\tstop_after_step\tcurvature_steps"]
    manifest.append("eos_shared_base\t0\tstandard\t2000\t")
    for fork, stop, checkpoints, arms in PROGRAMS:
        for label, multiplier in arms:
            config = config_for(fork, stop, label, multiplier)
            path = args.out / f"{config['run_id']}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            manifest.append("\t".join((config["run_id"], str(fork), str(multiplier),
                                       str(stop), ",".join(map(str, checkpoints)))))
    (args.out / "manifest.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
