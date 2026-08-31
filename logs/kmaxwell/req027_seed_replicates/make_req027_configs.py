"""REQ-027 seed-replicate configs for the REQ-026 batch x kernel finding.

6 arms = REQ-026 configs with `seed` changed only (REQ-026 was seed 0):
  4x batch: {muon mu0, bimaxwell record} x seeds {1,2}   (4 arms)
  1x batch: {muon mu0, bimaxwell record} x seed {1}       (2 arms)
Identical to REQ-026 otherwise (same shared step-2000 state, token-aligned
skip_batches, LR schedule) EXCEPT checkpoints dumped at +750 ONLY (every=2750).
Schema copied from make_eos_state_dependence_configs.py common()/config_for().
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

FORK = 2000
STOP = 2750
BASE_BATCH = 524288
CKPT_EVERY = 2750  # dumps only at step 2750 (= fork +750)

KERNELS = {
    "mu0":  {"optimizer": "muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.0}},
    "bimax": {"optimizer": "bimaxwell_muon",
              "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                              "fast_decay": 0.85, "slow_decay": 0.98,
                              "fast_weight": 0.4385, "switch_step": 1000}},
}
BATCHES = {"b1x": BASE_BATCH, "b4x": 4 * BASE_BATCH}

# (batch, kernel, seed) for the 6 arms
ARMS = [
    ("b4x", "mu0", 1), ("b4x", "bimax", 1),
    ("b4x", "mu0", 2), ("b4x", "bimax", 2),
    ("b1x", "mu0", 1), ("b1x", "bimax", 1),
]


def common(run_id: str, batch_tokens: int, blocks_group: dict, seed: int) -> dict:
    return {
        "loop": "gpt_record",
        "run_id": run_id,
        "seed": seed,
        "require_world_size": 8,
        "train_steps": 3250,
        "batch_tokens": batch_tokens,
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
            {"pattern": r"^blocks\..*\.weight$", **blocks_group},
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
            {"name": "load_training_state",
             "hyperparams": {"state_dir": "eos_shared_state", "step": FORK,
                             "skip_batches": FORK * BASE_BATCH // batch_tokens}},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "checkpoint_model_at_cadence",
             "hyperparams": {"every": CKPT_EVERY, "dump_dir": f"dumps_{run_id}"}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125}},
        ],
        "teardown": [{"name": "mark_log_finished"}],
        "start_step": FORK,
        "stop_after_step": STOP,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = ["run_id\tbatch_tokens\tbatch_x\tkernel\tmu\tseed\tstart\tstop\tckpt"]
    for blabel, klabel, seed in ARMS:
        run_id = f"req027_{blabel}_{klabel}_s{seed}"
        cfg = common(run_id, BATCHES[blabel], KERNELS[klabel], seed)
        (args.out / f"{run_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        mu = KERNELS[klabel]["hyperparams"]["mu"]
        manifest.append("\t".join(map(str, [run_id, BATCHES[blabel], blabel[1:],
                                            klabel, mu, seed, FORK, STOP, "2750"])))
    (args.out / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print(f"wrote {len(ARMS)} configs to {args.out}")


if __name__ == "__main__":
    main()
