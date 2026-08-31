"""REQ-026 batch-size x momentum-kernel grid config generator.

Six 750-step continuations from the shared step-2000 state (same eos_shared_base
machinery as REQ-019/025). Grid = batch{1x,4x tokens per optimizer step} x
kernel{muon mu=0.0, muon mu=0.95, bimaxwell_muon record}. LR schedule UNCHANGED
across arms (standard cooldown, no multiplier). Checkpoints at +250/+500/+750.

Schema copied verbatim from make_eos_state_dependence_configs.py common()/config_for()
so it byte-matches the Track-3 loader; only batch_tokens and the blocks-group
optimizer differ per arm.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

FORK = 2000
STOP = 2750          # 750-step continuation
BASE_BATCH = 524288  # 1x tokens per optimizer step (microbatch_sequences=64 -> 8 accum)
CKPT_EVERY = 250     # dumps at 2250/2500/2750 (= fork +250/+500/+750) plus the fork

# blocks-group optimizer per kernel. lr/weight_decay held fixed across kernels so
# the only varied axis is the momentum kernel itself.
KERNELS = {
    "mu0":  {"optimizer": "muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.0}},
    "mu95": {"optimizer": "muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95}},
    "bimax": {"optimizer": "bimaxwell_muon",
              "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                              "fast_decay": 0.85, "slow_decay": 0.98,
                              "fast_weight": 0.4385, "switch_step": 1000}},
}
BATCHES = {"b1x": BASE_BATCH, "b4x": 4 * BASE_BATCH}


def common(run_id: str, batch_tokens: int, blocks_group: dict) -> dict:
    return {
        "loop": "gpt_record",
        "run_id": run_id,
        "seed": 0,
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
                             # token-aligned to the fork data position: skip the same
                             # ~1.05B tokens (FORK base-batches) regardless of batch size,
                             # so 1x skips 2000 batches and 4x skips 500.
                             "skip_batches": FORK * BASE_BATCH // batch_tokens}},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "checkpoint_model_at_cadence",
             "hyperparams": {"every": CKPT_EVERY, "dump_dir": f"dumps_{run_id}"}},
            # standard cooldown, NO fixed_eta_after -> LR schedule identical to base
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
    manifest = ["run_id\tbatch_tokens\tbatch_x\tkernel\toptimizer\tmu\tstart\tstop\tckpts"]
    for blabel, btok in BATCHES.items():
        for klabel, kspec in KERNELS.items():
            run_id = f"req026_{blabel}_{klabel}"
            cfg = common(run_id, btok, kspec)
            (args.out / f"{run_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            mu = kspec["hyperparams"]["mu"]
            manifest.append("\t".join(map(str, [
                run_id, btok, blabel[1:], klabel, kspec["optimizer"], mu,
                FORK, STOP, "2250,2500,2750"])))
    (args.out / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    print(f"wrote {len(BATCHES)*len(KERNELS)} configs to {args.out}")


if __name__ == "__main__":
    main()
