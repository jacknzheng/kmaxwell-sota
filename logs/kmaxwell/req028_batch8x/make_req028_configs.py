"""REQ-028 8x-batch point: 2 arms, seed 0, 8x batch x {muon mu0, bimaxwell record}.
Same shared step-2000 state/machinery as REQ-026/027; checkpoint +750 only.
Token-aligned skip_batches = FORK*BASE_BATCH//batch_tokens = 250 at 8x."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

FORK, STOP, BASE_BATCH, CKPT_EVERY = 2000, 2750, 524288, 2750
BATCH_8X = 8 * BASE_BATCH  # 4194304
KERNELS = {
    "mu0":  {"optimizer": "muon", "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.0}},
    "bimax": {"optimizer": "bimaxwell_muon",
              "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                              "fast_decay": 0.85, "slow_decay": 0.98,
                              "fast_weight": 0.4385, "switch_step": 1000}},
}
ARMS = [("mu0", 0), ("bimax", 0)]

def common(run_id, batch_tokens, blocks_group, seed):
    return {
        "loop": "gpt_record", "run_id": run_id, "seed": seed, "require_world_size": 8,
        "train_steps": 3250, "batch_tokens": batch_tokens, "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": r"^blocks\..*\.weight$", **blocks_group},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
            {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"},
            {"name": "load_training_state", "hyperparams": {"state_dir": "eos_shared_state", "step": FORK,
                                                           "skip_batches": FORK * BASE_BATCH // batch_tokens}},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "checkpoint_model_at_cadence", "hyperparams": {"every": CKPT_EVERY, "dump_dir": f"dumps_{run_id}"}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ],
        "post_optimizer": [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125}}],
        "teardown": [{"name": "mark_log_finished"}],
        "start_step": FORK, "stop_after_step": STOP,
    }

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    man = ["run_id\tbatch_tokens\tskip_batches\tkernel\tmu\tseed"]
    for klabel, seed in ARMS:
        rid = f"req028_b8x_{klabel}_s{seed}"
        cfg = common(rid, BATCH_8X, KERNELS[klabel], seed)
        (args.out / f"{rid}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        man.append("\t".join(map(str, [rid, BATCH_8X, FORK*BASE_BATCH//BATCH_8X, klabel,
                                        KERNELS[klabel]["hyperparams"]["mu"], seed])))
    (args.out / "manifest.tsv").write_text("\n".join(man) + "\n")
    print(f"wrote {len(ARMS)} configs; skip_batches={FORK*BASE_BATCH//BATCH_8X}, batch_tokens={BATCH_8X}")

if __name__ == "__main__":
    main()
