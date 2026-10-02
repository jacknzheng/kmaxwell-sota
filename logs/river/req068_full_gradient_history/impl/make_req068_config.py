"""REQ-068 baseline capture config generator. Standard Muon (mu=0.95) on hidden matrices + the canonical
auxiliary AdamW groups (embed/head/scalars) — the project's verified baseline (= bimaxwell.yaml structure
with plain `muon` on blocks; this is also REQ-072's muon-mom arm). Wires the capture pipeline:
  pre_optimizer : capture_full_gradients (every step) -> dump_training_state_at_steps [500,1500,2500]
                  -> cool_down_learning_rate
  teardown      : finalize_gradient_capture -> mark_log_finished
Raw gradient chunks + fork states go to --grad_dir / --state_dir on a DURABLE volume (set at launch), off Git.

Usage: make_req068_config.py --out CONFIG.yaml --grad_dir <durable>/grads --state_dir <durable>/forks
       [--seed 0] [--chunk_steps 50] [--train_steps 3250]
"""
import argparse, yaml
from pathlib import Path

FORK_STEPS = [500, 1500, 2500]


def build(seed, grad_dir, state_dir, chunk_steps, train_steps):
    return {
        "loop": "gpt_record", "run_id": f"req068_capture_s{seed}", "seed": seed,
        "require_world_size": 8, "train_steps": train_steps, "batch_tokens": 524288,
        "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95}},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
            {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"}, {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "capture_full_gradients",
             "hyperparams": {"out_dir": grad_dir, "chunk_steps": chunk_steps}},
            {"name": "dump_training_state_at_steps",
             "hyperparams": {"steps": FORK_STEPS, "dump_dir": state_dir}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries",
             "hyperparams": {"every": 125, "final_tenth_every": 25,
                             "dense_window": [2900, 3250], "dense_every": 10}},
        ],
        "teardown": [
            {"name": "finalize_gradient_capture"},
            {"name": "mark_log_finished"},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grad_dir", required=True); ap.add_argument("--state_dir", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk_steps", type=int, default=50)
    ap.add_argument("--train_steps", type=int, default=3250)
    a = ap.parse_args()
    cfg = build(a.seed, a.grad_dir, a.state_dir, a.chunk_steps, a.train_steps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {a.out} (seed {a.seed}, {a.train_steps} steps, chunk {a.chunk_steps}, forks {FORK_STEPS})")


if __name__ == "__main__":
    main()
