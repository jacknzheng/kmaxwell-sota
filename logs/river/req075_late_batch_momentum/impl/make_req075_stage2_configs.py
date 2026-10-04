"""REQ-075 Stage 2 config generator: the 256-update, 5-arm mechanism pilot from the REQ-073 B-mom @2500 fork.

All arms fork from the identical @2500 state and share ONE data-order seed: they start at the SAME token
(sequence 1,286,144 = 2512 B-batches = 157 16B-batches), so the B arm's tokens are a prefix-subset of the
16B arm's. 2500 is not divisible by 16, so exact alignment at the fork cursor (seq 1,280,000) is impossible;
all arms instead start at the nearest 16B-aligned boundary, a documented 12-B-batch (6.29M-token) gap after
the fork cursor into fresh post-fork data.

LR is held CONSTANT at the checkpoint value for this diagnostic: cool_down_learning_rate with
schedule_step_max=2500 holds eta at eta(2500)=0.3297, so Muon lr = 0.025*0.3297 = 0.008242 (half-LR arm:
0.0125*0.3297 = 0.004121). Momentum arms keep mu=0.95 + the inherited buffer; nomom arms set mu=0 + clear
the buffer. 16B arms accumulate 16x and scale grad by 1/16 (B token-sum convention). 4-stage capture (REQ-073)
every update. No LR schedule decay, no auto LR-vs-batch scaling.

Usage: make_req075_stage2_configs.py --out_dir DIR --dur DUR --fork_dir FORK
"""
import argparse, yaml
from pathlib import Path

B = 524288
FORK_STEP = 2500
UPDATES = 256
# per arm: (batch_tokens, mu, muon_base_lr, clear_buffer, scale_factor, skip_batches)
ARMS = {
    "B-mom":            (B,      0.95, 0.025,  False, 1.0,      2512),
    "B-nomom":          (B,      0.0,  0.025,  True,  1.0,      2512),
    "16B-mom":          (16 * B, 0.95, 0.025,  False, 1.0 / 16, 157),
    "16B-nomom":        (16 * B, 0.0,  0.025,  True,  1.0 / 16, 157),
    "16B-nomom-halfLR": (16 * B, 0.0,  0.0125, True,  1.0 / 16, 157),
}
SELECTED = ["blocks.5.attn.q.weight", "blocks.5.attn.k.weight", "blocks.5.mlp.proj.weight"]


def build(arm, fork_dir, dur):
    batch, mu, mlr, clear, scale, skip = ARMS[arm]
    a = f"{dur}/{arm}"
    return {
        "loop": "gpt_record", "run_id": f"req075_s2_{arm}", "seed": 0, "require_world_size": 8,
        "start_step": FORK_STEP, "train_steps": FORK_STEP + UPDATES,
        "batch_tokens": batch, "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin", "val_data": "data/fineweb10B/fineweb_val_*.bin",
        "val_tokens": 10485760, "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.0}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.0}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": mlr, "weight_decay": 0.0, "mu": mu}},
            {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.0}}],
        "setup": [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                  {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                  {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                  {"name": "broadcast_initial_parameters"},
                  {"name": "load_training_state", "hyperparams": {"state_dir": fork_dir, "step": FORK_STEP, "skip_batches": skip}},
                  {"name": "req075_set_muon_group", "hyperparams": {"mu": mu, "base_lr": mlr, "clear_buffer": clear}},
                  {"name": "tag_req073_capture", "hyperparams": {"names": SELECTED, "out_base": a, "chunk_steps": UPDATES}},
                  {"name": "validate_at_step_boundaries"}],
        "pre_optimizer": [{"name": "req073_pre"},
                          {"name": "scale_grad_for_4B", "hyperparams": {"factor": scale}},
                          {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7, "schedule_step_max": FORK_STEP}}],
        "post_optimizer": [{"name": "req073_post"}, {"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries", "hyperparams": {"every": 64}}],
        "teardown": [{"name": "finalize_req073"}, {"name": "mark_log_finished"}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--dur", required=True)
    ap.add_argument("--fork_dir", required=True)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        (a.out_dir / f"{arm}.yaml").write_text(yaml.safe_dump(build(arm, a.fork_dir, a.dur), sort_keys=False))
    print(f"wrote {len(ARMS)} Stage-2 arm configs -> {a.out_dir} (all start at seq 1,286,144; {UPDATES} updates)")


if __name__ == "__main__":
    main()
