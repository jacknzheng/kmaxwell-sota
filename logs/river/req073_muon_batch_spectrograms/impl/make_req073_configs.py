"""REQ-073 config generator: 4 arms (B/4B effective batch × Muon hidden-matrix momentum on/off), each capturing
the 4-stage pipeline for blocks.5 Q/K/mlp.proj. 4B = batch_tokens 2,097,152 (4× accumulation); divide the
accumulated grad by 4 (scale_grad_for_4B) to preserve the B token-sum optimizer convention. Muon LR fixed
0.025 (NOT auto-scaled with batch). Aux AdamW fixed (embed/head/scalars). wd=0 (stages 1-3 are pre-decay;
stage-4 displacement is then a clean no-decay update). Fork checkpoints @500/1500/2500 for noise/shadow
diagnostics (Muon group present, so dump_training_state works). 3250 updates.

Usage: make_req073_configs.py --out_dir DIR --dur DUR [--muon_lr 0.025] [--seed 0]
"""
import argparse, yaml
from pathlib import Path

SELECTED = [f"blocks.5.{t}.weight" for t in ("attn.q", "attn.k", "mlp.proj")]
FORKS = [500, 1500, 2500]
B = 524288
ARMS = {"B-mom": (B, 0.95, 1.0), "B-nomom": (B, 0.0, 1.0),
        "4B-mom": (4 * B, 0.95, 0.25), "4B-nomom": (4 * B, 0.0, 0.25)}


def build(arm, seed, dur, muon_lr):
    batch, mu, scale = ARMS[arm]
    a = f"{dur}/{arm}"
    c = {"loop": "gpt_record", "run_id": f"req073_{arm}_s{seed}", "seed": seed, "require_world_size": 8,
         "train_steps": 3250, "batch_tokens": batch, "microbatch_sequences": 64,
         "train_data": "data/fineweb10B/fineweb_train_*.bin", "val_data": "data/fineweb10B/fineweb_val_*.bin",
         "val_tokens": 10485760, "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
         "optimizer_groups": [
             {"pattern": r"^embed\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.7, "weight_decay": 0.0}},
             {"pattern": r"^proj\.weight$", "optimizer": "adamw", "hyperparams": {"lr": 0.004, "weight_decay": 0.0}},
             {"pattern": r"^blocks\..*\.weight$", "optimizer": "muon", "hyperparams": {"lr": muon_lr, "weight_decay": 0.0, "mu": mu}},
             {"pattern": ".*", "optimizer": "adamw", "hyperparams": {"lr": 0.015, "weight_decay": 0.0}}],
         "setup": [{"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
                   {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
                   {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
                   {"name": "broadcast_initial_parameters"},
                   {"name": "tag_req073_capture", "hyperparams": {"names": SELECTED, "out_base": a, "chunk_steps": 500}},
                   {"name": "validate_at_step_boundaries"}],
         "pre_optimizer": [{"name": "req073_pre"},
                           {"name": "scale_grad_for_4B", "hyperparams": {"factor": scale}},
                           {"name": "dump_training_state_at_steps", "hyperparams": {"steps": FORKS, "dump_dir": f"{a}/forks"}},
                           {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}}],
         "post_optimizer": [{"name": "req073_post"}, {"name": "print_training_progress"},
                            {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125, "final_tenth_every": 25}}],
         "teardown": [{"name": "finalize_req073"}, {"name": "mark_log_finished"}]}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--dur", required=True)
    ap.add_argument("--muon_lr", type=float, default=0.025); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        (a.out_dir / f"{arm}.yaml").write_text(yaml.safe_dump(build(arm, a.seed, a.dur, a.muon_lr), sort_keys=False))
    print(f"wrote {len(ARMS)} arm configs (B={B}, 4B={4*B}, muon_lr={a.muon_lr}) -> {a.out_dir}")


if __name__ == "__main__":
    main()
