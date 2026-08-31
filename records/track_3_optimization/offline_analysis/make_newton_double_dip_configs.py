"""Generate REQ-025's alpha-by-memory shared-state intervention grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FORKS = ((1500, 2250), (2000, 2750))
ALPHAS = (0.0, 0.25, 0.5)
KERNELS = ("record", "short")


def config_for(fork: int, stop: int, alpha: float, kernel: str,
               state_dir: str) -> dict:
    alpha_label = str(alpha).replace(".", "p")
    run_id = f"newton_dd_f{fork}_{kernel}_a{alpha_label}"
    muon_name = ("newton_bimaxwell_muon" if kernel == "record"
                 else "newton_short_ema_muon")
    muon_hparams = {
        "lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
        "newton_alpha": alpha, "precond_refresh_interval": 10,
        "precond_beta": 0.95, "precond_damping": 0.2,
    }
    if kernel == "record":
        muon_hparams.update(fast_decay=0.85, slow_decay=0.98,
                            fast_weight=0.4385, switch_step=1000)
    else:
        muon_hparams["decay"] = 0.85
    return {
        "loop": "gpt_record", "run_id": run_id, "seed": 0,
        "require_world_size": 8, "train_steps": 3250,
        "start_step": fork, "stop_after_step": stop,
        "batch_tokens": 524288, "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin",
        "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": r"^embed\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": r"^proj\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": r"^blocks\..*\.weight$", "optimizer": muon_name,
             "hyperparams": muon_hparams},
            {"pattern": ".*", "optimizer": "adamw",
             "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"}, {"name": "seed_then_initialize_parameters"},
            {"name": "attach_newton_muon_activation_stats"},
            {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"},
            {"name": "load_training_state", "hyperparams": {
                "state_dir": state_dir, "step": fork, "skip_batches": fork}},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "checkpoint_model_at_cadence", "hyperparams": {
                "every": 250, "dump_dir": f"dumps_{run_id}"}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125}},
        ],
        "teardown": [{"name": "mark_log_finished"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-dir", default="eos_shared_state")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = ["run_id\tfork_step\tstop_step\tkernel\talpha\tcurvature_steps"]
    for fork, stop in FORKS:
        for kernel in KERNELS:
            for alpha in ALPHAS:
                config = config_for(fork, stop, alpha, kernel, args.state_dir)
                (args.out / f"{config['run_id']}.yaml").write_text(
                    yaml.safe_dump(config, sort_keys=False))
                rows.append("\t".join((config["run_id"], str(fork), str(stop), kernel,
                                       str(alpha), f"{fork + 250},{fork + 500},{stop}")))
    (args.out / "manifest.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
