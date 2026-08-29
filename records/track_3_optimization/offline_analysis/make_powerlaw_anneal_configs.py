"""Generate the normalized scheduled-power-law momentum sweep configs.

The base run warms the shared EMA bank through step 1000 and dumps its full
state. Each grid run resumes that exact state and linearly interpolates from a
power-law kernel fit over ages 0..1000 to one fit over ages 0..3249.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optimizers.muon import finite_history_power_law_weights  # noqa: E402


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def kernel_stats(decays: list[float], weights: list[float], horizon: int,
                 bias_corrected: bool = True) -> dict:
    taps = [sum(w * (1 - beta) * beta ** k
                / ((1 - beta ** (horizon + 1)) if bias_corrected else 1)
                for beta, w in zip(decays, weights))
            for k in range(horizon + 1)]
    finite_mass = sum(taps)
    return {
        "dc_gain": sum(weights),
        "finite_mass": finite_mass,
        "finite_mean_age": sum(k * value for k, value in enumerate(taps)) / finite_mass,
        "momentum_flip_gain": sum(w * (1 - beta) / (1 + beta)
                                  for beta, w in zip(decays, weights)),
        "prepolar_flip_gain_mu_0p95": 0.05 + 0.95 * sum(
            w * (1 - beta) / (1 + beta) for beta, w in zip(decays, weights)),
        "minimum_fitted_tap": min(taps),
    }


def base_config(decays: list[float]) -> dict:
    return {
        "loop": "gpt_record", "stop_after_step": 1001,
        "run_id": "pl_anneal_base", "seed": 0, "require_world_size": 8,
        "train_steps": 3250, "batch_tokens": 524288,
        "microbatch_sequences": 64,
        "train_data": "data/fineweb10B/fineweb_train_*.bin",
        "val_data": "data/fineweb10B/fineweb_val_*.bin", "val_tokens": 10485760,
        "model": {"vocab_size": 50304, "num_layers": 12, "model_dim": 768},
        "optimizer_groups": [
            {"pattern": "^embed\\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.7, "weight_decay": 0.001}},
            {"pattern": "^proj\\.weight$", "optimizer": "adamw",
             "hyperparams": {"lr": 0.004, "weight_decay": 0.001}},
            {"pattern": "^blocks\\..*\\.weight$", "optimizer": "annealed_weights_muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                             "decays": decays,
                             "start_weights": [1.0] + [0.0] * (len(decays) - 1),
                             "end_weights": [1.0] + [0.0] * (len(decays) - 1),
                             "switch_step": 1000, "anneal_end_step": 3249,
                             "warm_streams_before_switch": True,
                             "bias_correct_streams": True}},
            {"pattern": ".*", "optimizer": "adamw",
             "hyperparams": {"lr": 0.015, "weight_decay": 0.001}},
        ],
        "setup": [
            {"name": "open_rank_zero_log"}, {"name": "load_validation_tokens"},
            {"name": "build_compiled_gpt"},
            {"name": "seed_then_initialize_parameters"},
            {"name": "assemble_grouped_optimizer"}, {"name": "open_training_batches"},
            {"name": "broadcast_initial_parameters"},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "dump_training_state_at_steps",
             "hyperparams": {"steps": [1000], "dump_dir": "warmstart_pl_anneal"}},
            {"name": "cool_down_learning_rate", "hyperparams": {"cooldown_frac": 0.7}},
        ],
        "post_optimizer": [{"name": "print_training_progress"},
                           {"name": "validate_at_step_boundaries",
                            "hyperparams": {"every": 125, "final_tenth_every": 25,
                                            "dense_window": [2900, 3250],
                                            "dense_every": 10}}],
        "teardown": [{"name": "mark_log_finished"}],
    }


def fork_config(decays: list[float], gamma_start: float, gamma_end: float) -> dict:
    cfg = base_config(decays)
    run_id = f"plann_g{tag(gamma_start)}_g{tag(gamma_end)}"
    cfg.update(run_id=run_id, start_step=1000)
    cfg.pop("stop_after_step")
    cfg["optimizer_groups"][2] = {
        "pattern": "^blocks\\..*\\.weight$", "optimizer": "annealed_weights_muon",
        "hyperparams": {
            "lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
            "decays": decays,
            "start_weights": finite_history_power_law_weights(
                decays, gamma_start, horizon=1000),
            "end_weights": finite_history_power_law_weights(
                decays, gamma_end, horizon=3249),
            "switch_step": 1000, "anneal_end_step": 3249,
            "warm_streams_before_switch": True,
            "bias_correct_streams": True,
        },
    }
    cfg["setup"].insert(-1, {
        "name": "load_training_state",
        "hyperparams": {"state_dir": "warmstart_pl_anneal", "step": 1000,
                        "skip_batches": 1000},
    })
    cfg["pre_optimizer"] = [cfg["pre_optimizer"][-1]]
    return cfg


def control_config(decays: list[float], optimizer: str) -> dict:
    cfg = base_config(decays)
    cfg.update(run_id=f"plann_{optimizer}_control", start_step=1000)
    cfg.pop("stop_after_step")
    if optimizer == "bimaxwell":
        spec = {"optimizer": "bimaxwell_muon",
                "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                                "fast_decay": 0.85, "slow_decay": 0.98,
                                "fast_weight": 0.4385, "switch_step": 1000}}
    elif optimizer == "single":
        spec = {"optimizer": "muon",
                "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95}}
    else:
        raise ValueError(optimizer)
    cfg["optimizer_groups"][2] = {"pattern": "^blocks\\..*\\.weight$", **spec}
    cfg["setup"].insert(-1, {
        "name": "load_training_state",
        "hyperparams": {"state_dir": "warmstart_pl_anneal", "step": 1000,
                        "skip_batches": 1000},
    })
    cfg["pre_optimizer"] = [cfg["pre_optimizer"][-1]]
    return cfg


def intern_kernel_control_config(decays: list[float], start_weights: list[float],
                                 end_weights: list[float], pr: int) -> dict:
    """Replay an intern PR's K-Maxwell kernel on the common step-1000 fork.

    PR359's other MuonH/LR/length changes are intentionally excluded so this
    remains a momentum-family control rather than a different training recipe.
    """
    cfg = base_config(decays)
    cfg.update(run_id=f"plann_pr{pr}_kernel_control", start_step=1000)
    cfg.pop("stop_after_step")
    cfg["optimizer_groups"][2] = {
        "pattern": "^blocks\\..*\\.weight$", "optimizer": "annealed_weights_muon",
        "hyperparams": {
            "lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
            "decays": decays, "start_weights": start_weights,
            "end_weights": end_weights, "switch_step": 1000,
            "anneal_end_step": 3250, "warm_streams_before_switch": False,
            "bias_correct_streams": False,
        },
    }
    cfg["setup"].insert(-1, {
        "name": "load_training_state",
        "hyperparams": {"state_dir": "warmstart_pl_anneal", "step": 1000,
                        "skip_batches": 1000},
    })
    cfg["pre_optimizer"] = [cfg["pre_optimizer"][-1]]
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--start", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--end", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ages = [10 ** x for x in [math.log10(0.1) + i *
            (math.log10(3300) - math.log10(0.1)) / 9 for i in range(10)]]
    decays = [age / (1 + age) for age in ages]
    configs = [("pl_anneal_base.yaml", base_config(decays)),
               ("plann_bimaxwell_control.yaml", control_config(decays, "bimaxwell")),
               ("plann_single_control.yaml", control_config(decays, "single"))]
    pr357_decays = [0.75, 0.8228524398549413, 0.8779303386257296,
                    0.9175985472180883, 0.9451809410725112,
                    0.9638939208460664, 0.976378696890324, 0.9846153846153846]
    pr357_start = [0.005094, 0.010188, 0.015282, 0.020376,
                   0.025470, 0.030564, 0.035658, 0.857369]
    pr357_end = [0.032262, 0.064524, 0.096786, 0.129047,
                 0.161309, 0.193571, 0.225833, 0.096669]
    pr359_decays = [0.75, 0.8469227055704172, 0.9107413502519109,
                    0.9495389483727874, 0.9719912250466763, 0.9846153846153846]
    pr359_start = [0.021003991507788214, 0.04200798301557643,
                   0.06301197452336464, 0.08401596603115286,
                   0.10501995753894107, 0.6849401273831768]
    pr359_end = [0.06301197452336468, 0.12602394904672937,
                 0.18903592357009405, 0.25204789809345873,
                 0.31505987261682344, 0.05482038214952978]
    configs += [
        ("plann_pr357_kernel_control.yaml", intern_kernel_control_config(
            pr357_decays, pr357_start, pr357_end, 357)),
        ("plann_pr359_kernel_control.yaml", intern_kernel_control_config(
            pr359_decays, pr359_start, pr359_end, 359)),
    ]
    configs += [(f"plann_g{tag(a)}_g{tag(b)}.yaml", fork_config(decays, a, b))
                for a in args.start for b in args.end]
    smoke = copy.deepcopy(fork_config(decays, 0.75, 1.0))
    smoke.update(run_id="plann_smoke", stop_after_step=1002)
    configs.append(("plann_smoke.yaml", smoke))
    for name, cfg in configs:
        with open(os.path.join(args.output_dir, name), "w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print(name)
    manifest = {}
    for a in args.start:
        for b in args.end:
            start_weights = finite_history_power_law_weights(decays, a, horizon=1000)
            end_weights = finite_history_power_law_weights(decays, b, horizon=3249)
            manifest[f"g{tag(a)}_g{tag(b)}"] = {
                "gamma_1000": a, "gamma_end": b,
                "at_1000": kernel_stats(decays, start_weights, 1000),
                "at_end": kernel_stats(decays, end_weights, 3249),
            }
    with open(os.path.join(args.output_dir, "plann_manifest.json"), "w") as handle:
        json.dump({"decays": decays, "runs": manifest}, handle, indent=2)


if __name__ == "__main__":
    main()
