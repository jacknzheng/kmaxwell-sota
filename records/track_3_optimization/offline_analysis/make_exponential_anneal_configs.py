"""Generate a same-state 2D sweep of scheduled single-EMA kernels."""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from offline_analysis.make_powerlaw_anneal_configs import base_config  # noqa: E402


def tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def fork_config(beta_start: float, beta_end: float) -> dict:
    # The pole ladder is irrelevant to this fork, but base_config provides the
    # exact same warmed model/optimizer checkpoint protocol as the power-law run.
    placeholder_decays = [0.5]
    cfg = copy.deepcopy(base_config(placeholder_decays))
    cfg.update(run_id=f"expann_b{tag(beta_start)}_b{tag(beta_end)}",
               start_step=1000)
    cfg.pop("stop_after_step")
    cfg["optimizer_groups"][2] = {
        "pattern": "^blocks\\..*\\.weight$",
        "optimizer": "annealed_decay_muon",
        "hyperparams": {
            "lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
            "beta_start": beta_start, "beta_end": beta_end,
            "switch_step": 1000, "anneal_end_step": 3249,
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
    parser.add_argument("--start", nargs="+", type=float,
                        default=[0.978, 0.982, 0.986])
    parser.add_argument("--end", nargs="+", type=float,
                        default=[0.952, 0.960, 0.968])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for start in args.start:
        for end in args.end:
            name = f"expann_b{tag(start)}_b{tag(end)}.yaml"
            with open(os.path.join(args.output_dir, name), "w") as handle:
                yaml.safe_dump(fork_config(start, end), handle, sort_keys=False)
            print(name)


if __name__ == "__main__":
    main()
