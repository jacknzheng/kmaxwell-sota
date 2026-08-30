"""Generate wide, type-balanced per-matrix LR intervention configs.

One seeded random ordering is drawn within each matrix type. Three cyclic
multiplier assignments then give every matrix each multiplier exactly once,
while every assignment contains four 0.6, four 1.0, and four 1.7 matrices per
type. The same assignments are used at both fork states, isolating state from
assignment.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.model_gpt import GPT
from optimizers.muon import order_params_like_record


MULTIPLIERS = (0.6, 1.0, 1.7)
FORKS = (1500, 2000)
ASSIGNMENT_SEED = 23023


def matrix_type(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[2:4])


def sorted_matrix_names() -> list[str]:
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    named = [(name, param) for name, param in model.named_parameters()
             if re.match(r"^blocks\..*\.weight$", name) and param.ndim == 2]
    ordered = order_params_like_record([param for _, param in named])
    name_of = {id(param): name for name, param in named}
    return [name_of[id(param)] for param in ordered]


def make_assignments(names: list[str]) -> list[dict[str, float]]:
    rng = random.Random(ASSIGNMENT_SEED)
    by_type = {kind: [name for name in names if matrix_type(name) == kind]
               for kind in sorted({matrix_type(name) for name in names})}
    assignments = [dict() for _ in range(3)]
    for kind, members in by_type.items():
        assert len(members) == 12, (kind, len(members))
        rng.shuffle(members)
        for position, name in enumerate(members):
            base = position % 3
            for assignment_index in range(3):
                assignments[assignment_index][name] = MULTIPLIERS[
                    (base + assignment_index) % 3]
    for assignment in assignments:
        for kind, members in by_type.items():
            counts = {multiplier: sum(assignment[name] == multiplier
                                      for name in members)
                      for multiplier in MULTIPLIERS}
            assert counts == {0.6: 4, 1.0: 4, 1.7: 4}, (kind, counts)
    for name in names:
        assert sorted(assignment[name] for assignment in assignments) == [0.6, 1.0, 1.7]
    return assignments


def common(run_id: str, fork: int, multipliers: list[float]) -> dict:
    return {
        "loop": "gpt_record", "run_id": run_id, "seed": 0,
        "require_world_size": 8, "train_steps": 3250,
        "start_step": fork, "stop_after_step": fork + 850,
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
            {"pattern": r"^blocks\..*\.weight$",
             "optimizer": "per_matrix_lr_muon",
             "hyperparams": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95,
                              "fast_decay": 0.85, "slow_decay": 0.98,
                              "fast_weight": 0.4385, "switch_step": 1000,
                              "lr_multipliers": multipliers}},
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
             "hyperparams": {"state_dir": "eos_shared_state", "step": fork,
                              "skip_batches": fork}},
            {"name": "validate_at_step_boundaries"},
        ],
        "pre_optimizer": [
            {"name": "dump_training_state_at_steps",
             "hyperparams": {"steps": [fork],
                              "dump_dir": f"gate_{run_id}"}},
            {"name": "checkpoint_model_at_cadence",
             "hyperparams": {"every": 125,
                              "dump_dir": f"dumps_{run_id}"}},
            {"name": "set_learning_rate_stairs",
             "hyperparams": {"stairs": [[fork, 1.0]]}},
            {"name": "log_learning_rates_at_steps",
             "hyperparams": {"steps": [fork, fork + 1]}},
        ],
        "post_optimizer": [
            {"name": "print_training_progress"},
            {"name": "validate_at_step_boundaries",
             "hyperparams": {"every": 125}},
        ],
        "teardown": [{"name": "mark_log_finished"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = sorted_matrix_names()
    assignments = make_assignments(names)
    metadata = {
        "seed": ASSIGNMENT_SEED,
        "design": "randomized within type, then three cyclic multiplier rotations",
        "sorted_names": names,
        "assignments": assignments,
    }
    (args.out / "assignments.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = ["sorted_index\tname\tmatrix_type\tassignment_0\tassignment_1\tassignment_2"]
    for index, name in enumerate(names):
        rows.append("\t".join((str(index), name, matrix_type(name),
                               *(str(assignment[name]) for assignment in assignments))))
    (args.out / "assignments.tsv").write_text("\n".join(rows) + "\n")

    manifest = ["run_id\tfork\tassignment\tstop\tcurvature_steps"]
    for fork in FORKS:
        stop = fork + 850
        curvature_steps = list(range(stop - 500, stop + 1, 125))
        for assignment_index, assignment in enumerate(assignments):
            run_id = f"req023_f{fork}_a{assignment_index}"
            multipliers = [assignment[name] for name in names]
            config = common(run_id, fork, multipliers)
            (args.out / f"{run_id}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, width=100))
            manifest.append("\t".join((run_id, str(fork), str(assignment_index),
                                       str(stop), ",".join(map(str, curvature_steps)))))
    (args.out / "manifest.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
