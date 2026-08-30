from __future__ import annotations

import os
import sys
from typing import Any

import yaml


def parse_config(argv: list[str]) -> dict[str, Any]:
    """Loads the YAML config and applies trailing key=value overrides.

    Overrides target existing top-level scalar keys only (e.g. seed=1) and are
    parsed with yaml.safe_load, so seed sweeps never edit configs.

    Raises:
        SystemExit: On missing arguments or an override that is not a known
            top-level scalar key.
    """
    if len(argv) < 2:
        raise SystemExit("usage: run.py <config.yaml> [key=value ...]")
    with open(argv[1]) as f:
        config = yaml.safe_load(f)
    for override in argv[2:]:
        key, sep, raw_value = override.partition("=")
        if not sep or key not in config:
            raise SystemExit(f"unknown override {override!r}: overrides must be existing"
                             f" top-level keys, e.g. seed=1")
        if isinstance(config[key], (dict, list)):
            raise SystemExit(f"override {key!r} targets a nested section; edit the YAML instead")
        value = yaml.safe_load(raw_value)
        if isinstance(value, (dict, list)):
            raise SystemExit(f"override {key!r} must be a scalar, got {value!r}")
        config[key] = value
    return config


def main() -> None:
    """Entry point for YAML-configured Track-3 runs.

    Usage (from repo root, so data globs resolve):
        torchrun --standalone --nproc_per_node=8 \\
            records/track_3_optimization/run.py records/track_3_optimization/configs/<run>.yaml [seed=N ...]

    The config is fully resolved -- optimizer names, hook names, hyperparameter
    signatures, regexes -- before CUDA or the process group is touched, so typos
    fail at launch.
    """
    config = parse_config(sys.argv)
    if config.get("loop") != "gpt_record":
        raise SystemExit(f"unsupported loop {config.get('loop')!r}; this runner runs 'gpt_record'")

    from harness.hooks import bind_sites
    from optimizers import check_optimizer_groups_resolve
    check_optimizer_groups_resolve(config)
    bind_sites(config)   # launch-time resolution; the loop binds its own copy

    import torch
    import torch.distributed as dist
    from harness import loop

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    loop.run(config)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
