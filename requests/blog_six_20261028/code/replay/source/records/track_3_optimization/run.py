from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


# keys a launcher may add on the command line although the YAML lacks them.
# stop_step=N: execute updates 0..N-1 only (the LR clock still uses train_steps);
# an anchor listed at N is written after update N-1; teardown then barriers all ranks.


def dist_timeout():
    """Collective timeout for init_process_group, from env DIST_TIMEOUT_S (default 1800 s;
    the capture launch sets 3600). Ranks compile with inductor independently before the
    first collective, so the 600 s NCCL default can expire at step 0 (GPU pilot, 2026-09-30)."""
    import datetime
    return datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_S", "1800")))


def _merge_config(base: dict, updates: dict) -> dict:
    # Different tagged variants must not retain fields from the old variant.
    if any(key in updates and key in base and updates[key] != base[key]
           for key in ("kind", "optimizer")):
        return deepcopy(updates)
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, ancestors: tuple[Path, ...] = (), *,
                base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve relative YAML parents in order; mappings merge, lists replace.

    optimizer_group_overrides merge a group's mapping by its exact pattern.
    No YAML control keys reach the executable config or checkpoint protocol.
    """
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"YAML inheritance cycle at {path}")
    with path.open() as stream:
        local = yaml.safe_load(stream)
    if not isinstance(local, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    parents = local.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError("extends must be a path or list of paths")
    config = deepcopy(base) if base is not None else {}
    for parent in parents:
        config = _merge_config(config, load_config(path.parent / parent, (*ancestors, path)))
    changes = local.pop("optimizer_group_overrides", [])
    config = _merge_config(config, local)
    if not isinstance(changes, list):
        raise ValueError("optimizer_group_overrides must be a list")
    seen = set()
    for change in changes:
        if not isinstance(change, dict) or "pattern" not in change:
            raise ValueError("group override requires an exact pattern")
        pattern = change["pattern"]
        matches = [i for i, group in enumerate(config.get("optimizer_groups", []))
                   if group["pattern"] == pattern]
        if len(matches) != 1 or pattern in seen:
            raise ValueError(f"unknown, duplicate or ambiguous group override: {pattern}")
        seen.add(pattern)
        index = matches[0]
        config["optimizer_groups"][index] = _merge_config(config["optimizer_groups"][index], change)
    return config


def load_config_chain(paths: list[str | Path]) -> dict[str, Any]:
    """Compose independent fragments left to right, without parent coupling.

    For example: base.yaml maxentslop.yaml capture.yaml. Later fragments may
    override fields or named optimizer groups supplied by earlier fragments.
    """
    if not paths:
        raise ValueError("config chain must contain at least one YAML")
    config = {}
    for path in paths:
        config = load_config(path, base=config)
    return config


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
    paths, overrides = [], []
    for argument in argv[1:]:
        if "=" in argument:
            overrides.append(argument)
        elif overrides:
            raise SystemExit("YAML fragments must precede key=value overrides")
        else:
            paths.append(argument)
    config = load_config_chain(paths)
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep or (key not in config and key != "stop_step"):
            raise SystemExit(f"unknown override {override!r}: overrides must be existing"
                             f" top-level keys, e.g. seed=1")
        if isinstance(config.get(key), (dict, list)):
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
            --module records.track_3_optimization.run records/track_3_optimization/configs/<run>.yaml [seed=N ...]

    The config is fully resolved -- optimizer names, hook names, hyperparameter
    signatures, regexes -- before CUDA or the process group is touched, so typos
    fail at launch.
    """
    config = parse_config(list(sys.argv))
    if config.get("loop") != "gpt_record":
        raise SystemExit("this runner requires loop: gpt_record")
    from records.track_3_optimization.harness.hooks import bind_sites
    from records.track_3_optimization.harness.optimizers import check_optimizer_groups_resolve
    check_optimizer_groups_resolve(config)
    bind_sites(config)   # launch-time resolution; the loop binds its own copy

    import torch
    import torch.distributed as dist
    from records.track_3_optimization.harness import loop

    if config.get("determinism") or os.environ.get("CAPTURE_DETERMINISM") == "1":
        from records.track_3_optimization.harness import determinism_policy
        determinism_policy.apply()
        determinism_policy.verify(require_launch_env=True)

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device, timeout=dist_timeout())
    dist.barrier()
    loop.run(config)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
