from __future__ import annotations

import glob
import os
from typing import Any

import torch
from torch import nn

# PR #340's submission recipe -- the single definition every fallback references
PR340_BIMAXWELL = dict(fast_decay=0.85, slow_decay=0.98, fast_weight=0.4385, switch_step=1000)


def write_file_atomically(payload: Any, path: str) -> None:
    """torch.save to a sibling tmp file, then atomically replace path, so a crash
    never leaves a truncated file under the final name."""
    tmp_path = str(path) + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def shard_path(dump_dir: str, muon_step: int, rank: int, world_size: int) -> str:
    """The canonical per-rank buffer shard filename for one pinned step."""
    return os.path.join(dump_dir, f"secant_state_step{muon_step:06d}_rank{rank}of{world_size}.pt")


def write_secant_shard(dump_dir: str, payload: dict[str, Any]) -> str:
    """Writes one rank's buffer shard for its pinned step (path from the payload).

    Format v2 payload: {format_version: 2, muon_step, k, betas, rank, world_size,
    ladder: {num_buffers, min_decay, max_decay, pin_exact_rates},
    solve: {truncated_rank_accumulated_parameter_displacements, truncated_rank_gmres},
    optimizer_recipe: {optimizer, hyperparams}, w, stats,
    params: {idx: {sorted_index, name, shape, m (k,n), q (k,n), grad, x}}}.

    Returns:
        The written path.
    """
    os.makedirs(dump_dir, exist_ok=True)
    path = shard_path(dump_dir, payload["muon_step"], payload["rank"], payload["world_size"])
    write_file_atomically(payload, path)
    return path


def read_secant_shards(dump_dir: str, muon_step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Loads all rank shards for one step, accepting format v2 and the legacy v1
    (no format_version; carries rcond and a bimaxwell dict) so old dumps on disk
    stay analyzable.

    Args:
        dump_dir: Directory holding secant_state_step*_rank*.pt shards.
        muon_step: The pinned step to load.

    Returns:
        (records, meta): records is a list ordered by the trainer's sorted
        parameter index, each with name, shape and CPU tensors m (k,n), q (k,n),
        grad (n, flattened), x (n, flattened). meta carries muon_step, k, betas,
        w, stats, ladder, solve, and bimaxwell (fast/slow rates + convex weight,
        derived from the optimizer recipe for v2 shards).

    Raises:
        AssertionError: On missing shards or cross-shard disagreement.
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, f"secant_state_step{muon_step:06d}_rank*.pt")))
    assert paths, f"no shards found for step {muon_step} in {dump_dir}"
    by_index: dict[int, dict[str, Any]] = {}
    meta: dict[str, Any] | None = None
    world_size: int | None = None
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if meta is None:
            version = shard.get("format_version", 1)
            if version == 1:
                bimaxwell = shard["bimaxwell"]
                ladder: dict[str, Any] = dict(num_buffers=shard["k"])
                solve: dict[str, Any] = dict(rcond=shard.get("rcond"))
            else:
                recipe = shard.get("optimizer_recipe") or {}
                hp = recipe.get("hyperparams") or {}
                # only a bimaxwell shard truly carries these constants; other
                # optimizers get None rather than fabricated defaults
                bimaxwell = (dict(bf=hp.get("fast_decay", PR340_BIMAXWELL["fast_decay"]),
                                  bs=hp.get("slow_decay", PR340_BIMAXWELL["slow_decay"]),
                                  w=hp.get("fast_weight", PR340_BIMAXWELL["fast_weight"]),
                                  start=hp.get("switch_step", PR340_BIMAXWELL["switch_step"]))
                             if recipe.get("optimizer") == "bimaxwell_muon" else None)
                ladder = shard["ladder"]
                solve = shard["solve"]
            meta = dict(format_version=version, muon_step=shard["muon_step"], k=shard["k"],
                        betas=shard["betas"], w=shard["w"], stats=shard["stats"],
                        ladder=ladder, solve=solve, bimaxwell=bimaxwell,
                        optimizer_recipe=shard.get("optimizer_recipe"))
            world_size = shard["world_size"]
        else:
            assert torch.equal(meta["betas"], shard["betas"]) and meta["k"] == shard["k"]
            assert meta["muon_step"] == shard["muon_step"]
        for idx, record in shard["params"].items():
            assert int(idx) not in by_index, f"parameter index {idx} appears in two shards"
            by_index[int(idx)] = record
    assert len(paths) == world_size, f"expected {world_size} shards, found {len(paths)}"
    indices = sorted(by_index)
    assert indices == list(range(len(indices))), "missing parameter shards"
    records = [by_index[i] for i in indices]
    for record in records:
        record["grad"] = record["grad"].float().flatten()
        record["x"] = record["x"].float().flatten()
    return records, meta


def write_probe_history(dump_dir: str, payload: dict[str, Any]) -> str:
    """Atomically writes probe_history.pt: {format_version, k, betas, ladder, solve,
    probes: [{muon_step, G (fp64 (2k+1)^2), w, stats}]}. Every probe keeps the full
    Gram (~320 KB at k=100), enough to replay solves under any rank choice offline.

    Returns:
        The written path.
    """
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, "probe_history.pt")
    write_file_atomically(payload, path)
    return path


def read_probe_history(path: str) -> dict[str, Any]:
    """Loads a probe_history.pt written by write_probe_history."""
    return torch.load(path, map_location="cpu", weights_only=False)


def write_model_checkpoint(dump_dir: str, muon_step: int, model: nn.Module) -> str:
    """Writes a CPU state-dict checkpoint of the model at x_t (call before updates).

    Returns:
        The written path.
    """
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"model_step{muon_step:06d}.pt")
    write_file_atomically({key: value.cpu() for key, value in model.state_dict().items()}, path)
    return path
