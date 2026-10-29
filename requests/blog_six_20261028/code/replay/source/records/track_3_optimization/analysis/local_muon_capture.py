"""Read-only dense observer; all ranks gather tokens BEFORE optimizer collectives."""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from records.track_3_optimization.analysis.local_muon import (
    arnoldi_response, converged_response, frozen_hessian_block, local_response)
from .response_decay import decay_shift, decay_readout


DEFAULTS = dict(mode="newton_schulz", every=1, matrices="all", krylov_dim=32,
                seed=0, residual_tol=.02, real_tol=1e-7, substantial_fraction=.1,
                convergence_dims=None, convergence_starts=2, eigenvalue_rtol=.01,
                windows=None, microbatch_sequences=None, loss_batch="training",
                validation_tokens=524288, validation_offset=0, loss_normalization="sum",
                warm_start=False, include_weight_decay=True)


def validate_spec(spec, muon, names):
    if not isinstance(spec, dict) or set(spec) - set(DEFAULTS):
        raise ValueError("invalid local_muon_operator configuration")
    spec = {**DEFAULTS, **spec}
    if type(spec["warm_start"]) is not bool:
        raise ValueError("warm_start must be boolean")
    if type(spec["include_weight_decay"]) is not bool:
        raise ValueError("include_weight_decay must be boolean")
    if spec["warm_start"] and (spec["convergence_dims"] is None or spec["loss_batch"] != "fixed_validation"):
        raise ValueError("warm starts require convergence checks and a fixed validation probe")
    for key in ("every", "krylov_dim"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError("local Muon cadence/dimension must be positive integers")
    if spec["microbatch_sequences"] is not None and (
            type(spec["microbatch_sequences"]) is not int or spec["microbatch_sequences"] < 1):
        raise ValueError("invalid Hessian microbatch")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("local Muon seed must be a nonnegative integer")
    dims = spec["convergence_dims"]
    if dims is not None and (not isinstance(dims, list) or len(dims) < 2 or
            any(type(k) is not int or k < 1 for k in dims) or
            any(a >= b for a, b in zip(dims, dims[1:]))):
        raise ValueError("convergence dimensions must be increasing positive integers")
    if type(spec["convergence_starts"]) is not int or spec["convergence_starts"] < 2:
        raise ValueError("convergence requires at least two starts")
    if not 0 < spec["eigenvalue_rtol"] < 1:
        raise ValueError("invalid eigenvalue convergence tolerance")
    windows = spec["windows"]
    if windows is not None and (not isinstance(windows, list) or not windows or any(
            not isinstance(w, list) or len(w) != 2 or
            any(type(t) is not int or t < 0 for t in w) or w[0] > w[1] for w in windows)):
        raise ValueError("invalid local Muon burst windows")
    if windows is not None and any(a[1] >= b[0] for a, b in zip(windows, windows[1:])):
        raise ValueError("local Muon burst windows must be ordered and disjoint")
    if spec["loss_batch"] not in ("training", "fixed_validation"):
        raise ValueError("unknown local Muon loss batch")
    if (type(spec["validation_tokens"]) is not int or spec["validation_tokens"] < 1024 or
            spec["validation_tokens"] % 1024 or type(spec["validation_offset"]) is not int or
            spec["validation_offset"] < 0 or spec["loss_normalization"] not in ("sum", "mean")):
        raise ValueError("invalid validation probe/normalization")
    if spec["mode"] not in ("newton_schulz", "ideal_polar"):
        raise ValueError("unknown local Muon derivative mode")
    if not (0 < spec["residual_tol"] < 1 and 0 <= spec["real_tol"] < 1 and
            0 <= spec["substantial_fraction"] <= 1):
        raise ValueError("invalid local Muon tolerances")
    available = {names[id(p)] for p in muon.sorted_params()}
    selected = spec["matrices"]
    if selected != "all" and (not isinstance(selected, list) or not selected or
            any(not isinstance(n, str) or n not in available for n in selected) or
            len(set(selected)) != len(selected)):
        raise ValueError("local Muon matrix selection must name unique Muon matrices")
    return spec


def readout_due(spec, update):
    return update % spec["every"] == 0 and (spec["windows"] is None or
        any(a <= update <= b for a, b in spec["windows"]))


def global_batch(batch, world):
    if not dist.is_available() or not dist.is_initialized():
        if world != 1:
            raise ValueError("global Hessian batch requires initialized distribution")
        return batch
    if dist.get_world_size() != world:
        raise ValueError("Hessian and training world size differ")
    result = []
    for tensor in batch:
        pieces = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(pieces, tensor.contiguous())
        result.append(torch.cat(pieces, dim=0))
    return tuple(result)


def fixed_validation_batch(config, device, tokens=524288, offset=0):
    """Deterministic contiguous, non-wrapping validation probe; nested by size.

    Offset counts input tokens, not bytes. Disjoint probes leave one boundary
    token between intervals so their input/target tokens do not overlap.
    """
    if type(tokens) is not int or tokens < 1024 or tokens % 1024 or type(offset) is not int or offset < 0:
        raise ValueError("validation tokens must be a positive multiple of 1024")
    files = sorted(Path(p) for p in glob.glob(config["val_data"]))
    if not files:
        raise ValueError("fixed validation Hessian needs a validation shard")
    path = files[0]
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1 or header[2] < offset + tokens + 1:
        raise ValueError("invalid fixed validation shard")
    raw = np.fromfile(path, dtype=np.uint16, count=tokens + 1, offset=1024 + 2 * offset)
    if len(raw) != tokens + 1:
        raise ValueError("incomplete fixed validation probe")
    return (torch.from_numpy(raw[:-1].copy()).to(device=device, dtype=torch.int32).view(-1, 1024),
            torch.from_numpy(raw[1:].copy()).to(device=device, dtype=torch.int64).view(-1, 1024))


def prepare_readout(config, state, muon, ctx, due):
    ctx["local_muon_results"] = {}
    ctx["local_muon_vectors"] = {}
    ctx["local_muon_expected"] = set()
    warm_cache = ctx.setdefault("local_muon_warm_cache", {})
    if not due:
        return None
    # Same actual tokens, all ranks, no loader advance; gather outside the
    # sharded observer so different Arnoldi breakdowns cannot deadlock NCCL.
    parameters = ctx["before"]
    buffers = {n: b.detach().clone() for n, b in state["model"].named_buffers()}
    spec = ctx["local_muon_spec"]
    batch = (fixed_validation_batch(config, next(iter(parameters.values())).device,
                                   spec["validation_tokens"], spec["validation_offset"])
             if spec["loss_batch"] == "fixed_validation" else
             global_batch(state["train_batches"].last, state["world_size"]))
    t = state["step"]
    ctx["local_muon_expected"] = {
        ctx["names"][id(p)] for i, p in enumerate(muon.sorted_params())
        if i % state["world_size"] == state["rank"] and
        (spec["matrices"] == "all" or ctx["names"][id(p)] in spec["matrices"])}

    def observe(p, gradient, momentum, kernel, iteration, saved_state):
        if iteration != t:
            raise ValueError("local Muon momentum/readout clocks differ")
        name = ctx["names"][id(p)]
        if spec["matrices"] != "all" and name not in spec["matrices"]:
            return
        index = next(i for i, other in enumerate(muon.sorted_params()) if other is p)
        hvp = frozen_hessian_block(state["model"], parameters, buffers, name, batch,
                                  spec["microbatch_sequences"] or int(config["microbatch_sequences"]),
                                  spec["loss_normalization"])
        groups = [g for g in muon.param_groups if any(other is p for other in g['params'])]
        if len(groups) != 1:
            raise ValueError('local decay readout requires one owning parameter group')
        wd = float(groups[0]['weight_decay'])
        normalization_tokens = int(config['batch_tokens']) if spec['loss_normalization'] == 'mean' else 1
        shift = decay_shift(wd, normalization_tokens, spec['include_weight_decay'])
        apply = local_response(hvp, momentum, spec["mode"], weight_decay=wd,
                               normalization_tokens=normalization_tokens,
                               include_weight_decay=spec['include_weight_decay'])
        scope = (spec["mode"], spec["loss_batch"], spec["loss_normalization"],
                 spec["validation_tokens"], spec["validation_offset"], str(config.get("val_data")),
                 tuple(p.shape), spec['include_weight_decay'], wd, normalization_tokens)
        previous = warm_cache.get(name)
        use_warm = (spec["warm_start"] and previous is not None and
                    previous["update"] == t - 1 and previous["scope"] == scope)
        initial_vector = previous["vector"] if use_warm else None
        devices = [p.device.index] if p.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            arguments = dict(device=p.device, seed=spec["seed"] + t * len(muon.sorted_params()) + index,
                residual_tol=spec["residual_tol"], real_tol=spec["real_tol"],
                substantial_fraction=spec["substantial_fraction"])
            if spec["convergence_dims"] is not None:
                summary, vector = converged_response(apply, tuple(p.shape), **arguments,
                    krylov_dims=spec["convergence_dims"], starts=spec["convergence_starts"],
                    eigenvalue_rtol=spec["eigenvalue_rtol"], initial_vector=initial_vector)
            else:
                summary, vector = arnoldi_response(apply, tuple(p.shape), **arguments,
                                                  krylov_dim=spec["krylov_dim"])
        summary.update(update=t, matrix=name, owner_rank=state["rank"], mode=spec["mode"],
                       include_weight_decay=spec['include_weight_decay'], weight_decay=wd,
                       decay_normalization_tokens=normalization_tokens,
                       training_batch_tokens=int(config['batch_tokens']),
                       decay_is_instantaneous=True, hyperball_included=False,
                       warm_start_used=bool(use_warm),
                       warm_start_update=previous["update"] if use_warm else None,
                       warm_start_mu=previous["mu"] if use_warm else None,
                       shape_multiplier=max(1, p.shape[0] / p.shape[1]) ** .5,
                       momentum_clock="actual post-insertion pre-polar M_t",
                       weight_clock="full model before optimizer update t",
                       loss=spec["loss_batch"], loss_normalization=spec["loss_normalization"],
                       validation_offset=spec["validation_offset"] if spec["loss_batch"] == "fixed_validation" else None,
                       batch_sequences=len(batch[0]), tokens=int(batch[1].numel()),
                       hessian_backend="eager FP32 loss; math SDPA; autograd Hessian block",
                       momentum_norm=float(momentum.double().norm()),
                       kernel_weights=kernel.weights.cpu().tolist(),
                       normalize_available=kernel.normalize_available,
                       excludes=["learning_rate", "lag0_coefficient", "kernel_derivative", "hyperball"] +
                                ([] if spec['include_weight_decay'] else ['weight_decay']),
                       precision="FP32 Hessian; BF16 NS primals with linear FP32 tangents"
                                 if spec["mode"] == "newton_schulz" else "FP32 Hessian; FP64 ideal polar derivative")
        summary.update(decay_readout(summary['mu_estimate'], [shift]))
        ctx["local_muon_results"][name] = summary
        warm_cache.pop(name, None)
        if summary.get("empirical_convergence") and vector is not None:
            warm_cache[name] = dict(update=t, scope=scope, vector=vector.clone(),
                                    mu=summary["mu_estimate"])
        if vector is not None:
            # Real eigenpair selected by imag tolerance. Keep both components
            # rather than silently discarding a nonzero imaginary component.
            ctx["local_muon_vectors"][name] = dict(real=vector.real.float(), imag=vector.imag.float())
    return observe


def finish_readout(ctx, update, rank):
    summaries = ctx["local_muon_results"]
    if set(summaries) != ctx["local_muon_expected"]:
        raise ValueError("local Muon readout is missing expected owned matrices")
    if not summaries:
        return
    receipt = ctx["store"].save("local_muon", update,
                                dict(update=update, rank=rank, matrices=summaries,
                                     leading_positive_vectors=ctx["local_muon_vectors"]))
    with (ctx["root"] / f"local_muon_rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(update=update, rank=rank, file=receipt["file"],
                                    sha256=receipt["sha256"], bytes=receipt["bytes"],
                                    matrices=summaries), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    path = ctx["root"] / f"local_muon_rank{rank}.csv"
    columns = ["update", "matrix", "owner_rank", "mode", "status", "mu_estimate",
               "leading_positive_candidate", "leading_positive_residual", "shape_multiplier",
               "krylov_dimension", "matvec_calls", "tokens", "loss", "loss_normalization",
               "validation_offset", "warm_start_used", "warm_start_update", "warm_start_mu",
               "include_weight_decay", "weight_decay", "decay_normalization_tokens",
               "mu_polar_component", "weight_decay_shift"]
    columns += ["most_negative_real_ritz", "max_abs_imag_ritz"]
    new = not path.exists()
    if not new:
        with path.open(newline='') as stream:
            if next(csv.reader(stream), None) != columns:
                raise ValueError('local Muon CSV schema changed; use a new output path')
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new:
            writer.writeheader()
        for summary in summaries.values():
            writer.writerow({k: summary[k] for k in columns})
        stream.flush(); os.fsync(stream.fileno())
    ctx["local_muon_vectors"] = {}
    ctx["local_muon_results"] = {}
