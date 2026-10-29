"""GPU qualification on an actual saved dense checkpoint, not synthetic weights."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from records.track_3_optimization.analysis.local_muon import (
    arnoldi_response, converged_response, frozen_hessian_block, local_response)
from records.track_3_optimization.harness.capture_hooks import batch_digest
from records.track_3_optimization.harness.model_gpt import GPT
from records.track_3_optimization.analysis.local_muon_capture import fixed_validation_batch
from .response_decay import add_decay_flag, checkpoint_decay, decay_shift, decay_readout


def main():
    parser = argparse.ArgumentParser()
    add_decay_flag(parser)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", type=int, default=1250)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--sequences", type=int,
                        help="Test-only prefix of saved batch; default is full global batch")
    parser.add_argument("--fixed-validation", action="store_true")
    parser.add_argument("--tokens", type=int, default=524288)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--matrix", help="Actual owned matrix; default first owned matrix")
    parser.add_argument("--normalization", choices=["sum", "mean"], default="mean")
    parser.add_argument("--matmul-precision", choices=["highest", "high"], default="highest")
    parser.add_argument("--compare-precision", action="store_true")
    parser.add_argument("--warm-start-from", type=Path,
                        help="Previous checkpoint pilot directory on the identical fixed probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.rank)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision(args.matmul_precision)
    before = torch.load(args.capture / f"rank0/boundary_{args.update:06d}.pt",
                        weights_only=True, map_location="cpu")
    owned = (before if args.rank == 0 else torch.load(
        args.capture / f"rank{args.rank}/boundary_{args.update:06d}.pt",
        weights_only=True, map_location="cpu"))
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    model.load_state_dict({**before["model_before"], **before["buffers"]}, strict=True)
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    receipt = before["batch_receipts"]["used"]
    path = Path("/scratch/datasets/fineweb10B") / receipt["shard_name"]
    # A single contiguous saved-batch slice, not random training from the volume.
    raw = np.fromfile(path, dtype=np.uint16, count=receipt["batch_tokens"] + 1,
                      offset=1024 + 2 * receipt["batch_start_in_shard"])
    batches = []
    local = receipt["batch_tokens"] // receipt["world_size"]
    for rank in range(receipt["world_size"]):
        buf = torch.from_numpy(raw[rank * local:(rank + 1) * local + 1].copy())
        batches.append((buf[:-1].to(torch.int32).view(-1, 1024),
                        buf[1:].to(torch.int64).view(-1, 1024)))
    assert batch_digest(batches[0]) == receipt["sha256_this_rank"]
    batch = tuple(torch.cat([b[k] for b in batches]).cuda() for k in range(2))
    if args.sequences is not None:
        if not 1 <= args.sequences <= len(batch[0]):
            raise ValueError("invalid test batch prefix")
        batch = tuple(b[:args.sequences] for b in batch)
    if args.fixed_validation:
        if args.sequences is not None:
            raise ValueError("choose training prefix or fixed validation, not both")
        batch = fixed_validation_batch(dict(val_data="/scratch/datasets/fineweb10B/fineweb_val_*.bin"),
                                       "cuda", args.tokens, args.offset)
    momenta = owned["post_insertion_prepolar_momentum"]
    # First and last owned matrices cover materially different Hessian depths.
    selected = sorted(momenta)
    name = args.matrix or selected[0]
    if name not in momenta:
        raise ValueError("selected matrix is not owned by this rank")
    momentum = momenta[name].cuda()
    coefficients, training_tokens = (checkpoint_decay(before, [name]) if args.include_weight_decay
        else ({name: 0.}, receipt['batch_tokens']))
    normalization_tokens = training_tokens if args.normalization == 'mean' else 1
    shift = decay_shift(coefficients[name], normalization_tokens, args.include_weight_decay)
    del before, owned, momenta
    hvp = frozen_hessian_block(model, params, buffers, name, batch, args.microbatch,
                              args.normalization)
    apply = local_response(hvp, momentum, weight_decay=coefficients[name],
                           normalization_tokens=normalization_tokens,
                           include_weight_decay=args.include_weight_decay)
    generator = torch.Generator(device="cuda").manual_seed(921)
    v = torch.randn(momentum.shape, generator=generator, device="cuda", dtype=torch.float64)
    v /= v.norm()
    row = dict(stage="initializing", update=args.update, matrix=name, rank=args.rank, tokens=batch[1].numel(),
               full_training_batch_tokens=receipt["batch_tokens"],
               include_weight_decay=args.include_weight_decay, weight_decay=coefficients[name],
               weight_decay_shift=shift, decay_normalization_tokens=normalization_tokens,
               decay_is_instantaneous=True, hyperball_included=False,
               test_only_batch_prefix=args.sequences is not None,
               loss=("fixed_validation" if args.fixed_validation else "training"),
               loss_normalization=args.normalization, validation_offset=args.offset,
               batch_sha256=batch_digest(batch), matmul_precision=args.matmul_precision,
               derivative_precision="FP32 Hessian; BF16 NS primals, linear FP32 tangents",
               microbatch=args.microbatch, torch_version=torch.__version__,
               gpu=torch.cuda.get_device_name(args.rank))
    initial_vector = None
    if args.warm_start_from is not None:
        previous = json.loads((args.warm_start_from / f"pilot_rank{args.rank}.json").read_text())
        if not args.fixed_validation or previous["update"] != args.update - 1 or any(
                previous.get(k) != row.get(k) for k in (
                    "matrix", "rank", "tokens", "validation_offset", "batch_sha256",
                    "loss", "loss_normalization", "matmul_precision", "derivative_precision",
                    "include_weight_decay", "weight_decay_shift", "decay_normalization_tokens")):
            raise ValueError("warm pilot requires adjacent checkpoint and identical operator/probe definition")
        if not previous.get("convergence", {}).get("empirical_convergence"):
            raise ValueError("warm pilot requires a qualified previous eigenpair")
        initial_vector = torch.load(args.warm_start_from / f"leading_rank{args.rank}.pt",
                                    weights_only=True, map_location="cpu")
    row["warm_start_update"] = args.update - 1 if initial_vector is not None else None
    def save():
        tmp = args.output / f"pilot_rank{args.rank}.tmp"
        tmp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output / f"pilot_rank{args.rank}.json")
    try:
        save()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        a = apply(v)
        torch.cuda.synchronize()
        row.update(matvec_seconds=time.perf_counter() - start,
                   peak_memory_bytes=torch.cuda.max_memory_allocated(),
                   response_norm=float(a.norm()), stage="one_full_batch_matvec_complete")
        save()
        start = time.perf_counter()
        repeated = apply(v)
        row["repeatability_relative_error"] = float((a - repeated).norm() / a.norm())
        # Quantized tangents can violate linearity; this is important for Arnoldi.
        row["homogeneity_relative_error"] = float((apply(2 * v) - 2 * a).norm() / (2 * a.norm()))
        w = torch.randn(momentum.shape, generator=generator, device="cuda", dtype=torch.float64)
        w /= w.norm()
        aw = apply(w)
        row["additivity_relative_error"] = float((apply(v + w) - a - aw).norm() / (a + aw).norm())
        if args.convergence and max(row["repeatability_relative_error"],
                row["homogeneity_relative_error"], row["additivity_relative_error"]) > .0001:
            raise ValueError("operator linearity/repeatability exceeds solver residual tolerance")
        if args.compare_precision:
            torch.set_float32_matmul_precision("highest" if args.matmul_precision == "high" else "high")
            alternate_start = time.perf_counter()
            alternate = apply(v)
            torch.cuda.synchronize()
            row["alternate_precision_seconds"] = time.perf_counter() - alternate_start
            row["precision_response_relative_error"] = float((a - alternate).norm() / alternate.norm())
            torch.set_float32_matmul_precision(args.matmul_precision)
        save()
        if args.convergence:
            calls = 0
            def measured(v):
                nonlocal calls
                value = apply(v)
                calls += 1
                if calls % 16 == 0:
                    row["completed_operator_applications"] = calls
                    save()
                return value
            result, vector = converged_response(measured, tuple(momentum.shape), device="cuda",
                krylov_dims=[4, 8, 16, 32, 64], starts=2, seed=921,
                eigenvalue_rtol=.001, residual_tol=.0001, initial_vector=initial_vector)
            result.update(decay_readout(result['mu_estimate'], [shift]),
                          include_weight_decay=args.include_weight_decay)
            row["convergence"] = result
            if result["empirical_convergence"]:
                torch.save(vector, args.output / f"leading_rank{args.rank}.pt")
        row.update(stage="complete", qualification_seconds=time.perf_counter() - start)
        save()
        print(json.dumps(row), flush=True)
    except BaseException as exc:
        row.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
