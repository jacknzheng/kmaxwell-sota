import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import numpy as np

from .coupled_muon import (
    FrozenMuonHessian,
    ShardedVector,
    CoupledResponse,
    qualify_eigenpair,
)
from .validation_probe import fixed_validation_batch
from .response_decay import add_decay_flag, checkpoint_decay, decay_shift, decay_readout
from .hvp_systems import StagedHessian
from .frozen_head_loss import FrozenHeadFusionGPT
from ..harness.model_gpt import GPT


def get_metrics_snapshot_path(capture: Path, rank: int, update: int) -> Path:
    """Prefer schema-3 snapshot names; preserve reads of immutable schema-2 captures."""
    path = capture / f"rank{rank}/metrics_snapshot_{update:06d}.pt"
    return path if path.exists() else capture / f"rank{rank}/boundary_{update:06d}.pt"


def largest_three_disjoint_probe(available, world=8, sequence=1024):
    """Align rank slices; leave a target token between nonoverlapping probes."""
    result = ((available - 3) // (3 * world * sequence)) * world * sequence
    if result < world * sequence:
        raise ValueError("Validation pool too small for three distributed probes")
    return result


def start_check_policy(output, update, has_warm):
    """Optional run-local policy, reread by each new checkpoint child.

    No policy preserves the original two-start behavior. This changes solver
    auditing only, never the operator, token probe, or residual diagnostics.
    """
    path = output.parent / "mu_start_policy.json"
    if not path.exists():
        return dict(fresh_start=True, previous_mu=None, jump_rtol=0.2)
    policy = json.loads(path.read_text())
    if policy["format"] != 1 or policy["mode"] != "sparse_fresh_checks":
        raise ValueError("Unknown Mu start-check policy")
    threshold = float(policy["jump_relative_threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid Mu jump threshold")
    previous = None
    csv_path = output.parent / "mu.csv"
    if csv_path.exists():
        with csv_path.open() as stream:
            earlier = [r for r in csv.DictReader(stream) if int(r["update"]) < update and r["mu"]]
        if earlier:
            previous = float(max(earlier, key=lambda r: int(r["update"]))["mu"])
    # A scheduled LR step predicts a Mu jump. Audit only a deviation from the
    # saved-kernel prediction's ratio, not the expected jump itself.
    ratio = float(policy.get("expected_mu_ratio_by_update", {}).get(str(update), 1.0))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid predicted Mu endpoint ratio")
    if previous is not None:
        previous *= ratio
    return dict(
        fresh_start=not has_warm or update in policy["audit_updates"],
        previous_mu=previous,
        jump_rtol=threshold,
    )


def compatible_resume(previous, current, old_pins, new_pins):
    keys = (
        "update",
        "normalization",
        "operator",
        "precision",
        "residual_tolerance",
        "eigenvalue_rtol",
        "token_rtol",
        "total_muon_parameters",
        "torch",
    )
    decay_keys = (
        "include_weight_decay",
        "weight_decay_by_matrix",
        "training_batch_tokens",
        "decay_normalization_tokens",
        "hvp_mode",
        "head_backend",
        "matmul_precision",
        "gemm_precision",
    )
    return (
        old_pins == new_pins
        and all(previous[k] == current[k] for k in keys)
        and all(previous.get(k) == current.get(k) for k in decay_keys)
        and json.dumps(previous["matrix_layouts"]) == json.dumps(current["matrix_layouts"])
        and (
            "operator_hashes" not in previous
            or previous["operator_hashes"] == current["operator_hashes"]
        )
    )


def requested_probe_results(sizes, probe):
    """Measure every requested size; qualification is a readout, not a veto.

    None means an explicit resource/checkpoint boundary. A completed but
    numerically unqualified estimate must not prevent measuring larger probes.
    """
    for tokens in sizes:
        result = probe(tokens, 0)
        if result is None:
            return
        yield result


def main():
    """Estimate the coupled response at one saved checkpoint using torchrun.

    Ranks split validation tokens for Hessian-vector products and own disjoint
    Muon matrices for polar JVPs and Arnoldi vectors. Adam weights, momentum and
    the selected validation probe stay fixed throughout each solve.
    """
    parser = argparse.ArgumentParser()
    add_decay_flag(parser)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", type=int, default=1250)
    parser.add_argument(
        "--token-sizes",
        default="131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432",
    )
    parser.add_argument("--dimensions", default="8,16,32,64,128,256")
    parser.add_argument("--microbatches", default="1,2,4,8")
    parser.add_argument(
        "--cache-batches",
        type=int,
        default=0,
        help="Optional graph retention; 0 avoids previously measured OOM",
    )
    parser.add_argument("--hvp-mode", choices=("eager", "trace", "compile"), default="eager")
    parser.add_argument("--head-backend", choices=("original", "cached"), default="original")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--gemm-precision", choices=("fp32", "bf16_fp32"), default="fp32")
    parser.add_argument(
        "--fixed-probes",
        action="store_true",
        help="Evaluate requested nested sizes only; do not claim independent sampling convergence",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Directory with current_vector_rankR.pt; independent fresh start remains mandatory",
    )
    parser.add_argument(
        "--linearity-diagnostic",
        action="store_true",
        help="Compatibility flag: precision-induced nonlinearity is always diagnostic, not a veto",
    )
    parser.add_argument("--token-rtol", type=float, default=0.01)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--pod-expires-at", type=float)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    if args.hvp_mode == "eager" and args.gemm_precision != "fp32":
        raise ValueError("BF16 GEMMs require staged HVPs")
    if args.hvp_mode != "eager" and args.cache_batches:
        raise ValueError("Staging cannot retain first-gradient graphs")
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 8:
        raise ValueError("This benchmark requires eight ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    start_wall = time.perf_counter()
    previous = (
        json.loads((args.resume_from / "progress.json").read_text()) if args.resume_from else None
    )
    row = dict(
        stage="loading",
        update=args.update,
        gpu=torch.cuda.get_device_name(),
        world=world,
        torch=torch.__version__,
        normalization="global token mean",
        operator=(
            "blockdiag(s_i Dmsign(M_i)) H_Muon,Muon + diag(lambda_i/N_train); Adam frozen"
            if args.include_weight_decay
            else "blockdiag(s_i Dmsign(M_i)) H_Muon,Muon; Adam frozen"
        ),
        include_weight_decay=args.include_weight_decay,
        precision=f"{args.gemm_precision} Hessian/{args.matmul_precision}; BF16 NS primals/FP32 linear tangents",
        hvp_mode=args.hvp_mode,
        head_backend=args.head_backend,
        matmul_precision=args.matmul_precision,
        gemm_precision=args.gemm_precision,
        accumulation_dtype="float32",
        reduced_precision_reduction=False,
        residual_tolerance=None,
        residual_gating=False,
        eigenvalue_rtol=0.001,
        token_rtol=args.token_rtol,
        adam_training_changed=False,
        probes=[],
        tuning=[],
        pilot={},
    )
    source = Path(__file__).resolve()
    row["study_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    row["operator_hashes"] = {
        str(p.relative_to(source.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            source.with_name("coupled_muon.py"),
            source.with_name("muon_jvp.py"),
            source.with_name("validation_probe.py"),
            source.with_name("response_decay.py"),
            source.with_name("hvp_systems.py"),
            source.with_name("frozen_head_loss.py"),
            source.with_name("bf16_gemm.py"),
            source.parent.parent / "harness/model_gpt.py",
        )
    }
    row["original_started_unix"] = (
        (
            previous.get("original_started_unix")
            or json.loads((args.resume_from / "process.json").read_text())["started"]
        )
        if previous
        else time.time()
    )
    wandb_run = None
    if rank == 0:
        import wandb

        wandb_run = wandb.init(
            project="critroll-coupled-muon",
            mode="offline",
            dir=str(args.output),
            config=vars(args),
        )
        (args.output / "wandb_dir.txt").write_text(wandb_run.dir.rsplit("/files", 1)[0] + "\n")

    def save():
        if rank == 0:
            row["elapsed_seconds"] = time.perf_counter() - start_wall
            row["total_wall_seconds"] = time.time() - row["original_started_unix"]
            temp = args.output / "progress.tmp"
            temp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            temp.replace(args.output / "progress.json")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "stage",
                            "elapsed_seconds",
                            "current_probe",
                            "current_search",
                        )
                    }
                ),
                flush=True,
            )

    save()
    before = torch.load(
        get_metrics_snapshot_path(args.capture, 0, args.update),
        map_location="cpu",
        weights_only=True,
    )
    owned = (
        before
        if rank == 0
        else torch.load(
            get_metrics_snapshot_path(args.capture, rank, args.update),
            map_location="cpu",
            weights_only=True,
        )
    )
    momenta = {n: m.cuda() for n, m in owned["post_insertion_prepolar_momentum"].items()}
    local_layout = [(n, tuple(m.shape)) for n, m in sorted(momenta.items())]
    layouts = [None] * world
    dist.all_gather_object(layouts, local_layout)
    space = ShardedVector(layouts, rank, "cuda")
    if len(space.names) != 72:
        raise ValueError("Checkpoint does not bind all 72 Muon matrices")
    if args.include_weight_decay:
        coefficients, training_tokens = checkpoint_decay(before, space.names)
    else:
        coefficients = {n: 0.0 for n in space.names}
        training_tokens = (
            before["training_batch_tokens"]
            if before.get("format") == 3
            else before["batch_receipts"]["used"]["batch_tokens"]
        )
    shifts = [
        decay_shift(v, training_tokens, args.include_weight_decay) for v in coefficients.values()
    ]
    row.update(
        weight_decay_by_matrix=coefficients,
        training_batch_tokens=training_tokens,
        decay_normalization_tokens=training_tokens,
        decay_is_instantaneous=True,
        hyperball_included=False,
    )
    model_class = GPT if args.head_backend == "original" else FrozenHeadFusionGPT
    model = model_class(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    model.load_state_dict({**before["model_before"], **before["buffers"]}, strict=True)
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    del before, owned
    row["matrix_layouts"] = layouts
    row["total_muon_parameters"] = sum(space.sizes)
    row["max_basis_shard_bytes_per_vector"] = space.width * 4
    reusable = {}
    if previous:
        old_manifest = json.loads((args.resume_from / "input_manifest.json").read_text())
        new_manifest = json.loads((args.output / "input_manifest.json").read_text())
        if not compatible_resume(previous, row, old_manifest["pins"], new_manifest["pins"]):
            raise ValueError("Resume requires identical checkpoint/data pins and operator/gates")
        if "operator_hashes" not in previous and old_manifest["source_head"] != "fc93b33":
            raise ValueError("Legacy resume source is not the known unchanged operator")
        # Preserve completed measurements, including their failed qualification
        # flags, so continuation never repeats them just to unlock larger sizes.
        row["probes"] = list(previous["probes"])
        reusable = {(p["tokens"], p["offset"]): p for p in row["probes"]}
        row["resume_from"] = str(args.resume_from)
    save()

    def batch(tokens, offset):
        if tokens % (world * 1024):
            raise ValueError("Probe must split into equal whole sequences")
        return fixed_validation_batch(
            {"val_data": args.validation},
            "cuda",
            tokens // world,
            offset + rank * tokens // world,
        )

    program_cache, metadata_cache = {}, {}
    backend = "reverse"

    def make_hessian(b, microbatch, tokens, cache_batches=0):
        kwargs = dict(global_tokens=tokens, cache_batches=cache_batches, backend=backend)
        if args.hvp_mode == "eager":
            return FrozenMuonHessian(model, params, buffers, space.names, b, microbatch, **kwargs)
        h = StagedHessian(
            model,
            params,
            buffers,
            space.names,
            b,
            microbatch,
            mode=args.hvp_mode,
            gemm_precision=args.gemm_precision,
            **kwargs,
        )
        h.programs.update(program_cache)
        h.graph_metadata.update(metadata_cache)
        return h

    def close_hessian(h):
        if isinstance(h, StagedHessian):
            program_cache.update(h.programs)
            metadata_cache.update(h.graph_metadata)
            row["hvp_graph_metadata"] = metadata_cache
        h.close()

    # Reserve 128 Krylov vectors PLUS peak HVP workspace. Keeping unused VRAM
    # solely to claim utilization is not an optimization.
    reserve = max(int(k) for k in args.dimensions.split(",")) * space.width * 4 + 4 * 1024**3
    v = space.random(921)
    direction_flat = torch.empty(world * space.width, device="cuda")
    dist.all_gather_into_tensor(direction_flat, v)
    direction = space.unpack(direction_flat, True)
    directions = tuple(direction[n] for n in space.names)
    tune_batch = batch(131072, 0)  # 16 sequences/rank
    # Forward-over-reverse is an exact alternative, not a Hessian surrogate.
    # Keep only if it matches the reference and actually improves throughput.
    forward_check = {}
    reference = FrozenMuonHessian(
        model, params, buffers, space.names, tune_batch, 1, global_tokens=131072
    )
    alternate = FrozenMuonHessian(
        model,
        params,
        buffers,
        space.names,
        tune_batch,
        1,
        global_tokens=131072,
        backend="forward",
    )
    try:
        if args.hvp_mode != "eager":
            raise NotImplementedError(
                "Use already benchmarked reverse AD for staged speedup comparison"
            )
        tick = time.perf_counter()
        expected = reference(directions)
        torch.cuda.synchronize()
        reverse_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        actual = alternate(directions)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - tick
        error = (
            sum(float((a.double() - e.double()).square().sum()) for a, e in zip(actual, expected))
            / max(sum(float(e.double().square().sum()) for e in expected), 1e-30)
        ) ** 0.5
        forward_check = dict(
            relative_error=error,
            reverse_seconds=reverse_seconds,
            forward_seconds=forward_seconds,
            supported=True,
        )
        del expected, actual
    except (RuntimeError, NotImplementedError) as exc:
        forward_check = dict(supported=False, error=str(exc))
    finally:
        reference.close()
        alternate.close()
        del reference, alternate
        torch.cuda.empty_cache()
    reports = [None] * world
    dist.all_gather_object(reports, forward_check)
    if all(
        r.get("supported")
        and r["relative_error"] <= 1e-4
        and r["forward_seconds"] < r["reverse_seconds"] * 0.98
        for r in reports
    ):
        backend = "forward"
    row["hvp_backend_comparison"] = dict(reports=reports, selected=backend)
    row["stage"] = "hvp_backend_check"
    save()
    candidates = [int(k) for k in args.microbatches.split(",")]
    best = None
    for micro in candidates:
        if row["tuning"]:
            last = row["tuning"][-1]
            # Never deliberately hit a known OOM. Prior 8-sequence timing and
            # nearly linear activation-memory scaling predict 16 cannot fit.
            valid_reports = [r for r in last.get("reports", []) if r.get("peak_bytes")]
            if valid_reports and max(r["peak_bytes"] for r in valid_reports) * micro / last[
                "microbatch"
            ] + reserve > min(r["total_bytes"] for r in valid_reports):
                row["tuning"].append(
                    dict(
                        microbatch=micro,
                        viable=False,
                        skipped="predicted workspace exceeds HBM plus Krylov reserve",
                        reports=[],
                    )
                )
                save()
                continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        hessian = make_hessian(tune_batch, micro, 131072)
        failure, duration, peak = 0, None, 0
        try:
            # Warm-up and a separately timed second full local token pass.
            for repeat in range(2):
                tick = time.perf_counter()
                values = hessian(directions)
                torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                del values
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            failure = 1
        finally:
            close_hessian(hessian)
            del hessian
            torch.cuda.empty_cache()
        reports = [None] * world
        dist.all_gather_object(
            reports,
            dict(
                rank=rank,
                oom=bool(failure),
                seconds=duration,
                peak_bytes=peak,
                total_bytes=torch.cuda.get_device_properties(rank).total_memory,
            ),
        )
        viable = not any(r["oom"] or r["peak_bytes"] + reserve > r["total_bytes"] for r in reports)
        worst = max(r["seconds"] or float("inf") for r in reports)
        row["tuning"].append(dict(microbatch=micro, viable=viable, reports=reports))
        if viable and (best is None or worst < best[0]):
            best = worst, micro
        row["stage"] = "tuning_microbatch"
        save()
    if best is None:
        raise RuntimeError("No microbatch fits the HVP plus Krylov reserve")
    micro = best[1]
    row["microbatch"] = micro
    del tune_batch, direction, direction_flat, directions
    torch.cuda.empty_cache()

    def operator(tokens, offset, cache_batches=0):
        b = batch(tokens, offset)
        hashes = [None] * world
        digest = hashlib.sha256(
            b[0].cpu().numpy().tobytes() + b[1].cpu().numpy().tobytes()
        ).hexdigest()
        dist.all_gather_object(hashes, digest)
        h = make_hessian(b, micro, tokens, cache_batches)
        return CoupledResponse(
            space,
            h,
            momenta,
            weight_decay={n: coefficients[n] for n in momenta},
            normalization_tokens=training_tokens,
            include_weight_decay=args.include_weight_decay,
        ), hashes

    apply, hashes = operator(131072, 0)
    row["stage"] = "numerical_pilot"
    save()
    a = apply(v)
    repeat = apply(v)
    twice = apply(2 * v)
    w = space.random(1523)
    aw = apply(w)
    combined = apply(v + w)
    row["pilot"] = dict(
        repeatability=space.norm(a - repeat) / space.norm(a),
        homogeneity=space.norm(twice - 2 * a) / (2 * space.norm(a)),
        additivity=space.norm(combined - a - aw) / space.norm(a + aw),
        matvec_seconds=apply.seconds / apply.calls,
        batch_hashes=hashes,
    )
    save()
    row["pilot"]["linearity_diagnostic_only"] = True
    # Measure retained-graph speedup on the SAME direction and token probe.
    baseline = apply.seconds / apply.calls
    close_hessian(apply.hessian)
    del apply, a, repeat, twice, w, aw, combined
    torch.cuda.empty_cache()
    cache_batches = 0
    cached, _ = operator(131072, 0, cache_batches=args.cache_batches)
    torch.cuda.reset_peak_memory_stats()
    # Test graph retention locally before a collective response. If any rank
    # cannot fit it, all ranks discard it before entering the next collective.
    gathered = torch.empty(world * space.width, device="cuda")
    dist.all_gather_into_tensor(gathered, v)
    d = space.unpack(gathered, True)
    directions = tuple(d[n] for n in space.names)
    failure, elapsed, peak = False, None, 0
    try:
        values = cached.hessian(directions)
        del values
        t0 = time.perf_counter()
        values = cached.hessian(directions)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        del values
        peak = torch.cuda.max_memory_allocated()
    except torch.cuda.OutOfMemoryError:
        failure = True
    reports = [None] * world
    dist.all_gather_object(reports, dict(rank=rank, oom=failure, peak_bytes=peak, seconds=elapsed))
    if (
        args.cache_batches
        and not any(r["oom"] for r in reports)
        and max(r["peak_bytes"] for r in reports) + reserve
        < torch.cuda.get_device_properties(rank).total_memory
        and max(r["seconds"] for r in reports) < baseline * 0.98
    ):
        cache_batches = args.cache_batches
    row["graph_cache_benchmark"] = dict(
        reports=reports, baseline_seconds=baseline, selected_cache_batches=cache_batches
    )
    close_hessian(cached.hessian)
    del cached, v, gathered, d, directions
    torch.cuda.empty_cache()
    if args.pilot_only:
        row["stage"] = "pilot_complete"
        save()
        dist.destroy_process_group()
        return

    dimensions = tuple(int(k) for k in args.dimensions.split(","))
    available = int(np.fromfile(args.validation, dtype=np.int32, count=256)[2])
    largest_disjoint = largest_three_disjoint_probe(available, world)
    sizes = tuple(sorted(set(min(int(k), largest_disjoint) for k in args.token_sizes.split(","))))
    row["validation_tokens_available"] = available
    row["maximum_three_disjoint_probe_tokens"] = largest_disjoint
    row["effective_token_sizes"] = sizes
    nested = []
    warm = None
    if args.warm_start:
        warm = torch.load(
            args.warm_start / f"current_vector_rank{rank}.pt",
            weights_only=True,
            map_location="cuda",
        )
        if warm.shape != (space.width,) or not torch.isfinite(warm).all():
            raise ValueError("Invalid warm-start vector")
        row["warm_start"] = str(args.warm_start)
    resume_required = False
    policy_packet = [
        start_check_policy(args.output, args.update, warm is not None) if rank == 0 else None
    ]
    dist.broadcast_object_list(policy_packet, src=0)
    start_policy = policy_packet[0]
    row["start_check_policy"] = start_policy

    def probe(tokens, offset):
        nonlocal warm, resume_required
        apply, hashes = operator(tokens, offset, cache_batches)
        if (tokens, offset) in reusable:
            result = reusable[(tokens, offset)]
            if result["batch_hashes"] != hashes:
                raise ValueError("Resumed validation bytes differ")
            if offset == 0:
                warm = torch.load(
                    args.resume_from / f"leading_N{tokens}_o0_rank{rank}.pt",
                    weights_only=True,
                    map_location="cuda",
                )
                if warm.shape != (space.width,) or not torch.isfinite(warm).all():
                    raise ValueError("Invalid resumed leading vector")
            close_hessian(apply.hessian)
            del apply
            save()
            return result
        if args.pod_expires_at:
            completed = row["probes"]
            reference = max(completed, key=lambda p: p["tokens"]) if completed else None
            estimate = (
                max(
                    600.0,
                    reference["seconds"] * tokens / reference["tokens"] * (1.6 if offset else 1.3),
                )
                if reference
                else 600.0
            )
            flag = torch.tensor(
                int(rank == 0 and time.time() + estimate + 900 > args.pod_expires_at),
                device="cuda",
            )
            dist.broadcast(flag, 0)
            if flag.item():
                resume_required = True
                row["resume_request"] = dict(
                    tokens=tokens, offset=offset, estimated_seconds=estimate
                )
                close_hessian(apply.hessian)
                del apply
                save()
                return None
        row["stage"] = "eigensolver"
        row["current_probe"] = dict(tokens=tokens, offset=offset)
        row.pop("current_search", None)
        tick = time.perf_counter()

        def progress(search, response=apply):
            row["current_search"] = {
                **search,
                "operator_calls": response.calls,
                "operator_seconds": response.seconds,
                "tokens": tokens,
                "offset": offset,
            }
            if rank == 0 and search["mu"]:
                wandb_run.log(
                    dict(
                        objective=search["mu"],
                        mu=search["mu"],
                        residual=search["residual"],
                        tokens=tokens,
                        offset=offset,
                        arnoldi_dimension=search["dimension"],
                        seed=search["seed"],
                    )
                )
            save()

        def checkpoint(search, vector):
            temp = args.output / f"current_vector_rank{rank}.tmp"
            torch.save(vector.cpu(), temp)
            temp.replace(args.output / f"current_vector_rank{rank}.pt")

        result, vector = qualify_eigenpair(
            apply,
            space,
            dimensions,
            progress=progress,
            initial=warm if offset == 0 else None,
            checkpoint=checkpoint,
            **(start_policy if offset == 0 else dict(fresh_start=True)),
        )
        result.update(
            tokens=tokens,
            offset=offset,
            seconds=time.perf_counter() - tick,
            operator_calls=apply.calls,
            operator_seconds=apply.seconds,
            batch_hashes=hashes,
        )
        result.update(
            decay_readout(result["mu"], shifts),
            include_weight_decay=args.include_weight_decay,
        )
        row["probes"].append(result)
        if vector is not None:
            torch.save(vector.cpu(), args.output / f"leading_N{tokens}_o{offset}_rank{rank}.pt")
            if offset == 0:
                warm = vector.detach()
        if rank == 0:
            path = args.output / "convergence.csv"
            with path.open("w") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "tokens",
                        "offset",
                        "qualified",
                        "mu",
                        "mu_polar_component",
                        "include_weight_decay",
                        "weight_decay_shift",
                        "seconds",
                        "operator_calls",
                        "operator_seconds",
                        "seed_relative_spread",
                    ],
                )
                writer.writeheader()
                writer.writerows({k: p.get(k) for k in writer.fieldnames} for p in row["probes"])
        close_hessian(apply.hessian)
        del apply, vector
        torch.cuda.empty_cache()
        save()
        return result

    converged = False
    for current in requested_probe_results(sizes, probe):
        tokens = current["tokens"]
        nested.append(current)
        if not current["qualified"]:
            row.setdefault("unqualified_probe_tokens", []).append(tokens)
            save()
            continue
        if (
            not args.fixed_probes
            and tokens >= 524288
            and len(nested) >= 3
            and all(p["qualified"] for p in nested[-3:])
            and all(
                abs(a["mu"] - b["mu"]) / b["mu"] <= args.token_rtol
                for a, b in zip(nested[-3:-1], nested[-2:])
            )
        ):
            replicas = [current]
            for offset in (tokens + 1, 2 * (tokens + 1)):
                replica = probe(tokens, offset)
                if replica is None:
                    break
                replicas.append(replica)
            if len(replicas) != 3:
                break
            values = [p["mu"] for p in replicas]
            spread = (max(values) - min(values)) / max(values) if all(values) else None
            row["sampling_gate"] = dict(
                tokens=tokens,
                spread=spread,
                qualified=all(p["qualified"] for p in replicas)
                and spread is not None
                and spread <= args.token_rtol,
            )
            if row["sampling_gate"]["qualified"]:
                converged = True
                break
    row["empirical_convergence"] = converged
    if args.fixed_probes:
        row["nested_probe_comparison"] = [
            dict(tokens=p["tokens"], mu=p["mu"], qualified=p["qualified"]) for p in nested
        ]
        row["fixed_probes_complete"] = len(nested) == len(sizes)
        row["stage"] = (
            "fixed_probes_complete" if row["fixed_probes_complete"] else "incomplete_convergence"
        )
    else:
        row["stage"] = (
            "resume_required"
            if resume_required
            else ("complete" if converged else "incomplete_convergence")
        )
    save()
    if rank == 0:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
