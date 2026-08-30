"""Per-weight-matrix top curvature of the loss Hessian at saved checkpoints.

For each weight matrix m (the 72 Muon block matrices plus embed.weight and
proj.weight), this measures the top eigenvalue of the DIAGONAL BLOCK of the
loss Hessian: perturbations live only in matrix m, and only matrix m's
component of H v is read back. Method: Lanczos with full reorthogonalization,
seeded at the matrix's own gradient block, exact HVPs by double backprop over
a fixed val-token set (identical tokens at every checkpoint and on every
rank, so cross-checkpoint ratios are not data noise).

Also recorded per matrix: alpha_1 (curvature along the matrix's own gradient
direction -- the first Lanczos Rayleigh quotient, free) and the curvature
along the exact polar factor of the gradient block (one extra HVP; the
direction Muon would actually step, before momentum). Polar is skipped for
embed/proj (AdamW parameters, no polar step).

Parallelism: every rank loads the full model and the SAME tokens; matrices
are split across ranks round-robin; zero collectives during compute. Each
rank writes its own JSON shard; rank 0 merges after a barrier.

Run (whole box, ~6 min per checkpoint at 128k tokens):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone \
        --nproc_per_node=8 records/track_3_optimization/offline_analysis/\
measure_per_matrix_curvature.py --dump_dir secant_dumps_foundation \
        --steps 250 500 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.data_fineweb import iterate_batches_single_process  # noqa: E402
from check_secant_direction_with_hvp import (BATCH_TOKENS,  # noqa: E402
                                             load_model_at_checkpoint,
                                             start_distributed_if_launched)


def target_matrix_names() -> list[str]:
    """The 72 Muon block matrices in model order, then embed and LM head."""
    from harness.model_gpt import GPT
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    names = [n for n, p in model.named_parameters()
             if re.match(r"^blocks\..*\.weight$", n) and p.ndim == 2]
    return names + ["embed.weight", "proj.weight"]


def batched_block_hvp(model, params: list[torch.nn.Parameter],
                      vecs: list[Tensor | None], data_glob: str,
                      total_tokens: int, mbs: int) -> list[Tensor | None]:
    """One pass over the token set computing H_mm v_m for every matrix at once.

    vecs[i] is a tensor shaped like params[i] or None (skip). The forward and
    the create_graph backward are shared; each matrix costs one extra backward.
    Returns tensors in the trainer's per-BATCH_TOKENS sum scale.
    """
    live = [i for i, v in enumerate(vecs) if v is not None]
    out: list[Tensor | None] = [
        torch.zeros(params[i].shape, dtype=torch.float32, device=params[i].device)
        if vecs[i] is not None else None for i in range(len(params))]
    tokens_seen = 0
    for inputs, targets in iterate_batches_single_process(
            data_glob, total_tokens, mbs, shard_rank=0, shard_world=1):
        with sdpa_kernel([SDPBackend.MATH]):
            loss = model(inputs, targets)
        g1 = torch.autograd.grad(loss, [params[i] for i in live], create_graph=True)
        for j, i in enumerate(live):
            dot = (g1[j] * vecs[i]).sum()
            keep = j < len(live) - 1
            hv = torch.autograd.grad(dot, params[i], retain_graph=keep)[0]
            out[i] += hv.detach().float()
            del dot, hv
        tokens_seen += inputs.numel()
        del loss, g1
    scale = BATCH_TOKENS / tokens_seen
    for i in live:
        out[i] *= scale
    return out


def block_gradients(model, params: list[torch.nn.Parameter], data_glob: str,
                    total_tokens: int, mbs: int) -> list[Tensor]:
    """Single-backward gradient blocks over the same fixed token set."""
    g = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    tokens_seen = 0
    for inputs, targets in iterate_batches_single_process(
            data_glob, total_tokens, mbs, shard_rank=0, shard_world=1):
        loss = model(inputs, targets)
        for acc, gp in zip(g, torch.autograd.grad(loss, params)):
            acc += gp.detach().float()
        tokens_seen += inputs.numel()
    for acc in g:
        acc *= BATCH_TOKENS / tokens_seen
    return g


def exact_polar(g: Tensor) -> Tensor:
    """U V^T from the SVD of the gradient block, normalized to unit Frobenius."""
    u, s, vh = torch.linalg.svd(g.double(), full_matrices=False)
    p = (u @ vh).float()
    return p / p.norm()


def top_ritz(alphas: list[float], offdiags: list[float]) -> tuple[float, float]:
    """(top eigenvalue, residual bound) of the Lanczos tridiagonal."""
    m = len(alphas)
    t = torch.zeros(m, m, dtype=torch.float64)
    for i, a in enumerate(alphas):
        t[i, i] = a
    for i, b in enumerate(offdiags[:m - 1]):
        t[i, i + 1] = t[i + 1, i] = b
    evals, evecs = torch.linalg.eigh(t)
    return float(evals[-1]), abs(float(evecs[-1, -1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--data", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--tokens", type=int, default=128 * 1024)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out_tag", type=str, default="per_matrix_curvature")
    ap.add_argument("--no_dist", action="store_true",
                    help="no process group: ranks are fully independent (no "
                         "collectives exist in this script's compute path, and "
                         "rank finish times can differ by more than the NCCL "
                         "watchdog timeout); the merge polls for shard files")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    if args.no_dist and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    else:
        rank, world_size = start_distributed_if_launched()
    all_names = target_matrix_names()
    my_names = all_names[rank::world_size]
    results: dict[str, dict] = {}
    for step in args.steps:
        t0 = time.time()
        model_path = os.path.join(args.dump_dir, f"model_step{step:06d}.pt")
        model, params = load_model_at_checkpoint(model_path, [], my_names)
        g = block_gradients(model, params, args.data, args.tokens, args.mbs)
        seeds = []
        for gp in g:
            n = gp.norm()
            seeds.append(gp / n if float(n) > 0 else
                         torch.nn.functional.normalize(torch.randn_like(gp).flatten(),
                                                       dim=0).view_as(gp))
        basis = [[s] for s in seeds]          # per-matrix Lanczos bases
        alphas = [[] for _ in params]
        offdiags = [[] for _ in params]
        active = list(range(len(params)))
        for it in range(args.iters):
            vecs = [basis[i][-1] if i in active else None for i in range(len(params))]
            hv = batched_block_hvp(model, params, vecs, args.data, args.tokens, args.mbs)
            still = []
            for i in active:
                w = hv[i]
                a = float((w * basis[i][-1]).sum())
                alphas[i].append(a)
                for _ in range(2):                     # full reorth, twice
                    for b in basis[i]:
                        w -= (w * b).sum() * b
                beta = float(w.norm())
                if beta < 1e-8 * max(abs(a), 1e-30) or it == args.iters - 1:
                    continue
                offdiags[i].append(beta)
                basis[i].append(w / beta)
                still.append(i)
            active = still
            if rank == 0:
                print(f"step {step} lanczos round {it + 1}/{args.iters}: "
                      f"{len(active)} active ({time.time() - t0:.0f}s)", flush=True)
            if not active:
                break
        polars = [exact_polar(gp) if not my_names[i].startswith(("embed", "proj"))
                  else None for i, gp in enumerate(g)]
        hv_p = batched_block_hvp(model, params, polars, args.data, args.tokens, args.mbs)
        step_out = {}
        for i, name in enumerate(my_names):
            lam, tail = top_ritz(alphas[i], offdiags[i])
            step_out[name] = dict(
                top_eigenvalue=lam,
                residual_tail=tail,
                curvature_along_gradient=alphas[i][0],
                curvature_along_polar=(float((hv_p[i] * polars[i]).sum())
                                       if polars[i] is not None else None),
                gradient_block_norm=float(g[i].norm()),
                alphas=alphas[i], offdiags=offdiags[i],
                shape=list(params[i].shape))
        results[str(step)] = dict(step=step, tokens=args.tokens,
                                  seconds=round(time.time() - t0, 1),
                                  matrices=step_out)
        if rank == 0:
            print(f"step {step} done in {results[str(step)]['seconds']}s", flush=True)
        del model, params, g, seeds, basis, hv_p, polars
        torch.cuda.empty_cache()
    shard_path = os.path.join(args.dump_dir,
                              f"{args.out_tag}_rank{rank}of{world_size}.json")
    with open(shard_path + ".tmp", "w") as f:
        json.dump(results, f)
    os.replace(shard_path + ".tmp", shard_path)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    elif world_size > 1 and rank == 0:
        expected = [os.path.join(args.dump_dir,
                                 f"{args.out_tag}_rank{r}of{world_size}.json")
                    for r in range(world_size)]
        while not all(os.path.exists(p) for p in expected):
            time.sleep(20)
    if rank == 0:
        merged: dict[str, dict] = {}
        for path in sorted(glob.glob(os.path.join(
                args.dump_dir, f"{args.out_tag}_rank*of{world_size}.json"))):
            with open(path) as f:
                shard = json.load(f)
            for step_key, entry in shard.items():
                merged.setdefault(step_key, dict(step=entry["step"],
                                                 tokens=entry["tokens"],
                                                 matrices={}))
                merged[step_key]["matrices"].update(entry["matrices"])
        out_path = os.path.join(args.dump_dir, f"{args.out_tag}.json")
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=1)
        print(f"wrote {out_path}", flush=True)
        print("PERMATRIX_DONE", flush=True)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
