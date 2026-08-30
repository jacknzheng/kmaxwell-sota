"""Global block-spectral generalized sharpness of the loss Hessian via Frank-Wolfe.

This measures the PAPER-FAITHFUL generalized sharpness of a saved checkpoint:

    S(x) = max_v  v^T H v      subject to  ||v_m||_2 <= r_m for every Muon block m,

where H is the loss Hessian at the checkpoint over a fixed val-token set, the
optimization variable v = (v_m) carries one matrix per Muon block matrix (the 72
matrices matching ^blocks\\..*\\.weight$ with ndim==2 -- NOT embed.weight /
proj.weight, which are AdamW parameters outside the Muon spectral domain), and
each block is constrained by its SPECTRAL (operator-2) norm. The feasible set is
the product of spectral-norm balls; the objective is maximized over that whole
product JOINTLY, so this is a single global quantity, NOT a collection of
independent per-matrix diagonal-block maxima.

Radius. Each block uses unit radius r_m = 1.0 by default (--radius). Unit radius
makes the ball the set of matrices with all singular values <= 1 (the natural
"spectral unit ball" Muon steps live in); the objective then reports the largest
curvature reachable by a joint unit-spectral perturbation. Any other constant
radius rescales the objective by r^2 (q is a quadratic form), so the default is a
pure convention -- cross-checkpoint ratios are radius-invariant.

Method: Frank-Wolfe (conditional gradient). Each iteration uses EXACTLY ONE joint
Hessian-vector product that couples all blocks through a single scalar dot:

    loss = model(inputs, targets)                     # one forward (SDPBackend.MATH)
    g1   = autograd.grad(loss, ALL_params, create_graph=True)
    dot  = sum_n (g1[n] * v[n]).sum()                 # sum over EVERY block couples them
    Hv   = autograd.grad(dot, ALL_params)             # (Hv)_m = sum_n H_mn v_n, cross terms in

accumulated over the token minibatches and rescaled by BATCH_TOKENS / tokens_seen
exactly as the reference tools do. The objective gradient is grad_q = 2 H v, so
the per-block linear-minimization oracle (LMO) input is G_m = (Hv)_m (the factor
2 is irrelevant to the argmax). For maximizing <G_m, s_m> over {||s_m||_2 <= r_m}
the maximizer is s_m = r_m U_m V_m^T from the thin SVD G_m = U_m Sigma_m V_m^T
(spectral-UNIT polar, all singular values 1, scaled by r_m -- this is NOT the
Frobenius-normalized exact_polar used in the reference tool). The FW step uses the
standard schedule gamma_k = 2/(k+2), which needs no line search and therefore no
second HVP, keeping the cost at one joint HVP per iteration. The objective
q(v)=<v, Hv> is reported after every iteration by reusing that iteration's Hv (no
extra HVP).

Parallelism. Cross-block coupling forbids splitting matrices across ranks (unlike
the per-matrix diagonal tool). Instead we DATA-PARALLEL shard the fixed token set
across ranks (shard_rank=rank, shard_world=world_size), compute the partial joint
Hv on each rank, and all_reduce(SUM) the Hv blocks every iteration so every rank
holds the identical full joint Hv and evolves an identical v. Initializations are
computed identically on every rank (deterministic seeds / all-reduced gradient),
so no v broadcast is needed. --no_dist / single process runs on one GPU over the
full token set. Rank 0 writes the JSON and TSV artifacts.

Euclidean-for-scale. For scale reference only, we also record the reference tool's
per-matrix DIAGONAL-block top Hessian eigenvalue (Lanczos), and report its max
over the 72 blocks. This is a DIAGONAL quantity, fundamentally different from the
JOINT spectral-ball maximum computed by FW; the two are NOT expected to agree and
no such agreement is claimed. Skip with --no_euclidean.

Run (whole box, mirrors the reference tools):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone \\
        --nproc_per_node=8 records/track_3_optimization/offline_analysis/\\
measure_generalized_sharpness_fw.py --dump_dir secant_dumps_foundation \\
        --steps 250 3000 --tokens 131072 --iters_list 5 10 20 50 --restarts 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch import Tensor

# Same path pattern as the reference tools: make the harness package importable
# when this file lives under records/.../offline_analysis/. Harness imports are
# done lazily (inside functions) so this module imports cleanly on a CPU box with
# no harness / no GPU, which the unit tests rely on.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Pure, model-free primitives (unit-tested on CPU without a GPU or the harness)
# --------------------------------------------------------------------------- #
def muon_matrix_names() -> list[str]:
    """The 72 Muon block matrices in model order (spectral domain of the study).

    These are exactly the ndim==2 parameters matching ^blocks\\..*\\.weight$.
    embed.weight and proj.weight are AdamW parameters and are deliberately
    EXCLUDED -- they are outside the Muon spectral-ball domain.
    """
    import re

    from harness.model_gpt import GPT
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    return [n for n, p in model.named_parameters()
            if re.match(r"^blocks\..*\.weight$", n) and p.ndim == 2]


def polar_lmo(g: Tensor, r: float = 1.0) -> Tensor:
    """FW linear-minimization oracle for the spectral-norm ball {||s||_2 <= r}.

    argmax_{||s||_2 <= r} <g, s> = r * U V^T where g = U Sigma V^T (thin SVD).
    The result has EVERY singular value equal to r (spectral norm exactly r), so
    <g, polar_lmo(g, 1)> = ||g||_nuclear. This is distinct from the reference
    tool's exact_polar, which divides U V^T by its Frobenius norm.
    """
    u, _, vh = torch.linalg.svd(g.double(), full_matrices=False)
    return (r * (u @ vh)).to(g.dtype)


def autograd_joint_hvp(loss_fn, params: list[Tensor],
                       vecs: list[Tensor]) -> list[Tensor]:
    """One JOINT Hessian-vector product with all cross-block terms present.

    (Hv)_m = sum_n H_mn v_n for every block m, from a SINGLE second backward over
    a scalar that sums the g1 . v dot across EVERY block -- that shared scalar is
    exactly what couples the blocks. loss_fn() returns a scalar depending on
    params; params are leaf tensors with requires_grad. Returned tensors are
    detached and shaped like params.
    """
    loss = loss_fn()
    g1 = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g1[n] * vecs[n]).sum() for n in range(len(params)))
    hv = torch.autograd.grad(dot, params)
    return [h.detach() for h in hv]


def autograd_diagonal_hvp(loss_fn, params: list[Tensor],
                          vecs: list[Tensor]) -> list[Tensor]:
    """DIAGONAL-only HVP (Hv)_m = H_mm v_m, the reference tool's per-matrix read.

    Provided ONLY as the negative control for the cross-block coupling test: it
    reads back block m's own component from block m's own dot and is an INCORRECT
    substitute for the joint HVP in this study.
    """
    loss = loss_fn()
    g1 = torch.autograd.grad(loss, params, create_graph=True)
    out: list[Tensor] = []
    for m in range(len(params)):
        dot_m = (g1[m] * vecs[m]).sum()
        keep = m < len(params) - 1
        hv = torch.autograd.grad(dot_m, params[m], retain_graph=keep)[0]
        out.append(hv.detach())
    return out


def make_quadratic_loss(hessian: Tensor, params: list[Tensor]):
    """A quadratic loss 0.5 * x^T H x over x = concat(flattened params).

    Its Hessian is exactly `hessian` regardless of the params' values, so it
    drives the pure-tensor tests of the joint-HVP and FW machinery without a
    model or GPU. Cross-block coupling lives in the off-diagonal blocks of H.
    """
    def loss_fn() -> Tensor:
        x = torch.cat([p.reshape(-1) for p in params])
        return 0.5 * (x @ (hessian @ x))
    return loss_fn


def frank_wolfe(hvp_fn, v0: list[Tensor], radii: list[float],
                max_iters: int) -> dict:
    """Maximize q(v) = v^T H v over the product of spectral balls via Frank-Wolfe.

    hvp_fn(v) -> [ (Hv)_m ] is the JOINT HVP oracle (H fixed). One call per
    recorded objective point; the LMO and the step both reuse that single Hv, so
    the cost is one joint HVP per iteration. Step schedule gamma_k = 2/(k+2)
    (no line search, no second HVP).

    objective_trace[k] = <v_k, H v_k> is the objective at v_k for k=0..max_iters,
    so the value after K FW steps is read off directly as objective_trace[K].

    Returns dict(objective_trace, v) where v is v_{max_iters}.
    """
    v = [x.clone() for x in v0]
    trace: list[float] = []
    for k in range(max_iters + 1):
        hv = hvp_fn(v)                                   # the k-th joint HVP
        trace.append(float(sum((vi * hi).sum() for vi, hi in zip(v, hv))))
        if k == max_iters:
            break
        s = [polar_lmo(hv[m], radii[m]) for m in range(len(v))]
        gamma = 2.0 / (k + 2.0)
        v = [v[m] + gamma * (s[m] - v[m]) for m in range(len(v))]
    return dict(objective_trace=trace, v=v)


def init_v(seed_grads: list[Tensor], radii: list[float], restart: int,
           base_seed: int = 1337) -> list[Tensor]:
    """Initialization on the ball, identical on every rank for a given restart.

    restart 0 seeds from the block gradient's spectral-polar factor scaled to the
    ball (the canonical, deterministic start). restart >= 1 uses a random
    spectral-unit matrix from a CPU generator seeded by (base_seed + restart), so
    all ranks reproduce the identical tensor with no broadcast.
    """
    if restart == 0:
        return [polar_lmo(g, radii[m]) for m, g in enumerate(seed_grads)]
    # CPU generator for cross-rank-reproducible randomness; the result MUST be
    # moved onto the seed-grad device/dtype (a bare .to(dtype) leaves it on CPU
    # and the joint HVP then mixes cpu v with cuda grads -> device-mismatch crash).
    gen = torch.Generator().manual_seed(base_seed + restart)
    out: list[Tensor] = []
    for m, g in enumerate(seed_grads):
        rnd = torch.randn(g.shape, generator=gen, dtype=torch.float64)
        out.append(polar_lmo(rnd, radii[m]).to(device=g.device, dtype=g.dtype))
    return out


# --------------------------------------------------------------------------- #
# Model-based joint HVP (data-parallel token sharding) -- GPU path
# --------------------------------------------------------------------------- #
def block_gradients(model, params: list[torch.nn.Parameter], data_glob: str,
                    total_tokens: int, mbs: int, rank: int,
                    world_size: int) -> list[Tensor]:
    """Sharded, all-reduced single-backward gradient blocks over the fixed set.

    In the trainer's per-BATCH_TOKENS sum scale, identical on every rank.
    """
    from harness.data_fineweb import iterate_batches_single_process

    g = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    tokens_seen = 0
    for inputs, targets in iterate_batches_single_process(
            data_glob, total_tokens, mbs, shard_rank=rank, shard_world=world_size):
        loss = model(inputs, targets)
        for acc, gp in zip(g, torch.autograd.grad(loss, params)):
            acc += gp.detach().float()
        tokens_seen += inputs.numel()
    counts = torch.tensor([tokens_seen], device=params[0].device)
    if world_size > 1:
        for acc in g:
            dist.all_reduce(acc)
        dist.all_reduce(counts)
    tokens_seen = int(counts[0])
    assert tokens_seen > 0, "gradient glob matched no tokens"
    for acc in g:
        acc *= _BATCH_TOKENS() / tokens_seen
    return g


def joint_block_hvp(model, params: list[torch.nn.Parameter], vecs: list[Tensor],
                    data_glob: str, total_tokens: int, mbs: int, rank: int,
                    world_size: int) -> list[Tensor]:
    """ONE joint HVP over the sharded token set: (Hv)_m = sum_n H_mn v_n.

    Forward + create_graph backward per microbatch, then a single second backward
    over the ALL-BLOCK dot (the cross-block coupling). Partial Hv is all-reduced
    so every rank returns the identical global Hv, rescaled to the trainer's
    per-BATCH_TOKENS sum scale.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from harness.data_fineweb import iterate_batches_single_process

    out = [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in params]
    tokens_seen = 0
    for inputs, targets in iterate_batches_single_process(
            data_glob, total_tokens, mbs, shard_rank=rank, shard_world=world_size):
        # fused SDPA kernels do not support double backprop; force the math path
        with sdpa_kernel([SDPBackend.MATH]):
            loss = model(inputs, targets)
        g1 = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g1[n] * vecs[n]).sum() for n in range(len(params)))
        hv = torch.autograd.grad(dot, params)
        for acc, h in zip(out, hv):
            acc += h.detach().float()
        tokens_seen += inputs.numel()
        del loss, g1, dot, hv
    counts = torch.tensor([tokens_seen], device=params[0].device)
    if world_size > 1:
        for acc in out:
            dist.all_reduce(acc)
        dist.all_reduce(counts)
    tokens_seen = int(counts[0])
    assert tokens_seen > 0, "hvp glob matched no tokens"
    scale = _BATCH_TOKENS() / tokens_seen
    for acc in out:
        acc *= scale
    return out


def _BATCH_TOKENS() -> int:
    """The trainer's per-step token count (gradient/HVP scale), from the ref."""
    from check_secant_direction_with_hvp import BATCH_TOKENS
    return BATCH_TOKENS


def euclidean_top_ritz(model, params, my_names, data_glob, total_tokens, mbs,
                       iters: int) -> dict:
    """Reference per-matrix DIAGONAL-block top eigenvalue (Lanczos), for SCALE ONLY.

    Reuses the reference tool's own block_gradients / batched_block_hvp / top_ritz
    on the already-loaded model+params. This is a per-matrix DIAGONAL quantity and
    is fundamentally different from the joint FW spectral-ball maximum; recorded
    only so FW numbers can be read against a familiar magnitude. Runs single-rank
    over the full token set (shard_world=1 inside the reference helpers), so it is
    computed on the caller only. No agreement with FW is expected or claimed.
    """
    from measure_per_matrix_curvature import (batched_block_hvp,
                                              block_gradients as ref_block_grads,
                                              top_ritz)

    g = ref_block_grads(model, params, data_glob, total_tokens, mbs)
    seeds = []
    for gp in g:
        n = gp.norm()
        seeds.append(gp / n if float(n) > 0 else
                     torch.nn.functional.normalize(
                         torch.randn_like(gp).flatten(), dim=0).view_as(gp))
    basis = [[s] for s in seeds]
    alphas: list[list[float]] = [[] for _ in params]
    offdiags: list[list[float]] = [[] for _ in params]
    active = list(range(len(params)))
    for it in range(iters):
        vecs = [basis[i][-1] if i in active else None for i in range(len(params))]
        hv = batched_block_hvp(model, params, vecs, data_glob, total_tokens, mbs)
        still = []
        for i in active:
            w = hv[i]
            a = float((w * basis[i][-1]).sum())
            alphas[i].append(a)
            for _ in range(2):
                for b in basis[i]:
                    w -= (w * b).sum() * b
            beta = float(w.norm())
            if beta < 1e-8 * max(abs(a), 1e-30) or it == iters - 1:
                continue
            offdiags[i].append(beta)
            basis[i].append(w / beta)
            still.append(i)
        active = still
        if not active:
            break
    per_matrix = {}
    for i, name in enumerate(my_names):
        lam, tail = top_ritz(alphas[i], offdiags[i])
        per_matrix[name] = dict(top_eigenvalue=lam, residual_tail=tail)
    vals = [v["top_eigenvalue"] for v in per_matrix.values()]
    return dict(per_matrix=per_matrix, max_top_eigenvalue=max(vals) if vals else None,
                note="per-matrix DIAGONAL-block Lanczos top eigenvalue; for scale "
                     "only; NOT expected to equal the joint FW spectral-ball max")


# --------------------------------------------------------------------------- #
# Per-checkpoint driver
# --------------------------------------------------------------------------- #
def run_checkpoint(model, params, my_names, args, rank, world_size) -> dict:
    """FW generalized sharpness for one loaded checkpoint (all restarts)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    radii = [float(args.radius)] * len(params)
    seed_grads = block_gradients(model, params, args.data, args.tokens,
                                 args.mbs, rank, world_size)
    max_iters = max(max(args.iters_list), 50 if args.restarts > 1 else 0)

    def hvp_fn(v):
        return joint_block_hvp(model, params, v, args.data, args.tokens,
                               args.mbs, rank, world_size)

    restarts_out = []
    for restart in range(args.restarts):
        v0 = init_v(seed_grads, radii, restart)
        res = frank_wolfe(hvp_fn, v0, radii, max_iters)
        trace = res["objective_trace"]
        at_k = {str(k): trace[k] for k in args.iters_list if k < len(trace)}
        restarts_out.append(dict(restart=restart, objective_trace=trace,
                                 objective_at_k=at_k))
        if rank == 0:
            print(f"  restart {restart}: q(v_0)={trace[0]:.4e} "
                  f"q(v_{max_iters})={trace[-1]:.4e} ({time.time() - t0:.0f}s)",
                  flush=True)

    # iters_list read off the shared (restart 0) trajectory
    shared = restarts_out[0]["objective_trace"]
    objective_at_k = {str(k): shared[k] for k in args.iters_list if k < len(shared)}
    ks = sorted(int(k) for k in objective_at_k)
    rel_changes = {}
    for a, b in zip(ks[:-1], ks[1:]):
        base = objective_at_k[str(a)]
        rel_changes[f"{a}->{b}"] = ((objective_at_k[str(b)] - base) / base
                                    if base != 0 else float("nan"))

    # spread across restarts at K=50 (or the max iter if fewer)
    k_spread = min(50, max_iters)
    finals = [ro["objective_trace"][k_spread] for ro in restarts_out
              if k_spread < len(ro["objective_trace"])]
    spread = None
    if len(finals) > 1:
        mean = sum(finals) / len(finals)
        std = (sum((x - mean) ** 2 for x in finals) / len(finals)) ** 0.5
        spread = dict(k=k_spread, n=len(finals), mean=mean, std=std,
                      min=min(finals), max=max(finals), values=finals,
                      single_vs_ensemble=dict(single=finals[0], ensemble_mean=mean))

    euclid = None
    if not args.no_euclidean:
        # The scale reference does NO collectives (shard_world=1, rank 0 only), so
        # ranks 1..N-1 must be held at a barrier while rank 0 runs it -- otherwise
        # they race ahead into the next checkpoint's all_reduce and desync (or trip
        # the NCCL watchdog). The model is still loaded here, so this is cheap for
        # the waiters. If rank 0's Lanczos can exceed the distributed timeout, run
        # under --no_euclidean and take the scale from measure_per_matrix_curvature.
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        if rank == 0:
            euclid = euclidean_top_ritz(model, params, my_names, args.data,
                                        args.tokens, args.mbs, args.euclidean_ritz_iters)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

    peak_mem = (int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else None)
    return dict(
        radius=float(args.radius), tokens=args.tokens, max_iters=max_iters,
        iters_list=args.iters_list, restarts=args.restarts,
        objective_at_k=objective_at_k, relative_changes=rel_changes,
        restart_traces=[ro["objective_trace"] for ro in restarts_out],
        spread_at_k50=spread, euclidean_for_scale=euclid,
        peak_cuda_bytes=peak_mem, seconds=round(time.time() - t0, 1),
        gradient_block_norms=[float(g.norm()) for g in seed_grads],
    )


def write_artifacts(merged: dict, args) -> None:
    """Rank-0 JSON + summary.tsv + objective_trace.tsv in the artifact layout."""
    art = args.artifact_dir
    for sub in ("", "configs", "raw"):
        os.makedirs(os.path.join(art, sub), exist_ok=True)
    with open(os.path.join(art, "raw", f"{args.out_tag}.json"), "w") as f:
        json.dump(merged, f, indent=1)
    with open(os.path.join(art, "configs", f"{args.out_tag}.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    # also drop the merged JSON next to the checkpoints, mirroring the ref tools
    with open(os.path.join(args.dump_dir, f"{args.out_tag}.json"), "w") as f:
        json.dump(merged, f, indent=1)

    with open(os.path.join(art, "summary.tsv"), "w") as f:
        cols = ["step", "radius", "tokens"] + [f"q@{k}" for k in args.iters_list] + \
               ["spread_k", "spread_mean", "spread_std", "spread_min", "spread_max",
                "euclid_max_top_eig", "peak_cuda_bytes", "seconds"]
        f.write("\t".join(cols) + "\n")
        for step in sorted(merged, key=int):
            e = merged[step]
            sp = e.get("spread_at_k50") or {}
            eu = e.get("euclidean_for_scale") or {}
            row = [step, e["radius"], e["tokens"]] + \
                  [e["objective_at_k"].get(str(k), "") for k in args.iters_list] + \
                  [sp.get("k", ""), sp.get("mean", ""), sp.get("std", ""),
                   sp.get("min", ""), sp.get("max", ""),
                   eu.get("max_top_eigenvalue", ""), e.get("peak_cuda_bytes", ""),
                   e["seconds"]]
            f.write("\t".join(str(x) for x in row) + "\n")

    with open(os.path.join(art, "objective_trace.tsv"), "w") as f:
        f.write("step\trestart\titer\tobjective\n")
        for step in sorted(merged, key=int):
            for r_i, trace in enumerate(merged[step]["restart_traces"]):
                for it, val in enumerate(trace):
                    f.write(f"{step}\t{r_i}\t{it}\t{val}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True,
                    help="checkpoint steps, e.g. an early and a late one")
    ap.add_argument("--data", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--radius", type=float, default=1.0,
                    help="per-block spectral-norm ball radius r_m (default 1.0)")
    ap.add_argument("--iters_list", type=int, nargs="+", default=[5, 10, 20, 50],
                    help="FW budgets read off ONE shared trajectory (restart 0)")
    ap.add_argument("--restarts", type=int, default=5,
                    help="independent inits run to K=50 for the spread report")
    ap.add_argument("--no_euclidean", action="store_true",
                    help="skip the per-matrix diagonal top-Ritz scale reference")
    ap.add_argument("--euclidean_ritz_iters", type=int, default=8)
    ap.add_argument("--out_tag", type=str, default="fw_generalized_sharpness")
    ap.add_argument("--artifact_dir", type=str,
                    default="logs/kmaxwell/req019_fw_calibration")
    ap.add_argument("--no_dist", action="store_true",
                    help="no process group; single-process over the full token set")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this measurement needs a GPU for the HVPs"

    from check_secant_direction_with_hvp import (load_model_at_checkpoint,
                                                 start_distributed_if_launched)
    if args.no_dist:
        rank, world_size = 0, 1
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    else:
        rank, world_size = start_distributed_if_launched()

    all_names = muon_matrix_names()
    assert len(all_names) == 72, f"expected 72 Muon block matrices, got {len(all_names)}"

    merged: dict[str, dict] = {}
    for step in args.steps:
        if rank == 0:
            print(f"step {step}: FW generalized sharpness, {len(all_names)} blocks, "
                  f"radius={args.radius}, {args.tokens} tokens, {world_size} ranks",
                  flush=True)
        model_path = os.path.join(args.dump_dir, f"model_step{step:06d}.pt")
        model, params = load_model_at_checkpoint(model_path, [], all_names)
        merged[str(step)] = dict(step=step,
                                 **run_checkpoint(model, params, all_names,
                                                  args, rank, world_size))
        if rank == 0:
            e = merged[str(step)]
            print(f"step {step} done in {e['seconds']}s  q@K={e['objective_at_k']}",
                  flush=True)
        del model, params
        torch.cuda.empty_cache()

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        write_artifacts(merged, args)
        print(f"wrote artifacts under {args.artifact_dir}", flush=True)
        print("FW_SHARPNESS_DONE", flush=True)
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
