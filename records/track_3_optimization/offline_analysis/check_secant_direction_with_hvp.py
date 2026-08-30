from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.data_fineweb import iterate_batches_single_process
from harness.model_gpt import GPT
from optimizers.muon import zeropower_via_newtonschulz5
from secant_gmres_solver.diagnostics import summarize_gram_diagnostics
from secant_gmres_solver.serialization import PR340_BIMAXWELL, read_secant_shards
from secant_gmres_solver.solve import solve_secant_least_squares, split_secant_gram
from secant_gmres_solver.streams import GRAM_CHUNK, accumulate_secant_gram, secant_direction

BATCH_TOKENS = 8 * 64 * 1024   # the trainer's per-step token count; defines gradient scale
NESTEROV_MU = 0.95             # the record's momentum rate, for the comparator directions
Record = dict[str, Any]


def start_distributed_if_launched() -> tuple[int, int]:
    """Under torchrun, binds the local GPU and joins an NCCL group so the data
    passes shard across every GPU; a plain python launch stays single-process.

    Returns:
        (rank, world_size); (0, 1) when not launched by torchrun.
    """
    if "RANK" not in os.environ:
        return 0, 1
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size()


def build_secant_gram(records: list[Record], gradient_per_record: list[Tensor],
                      device: torch.device) -> Tensor:
    """The (2k+1)^2 fp64 Gram over all records, with the given per-record gradient."""
    k = records[0]["m"].shape[0]
    G = torch.zeros(2 * k + 1, 2 * k + 1, dtype=torch.float64, device=device)
    for record, g in zip(records, gradient_per_record):
        accumulate_secant_gram(G, record["m"].to(device), record["q"].to(device),
                               g.to(device), record["x"].to(device), GRAM_CHUNK)
    return G.cpu()


def load_model_at_checkpoint(model_path: str, records: list[Record],
                             muon_names: list[str] | None = None) -> tuple[GPT, list[torch.nn.Parameter]]:
    """Rebuilds the GPT at x_t and binds the Muon parameters in shard order.

    Binding comes from records (with an exact-x consistency assert) when they are
    given, else from muon_names -- the path non-zero ranks take, since only rank 0
    holds the 68 GB of buffer records.

    Raises:
        AssertionError: When the checkpoint disagrees with the shard's x tensors.
    """
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    sd = {key.removeprefix("_orig_mod."): v for key, v in sd.items()}  # tolerate compiled dumps
    model.load_state_dict(sd)
    model = model.cuda()
    named = dict(model.named_parameters())
    if not records:
        return model, [named[name] for name in muon_names or []]
    muon_plist = []
    for record in records:
        p = named[record["name"]]
        assert tuple(p.shape) == tuple(record["shape"])
        mismatch = (p.detach().float().flatten().cpu() - record["x"]).abs().max()
        assert float(mismatch) == 0.0, f"model checkpoint disagrees with shard x for {record['name']}"
        muon_plist.append(p)
    return model, muon_plist


def choose_hvp_directions(G: Tensor, records: list[Record], k: int, basis_rank: int,
                          i_fast: int, i_slow: int,
                          device: torch.device) -> dict[str, Any]:
    """Picks the HVP targets: the top basis_rank orthonormal displacement
    directions Q = S C (C = V_r Sigma_r^-1 from the S Gram) plus the two
    normalized exact-rate raw columns.

    Returns:
        Dict with C_dev ((k, r) fp32 on device), dirs (per record (r+2, n_p)
        fp32 on device), r, s_norm_fast, s_norm_slow, and the S spectrum
        (evals_S descending-order indices in "order").
    """
    _, _, StS, _, _, _ = split_secant_gram(G, k)
    evals_S, V_S = torch.linalg.eigh(StS)
    order = torch.argsort(evals_S, descending=True)
    kept = [int(i) for i in order if float(evals_S[i]) > 0][:basis_rank]
    r = len(kept)
    C_dev = (V_S[:, kept] / evals_S[kept].clamp_min(1e-300).sqrt()).to(torch.float32).to(device)
    s_norm_fast = float(StS[i_fast, i_fast]) ** 0.5
    s_norm_slow = float(StS[i_slow, i_slow]) ** 0.5
    dirs = []
    for record in records:
        s_stack = record["q"].to(device) - record["x"].to(device)
        d = torch.empty(r + 2, s_stack.size(1), dtype=torch.float32, device=device)
        d[:r] = C_dev.mT @ s_stack
        d[r] = s_stack[i_fast] / max(s_norm_fast, 1e-300)
        d[r + 1] = s_stack[i_slow] / max(s_norm_slow, 1e-300)
        dirs.append(d)
        del s_stack
    return dict(C_dev=C_dev, dirs=dirs, r=r, s_norm_fast=s_norm_fast,
                s_norm_slow=s_norm_slow, evals_S=evals_S, order=order)


def collect_gradient_and_hvps(model: GPT, muon_plist: list[torch.nn.Parameter],
                              dirs: list[Tensor], data_glob: str, total_tokens: int,
                              microbatch_sequences: int, cache_microbatches: int,
                              rank: int = 0, world_size: int = 1) -> dict[str, Any]:
    """One data pass: the large-batch gradient, exact HVPs for every direction,
    and (optionally, rank 0) per-microbatch gradient caches for the bootstrap.
    With world_size > 1 the microbatch stream shards across ranks and the
    accumulators are all-reduced, so every rank returns the identical global sums.

    Returns:
        Dict with g_hat (per-param flattened large-batch gradient) and hvp (per
        record, (n_dirs, n_p) exact HVPs), both rescaled to the trainer's
        per-BATCH_TOKENS sum scale; grad_cache (up to cache_microbatches raw
        per-microbatch gradient lists from this rank's shard, unscaled);
        tokens_seen and n_microbatches, global.
    """
    device = torch.device("cuda")
    model.train()
    n_dirs = dirs[0].size(0)
    g_hat = [torch.zeros(p.numel(), dtype=torch.float32, device=device) for p in muon_plist]
    hvp = [torch.zeros_like(d) for d in dirs]
    grad_cache: list[list[Tensor]] = []
    tokens_seen = 0
    n_microbatches = 0
    for inputs, targets in iterate_batches_single_process(data_glob, total_tokens,
                                                          microbatch_sequences,
                                                          shard_rank=rank,
                                                          shard_world=world_size):
        # the fused SDPA kernels do not support double backprop; force the math path
        with sdpa_kernel([SDPBackend.MATH]):
            loss = model(inputs, targets)
        g1 = torch.autograd.grad(loss, muon_plist, create_graph=True)
        for acc, g in zip(g_hat, g1):
            acc += g.detach().float().flatten()
        if len(grad_cache) < cache_microbatches:
            grad_cache.append([g.detach().float().flatten().cpu() for g in g1])
        for j in range(n_dirs):
            dot = sum((g.flatten() * d[j]).sum() for g, d in zip(g1, dirs))
            hv = torch.autograd.grad(dot, muon_plist, retain_graph=(j < n_dirs - 1))
            for acc, h in zip(hvp, hv):
                acc[j] += h.detach().float().flatten()
        tokens_seen += inputs.numel()
        n_microbatches += 1
        del loss, g1
    if world_size > 1:
        for acc in g_hat:
            dist.all_reduce(acc)
        for acc in hvp:
            dist.all_reduce(acc)
        counts = torch.tensor([tokens_seen, n_microbatches], device=device)
        dist.all_reduce(counts)
        tokens_seen, n_microbatches = int(counts[0]), int(counts[1])
    assert tokens_seen > 0
    scale = BATCH_TOKENS / tokens_seen
    for acc in g_hat:
        acc *= scale
    for acc in hvp:
        acc *= scale
    return dict(g_hat=g_hat, hvp=hvp, grad_cache=grad_cache, tokens_seen=tokens_seen,
                n_microbatches=n_microbatches)


def measure_identity_residuals(records: list[Record], hvp: list[Tensor], C_dev: Tensor,
                               r: int, i_fast: int, i_slow: int, s_norm_fast: float,
                               s_norm_slow: float, device: torch.device) -> Tensor:
    """||Y c - H S c|| / ||H S c|| per HVP direction: the online prediction of
    each direction's Hessian image against its true large-batch HVP."""
    n_dirs = r + 2
    true_sq = torch.zeros(n_dirs, dtype=torch.float64)
    err_sq = torch.zeros(n_dirs, dtype=torch.float64)
    for record, d_hvp in zip(records, hvp):
        y_stack = record["m"].to(device) - record["grad"].to(device)
        pred = torch.empty_like(d_hvp)
        pred[:r] = C_dev.mT @ y_stack
        pred[r] = y_stack[i_fast] / max(s_norm_fast, 1e-300)
        pred[r + 1] = y_stack[i_slow] / max(s_norm_slow, 1e-300)
        true_sq += d_hvp.double().square().sum(dim=1).cpu()
        err_sq += (pred.double() - d_hvp.double()).square().sum(dim=1).cpu()
        del y_stack, pred
    return (err_sq / true_sq.clamp_min(1e-300)).sqrt()


def solve_hvp_reference(hvp: list[Tensor], g_hat: list[Tensor], r: int) -> dict[str, Any]:
    """The restricted Newton reference: z* = argmin ||g_hat + (HQ) z|| from the
    accumulated HVPs, via a floored pseudo-inverse of the tiny normal matrix."""
    M = torch.zeros(r, r, dtype=torch.float64)
    ug = torch.zeros(r, dtype=torch.float64)
    for d_hvp, g in zip(hvp, g_hat):
        M += (d_hvp[:r].double() @ d_hvp[:r].double().mT).cpu()
        ug += (d_hvp[:r].double() @ g.double()).cpu()
    z = -(torch.linalg.pinv(M, rcond=1e-12) @ ug)
    gg_hat = float(sum((g.double() @ g.double()) for g in g_hat).cpu())
    res2 = gg_hat + 2 * float(z @ ug) + float(z @ (M @ z))
    return dict(z=z, res_ratio=max(res2, 0.0) ** 0.5 / gg_hat ** 0.5, gg_hat=gg_hat)


def collect_holdout_gradient(muon_plist: list[torch.nn.Parameter], model: GPT,
                             data_glob: str, total_tokens: int, microbatch_sequences: int,
                             rank: int = 0,
                             world_size: int = 1) -> tuple[list[Tensor] | None, int]:
    """The (sharded, all-reduced) gradient over a disjoint holdout shard, in the
    trainer's BATCH_TOKENS sum scale. Returns (None, tokens) off rank 0."""
    device = torch.device("cuda")
    g_holdout = [torch.zeros(p.numel(), dtype=torch.float32, device=device) for p in muon_plist]
    holdout_tokens = 0
    for inputs, targets in iterate_batches_single_process(data_glob, total_tokens,
                                                          microbatch_sequences,
                                                          shard_rank=rank,
                                                          shard_world=world_size):
        loss = model(inputs, targets)
        for acc, g in zip(g_holdout, torch.autograd.grad(loss, muon_plist)):
            acc += g.detach().float().flatten()
        holdout_tokens += inputs.numel()
    if world_size > 1:
        for acc in g_holdout:
            dist.all_reduce(acc)
        counts = torch.tensor([holdout_tokens], device=device)
        dist.all_reduce(counts)
        holdout_tokens = int(counts[0])
    assert holdout_tokens > 0, "holdout glob matched no tokens"
    for acc in g_holdout:
        acc *= BATCH_TOKENS / holdout_tokens
    return (g_holdout if rank == 0 else None), holdout_tokens


def measure_holdout_residual(records: list[Record], G: Tensor, k: int,
                             w_by_label: dict[str, Tensor | None],
                             g_holdout: list[Tensor], holdout_tokens: int) -> dict[str, Any]:
    """||g_holdout + Y w|| / ||g_holdout|| for each fitted w, with Y from the
    dumped minibatch: whether the fitted combination cancels an independent
    gradient estimate, or just the noise it was fitted on. Rank-0 Gram algebra."""
    device = torch.device("cuda")
    YtY = split_secant_gram(G, k)[0]
    c_h = torch.zeros(k, dtype=torch.float64)
    gg_h = 0.0
    for record, g_h in zip(records, g_holdout):
        y_stack = record["m"].to(device) - record["grad"].to(device)
        c_h += (y_stack.double() @ g_h.double()).cpu()
        gg_h += float(g_h.double() @ g_h.double())
        del y_stack
    result: dict[str, Any] = {"tokens": holdout_tokens}
    for label, w in w_by_label.items():
        if w is None:
            result[label] = float("nan")
            continue
        res2 = max(gg_h + 2 * float(w @ c_h) + float(w @ (YtY @ w)), 0.0)
        result[label] = (res2 / gg_h) ** 0.5
    return result


def compare_directions(records: list[Record], dirs: list[Tensor],
                       g_hat: list[Tensor], z: Tensor, r: int,
                       w_mb: Tensor | None, w_lb: Tensor | None,
                       bimaxwell: dict[str, float], i_fast: int, i_slow: int,
                       i_single: int, device: torch.device) -> dict[str, Any]:
    """Global and per-block cosines of every candidate pre-polar input against
    the restricted Newton reference input -Q z*."""
    z32 = z.to(torch.float32).to(device)
    labels = ["p_ref", "p_secant_mb", "p_secant_lb", "d_bimaxwell", "d_single_ema",
              "g_minibatch", "g_largebatch"]
    n_lab = len(labels)
    dots = torch.zeros(n_lab, n_lab, dtype=torch.float64)
    per_block: dict[tuple[str, str], list[float]] = {
        ("p_secant_mb", "p_ref"): [], ("d_bimaxwell", "p_ref"): [], ("g_minibatch", "p_ref"): []}
    for record, d_basis, g in zip(records, dirs, g_hat):
        g_t = record["grad"].to(device)
        x_t = record["x"].to(device)
        q_stack = record["q"].to(device)
        y_fast = record["m"][i_fast].to(device) - g_t
        y_slow = record["m"][i_slow].to(device) - g_t
        vecs = torch.empty(n_lab, g_t.numel(), dtype=torch.float32, device=device)
        vecs[0] = -(d_basis[:r].mT @ z32)                      # -Q z*: the estimand's input
        vecs[1] = secant_direction(q_stack, x_t, w_mb) if w_mb is not None else 0
        vecs[2] = secant_direction(q_stack, x_t, w_lb) if w_lb is not None else 0
        vecs[3] = g_t + NESTEROV_MU * (bimaxwell["w"] * y_fast + (1 - bimaxwell["w"]) * y_slow)
        vecs[4] = g_t + NESTEROV_MU * (record["m"][i_single].to(device) - g_t)
        vecs[5] = g_t
        vecs[6] = g
        blk = (vecs.double() @ vecs.double().mT).cpu()
        dots += blk
        for (label_a, label_b) in per_block:
            ia, ib = labels.index(label_a), labels.index(label_b)
            denom = (blk[ia, ia] * blk[ib, ib]).sqrt()
            per_block[(label_a, label_b)].append(
                float(blk[ia, ib] / denom) if float(denom) > 0 else float("nan"))
        del vecs, q_stack
    norms = dots.diagonal().sqrt()
    cosines = dots / (norms[:, None] * norms[None, :]).clamp_min(1e-300)
    return dict(labels=labels, cosines=cosines, per_block=per_block)


def reconstruct_resampled_gram(u: Tensor, Mg: Tensor, Sg: Tensor, P: Tensor,
                               G: Tensor, num_buffers: int) -> Tensor:
    """The secant Gram for the resampled gradient g* = sum_j u_j g_j, assembled
    from reduced statistics only: M'M and S'M are recovered from G via
    M = Y + g 1', then every g-dependent block is rebuilt around g*.

    Args:
        u: (n_pop,) resample weights (they carry the batch-scale factor).
        Mg: (k, n_pop) projections m_i . g_j.
        Sg: (k, n_pop) projections s_i . g_j.
        P: (n_pop, n_pop) pairwise gradient Gram g_i . g_j.
        G: The original (2k+1)^2 Gram built with the dumped minibatch gradient.
        num_buffers: k.

    Returns:
        The (2k+1)^2 fp64 Gram of [Y*, S, g*] with Y* = M - g* 1'.
    """
    k = num_buffers
    YtY, YtS, StS, Ytg, Stg, gg = split_secant_gram(G, k)
    ones = torch.ones(k, dtype=torch.float64)
    MtM = YtY + torch.outer(Ytg, ones) + torch.outer(ones, Ytg) + gg * torch.outer(ones, ones)
    StM = YtS.mT + torch.outer(Stg, ones)
    Mg_star = Mg @ u
    Sg_star = Sg @ u
    gg_star = float(u @ (P @ u))
    YtY_star = MtM - torch.outer(Mg_star, ones) - torch.outer(ones, Mg_star) \
        + gg_star * torch.outer(ones, ones)
    Ytg_star = Mg_star - gg_star * ones
    StY_star = StM - torch.outer(Sg_star, ones)
    G_star = torch.zeros(2 * k + 1, 2 * k + 1, dtype=torch.float64)
    G_star[:k, :k] = YtY_star
    G_star[:k, k:2 * k] = StY_star.mT
    G_star[k:2 * k, :k] = StY_star
    G_star[k:2 * k, k:2 * k] = StS
    G_star[:k, 2 * k] = Ytg_star
    G_star[2 * k, :k] = Ytg_star
    G_star[k:2 * k, 2 * k] = Sg_star
    G_star[2 * k, k:2 * k] = Sg_star
    G_star[2 * k, 2 * k] = gg_star
    return G_star


def bootstrap_direction_stability(records: list[Record], grad_cache: list[list[Tensor]],
                                  G: Tensor, k: int, r1: int, r2: int,
                                  microbatch_sequences: int, resamples: int,
                                  seed: int) -> dict[str, Any] | None:
    """Microbatch bootstrap of the solve: resample the cached gradients with
    replacement, re-solve entirely in reduced (Gram) statistics, and report the
    cosine spread of the pre- and post-polar directions against the population
    solve. Wide spread = the direction is fit to noise, not curvature."""
    device = torch.device("cuda")
    n_pop = len(grad_cache)
    if n_pop < 8:
        return None
    # reduced statistics: M'g_j, S'g_j per cached microbatch, and P = g_i'g_j
    Mg = torch.zeros(k, n_pop, dtype=torch.float64)
    Sg = torch.zeros(k, n_pop, dtype=torch.float64)
    P = torch.zeros(n_pop, n_pop, dtype=torch.float64)
    for record_index, record in enumerate(records):
        m_stack = record["m"].to(device)
        s_stack = record["q"].to(device) - record["x"].to(device)
        stacked = torch.stack([grad_cache[j][record_index] for j in range(n_pop)]).to(device)
        Mg += (m_stack.double() @ stacked.double().mT).cpu()
        Sg += (s_stack.double() @ stacked.double().mT).cpu()
        P += (stacked.double() @ stacked.double().mT).cpu()
        del m_stack, s_stack, stacked
    # cached grads are sums over one microbatch (mbs*1024 tokens); the streams M
    # are in the per-BATCH_TOKENS sum scale, so resample weight vectors carry the
    # scale: g* = sum_j u_j g_j with sum_j u_j = BATCH_TOKENS/(mbs*1024)
    scale_mb = BATCH_TOKENS / (microbatch_sequences * 1024)
    _, _, StS, _, _, _ = split_secant_gram(G, k)

    def solve_for_weight_vector(u: Tensor) -> Tensor | None:
        """Resample weights u (n_pop, summing to scale_mb) -> the solve on Y* = M - g* 1'."""
        G_star = reconstruct_resampled_gram(u, Mg, Sg, P, G, k)
        return solve_secant_least_squares(G_star, k, r1, r2)[0]

    uniform = torch.full((n_pop,), scale_mb / n_pop, dtype=torch.float64)
    w_pop = solve_for_weight_vector(uniform)
    if w_pop is None:
        return None
    generator = torch.Generator().manual_seed(1337 + seed)
    resampled_w = []
    for _ in range(resamples):
        draws = torch.randint(0, n_pop, (n_pop,), generator=generator)
        u = torch.bincount(draws, minlength=n_pop).to(torch.float64) / n_pop * scale_mb
        resampled_w.append(solve_for_weight_vector(u))

    pre_cos: list[float] = []
    norm_pop = max(float(w_pop @ (StS @ w_pop)), 0.0) ** 0.5
    for w_star in resampled_w:
        if w_star is None:
            pre_cos.append(float("nan"))
            continue
        norm_star = max(float(w_star @ (StS @ w_star)), 0.0) ** 0.5
        pre_cos.append(float(w_pop @ (StS @ w_star)) / max(norm_pop * norm_star, 1e-300))

    # post-polar: per record, polar the population input once, then each resample
    post_dot = torch.zeros(len(resampled_w), dtype=torch.float64)
    post_nn = torch.zeros(len(resampled_w), dtype=torch.float64)
    pop_nn = 0.0
    for record in records:
        q_stack = record["q"].to(device)
        x_t = record["x"].to(device)
        shape = record["shape"]
        pop_polar = zeropower_via_newtonschulz5(
            secant_direction(q_stack, x_t, w_pop).view(shape)).float()
        pop_nn += float(pop_polar.double().square().sum())
        for j, w_star in enumerate(resampled_w):
            if w_star is None:
                continue
            star_polar = zeropower_via_newtonschulz5(
                secant_direction(q_stack, x_t, w_star).view(shape)).float()
            post_dot[j] += float((pop_polar.double() * star_polar.double()).sum())
            post_nn[j] += float(star_polar.double().square().sum())
        del q_stack, pop_polar
    post_cos = [(float(post_dot[j]) / max((pop_nn * float(post_nn[j])) ** 0.5, 1e-300))
                if resampled_w[j] is not None else float("nan")
                for j in range(len(resampled_w))]

    def mean_std(values: list[float]) -> tuple[float, float]:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return (float("nan"), float("nan"))
        mean = sum(finite) / len(finite)
        return (mean, (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5)

    return dict(resamples=resamples, population=n_pop,
                pre_polar_cos=pre_cos, post_polar_cos=post_cos,
                pre_polar_cos_mean_std=mean_std(pre_cos),
                post_polar_cos_mean_std=mean_std(post_cos))


def quartile(values: list[float], q: float) -> float:
    """The q-quantile of the finite entries (nearest-rank), NaN when none."""
    finite = sorted(v for v in values if math.isfinite(v))
    return finite[int(q * (len(finite) - 1))] if finite else float("nan")


def main() -> None:
    """Offline per-checkpoint diagnostic for the secant-GMRES construction.

    NOT part of any scored trainer -- Track 3 permits one forward/backward per
    optimizer step; this runs on frozen checkpoints written by
    configs/bimaxwell_record_secant.yaml. Per checkpoint, in order:

    1. Model fit -- does the quadratic/secant model hold at all? The Gram-only
       block (epsilon_sym, cross/restricted spectra, leave-one-decay-out
       prediction and residual, drop-one-scale sensitivity, ||a*||), then the
       direct measurement: exact large-batch HVPs via double backprop, giving
       ||Y c - H S c|| / ||H S c|| per direction.
    2. Signal vs noise -- is GMRES fitting curvature or canceling noise?
       Held-out-shard residual, the big-batch re-solve, and a microbatch
       bootstrap of the pre- and post-polar directions.
    3. The estimand test: cosines of the online direction, the big-batch
       variant, Bi-Maxwell, single-EMA momentum, and the raw gradients against
       the restricted Newton reference -Q z*, z* = argmin ||g_hat + H Q z||.

    All gradient-like vectors stay in the trainer's scale (summed loss per
    524288-token batch); cosines and ratios are scale-free.

    The gradient/HVP data passes shard across every GPU under torchrun (rank 0
    additionally owns the 68 GB buffer records, the Gram work, the bootstrap,
    and the report); a plain python launch uses one GPU.

    Run (whole box):
        torchrun --standalone --nproc_per_node=8 \\
            records/track_3_optimization/offline_analysis/check_secant_direction_with_hvp.py \\
            --dump_dir secant_dumps --step 250 --holdout_data 'data/fineweb10B/fineweb_train_000020.bin'
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, default="secant_dumps")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/fineweb10B/fineweb_val_*.bin",
                    help="tokens for the large-batch gradient and HVPs (val shards by "
                         "default, so the sample is independent of the dumped minibatch)")
    ap.add_argument("--holdout_data", type=str, default=None,
                    help="a DISJOINT shard glob for the held-out residual; omit to skip")
    ap.add_argument("--holdout_tokens", type=int, default=1024 * 1024)
    ap.add_argument("--tokens", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--mbs", type=int, default=16, help="microbatch, in 1024-token sequences")
    ap.add_argument("--basis_rank", type=int, default=16,
                    help="top displacement directions of S that get true HVPs")
    ap.add_argument("--rank_displacements", type=int, default=None,
                    help="r1 override; defaults to the trainer's dumped value (else 32)")
    ap.add_argument("--rank_gmres", type=int, default=None,
                    help="r2 override; defaults to the trainer's dumped value (else 8)")
    ap.add_argument("--bootstrap_resamples", type=int, default=8, help="0 disables the bootstrap")
    ap.add_argument("--bootstrap_microbatches", type=int, default=64,
                    help="bootstrap population size; caches ~340 MB of host RAM each")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this diagnostic needs a GPU for the HVPs"
    rank, world_size = start_distributed_if_launched()
    device = torch.device("cuda")
    t_start = time.time()
    phase_seconds: dict[str, float] = {}
    t_phase = time.time()

    def mark_phase(name: str) -> None:
        nonlocal t_phase
        phase_seconds[name] = round(time.time() - t_phase, 1)
        t_phase = time.time()
    model_path = args.model_path or os.path.join(args.dump_dir, f"model_step{args.step:06d}.pt")

    records: list[Record] = []
    G = None
    basis: dict[str, Any] = {}
    if rank == 0:
        records, meta = read_secant_shards(args.dump_dir, args.step)
        mark_phase("load_buffer_shards")
        k = meta["k"]
        betas = meta["betas"].to(torch.float64)
        solve_recipe = meta.get("solve") or {}
        r1 = args.rank_displacements or solve_recipe.get(
            "truncated_rank_accumulated_parameter_displacements") or 32
        r2 = args.rank_gmres or solve_recipe.get("truncated_rank_gmres") or 8
        # comparator constants: from the shard when it is a bimaxwell run, else PR #340's
        bimaxwell = meta["bimaxwell"] or dict(bf=PR340_BIMAXWELL["fast_decay"],
                                              bs=PR340_BIMAXWELL["slow_decay"],
                                              w=PR340_BIMAXWELL["fast_weight"],
                                              start=PR340_BIMAXWELL["switch_step"])
        i_fast = int((betas - bimaxwell["bf"]).abs().argmin())
        i_slow = int((betas - bimaxwell["bs"]).abs().argmin())
        i_single = int((betas - NESTEROV_MU).abs().argmin())
        print(f"step {args.step}: k={k} r1={r1} r2={r2} params={len(records)} "
              f"world={world_size} columns fast={i_fast} slow={i_slow} "
              f"nearest-0.95={i_single} (beta={float(betas[i_single]):.4f})")
        model, muon_plist = load_model_at_checkpoint(model_path, records)
        mark_phase("load_model")

        # ---- replicate the online Gram, solve, and the Gram-only diagnostics ----
        G = build_secant_gram(records, [record["grad"] for record in records], device)
        w_mb, stats_mb = solve_secant_least_squares(G, k, r1, r2)
        w_check = float("nan")
        if meta["w"] is not None and w_mb is not None:
            w_check = float((w_mb - meta["w"].to(torch.float64)).abs().max())
        print(f"online solve replicated: rank_disp={stats_mb['rank_displacements']} "
              f"rank_gmres={stats_mb['rank_gmres']} res_ratio={stats_mb['res_ratio']:.4f} "
              f"coeff_norm={stats_mb['coefficient_norm']:.3e} max|w - w_dumped|={w_check:.3e}")
        gram_block = summarize_gram_diagnostics(G, k, r1, r2)
        print(f"model fit (Gram-only): eps_sym={gram_block['epsilon_sym']:.4f} "
              f"restricted spectrum [{gram_block['symmetrized_cross_spectrum_min']:.3e}, "
              f"{gram_block['symmetrized_cross_spectrum_max']:.3e}] "
              f"negative_fraction={gram_block['restricted_negative_fraction']:.3f}")
        lodo = [v for v in gram_block["left_out_prediction_error"] if math.isfinite(v)]
        drop = [v for v in gram_block["drop_one_scale_direction_cos"] if math.isfinite(v)]
        if lodo:
            print(f"leave-one-decay-out prediction error: median={quartile(lodo, 0.5):.4f} "
                  f"max={max(lodo):.4f}")
        if drop:
            print(f"drop-one direction cos: min={min(drop):.4f}")
        mark_phase("gram_solve_and_diagnostics")
        basis = choose_hvp_directions(G, records, k, args.basis_rank, i_fast, i_slow, device)
        mark_phase("build_hvp_basis")
        shared = [[record["name"] for record in records], basis["r"], k]
    else:
        shared = [None, None, None]
    if world_size > 1:
        dist.broadcast_object_list(shared, src=0)
    muon_names, r, k = shared
    if rank != 0:
        model, muon_plist = load_model_at_checkpoint(model_path, [], muon_names)
        basis = dict(dirs=[torch.empty(r + 2, p.numel(), dtype=torch.float32, device=device)
                           for p in muon_plist])
    if world_size > 1:
        for direction_stack in basis["dirs"]:
            dist.broadcast(direction_stack, src=0)

    # ---- data passes: every rank shares the work; accumulators come back global ----
    cache_microbatches = args.bootstrap_microbatches \
        if (args.bootstrap_resamples > 0 and rank == 0) else 0
    pass1 = collect_gradient_and_hvps(model, muon_plist, basis["dirs"], args.data,
                                      args.tokens, args.mbs, cache_microbatches,
                                      rank, world_size)
    mark_phase("gradient_and_hvp_pass")
    if rank == 0:
        print(f"large batch: {pass1['tokens_seen']} tokens in {pass1['n_microbatches']} "
              f"microbatches across {world_size} GPUs, {r + 2} HVP directions, "
              f"{len(pass1['grad_cache'])} microbatch grads cached")
    g_holdout, holdout_tokens = (None, 0)
    if args.holdout_data:
        g_holdout, holdout_tokens = collect_holdout_gradient(
            muon_plist, model, args.holdout_data, args.holdout_tokens, args.mbs,
            rank, world_size)
        mark_phase("holdout_gradient_pass")
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return

    # ---- rank 0 only: everything Gram-, record-, and report-shaped ----
    rel_res = measure_identity_residuals(records, pass1["hvp"], basis["C_dev"], r,
                                         i_fast, i_slow, basis["s_norm_fast"],
                                         basis["s_norm_slow"], device)
    reference = solve_hvp_reference(pass1["hvp"], pass1["g_hat"], r)
    G_lb = build_secant_gram(records, pass1["g_hat"], device)
    w_lb, stats_lb = solve_secant_least_squares(G_lb, k, r1, r2)
    holdout = None
    if g_holdout is not None:
        holdout = measure_holdout_residual(records, G, k,
                                           {"w_minibatch": w_mb, "w_largebatch": w_lb},
                                           g_holdout, holdout_tokens)
        print(f"held-out residual ({holdout['tokens']} tokens): "
              f"minibatch-w {holdout['w_minibatch']:.4f}  big-batch-w {holdout['w_largebatch']:.4f}")
    comparison = compare_directions(records, basis["dirs"], pass1["g_hat"],
                                    reference["z"], r, w_mb, w_lb, bimaxwell,
                                    i_fast, i_slow, i_single, device)
    mark_phase("residuals_reference_and_comparisons")
    bootstrap = None
    if args.bootstrap_resamples > 0 and w_mb is not None:
        bootstrap = bootstrap_direction_stability(records, pass1["grad_cache"], G, k,
                                                  r1, r2, args.mbs,
                                                  args.bootstrap_resamples, args.step)
        if bootstrap:
            print(f"bootstrap ({bootstrap['population']} microbatches, "
                  f"{bootstrap['resamples']} resamples): "
                  f"pre-polar cos {bootstrap['pre_polar_cos_mean_std'][0]:.4f}"
                  f"+/-{bootstrap['pre_polar_cos_mean_std'][1]:.4f}  "
                  f"post-polar cos {bootstrap['post_polar_cos_mean_std'][0]:.4f}"
                  f"+/-{bootstrap['post_polar_cos_mean_std'][1]:.4f}")

    mark_phase("bootstrap")
    # ---- report ----
    labels = comparison["labels"]
    cosines = comparison["cosines"]
    result = dict(
        step=args.step, k=k, rank_displacements=r1, rank_gmres=r2, basis_rank=r,
        tokens=pass1["tokens_seen"], w_replication_max_err=w_check,
        online_solve=stats_mb, online_solve_largebatch=stats_lb,
        reference_res_ratio=reference["res_ratio"],
        gram_diagnostics=gram_block,
        S_spectrum_top=[float(basis["evals_S"][i]) for i in basis["order"][:12]],
        secant_identity_rel_res_basis=[float(v) for v in rel_res[:r]],
        secant_identity_rel_res_exact_085=float(rel_res[r]),
        secant_identity_rel_res_exact_098=float(rel_res[r + 1]),
        holdout=holdout,
        bootstrap=bootstrap,
        cosine_labels=labels,
        cosine_matrix=[[float(c) for c in row] for row in cosines],
        per_block_cos_quartiles={f"{a}~{b}": [quartile(v, 0.25), quartile(v, 0.5), quartile(v, 0.75)]
                                 for (a, b), v in comparison["per_block"].items()},
        runtime_sec=time.time() - t_start,
        phase_seconds=phase_seconds,
    )
    print("\nsecant identity, relative residual ||Y c - H S c|| / ||H S c||:")
    print("  basis directions (descending S spectrum): "
          + " ".join(f"{float(v):.3f}" for v in rel_res[:r]))
    print(f"  exact 0.85 column: {float(rel_res[r]):.3f}   exact 0.98 column: {float(rel_res[r+1]):.3f}")
    print("\nresidual ratios: online(mb) {:.4f}   online(lb) {:.4f}   HVP reference {:.4f}".format(
        stats_mb["res_ratio"], stats_lb["res_ratio"], reference["res_ratio"]))
    print("\ncosines to the restricted Newton reference input (p_ref):")
    for j, label in enumerate(labels[1:], start=1):
        print(f"  cos(p_ref, {label:14s}) = {float(cosines[0, j]):+.4f}")
    print("\nper-block cosine quartiles (25/50/75):")
    for key, v in result["per_block_cos_quartiles"].items():
        print(f"  {key}: {v[0]:+.3f} / {v[1]:+.3f} / {v[2]:+.3f}")
    out_path = args.out or os.path.join(args.dump_dir, f"analysis_step{args.step:06d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("phase seconds: " + " ".join(f"{name}:{sec}" for name, sec in phase_seconds.items()))
    print(f"\nwrote {out_path}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
