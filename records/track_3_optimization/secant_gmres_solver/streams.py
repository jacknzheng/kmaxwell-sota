from __future__ import annotations

import math

import torch
from torch import Tensor

GRAM_CHUNK = 1 << 19   # flat-dim chunk bounding every transient buffer in this package


def build_decay_ladder(num_buffers: int, min_decay: float = 0.8, max_decay: float = 0.995,
                       pinned_rates: tuple[float, ...] | None = (0.85, 0.98)) -> Tensor:
    """Builds the EMA decay ladder.

    Mean ages tau = beta/(1-beta) are log-spaced between the ages of min_decay and
    max_decay; the entry nearest each pinned rate is replaced by exactly that rate.
    Defaults give tau in [4, 199] with the Bi-Maxwell rates 0.85 and 0.98 exact,
    keeping stream-level comparisons against PR #340 direct.

    Args:
        num_buffers: k, the number of paired EMA streams (>= 2).
        min_decay: Smallest rate in the ladder, in (0, 1).
        max_decay: Largest rate in the ladder, in (min_decay, 1).
        pinned_rates: Rates to place exactly; rates outside [min_decay, max_decay]
            are skipped. None disables pinning.

    Returns:
        (k,) float64 tensor of decay rates.
    """
    assert num_buffers >= 2
    assert 0.0 < min_decay < max_decay < 1.0
    tau_min = min_decay / (1.0 - min_decay)
    tau_max = max_decay / (1.0 - max_decay)
    taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), num_buffers, dtype=torch.float64)
    betas = taus / (1 + taus)
    used: list[int] = []
    for rate in (pinned_rates or ()):
        if not (min_decay <= rate <= max_decay):
            continue
        distance = (betas - rate).abs()
        distance[used] = float("inf")
        j = int(distance.argmin())
        betas[j] = rate
        used.append(j)
    return betas


def build_one_minus_betas(ladder: Tensor, device: torch.device) -> Tensor:
    """The (k, 1) fp32 column of 1 - beta_i the advance kernel broadcasts over --
    the single definition shared by the optimizer and the recorder hook."""
    return (1.0 - ladder).to(device=device, dtype=torch.float32).view(-1, 1)


def advance_paired_averages(m_stack: Tensor, q_stack: Tensor, one_minus_betas_col: Tensor,
                            g_flat: Tensor, x_flat: Tensor) -> None:
    """Advances m_i <- beta_i m_i + (1-beta_i) g and q_i <- beta_i q_i + (1-beta_i) x, in place.

    Identical normalized weights on both stacks are what make m_i = g(q_i) exact
    under an affine gradient map g(x) = Hx + b (affine maps commute with weighted
    averages), so each pair supplies a secant y_i = m_i - g = H (q_i - x) = H s_i.
    First-touch initialization is the caller's job (or use advance_or_first_touch).

    Args:
        m_stack: (k, n) gradient EMAs, mutated.
        q_stack: (k, n) parameter EMAs, mutated.
        one_minus_betas_col: (k, 1) tensor of 1 - beta_i, same device/dtype family.
        g_flat: (n,) gradient at x_flat.
        x_flat: (n,) parameters the gradient was evaluated at.
    """
    m_stack.add_(one_minus_betas_col * (g_flat - m_stack))
    q_stack.add_(one_minus_betas_col * (x_flat - q_stack))


def advance_or_first_touch(shard_state: dict[str, Tensor], g_flat: Tensor, x_flat: Tensor,
                           one_minus_betas_col: Tensor, num_buffers: int) -> None:
    """Advances one parameter's paired stacks, creating them on first touch.

    First touch sets m_i = g and q_i = x for every stream, which gives both EMAs
    exactly normalized identical weights from step 1 onward (no debiasing). This
    is the single shared implementation used by both the SecantGmresMuon optimizer
    (state in optimizer state) and the record_paired_averages hook (state in the
    loop's state dict) -- do not fork it.

    Args:
        shard_state: Dict gaining/holding "grad_avgs" and "param_avgs", both (k, n).
        g_flat: (n,) gradient at x_flat.
        x_flat: (n,) parameters the gradient was evaluated at.
        one_minus_betas_col: (k, 1) tensor of 1 - beta_i on the same device.
        num_buffers: k.
    """
    if "grad_avgs" not in shard_state:
        shard_state["grad_avgs"] = g_flat.repeat(num_buffers, 1)
        shard_state["param_avgs"] = x_flat.repeat(num_buffers, 1)
    else:
        advance_paired_averages(shard_state["grad_avgs"], shard_state["param_avgs"],
                                one_minus_betas_col, g_flat, x_flat)


def accumulate_secant_gram(G: Tensor, m_stack: Tensor, q_stack: Tensor,
                           g_flat: Tensor, x_flat: Tensor, chunk: int = GRAM_CHUNK) -> None:
    """Accumulates G += Z^T Z in float64, Z rows = [y_1..y_k, s_1..s_k, g].

    y_i = m_i - g and s_i = q_i - x. Everything downstream (the solve, the
    diagnostics) consumes only this (2k+1, 2k+1) Gram, so no parameter-sized
    column is ever gathered. Chunked over the flat dimension so the transient Z
    buffer stays bounded regardless of k.

    Args:
        G: (2k+1, 2k+1) float64 accumulator, mutated.
        m_stack: (k, n) gradient EMAs.
        q_stack: (k, n) parameter EMAs.
        g_flat: (n,) current gradient.
        x_flat: (n,) current parameters.
        chunk: Flat-dimension chunk size.
    """
    k, n = m_stack.shape
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Z = torch.empty(2 * k + 1, b - a, dtype=m_stack.dtype, device=m_stack.device)
        torch.sub(m_stack[:, a:b], g_flat[a:b], out=Z[:k])
        torch.sub(q_stack[:, a:b], x_flat[a:b], out=Z[k:2 * k])
        Z[2 * k] = g_flat[a:b]
        G += (Z @ Z.mT).to(torch.float64)


def secant_direction(q_stack: Tensor, x_flat: Tensor, w: Tensor, chunk: int = GRAM_CHUNK) -> Tensor:
    """Forms the pre-polar Muon input p = -S w = -sum_i w_i (q_i - x).

    Args:
        q_stack: (k, n) parameter EMAs.
        x_flat: (n,) current parameters.
        w: (k,) stream-mixing weights from the solve.
        chunk: Flat-dimension chunk size.

    Returns:
        (n,) tensor with q_stack's dtype/device.
    """
    out = torch.empty_like(x_flat)
    wv = w.to(dtype=q_stack.dtype, device=q_stack.device)
    w_sum = wv.sum()
    n = x_flat.numel()
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        out[a:b] = w_sum * x_flat[a:b] - wv @ q_stack[:, a:b]
    return out
