from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def split_secant_gram(G: Tensor, num_buffers: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    """Splits the Gram of Z = [Y rows, S rows, g] into its named blocks.

    Args:
        G: (2k+1, 2k+1) Gram tensor.
        num_buffers: k.

    Returns:
        (YtY, YtS, StS, Ytg, Stg, gg): tensor views ((k,k), (k,k), (k,k), (k,), (k,))
        and the float g'g.
    """
    k = num_buffers
    return (G[:k, :k], G[:k, k:2 * k], G[k:2 * k, k:2 * k],
            G[:k, 2 * k], G[k:2 * k, 2 * k], float(G[2 * k, 2 * k]))


def _keep_top_eigenpairs(evals: Tensor, max_rank: int, eigenvalue_floor: float) -> list[int]:
    """Indices of the largest eigenvalues, at most max_rank, all >= floor * max."""
    lam_max = float(evals[-1])
    if not math.isfinite(lam_max) or lam_max <= 0:
        return []
    order = torch.argsort(evals, descending=True)
    return [int(i) for i in order if float(evals[i]) >= eigenvalue_floor * lam_max][:max_rank]


def solve_secant_least_squares(G: Tensor, num_buffers: int,
                               truncated_rank_accumulated_parameter_displacements: int,
                               truncated_rank_gmres: int,
                               eigenvalue_floor: float = 1e-12) -> tuple[Tensor | None, dict[str, Any]]:
    """Two-truncation restricted-Newton solve, entirely from the secant Gram.

    With paired-EMA displacements dX = S (columns s_i = q_i - x_t) and dG = Y
    (columns y_i = m_i - g_t), the affine model gives Y = H S. Restricting the
    Newton displacement to span(S):

      (i)  Truncated SVD of dX from its Gram: eigh(S'S) -> V, Sigma^2; keep the
           top r1 directions. This defines the orthonormal displacement basis
           D = S V Sigma^-1 without materializing it, and removes redundant
           history directions before Sigma^-1 can amplify them.
      (ii) The gradient responses G_D = Y V Sigma^-1 = H D are already known;
           solve a* = argmin ||g + G_D a|| by truncated pseudo-inverse of the
           whitened normal matrix N = G_D' G_D, keeping the top r2 eigenpairs --
           refusing to invert tiny or statistically unresolved responses is the
           substantive GMRES regularization.
      (iii) delta* = D a* = S w with w = V Sigma^-1 a*; Muon's pre-polar input
           is -delta* (formed by streams.secant_direction).

    Args:
        G: (2k+1, 2k+1) Gram of [Y rows, S rows, g]; any device, promoted to fp64.
        num_buffers: k.
        truncated_rank_accumulated_parameter_displacements: r1, SVD rank kept of S.
        truncated_rank_gmres: r2, eigenpairs kept when inverting G_D' G_D.
        eigenvalue_floor: Relative numerical floor inside both truncations.

    Returns:
        (w, stats): w is the (k,) float64 stream-mixing vector with delta = S w, or
        None when the solve is degenerate (stats["fallback_reason"] says why).
        stats keys: g_norm, rank_displacements, rank_gmres, displacement_kept_cond,
        whitened_normal_cond, res_ratio (||g + Y w|| / ||g||), delta_dot_g,
        delta_norm, coefficient_norm (||a*||), fallback_reason.
    """
    G = G.to(torch.float64).cpu()
    k = num_buffers
    YtY, _, StS, Ytg, Stg, gg = split_secant_gram(G, k)
    stats: dict[str, Any] = {
        "g_norm": max(gg, 0.0) ** 0.5, "rank_displacements": 0, "rank_gmres": 0,
        "displacement_kept_cond": float("inf"), "whitened_normal_cond": float("inf"),
        "res_ratio": 1.0, "delta_dot_g": 0.0, "delta_norm": 0.0,
        "coefficient_norm": 0.0, "fallback_reason": None}

    def degenerate(reason: str) -> tuple[None, dict[str, Any]]:
        stats["fallback_reason"] = reason
        return None, stats

    if gg <= 0:
        return degenerate("nonpositive_gradient_norm")
    try:
        displacement_evals, V = torch.linalg.eigh(StS)
    except Exception:
        return degenerate("eigh_failed")
    kept1 = _keep_top_eigenpairs(displacement_evals,
                                 truncated_rank_accumulated_parameter_displacements,
                                 eigenvalue_floor)
    if not kept1:
        return degenerate("empty_displacement_spectrum")
    V1 = V[:, kept1]                                   # (k, r1)
    inverse_sigma = displacement_evals[kept1].rsqrt()  # Sigma^-1 diagonal, (r1,)
    stats["rank_displacements"] = len(kept1)
    stats["displacement_kept_cond"] = float(displacement_evals[kept1[0]]
                                            / displacement_evals[kept1[-1]])

    whitened_normal = (V1.mT @ YtY @ V1) * (inverse_sigma[:, None] * inverse_sigma[None, :])
    whitened_rhs = (V1.mT @ Ytg) * inverse_sigma       # G_D' g
    try:
        response_evals, W = torch.linalg.eigh(whitened_normal)
    except Exception:
        return degenerate("eigh_failed")
    kept2 = _keep_top_eigenpairs(response_evals, truncated_rank_gmres, eigenvalue_floor)
    if not kept2:
        return degenerate("empty_gmres_spectrum")
    W2 = W[:, kept2]
    coefficients = -(W2 @ ((W2.mT @ whitened_rhs) / response_evals[kept2]))   # a*
    w = V1 @ (coefficients * inverse_sigma)
    if not bool(torch.isfinite(w).all()):
        return degenerate("nonfinite")
    stats["rank_gmres"] = len(kept2)
    stats["whitened_normal_cond"] = float(response_evals[kept2[0]] / response_evals[kept2[-1]])
    residual_sq = max(gg + 2 * float(w @ Ytg) + float(w @ (YtY @ w)), 0.0)
    stats["res_ratio"] = (residual_sq / gg) ** 0.5
    stats["delta_dot_g"] = float(w @ Stg)
    stats["delta_norm"] = max(float(w @ (StS @ w)), 0.0) ** 0.5
    stats["coefficient_norm"] = float(coefficients.norm())
    return w, stats
