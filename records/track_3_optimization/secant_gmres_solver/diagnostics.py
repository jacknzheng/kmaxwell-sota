from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .solve import _keep_top_eigenpairs, solve_secant_least_squares, split_secant_gram


def measure_symmetry_defect(G: Tensor, num_buffers: int) -> float:
    """Computes epsilon_sym = ||S'Y - Y'S||_F / ||S'Y + Y'S||_F.

    Zero for any exact symmetric Hessian with noiseless streams; near one when the
    cross-Gram is essentially antisymmetric, i.e. no common symmetric Hessian
    explains the buffer pairs even inside their own displacement subspace.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.

    Returns:
        The defect in [0, ~1], or NaN when the cross-Gram vanishes.
    """
    k = num_buffers
    _, YtS, _, _, _, _ = split_secant_gram(G.to(torch.float64), k)
    StY = YtS.mT
    numerator = (StY - StY.mT).norm()
    denominator = (StY + StY.mT).norm()
    return float(numerator / denominator) if float(denominator) > 0 else float("nan")


def estimate_symmetrized_cross_spectrum(G: Tensor, num_buffers: int) -> Tensor:
    """Eigenvalues (ascending) of 0.5 (S'Y + Y'S).

    Under Y = HS this is S'HS symmetrized: negative eigenvalues flag negative
    curvature (or model violation) in the raw, unwhitened stream coordinates.
    """
    k = num_buffers
    _, YtS, _, _, _, _ = split_secant_gram(G.to(torch.float64), k)
    StY = YtS.mT
    return torch.linalg.eigvalsh(0.5 * (StY + StY.mT))


def estimate_restricted_hessian_spectrum(G: Tensor, num_buffers: int, truncated_rank: int,
                                         eigenvalue_floor: float = 1e-12) -> Tensor:
    """Eigenvalues (ascending) of the whitened operator Sigma^-1 V' sym(S'Y) V Sigma^-1.

    On the orthonormal displacement basis D = S V Sigma^-1 this equals sym(D'HD)
    under the affine model: the restricted Hessian spectrum the solve actually
    inverts. Empty when the displacement spectrum is degenerate.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.
        truncated_rank: How many top displacement directions to whiten over.
        eigenvalue_floor: Relative numerical floor on the displacement spectrum.

    Returns:
        (r,) float64 eigenvalues, ascending; possibly empty.
    """
    k = num_buffers
    G = G.to(torch.float64).cpu()
    _, _, StS, _, _, _ = split_secant_gram(G, k)
    evals, V = torch.linalg.eigh(StS)
    kept = _keep_top_eigenpairs(evals, truncated_rank, eigenvalue_floor)
    if not kept:
        return torch.empty(0, dtype=torch.float64)
    V1 = V[:, kept]
    inverse_sigma = evals[kept].rsqrt()
    _, YtS, _, _, _, _ = split_secant_gram(G, k)
    symmetric_cross = 0.5 * (YtS.mT + YtS)
    whitened = (V1.mT @ symmetric_cross @ V1) * (inverse_sigma[:, None] * inverse_sigma[None, :])
    return torch.linalg.eigvalsh(whitened)


def delete_stream_from_gram(G: Tensor, num_buffers: int, stream_index: int) -> Tensor:
    """The Gram with decay stream i removed: rows/cols i and k+i deleted."""
    k = num_buffers
    keep = [j for j in range(k) if j != stream_index] \
         + [k + j for j in range(k) if j != stream_index] + [2 * k]
    idx = torch.tensor(keep)
    return G.index_select(0, idx).index_select(1, idx)


def predict_left_out_stream(G: Tensor, num_buffers: int, eigenvalue_floor: float = 1e-10) -> Tensor:
    """Leave-one-decay-out secant prediction error per stream.

    For each stream i: express s_i in the span of the other displacements
    (c = argmin ||s_i - S_-i c||, min-norm via a floored pseudo-inverse) and
    predict its gradient displacement as Y_-i c. Far more informative than the
    in-sample residual: a fitted operator that generalizes across timescales
    predicts held-out secants.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.
        eigenvalue_floor: Relative floor of the fit pseudo-inverse.

    Returns:
        (k,) float64 relative errors ||Y_-i c - y_i|| / ||y_i||; NaN where degenerate.
    """
    k = num_buffers
    G = G.to(torch.float64).cpu()
    YtY, _, StS, _, _, _ = split_secant_gram(G, k)
    errors = torch.full((k,), float("nan"), dtype=torch.float64)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        idx = torch.tensor(others)
        StS_oo = StS.index_select(0, idx).index_select(1, idx)
        StS_oi = StS.index_select(0, idx)[:, i]
        evals, V = torch.linalg.eigh(StS_oo)
        kept = _keep_top_eigenpairs(evals, k - 1, eigenvalue_floor)
        if not kept:
            continue
        Vk = V[:, kept]
        c = Vk @ ((Vk.mT @ StS_oi) / evals[kept])
        YtY_oo = YtY.index_select(0, idx).index_select(1, idx)
        YtY_oi = YtY.index_select(0, idx)[:, i]
        yy = float(YtY[i, i])
        if yy <= 0:
            continue
        error_sq = max(yy - 2 * float(c @ YtY_oi) + float(c @ (YtY_oo @ c)), 0.0)
        errors[i] = (error_sq / yy) ** 0.5
    return errors


def measure_left_out_residual(G: Tensor, num_buffers: int,
                              truncated_rank_accumulated_parameter_displacements: int,
                              truncated_rank_gmres: int,
                              eigenvalue_floor: float = 1e-12) -> Tensor:
    """Residual ratio of the full solve re-run with each stream deleted.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.
        truncated_rank_accumulated_parameter_displacements: r1 for the solve.
        truncated_rank_gmres: r2 for the solve.
        eigenvalue_floor: Relative numerical floor.

    Returns:
        (k,) float64 residual ratios; NaN where the reduced solve is degenerate.
    """
    k = num_buffers
    ratios = torch.full((k,), float("nan"), dtype=torch.float64)
    for i in range(k):
        _, stats = solve_secant_least_squares(
            delete_stream_from_gram(G, k, i), k - 1,
            truncated_rank_accumulated_parameter_displacements,
            truncated_rank_gmres, eigenvalue_floor)
        if stats["fallback_reason"] is None:
            ratios[i] = stats["res_ratio"]
    return ratios


def compare_directions_dropping_one_scale(G: Tensor, num_buffers: int,
                                          truncated_rank_accumulated_parameter_displacements: int,
                                          truncated_rank_gmres: int,
                                          eigenvalue_floor: float = 1e-12) -> Tensor:
    """cos(delta(w_full), delta(w_without_stream_i)) per stream, in Gram space.

    Values far below 1 mean the final direction hinges on one timescale -- a small
    residual with extreme drop-one sensitivity indicates noise cancellation, not
    signal.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.
        truncated_rank_accumulated_parameter_displacements: r1 for the solve.
        truncated_rank_gmres: r2 for the solve.
        eigenvalue_floor: Relative numerical floor.

    Returns:
        (k,) float64 cosines; NaN where either solve is degenerate.
    """
    k = num_buffers
    G = G.to(torch.float64).cpu()
    _, _, StS, _, _, _ = split_secant_gram(G, k)
    w_full, _ = solve_secant_least_squares(
        G, k, truncated_rank_accumulated_parameter_displacements,
        truncated_rank_gmres, eigenvalue_floor)
    cosines = torch.full((k,), float("nan"), dtype=torch.float64)
    if w_full is None:
        return cosines
    norm_full = max(float(w_full @ (StS @ w_full)), 0.0) ** 0.5
    for i in range(k):
        w_wo, _ = solve_secant_least_squares(
            delete_stream_from_gram(G, k, i), k - 1,
            truncated_rank_accumulated_parameter_displacements,
            truncated_rank_gmres, eigenvalue_floor)
        if w_wo is None:
            continue
        embedded = torch.zeros(k, dtype=torch.float64)
        embedded[[j for j in range(k) if j != i]] = w_wo
        norm_wo = max(float(embedded @ (StS @ embedded)), 0.0) ** 0.5
        if norm_full > 0 and norm_wo > 0:
            cosines[i] = float(w_full @ (StS @ embedded)) / (norm_full * norm_wo)
    return cosines


def summarize_gram_diagnostics(G: Tensor, num_buffers: int,
                               truncated_rank_accumulated_parameter_displacements: int,
                               truncated_rank_gmres: int,
                               eigenvalue_floor: float = 1e-12,
                               include_left_out: bool = True) -> dict[str, Any]:
    """Runs the full Gram-only diagnostic block into one JSON-friendly dict.

    Why these diagnostics: before judging the secant-GMRES update, test the
    assumption Y ~= HS (symmetry defect, cross spectrum, restricted spectrum) and
    test whether the solve is fitting curvature rather than canceling noise
    (leave-one-decay-out prediction/residual, drop-one-scale sensitivity, and the
    coefficient norm inside the solve stats). All of it replays offline over
    saved probe Grams -- no data or GPU needed.

    Args:
        G: (2k+1, 2k+1) secant Gram.
        num_buffers: k.
        truncated_rank_accumulated_parameter_displacements: r1 for the solve/spectra.
        truncated_rank_gmres: r2 for the solve.
        eigenvalue_floor: Relative numerical floor.
        include_left_out: Whether to run the k-solve leave-one-out suite.

    Returns:
        Dict of floats/lists (plus the solve stats dict under "solve").
    """
    k = num_buffers
    G = G.to(torch.float64).cpu()
    _, stats = solve_secant_least_squares(
        G, k, truncated_rank_accumulated_parameter_displacements,
        truncated_rank_gmres, eigenvalue_floor)
    cross = estimate_symmetrized_cross_spectrum(G, k)
    restricted = estimate_restricted_hessian_spectrum(
        G, k, truncated_rank_accumulated_parameter_displacements, eigenvalue_floor)
    summary: dict[str, Any] = dict(
        solve=stats,
        epsilon_sym=measure_symmetry_defect(G, k),
        symmetrized_cross_spectrum_min=float(cross[0]) if len(cross) else float("nan"),
        symmetrized_cross_spectrum_max=float(cross[-1]) if len(cross) else float("nan"),
        restricted_hessian_spectrum=[float(v) for v in restricted],
        restricted_negative_fraction=(float((restricted < 0).sum()) / len(restricted)
                                      if len(restricted) else float("nan")),
    )
    if include_left_out:
        summary.update(
            left_out_prediction_error=[float(v) for v in predict_left_out_stream(G, k)],
            left_out_residual_ratio=[float(v) for v in measure_left_out_residual(
                G, k, truncated_rank_accumulated_parameter_displacements,
                truncated_rank_gmres, eigenvalue_floor)],
            drop_one_scale_direction_cos=[float(v) for v in compare_directions_dropping_one_scale(
                G, k, truncated_rank_accumulated_parameter_displacements,
                truncated_rank_gmres, eigenvalue_floor)],
        )
    return summary
