"""REQ-068 amended analysis: exact temporal geometry from 64x64 Gram matrices (CPU-testable).

Whole-model vector v_i (one per window step, millions of coords) -> temporal Gram K_ij = v_i . v_j and
cosine C_ij = K_ij/sqrt(K_ii K_jj), both 64x64. K is accumulated in parameter BLOCKS (streaming; never
loads all vectors at once). Everything downstream is 64x64 linear algebra on K:
 - dominant recurring directions via eig(K): temporal modes; signed coefficient time-course
   coeff[i,k] = <v_i, u_k> = sqrt(lambda_k) * eigvec[i,k] (u_k = sum_j eigvec[j,k] v_j / sqrt(lambda_k)).
 - centered version: subtract the window-mean vector (K_centered = J K J, J = I - 11^T/n).
 - out-of-sample: fit directions on first 32 (K[:32,:32]); project next 32 via cross-Gram K[32:,:32];
   held-out captured vs residual energy at ranks 1,2,4,8; baselines past-mean / pair-average / exp-average.
 - per-parameter-family contribution to K (block-diagonal sums).
Uncentered Euclidean whole-model K is primary; a layer-reweighted K is a separate labeled variant.
"""
from __future__ import annotations
import numpy as np


def temporal_gram(step_vectors, names=None, family_of=None):
    """step_vectors: list (len n) of dict {name: 1-D np.array (float64)}. Returns (K, per_family_K).
    Accumulates K_ij += sum over params of v_i[p].v_j[p], block by block (one param family at a time)."""
    n = len(step_vectors)
    if names is None:
        names = list(step_vectors[0].keys())
    K = np.zeros((n, n), dtype=np.float64)
    fam_K = {}
    for name in names:
        M = np.stack([step_vectors[i][name].astype(np.float64).ravel() for i in range(n)], axis=0)  # n x d_p
        blk = M @ M.T  # n x n contribution of this parameter block
        K += blk
        if family_of is not None:
            f = family_of(name)
            fam_K[f] = fam_K.get(f, np.zeros((n, n))) + blk
    return K, fam_K


def cosine_from_gram(K):
    d = np.sqrt(np.clip(np.diag(K), 0, None))
    nz = d > 0
    C = np.zeros_like(K)
    outer = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(outer > 0, K / outer, 0.0)
    return C, (~nz)  # C, mask of zero-norm steps


def gram_eig(K):
    """Eigendecomposition of symmetric PSD K (descending). Returns (lambda[n], eigvec n x n)."""
    w, V = np.linalg.eigh((K + K.T) / 2)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def signed_coefficients(lam, V):
    """coeff[i,k] = <v_i, u_k> = sqrt(max(lam_k,0)) * V[i,k]. Time-course of each temporal mode."""
    return V * np.sqrt(np.clip(lam, 0, None))[None, :]


def center_gram(K):
    """Centered Gram = J K J, J = I - 11^T/n (subtracts the window-mean vector in feature space)."""
    n = K.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return J @ K @ J


def lag_agreement(C, lags=(1, 2, 4, 8, 16, 32)):
    """Mean signed cosine at each lag from the cosine matrix (off-diagonals at that offset)."""
    n = C.shape[0]
    return {int(L): float(np.mean([C[i, i + L] for i in range(n - L)])) for L in lags if L < n}


def out_of_sample(K, ranks=(1, 2, 4, 8), fit=32):
    """Fit directions from the first `fit` steps (K[:fit,:fit]); evaluate how much of the remaining
    steps' energy the rank-r subspace captures (held-out), via the cross-Gram K[fit:, :fit].
    Returns per-rank captured/residual energy fractions on the held-out half, plus baselines."""
    n = K.shape[0]
    Kff = K[:fit, :fit]; Kef = K[fit:, :fit]; Kee_diag = np.diag(K[fit:, fit:])
    lam, V = gram_eig(Kff)  # fit-half temporal modes
    eval_energy = float(np.sum(Kee_diag))  # total held-out energy = sum ||v_i||^2
    out = {"fit": fit, "n_eval": n - fit, "eval_total_energy": eval_energy, "by_rank": {}}
    pos = lam > 1e-12
    for r in ranks:
        r = min(r, int(pos.sum()))
        if r == 0:
            out["by_rank"][r] = {"captured_frac": 0.0}; continue
        # u_k = sum_j V[j,k] v_j / sqrt(lam_k); <v_eval_i, u_k> = (Kef @ V[:,k]) / sqrt(lam_k)
        proj = (Kef @ V[:, :r]) / np.sqrt(lam[:r])[None, :]   # (n_eval x r)
        captured = float(np.sum(proj ** 2))
        out["by_rank"][r] = {"captured_frac": captured / eval_energy if eval_energy > 0 else 0.0,
                             "residual_frac": 1 - captured / eval_energy if eval_energy > 0 else 1.0}
    # baselines (held-out energy fraction captured by a single predicted direction)
    # past-mean: mean of first `fit` vectors; captured = sum_i <v_i, mhat>^2/||mhat||^2
    ones = np.ones(fit)
    mnorm2 = float(ones @ Kff @ ones) / fit**2            # ||mean||^2
    cross_mean = (Kef @ ones) / fit                        # <v_eval_i, mean>
    pm = float(np.sum(cross_mean ** 2) / mnorm2) / eval_energy if mnorm2 > 0 and eval_energy > 0 else 0.0
    out["baseline_past_mean_frac"] = pm
    # exp-average of first fit (decay 0.9), as a direction
    w = 0.9 ** (fit - 1 - np.arange(fit)); w /= w.sum()
    enorm2 = float(w @ Kff @ w); cross_e = Kef @ w
    out["baseline_exp_avg_frac"] = (float(np.sum(cross_e ** 2) / enorm2) / eval_energy
                                    if enorm2 > 0 and eval_energy > 0 else 0.0)
    return out
