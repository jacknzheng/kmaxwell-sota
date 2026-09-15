"""REQ-064 pretreatment feature primitives (adds to the REQ-057 S_i/G_i spectral probe):
  - Euclidean lambda_i via Lanczos (8 iters + convergence diagnostics) on the per-matrix Hessian block
    H_ii, plus lambda_i/||g_i||_F^2.
  - M3 actual-direction curvature: Rayleigh quotient of H_ii along the actual post-polar shape-scaled
    update direction (and along candidate selective a-changed directions).
Reuses REQ-057 HVP/FW primitives; symmetric-operator Lanczos with full reorthogonalization (per-matrix
dims are small enough). CPU-validated against known quadratics in test_req064_features.py.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


def _ip(a: Tensor, b: Tensor) -> float:
    return float((a * b).sum())


def _tridiag_top_eig(alphas: list[float], betas: list[float]) -> tuple[float, float]:
    """Top eigenvalue of the Lanczos tridiagonal T(alpha,beta) and the |last component| of its
    top eigenvector (for the Ritz residual estimate)."""
    k = len(alphas)
    T = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        T[i, i] = alphas[i]
        if i + 1 < k:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    evals, evecs = torch.linalg.eigh(T)
    top = int(torch.argmax(evals))
    return float(evals[top]), float(abs(evecs[-1, top]))


def lanczos_top_eig(matvec, shape, iters: int = 8, seed: int = 1337, dtype=torch.float64):
    """Top eigenvalue of a symmetric operator `matvec` (tensor of `shape` -> same shape), by Lanczos with
    full reorthogonalization. Returns (lambda_top, ritz_history, ritz_residual, n_iters_run)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(*shape, generator=g, dtype=dtype)
    q = q / (q.norm() + 1e-30)
    Q: list[Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    q_prev = torch.zeros_like(q)
    beta = 0.0
    ritz_hist: list[float] = []
    resid = float("inf")
    for k in range(iters):
        Q.append(q)
        w = matvec(q).to(dtype)
        alpha = _ip(w, q)
        alphas.append(alpha)
        w = w - alpha * q - beta * q_prev
        for qi in Q:  # full reorthogonalization
            w = w - _ip(w, qi) * qi
        beta = float(w.norm())
        top, last_comp = _tridiag_top_eig(alphas, betas)
        ritz_hist.append(top)
        resid = beta * last_comp  # Ritz residual ||H y - theta y|| estimate
        if beta < 1e-12:
            return top, ritz_hist, resid, k + 1
        q_prev = q
        q = w / beta
        betas.append(beta)
    return ritz_hist[-1], ritz_hist, resid, iters


def euclidean_lambda(diag_hvp_one_fn, i: int, grad_i: Tensor, iters: int = 8, seed: int = 1337) -> dict:
    """Euclidean top curvature of matrix i: lambda_i (Lanczos on H_ii) and lambda_i/||g_i||_F^2.
    diag_hvp_one_fn(i, v) returns H_ii @ v as a tensor of matrix i's shape."""
    matvec = lambda v: diag_hvp_one_fn(i, v)
    lam, hist, resid, nit = lanczos_top_eig(matvec, tuple(grad_i.shape), iters=iters, seed=seed)
    gF2 = float((grad_i.double() ** 2).sum())
    return {"lambda_i": lam, "lambda_over_gF2": (lam / gF2 if gF2 > 0 else float("nan")),
            "ritz_history": hist, "ritz_residual": resid, "iters": nit, "gF2": gF2,
            "converged": resid < 1e-3 * max(1.0, abs(lam))}


def rayleigh_along(diag_hvp_one_fn, i: int, direction: Tensor) -> float:
    """Rayleigh quotient <d, H_ii d> / <d, d> — curvature along a fixed direction (e.g. the actual update)."""
    d = direction.double()
    dn2 = _ip(d, d)
    if dn2 <= 0:
        return float("nan")
    Hd = diag_hvp_one_fn(i, d).double()
    return _ip(d, Hd) / dn2


def actual_direction_curvature(diag_hvp_one_fn, i: int, update_dir: Tensor,
                               candidate_dirs: dict[str, Tensor] | None = None) -> dict:
    """M3: curvature along the actual post-polar shape-scaled update, and along candidate selective
    directions (a_star/2, 2*a_star updates) computed with cloned buffers at identical W,g,data."""
    out = {"curv_actual": rayleigh_along(diag_hvp_one_fn, i, update_dir),
           "update_norm": float(update_dir.double().norm())}
    if candidate_dirs:
        for name, d in candidate_dirs.items():
            out[f"curv_{name}"] = rayleigh_along(diag_hvp_one_fn, i, d)
    return out
