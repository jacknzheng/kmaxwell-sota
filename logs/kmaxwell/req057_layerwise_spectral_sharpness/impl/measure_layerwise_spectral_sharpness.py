"""REQ-057: layer-wise, shape-weighted spectral sharpness of the loss Hessian, with cross-layer coupling.

Extends REQ-019's joint block-spectral Frank-Wolfe (FW) probe
(logs/kmaxwell/req019_fw_calibration/impl/measure_generalized_sharpness_fw.py) with the measurements
REQ-019 does NOT supply: shape-weighted radii, ISOLATED per-matrix sharpness S_i, radial normalization to the
sphere, the G denominators / Z diagnostic, and the cross-layer interaction decomposition.

Definitions (Islamov et al. v3, Def. 2.2 / Eq. 19; https://arxiv.org/html/2603.05002v3), on the Muon
subspace (the ndim==2 `^blocks\\..*\\.weight$` matrices; non-Muon params held fixed). For matrix i with
implemented shape factor and relative-LR multiplier folded into r_i:

    r_i        = sqrt(max(1, rows_i/cols_i)) * lr_mult_i          (actual Muon step radius per matrix)
    ||D||_op   = largest singular value ; ||g||_nuclear = sum of singular values
    N_r(D)     = max_i ||D_i||_op / r_i                            (the optimizer norm)
    S_joint    = max <D, H[D]>  s.t. N_r(D) = 1                    (joint, keeps cross terms)
    S_i        = max <D_i, H_ii[D_i]>  s.t. ||D_i||_op = r_i       (isolated, single matrix, freeze others)
    G_i        = r_i * ||g_i||_nuclear ; G_joint = sum_i G_i
    Z_i        = eta * S_i / (2 G_i) ; Z_joint = eta * S_joint / (2 G_joint)

S_i is a restricted isolated maximum, NOT a decomposition of S_joint (summing S_i does not give S_joint).
FW returns an interior iterate; the sphere objective is recomputed on the radially-normalized point D/N_r(D)
with an independently recomputed HVP. For a negative-definite restricted problem the sphere maximum is the
best feasible boundary witness (an interior zero is NOT the sphere max) -- we report the best feasible
restart, not the mean, and flag values near zero as unresolved rather than hiding an epsilon.

Loss reduction: the harness `model_gpt.py` returns a SUMMED cross-entropy. All S/G here are computed in a
MEAN-per-valid-token scale: the summed gradient and summed HVP are both divided by the actual valid-token
count (published as `token_scale`), never by the nominal BATCH_TOKENS=524288 label. S and exact-polar
directions are invariant to a consistent loss rescale (checked in the loss-scale test); raw S and G scale
linearly, lambda/||g||_F^2 scales inversely.

Pure primitives below are CPU-unit-tested (test_cpu_sharpness.py) on float64 toy Hessians with known
off-diagonal blocks; the model-based HVP is additionally validated against central gradient differences on
the real model at relative step sizes 0.005/0.01/0.02/0.04 (plateau + 5% agreement gate).
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Pure primitives (CPU-testable, no model / no GPU)
# --------------------------------------------------------------------------- #
def shape_radius(rows: int, cols: int, lr_mult: float = 1.0) -> float:
    """Implemented Muon step radius r_i = sqrt(max(1, rows/cols)) * relative-LR multiplier."""
    return math.sqrt(max(1.0, rows / cols)) * lr_mult


def op_norm(D: Tensor) -> float:
    """Largest singular value (operator-2 norm)."""
    return float(torch.linalg.svdvals(D.double())[0])


def nuclear_norm(g: Tensor) -> float:
    """Sum of singular values (nuclear norm)."""
    return float(torch.linalg.svdvals(g.double()).sum())


def polar_lmo(g: Tensor, r: float = 1.0) -> Tensor:
    """LMO for the spectral-norm ball {||s||_op <= r}: argmax <g,s> = r*U V^T (thin SVD g=U S V^T).
    Result has every singular value = r, so <g, polar_lmo(g,1)> = ||g||_nuclear."""
    u, _, vh = torch.linalg.svd(g.double(), full_matrices=False)
    return (r * (u @ vh)).to(g.dtype)


def n_r(D: list[Tensor], radii: list[float]) -> float:
    """The optimizer norm N_r(D) = max_i ||D_i||_op / r_i."""
    return max(op_norm(D[i]) / radii[i] for i in range(len(D)))


def radial_normalize(D: list[Tensor], radii: list[float]) -> list[Tensor]:
    """Scale D onto the sphere N_r(D)=1 (radial projection; a quadratic form's boundary witness)."""
    s = n_r(D, radii)
    if s <= 0:
        return [x.clone() for x in D]
    return [x / s for x in D]


def G_denominators(grads: list[Tensor], radii: list[float]) -> tuple[list[float], float]:
    """G_i = r_i * ||g_i||_nuclear (shape-weighted dual gradient norm); G_joint = sum_i G_i."""
    Gi = [radii[i] * nuclear_norm(grads[i]) for i in range(len(grads))]
    return Gi, float(sum(Gi))


def Z_stat(eta: float, S: float, G: float) -> float:
    """Candidate normalized-spectral-descent diagnostic Z = eta*S/(2G). NOT a proven stability boundary."""
    return float("nan") if G == 0 else eta * S / (2.0 * G)


def autograd_joint_hvp(loss_fn, params: list[Tensor], vecs: list[Tensor]) -> list[Tensor]:
    """Joint HVP (Hv)_m = sum_n H_mn v_n via one create_graph backward + one scalar-dot second backward."""
    loss = loss_fn()
    g1 = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g1[n] * vecs[n]).sum() for n in range(len(params)))
    hv = torch.autograd.grad(dot, params)
    return [h.detach() for h in hv]


def autograd_diagonal_hvp_one(loss_fn, params: list[Tensor], i: int, vi: Tensor) -> Tensor:
    """Isolated diagonal HVP H_ii[v_i]: move only matrix i, read back only its own component."""
    loss = loss_fn()
    g1 = torch.autograd.grad(loss, params, create_graph=True)
    dot_i = (g1[i] * vi).sum()
    hv = torch.autograd.grad(dot_i, params[i])[0]
    return hv.detach()


def frank_wolfe(hvp_fn, v0: list[Tensor], radii: list[float], max_iters: int,
                snapshot_at: list[int] | None = None) -> dict:
    """Maximize q(v)=<v,H v> over the product of spectral balls (radii r_i) via Frank-Wolfe.
    hvp_fn(v) -> joint [(Hv)_m]. objective_trace[k]=<v_k,H v_k> (BALL relaxation value); step
    gamma_k=2/(k+2). One HVP per iteration. Records iterate snapshots at the requested K (default {max})
    so the SPHERE objective can be recomputed per K from a single FW run (no re-run per K)."""
    snap = set(snapshot_at or [max_iters])
    v = [x.clone() for x in v0]
    trace: list[float] = []
    snapshots: dict[int, list[Tensor]] = {}
    for k in range(max_iters + 1):
        hv = hvp_fn(v)
        trace.append(float(sum((vi * hi).sum() for vi, hi in zip(v, hv))))
        if k in snap:
            snapshots[k] = [x.clone() for x in v]
        if k == max_iters:
            break
        s = [polar_lmo(hv[m], radii[m]) for m in range(len(v))]
        gamma = 2.0 / (k + 2.0)
        v = [v[m] + gamma * (s[m] - v[m]) for m in range(len(v))]
    return dict(objective_trace=trace, v=v, snapshots=snapshots)


def sphere_objective(hvp_fn, D: list[Tensor], radii: list[float]) -> float | None:
    """Sphere value <D',H D'> with D'=D/N_r(D), HVP recomputed independently on D'. Returns None for an
    interior-collapsed point (N_r(D)~0) -- an interior zero is NOT the sphere maximum and must not be
    reported as one; the caller falls back to a boundary witness."""
    s = n_r(D, radii)
    if s <= 1e-30:
        return None
    Dn = [x / s for x in D]
    hv = hvp_fn(Dn)
    return float(sum((vi * hi).sum() for vi, hi in zip(Dn, hv)))


def init_v(seed_grads: list[Tensor], radii: list[float], restart: int, base_seed: int = 1337) -> list[Tensor]:
    """Restart 0 = gradient spectral-polar; restart>=1 = deterministic random spectral-unit (per restart)."""
    if restart == 0:
        return [polar_lmo(g, radii[m]) for m, g in enumerate(seed_grads)]
    gen = torch.Generator().manual_seed(base_seed + restart)
    out = []
    for m, g in enumerate(seed_grads):
        rnd = torch.randn(g.shape, generator=gen, dtype=torch.float64)
        out.append(polar_lmo(rnd, radii[m]).to(device=g.device, dtype=g.dtype))
    return out


def fw_best_sphere(hvp_fn, seed_grads, radii, iters_list, restarts):
    """SPHERE sharpness estimate. For each restart run FW ONCE to max(iters_list), snapshotting the iterate
    at every K in iters_list. The sphere witness at K is the best (max) sphere objective over {v0 boundary
    point, snapshot@K}, so a negative-definite problem whose FW iterate collapses to the interior still gets
    a boundary witness (v0) instead of a spurious interior zero. Report the best feasible restart per K (a
    lower-bound witness, not a global certificate) and flag negative-curvature / interior-collapse cases."""
    iters_list = sorted(set(iters_list))
    max_iters = max(iters_list)
    per_restart = []
    for r in range(restarts):
        v0 = init_v(seed_grads, radii, r)
        res = frank_wolfe(hvp_fn, v0, radii, max_iters, snapshot_at=iters_list)
        v0_sphere = sphere_objective(hvp_fn, v0, radii)  # v0 is already on the boundary (polar_lmo)
        sphere_at_k, collapsed_at_k = {}, {}
        for K in iters_list:
            cand = [c for c in (v0_sphere, sphere_objective(hvp_fn, res["snapshots"][K], radii)) if c is not None]
            sphere_at_k[K] = max(cand) if cand else None
            collapsed_at_k[K] = sphere_objective(hvp_fn, res["snapshots"][K], radii) is None
        per_restart.append(dict(restart=r, ball_trace=res["objective_trace"],
                                sphere_at_k=sphere_at_k, interior_collapsed_at_k=collapsed_at_k))
    best_per_k = {}
    for K in iters_list:
        vals = [(ro["sphere_at_k"][K], ro["restart"]) for ro in per_restart if ro["sphere_at_k"][K] is not None]
        best_per_k[K] = max(vals)[0] if vals else None
    best_sphere = best_per_k[max_iters]
    return dict(best_sphere=best_sphere,
                negative_curvature=(best_sphere is not None and best_sphere <= 0),
                best_sphere_per_k=best_per_k, per_restart=per_restart,
                iters_list=iters_list, restarts=restarts, max_iters=max_iters)


def isolated_sharpness(diag_hvp_one_fn, seed_grad_i: Tensor, radius_i: float, i: int,
                       iters_list, restarts) -> dict:
    """Isolated S_i = max <D_i, H_ii D_i> s.t. ||D_i||_op = r_i (single matrix; others frozen). Runs FW on
    matrix i alone with the diagonal HVP H_ii, then the sphere best-boundary witness (as fw_best_sphere)."""
    def hvp1(vlist):
        return [diag_hvp_one_fn(i, vlist[0])]
    return fw_best_sphere(hvp1, [seed_grad_i], [radius_i], iters_list, restarts)


def cross_layer_decomposition(hvp_fn, d: list[Tensor], diag_hvp_one_fn) -> dict:
    """For a signed direction d (per matrix): c=<d,H d>, c_diag=sum_i <d_i,H_ii d_i>, c_cross=c-c_diag,
    c_i_joint=<d_i,(H d)_i>. diag_hvp_one_fn(i, d_i) -> H_ii[d_i]."""
    hv = hvp_fn(d)
    c = float(sum((di * hi).sum() for di, hi in zip(d, hv)))
    c_i_joint = [float((d[i] * hv[i]).sum()) for i in range(len(d))]
    c_diag_terms = [float((d[i] * diag_hvp_one_fn(i, d[i])).sum()) for i in range(len(d))]
    c_diag = float(sum(c_diag_terms))
    return dict(c=c, c_diag=c_diag, c_cross=c - c_diag,
                c_i_joint=c_i_joint, c_diag_terms=c_diag_terms)


def interaction_matrix(hvp_fn, directions: list[Tensor]) -> list[list[float]]:
    """Q_ij = <d_i, H_ij[d_j]> using embedded isolated directions. For each j, a joint HVP on the vector
    that is d_j in slot j and zero elsewhere returns (H[e_j d_j])_i = H_ij[d_j] in slot i; dot with d_i."""
    n = len(directions)
    Q = [[0.0] * n for _ in range(n)]
    for j in range(n):
        ej = [torch.zeros_like(directions[k]) for k in range(n)]
        ej[j] = directions[j]
        hv = hvp_fn(ej)  # (hv)_i = H_ij[d_j]
        for i in range(n):
            Q[i][j] = float((directions[i] * hv[i]).sum())
    return Q


def central_diff_hvp(grad_fn, params: list[Tensor], d: list[Tensor], eps: float) -> list[Tensor]:
    """Finite-difference HVP: (g(W+eps*d) - g(W-eps*d)) / (2 eps) ~= H[d]. grad_fn() returns the gradient
    list at the CURRENT params; caller perturbs in place. Uses first-order gradients only (no double
    backward), so it validates the autograd HVP against the real (flash-attention) forward."""
    with torch.no_grad():
        for p, di in zip(params, d):
            p.add_(di, alpha=eps)
    gp = grad_fn()
    with torch.no_grad():
        for p, di in zip(params, d):
            p.add_(di, alpha=-2 * eps)
    gm = grad_fn()
    with torch.no_grad():
        for p, di in zip(params, d):
            p.add_(di, alpha=eps)  # restore
    return [(a - b) / (2 * eps) for a, b in zip(gp, gm)]


def make_quadratic_loss(hessian: Tensor, params: list[Tensor]):
    """Toy loss 0.5 x^T H x over x=concat(flattened params); Hessian is exactly `hessian`. For CPU tests."""
    def loss_fn() -> Tensor:
        x = torch.cat([p.reshape(-1) for p in params])
        return 0.5 * (x @ (hessian @ x))
    return loss_fn


# --------------------------------------------------------------------------- #
# GPU driver stubs (model-based) -- filled in for the box run; kept out of CPU import path.
# The model-based joint HVP / block gradients reuse REQ-019's data-parallel sharded implementation
# (joint_block_hvp, block_gradients) but with (a) shape-weighted radii, (b) a MEAN-per-valid-token
# token_scale instead of BATCH_TOKENS/tokens_seen, and (c) isolated single-matrix FW + Q_ij + Z.
# See run_pilot() below; harness imports are lazy so this file imports on a CPU box for the unit tests.
# --------------------------------------------------------------------------- #
def muon_matrix_names():
    import re
    from harness.model_gpt import GPT
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768)
    return [n for n, p in model.named_parameters() if re.match(r"^blocks\..*\.weight$", n) and p.ndim == 2]


if __name__ == "__main__":
    print("This module is imported by the driver and the CPU tests. Run test_cpu_sharpness.py to validate.")
