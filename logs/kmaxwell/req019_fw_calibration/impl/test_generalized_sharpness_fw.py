"""Fast CPU-only tests for measure_generalized_sharpness_fw.

No GPU, no model, no harness: the joint-HVP and FW machinery are exercised on
tiny explicit synthetic Hessians via the module's pure, model-free primitives.

    python -m pytest test_generalized_sharpness_fw.py -q
"""
from __future__ import annotations

import itertools

import torch

from measure_generalized_sharpness_fw import (autograd_diagonal_hvp,
                                              autograd_joint_hvp, frank_wolfe,
                                              init_v, make_quadratic_loss,
                                              polar_lmo)

torch.manual_seed(0)


def _spectral_norm(m: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(m, ord=2))


def _nuclear_norm(m: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(m, ord="nuc"))


# --------------------------------------------------------------------------- #
# Test 1: polar_lmo is the spectral-ball LMO
# --------------------------------------------------------------------------- #
def test_polar_lmo_spectral_norm_and_nuclear_duality():
    g = torch.randn(5, 3, dtype=torch.float64)
    for r in (1.0, 0.5, 2.3):
        s = polar_lmo(g, r)
        assert abs(_spectral_norm(s) - r) < 1e-9, "LMO output must sit on the ball"
        # <g, r U V^T> = r * sum(sigma) = r * ||g||_nuclear
        assert abs(float((g * s).sum()) - r * _nuclear_norm(g)) < 1e-9

    # <g, polar_lmo(g,1)> == ||g||_nuclear exactly (the duality identity)
    s1 = polar_lmo(g, 1.0)
    assert abs(float((g * s1).sum()) - _nuclear_norm(g)) < 1e-9


def test_polar_lmo_maximizes_over_the_ball():
    g = torch.randn(4, 4, dtype=torch.float64)
    best = float((g * polar_lmo(g, 1.0)).sum())
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):
        c = torch.randn(4, 4, generator=gen, dtype=torch.float64)
        c = c / max(_spectral_norm(c), 1e-30)          # random feasible point
        assert float((g * c).sum()) <= best + 1e-9


# --------------------------------------------------------------------------- #
# Test 2: the JOINT HVP carries cross-block terms (anti-diagonal guard)
# --------------------------------------------------------------------------- #
def _two_block_hessian(coupling: float) -> torch.Tensor:
    """4x4 SPD-ish H over two 1x2 blocks with a KNOWN off-diagonal coupling."""
    h = torch.tensor([
        [3.0, 0.0, coupling, 0.0],
        [0.0, 2.0, 0.0, 0.5],
        [coupling, 0.0, 4.0, 0.0],
        [0.0, 0.5, 0.0, 1.5],
    ], dtype=torch.float64)
    return h


def test_joint_hvp_has_cross_block_terms():
    coupling = 1.7
    h = _two_block_hessian(coupling)
    # two blocks, each a 1x2 matrix (flattened length 2 -> total 4)
    params = [torch.zeros(1, 2, dtype=torch.float64, requires_grad=True),
              torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)]
    loss_fn = make_quadratic_loss(h, params)

    # perturb ONLY block 1; block-0 output must be driven purely by the coupling
    vecs = [torch.zeros(1, 2, dtype=torch.float64),
            torch.tensor([[1.0, 3.0]], dtype=torch.float64)]

    joint = autograd_joint_hvp(loss_fn, params, vecs)
    diag = autograd_diagonal_hvp(loss_fn, params, vecs)

    # diagonal-only sees H_00 * v_0 = 0 on block 0
    assert float(joint[0].norm()) > 1e-8, "joint HVP must couple block 1 into block 0"
    assert float(diag[0].norm()) < 1e-12, "diagonal HVP must be blind to block 1"
    assert float((joint[0] - diag[0]).norm()) > 1e-8

    # exact value check against the dense Hessian: (Hv)_0 = H[0:2, 2:4] @ v_1
    v_full = torch.cat([vecs[0].reshape(-1), vecs[1].reshape(-1)])
    hv_full = h @ v_full
    assert torch.allclose(joint[0].reshape(-1), hv_full[:2], atol=1e-10)
    assert torch.allclose(joint[1].reshape(-1), hv_full[2:], atol=1e-10)


def test_joint_hvp_matches_dense_matmul_generic():
    torch.manual_seed(3)
    h = torch.randn(5, 5, dtype=torch.float64)
    h = h + h.T                                        # symmetric
    params = [torch.zeros(2, 1, dtype=torch.float64, requires_grad=True),
              torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)]
    loss_fn = make_quadratic_loss(h, params)
    vecs = [torch.randn(2, 1, dtype=torch.float64),
            torch.randn(1, 3, dtype=torch.float64)]
    joint = autograd_joint_hvp(loss_fn, params, vecs)
    v_full = torch.cat([vecs[0].reshape(-1), vecs[1].reshape(-1)])
    hv_full = h @ v_full
    got = torch.cat([joint[0].reshape(-1), joint[1].reshape(-1)])
    assert torch.allclose(got, hv_full, atol=1e-10)


# --------------------------------------------------------------------------- #
# Test 3: FW ascends on a concave instance; matches the closed-form max
# --------------------------------------------------------------------------- #
def _diag_hvp_fn(diag_vals):
    """Joint HVP for a diagonal H over 1x1 blocks: (Hv)_m = h_m v_m."""
    h = torch.diag(torch.tensor(diag_vals, dtype=torch.float64))
    params = [torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
              for _ in diag_vals]
    loss_fn = make_quadratic_loss(h, params)
    return lambda v: autograd_joint_hvp(loss_fn, params, v)


def test_fw_monotone_on_concave_instance():
    # negative-definite diagonal H -> q concave, constrained max at interior 0
    diag_vals = [-1.0, -2.0, -0.5, -3.0]
    hvp_fn = _diag_hvp_fn(diag_vals)
    radii = [1.0] * len(diag_vals)
    v0 = [torch.ones(1, 1, dtype=torch.float64) for _ in diag_vals]  # a corner
    res = frank_wolfe(hvp_fn, v0, radii, max_iters=60)
    trace = res["objective_trace"]
    for a, b in zip(trace[:-1], trace[1:]):
        assert b >= a - 1e-9, "FW must be non-decreasing on a concave instance"
    # concave max over the ball is the unconstrained max at v=0 -> objective 0
    assert trace[-1] <= 0.0 + 1e-9
    assert abs(trace[-1]) < 1e-2, "FW must climb toward the constrained max (0)"


def test_fw_matches_closed_form_max_convex_box():
    # positive diagonal, 1x1 blocks -> ball is [-r, r] per block; the max of
    # sum h_i v_i^2 is at a corner v_i = +/- r with value sum h_i r^2.
    diag_vals = [1.0, 2.0, 0.75]
    r = 1.0
    hvp_fn = _diag_hvp_fn(diag_vals)
    radii = [r] * len(diag_vals)
    v0 = [torch.full((1, 1), 0.1, dtype=torch.float64) for _ in diag_vals]
    res = frank_wolfe(hvp_fn, v0, radii, max_iters=80)
    got = res["objective_trace"][-1]

    # brute-force over the corners of the box
    best = max(sum(h * (s * r) ** 2 for h, s in zip(diag_vals, signs))
               for signs in itertools.product((-1.0, 1.0), repeat=len(diag_vals)))
    assert abs(got - best) < 1e-3, f"FW {got} vs closed-form {best}"


def test_init_v_matches_seed_device_and_dtype():
    # every init (restart 0 = gradient polar; restart>=1 = random) must live on the
    # SAME device and dtype as the seed grads. A bare .to(dtype) leaves the random
    # inits on CPU and the (cuda) joint HVP then crashes on a device mismatch.
    seed_grads = [torch.randn(3, 2, dtype=torch.float32),
                  torch.randn(2, 4, dtype=torch.float32)]
    radii = [1.0, 1.0]
    for restart in (0, 1, 2):
        vs = init_v(seed_grads, radii, restart)
        for v, g in zip(vs, seed_grads):
            assert v.device == g.device, f"restart {restart}: device mismatch"
            assert v.dtype == g.dtype, f"restart {restart}: dtype mismatch"
        vs2 = init_v(seed_grads, radii, restart)          # reproducible per restart
        for a, b in zip(vs, vs2):
            assert torch.allclose(a, b)


def test_frank_wolfe_objective_is_vHv():
    # objective_trace[0] must equal <v_0, H v_0> exactly
    diag_vals = [1.5, -0.5]
    hvp_fn = _diag_hvp_fn(diag_vals)
    v0 = [torch.tensor([[0.3]], dtype=torch.float64),
          torch.tensor([[-0.7]], dtype=torch.float64)]
    res = frank_wolfe(hvp_fn, v0, [1.0, 1.0], max_iters=0)
    expected = 1.5 * 0.3 ** 2 + (-0.5) * 0.7 ** 2
    assert abs(res["objective_trace"][0] - expected) < 1e-12
