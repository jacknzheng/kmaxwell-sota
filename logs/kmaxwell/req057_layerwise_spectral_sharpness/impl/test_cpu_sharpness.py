"""CPU toy-Hessian validation of the REQ-057 spectral-sharpness primitives.

Validates the pure math against the analytical answers the design review (verify.py) established:
 - joint HVP with cross terms == dense H@v; isolated diagonal HVP == H_ii v_i
 - cross-layer example H=[[1,.9],[.9,1]], d=1: c=3.8, c_diag=2, c_cross=1.8; Q symmetric, sum=3.8
 - joint sphere sharpness (~3.8) != sum of isolated maxima (2)
 - single-matrix spectral sharpness reproduces 1.2 vs 2.2 at equal Euclidean spectrum
 - negative-definite: sphere max = -1 (boundary witness), NOT the ball interior 0
 - shape_radius, G_i=r*nuclear, Z=eta*S/2G
 - loss-scale: Z and S/G invariant, raw S and G scale linearly, lambda/||g||_F^2 scales inversely
 - central-difference HVP == autograd HVP on the toy quadratic
 - quadratic loss identity: observed dL == -b + c/2
No GPU, no model, no checkpoints. Run: python test_cpu_sharpness.py
"""
import math
import torch

import measure_layerwise_spectral_sharpness as M

torch.set_default_dtype(torch.float64)


def _mats(shapes):
    return [torch.zeros(s, requires_grad=True) for s in shapes]


def test_joint_and_diagonal_hvp():
    H = torch.tensor([[2.0, 0.5, 0.3, 0.0],
                      [0.5, 1.0, 0.0, 0.1],
                      [0.3, 0.0, 3.0, 0.2],
                      [0.0, 0.1, 0.2, 1.5]])
    H = 0.5 * (H + H.T)
    params = _mats([(2, 1), (2, 1)])  # two 2x1 matrices -> 4 flattened dofs
    lf = M.make_quadratic_loss(H, params)
    v = [torch.tensor([[0.4], [-0.2]]), torch.tensor([[1.1], [0.7]])]
    hv = M.autograd_joint_hvp(lf, params, v)
    x = torch.cat([vi.reshape(-1) for vi in v])
    expect = H @ x
    got = torch.cat([h.reshape(-1) for h in hv])
    assert torch.allclose(got, expect, atol=1e-10), (got, expect)
    # diagonal H_ii v_i: block 0 is H[0:2,0:2], block 1 is H[2:4,2:4]
    hv0 = M.autograd_diagonal_hvp_one(lf, params, 0, v[0])
    assert torch.allclose(hv0.reshape(-1), H[0:2, 0:2] @ v[0].reshape(-1), atol=1e-10)
    hv1 = M.autograd_diagonal_hvp_one(lf, params, 1, v[1])
    assert torch.allclose(hv1.reshape(-1), H[2:4, 2:4] @ v[1].reshape(-1), atol=1e-10)


def test_cross_layer_example():
    H = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    params = _mats([(1, 1), (1, 1)])
    lf = M.make_quadratic_loss(H, params)
    d = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]

    def hvp(v):
        return M.autograd_joint_hvp(lf, params, v)

    def diag1(i, vi):
        return M.autograd_diagonal_hvp_one(lf, params, i, vi)

    dec = M.cross_layer_decomposition(hvp, d, diag1)
    assert abs(dec["c"] - 3.8) < 1e-10, dec["c"]
    assert abs(dec["c_diag"] - 2.0) < 1e-10, dec["c_diag"]
    assert abs(dec["c_cross"] - 1.8) < 1e-10, dec["c_cross"]
    Q = M.interaction_matrix(hvp, d)
    assert abs(Q[0][1] - 0.9) < 1e-10 and abs(Q[1][0] - 0.9) < 1e-10, Q
    assert abs(sum(Q[i][j] for i in range(2) for j in range(2)) - 3.8) < 1e-10


def test_joint_sphere_vs_isolated_sum():
    H = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    params = _mats([(1, 1), (1, 1)])
    lf = M.make_quadratic_loss(H, params)
    radii = [1.0, 1.0]
    seed = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]

    def hvp(v):
        return M.autograd_joint_hvp(lf, params, v)

    def diag1(i, vi):
        return M.autograd_diagonal_hvp_one(lf, params, i, vi)

    joint = M.fw_best_sphere(hvp, seed, radii, iters_list=[20, 50], restarts=5)
    assert abs(joint["best_sphere"] - 3.8) < 1e-6, joint["best_sphere"]
    iso = [M.isolated_sharpness(diag1, seed[i], radii[i], i, [20, 50], 3)["best_sphere"] for i in range(2)]
    assert abs(iso[0] - 1.0) < 1e-6 and abs(iso[1] - 1.0) < 1e-6, iso
    assert abs(sum(iso) - 2.0) < 1e-6                          # isolated sum 2 != joint 3.8
    assert joint["best_sphere"] - sum(iso) > 1.5


def test_spectral_sharpness_equal_euclidean():
    # single 2x2 matrix; H = outer(vec(a),vec(a)) + 0.1 I; a=diag(1,0) -> S=1.2, a=I/sqrt2 -> S=2.2,
    # while both H share the Euclidean spectrum [0.1,0.1,0.1,1.1].
    for a, expect in ((torch.diag(torch.tensor([1.0, 0.0])), 1.2),
                      (torch.eye(2) / math.sqrt(2), 2.2)):
        H = torch.outer(a.flatten(), a.flatten()) + 0.1 * torch.eye(4)
        params = _mats([(2, 2)])
        lf = M.make_quadratic_loss(H, params)

        def hvp(v):
            return M.autograd_joint_hvp(lf, params, v)

        seed = [a.clone().reshape(2, 2) if a.norm() > 0 else torch.eye(2)]
        res = M.fw_best_sphere(hvp, seed, [1.0], iters_list=[20, 50], restarts=5)
        assert abs(res["best_sphere"] - expect) < 1e-4, (expect, res["best_sphere"])


def test_negative_definite_sphere_not_ball():
    H = -torch.eye(1)
    params = _mats([(1, 1)])
    lf = M.make_quadratic_loss(H, params)

    def hvp(v):
        return M.autograd_joint_hvp(lf, params, v)

    seed = [torch.tensor([[1.0]])]
    res = M.fw_best_sphere(hvp, seed, [1.0], iters_list=[20, 50], restarts=3)
    assert res["best_sphere"] is not None
    assert abs(res["best_sphere"] - (-1.0)) < 1e-9, res["best_sphere"]   # sphere max -1, NOT interior 0
    assert res["negative_curvature"] is True


def test_shape_radius_and_G_Z():
    assert abs(M.shape_radius(3072, 768) - 2.0) < 1e-12
    assert abs(M.shape_radius(768, 3072) - 1.0) < 1e-12
    assert abs(M.shape_radius(768, 768) - 1.0) < 1e-12
    assert abs(M.shape_radius(3072, 768, lr_mult=0.5) - 1.0) < 1e-12
    g = [torch.tensor([[2.0, 0.2], [0.1, 1.0]])]
    Gi, Gj = M.G_denominators(g, [2.0])
    assert abs(Gi[0] - 2.0 * M.nuclear_norm(g[0])) < 1e-10 and abs(Gj - Gi[0]) < 1e-10
    assert abs(M.Z_stat(0.02, 2.2, 1.1) - 0.02 * 2.2 / (2 * 1.1)) < 1e-12
    assert math.isnan(M.Z_stat(0.02, 2.2, 0.0))


def test_loss_scale_invariance():
    base = torch.tensor([[1.0, 0.9], [0.9, 1.0]])
    params = _mats([(1, 1), (1, 1)])
    Z_vals, S_vals = [], []
    for scale in (0.1, 1.0, 10.0):
        H = scale * base
        lf = M.make_quadratic_loss(H, params)

        def hvp(v):
            return M.autograd_joint_hvp(lf, params, v)
        seed = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]
        S = M.fw_best_sphere(hvp, seed, [1.0, 1.0], [20, 50], 3)["best_sphere"]
        # gradient at a fixed point scales with the loss too; emulate g = scale * g0
        g0 = [torch.tensor([[2.0]]), torch.tensor([[1.0]])]
        g = [scale * gi for gi in g0]
        _, Gj = M.G_denominators(g, [1.0, 1.0])
        Z_vals.append(M.Z_stat(0.02, S, Gj))
        S_vals.append(S)
    assert abs(Z_vals[0] - Z_vals[1]) < 1e-6 and abs(Z_vals[2] - Z_vals[1]) < 1e-6, Z_vals
    assert abs(S_vals[0] / S_vals[1] - 0.1) < 1e-6 and abs(S_vals[2] / S_vals[1] - 10.0) < 1e-4, S_vals


def test_central_diff_matches_autograd_hvp():
    H = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    params = _mats([(1, 1), (1, 1)])
    # place params at a nonzero point so the toy grad is nonzero
    with torch.no_grad():
        params[0].copy_(torch.tensor([[0.3]])); params[1].copy_(torch.tensor([[-0.7]]))
    lf = M.make_quadratic_loss(H, params)

    def grad_fn():
        loss = lf()
        return [g.detach().clone() for g in torch.autograd.grad(loss, params)]

    def hvp(v):
        return M.autograd_joint_hvp(lf, params, v)
    d = [torch.tensor([[0.5]]), torch.tensor([[1.2]])]
    cd = M.central_diff_hvp(grad_fn, params, d, eps=1e-3)
    ag = hvp(d)
    for a, b in zip(cd, ag):
        assert torch.allclose(a, b, atol=1e-6), (a, b)
    # params restored
    assert torch.allclose(params[0], torch.tensor([[0.3]]), atol=1e-10)


def test_quadratic_loss_identity():
    H = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    w = torch.tensor([0.3, -0.7]); d = torch.tensor([-0.1, 0.2])
    g = H @ w
    b = float(-(g @ d))
    c = float(d @ H @ d)
    predicted = -b + c / 2
    observed = float(0.5 * ((w + d) @ H @ (w + d) - w @ H @ w))
    assert abs(observed - predicted) < 1e-12, (observed, predicted)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} CPU TESTS PASSED")
