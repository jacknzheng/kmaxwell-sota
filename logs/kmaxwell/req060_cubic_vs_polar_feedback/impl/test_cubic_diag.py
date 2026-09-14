"""CPU validation of the REQ-060 cubic-vs-polar separation on float64 toys."""
import torch, math
import req060_cubic_diag as R
torch.set_default_dtype(torch.float64)


def test_eloss_zero_on_quadratic():
    torch.manual_seed(0); n = 6
    H = torch.randn(n, n); H = H @ H.T
    g = R.make_grad(H)
    c = torch.randn(n); d = torch.randn(n) * 0.1
    el = R.e_loss(g, c, d)
    assert el.norm() < 1e-10, el.norm()   # no 3rd derivative -> e_loss == 0
    print(f"PASS eloss_zero_on_quadratic (||e_loss||={el.norm():.2e})")


def test_eloss_cubic_scaling():
    torch.manual_seed(1); n = 5
    H = torch.randn(n, n); H = H @ H.T
    T = torch.randn(n, n, n); T = (T + T.transpose(0,1) + T.transpose(0,2) + T.transpose(1,2)
                                   + T.transpose(0,1).transpose(1,2) + T.transpose(0,2).transpose(1,2)) / 6  # symmetrize
    g = R.make_grad(H, T)
    c = torch.randn(n); d = torch.randn(n) * 0.1
    sc = R.quad_scaling(g, c, d)
    # e_loss = 1/2 T[.,d,d] exactly for a cubic (no O(d^4)); so scaling is exactly s^2
    r1 = sc[1.0]; r05 = sc[0.5]; r025 = sc[0.25]
    assert abs(r05 / r1 - 0.25) < 1e-6 and abs(r025 / r1 - 0.0625) < 1e-6, (r1, r05, r025)
    # cross-check against the analytic 1/2 T[.,d,d]
    analytic = 0.5 * torch.einsum('ijk,j,k->i', T, d, d)
    assert (R.e_loss(g, c, d) - analytic).norm() < 1e-10
    print(f"PASS eloss_cubic_scaling (s^2 exact; matches 1/2 T[.,d,d])")


def test_polar_separation_quadratic():
    # On a quadratic loss, e_loss=0 so E_loss_residual should be ~0, but E_map (polar nonlinearity) can be nonzero.
    torch.manual_seed(2); n = 4
    H = torch.randn(n, n); H = H @ H.T
    g = R.make_grad(H)
    c = torch.randn(n); d = torch.randn(n) * 0.05
    c0 = 0.05
    # Phi = a nonlinear (normalizing) map, like a shape-scaled unit projection: Phi(q)=q/||q||
    def Phi(q): return q / (q.norm() + 1e-12)
    q0 = torch.randn(n)
    # inputs from g(c +/- d): the mixture input shifts by c0 * (g(c+/-d)-g(c)) = c0*(+/-H d) on a quadratic
    Hd = H @ d
    qp = q0 + c0 * (g(c + d) - g(c)); qm = q0 + c0 * (g(c - d) - g(c))
    sep = R.polar_separation(Phi, q0, c0, Hd, qp, qm)
    # on a quadratic, q+/- == q0 +/- c0*Hd exactly, so E_total == E_map and residual ~ 0
    assert sep["E_loss_residual"].norm() < 1e-9, sep["E_loss_residual"].norm()
    assert sep["E_map"].norm() > 1e-6   # the polar map itself bends (nonzero E_map)
    print(f"PASS polar_separation_quadratic (residual={sep['E_loss_residual'].norm():.2e}, E_map={sep['E_map'].norm():.2e})")


def test_polar_separation_cubic_residual():
    # With a cubic loss, q+/- differ from q0 +/- c0*Hd by c0*e_loss -> nonzero E_loss_residual ~ DPhi[c0 e_loss].
    torch.manual_seed(3); n = 4
    H = torch.randn(n, n); H = H @ H.T
    T = torch.randn(n, n, n); T = (T + T.transpose(0,1)+T.transpose(0,2)+T.transpose(1,2)
                                   + T.transpose(0,1).transpose(1,2)+T.transpose(0,2).transpose(1,2))/6
    g = R.make_grad(H, T)
    c = torch.randn(n); d = torch.randn(n) * 0.05; c0 = 0.05
    def Phi(q): return q / (q.norm() + 1e-12)
    q0 = torch.randn(n); Hd = H @ d
    qp = q0 + c0 * (g(c + d) - g(c)); qm = q0 + c0 * (g(c - d) - g(c))
    sep = R.polar_separation(Phi, q0, c0, Hd, qp, qm)
    assert sep["E_loss_residual"].norm() > 1e-7, "cubic loss should produce nonzero residual"
    print(f"PASS polar_separation_cubic_residual (residual={sep['E_loss_residual'].norm():.2e} > 0)")


if __name__ == "__main__":
    for t in [test_eloss_zero_on_quadratic, test_eloss_cubic_scaling, test_polar_separation_quadratic, test_polar_separation_cubic_residual]:
        t()
    print("\nALL REQ-060 CPU TESTS PASSED")
