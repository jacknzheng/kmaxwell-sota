"""REQ-060 diagnostic core: separate loss-cubic feedback from Muon's polar-map nonlinearity.

e_loss(c,delta) = [g(c+delta)+g(c-delta)]/2 - g(c)  (even 2nd-difference of the gradient).
  Since g=grad(L), this is (1/2) grad^3(L)[delta,delta] + O(delta^4): the LOSS-CUBIC (third-derivative)
  feedback. It is ZERO for a purely quadratic loss and scales ~delta^2 (checked at scales 1/0.5/0.25).

Polar-map separation at a frozen center (Phi = implemented polar/shape map, q0 its center input,
c0 = coefficient of the current gradient in the mixture after its buffer update):
  E_total        = [Phi(q+) + Phi(q-)]/2 - Phi(q0)          q+/- built from g(c +/- delta)
  E_map          = [Phi(q0 + c0*H[delta]) + Phi(q0 - c0*H[delta])]/2 - Phi(q0)   (exists on a quadratic loss)
  E_loss_residual= E_total - E_map                          (the extra curvature from loss-cubic feedback)
For a smooth map the leading residual is D Phi(q0)[c0 * e_loss]. Validated on float64 quadratic (e_loss=0,
residual=0, E_map!=0) and weakly-cubic (e_loss ~delta^2, residual tracks D Phi[c0 e_loss]) toys before the
production finite-precision polar map. Pure-Python/torch; no model. Frozen-buffer diagnostic, not a full replay.
"""
from __future__ import annotations
import torch


def e_loss(grad_fn, c, delta):
    """[g(c+delta)+g(c-delta)]/2 - g(c). grad_fn(x)->gradient at x (list or tensor)."""
    gp, gm, g0 = grad_fn(c + delta), grad_fn(c - delta), grad_fn(c)
    return 0.5 * (gp + gm) - g0


def quad_scaling(grad_fn, c, delta, scales=(1.0, 0.5, 0.25)):
    """Return ||e_loss(c, s*delta)|| for each scale; ~s^2 above the noise floor confirms cubic-feedback origin."""
    return {s: float(e_loss(grad_fn, c, s * delta).norm()) for s in scales}


def polar_separation(Phi, q0, c0, Hdelta, qplus, qminus):
    """E_total, E_map, E_loss_residual (see module docstring). Phi: tensor->tensor; q+/- precomputed inputs."""
    E_total = 0.5 * (Phi(qplus) + Phi(qminus)) - Phi(q0)
    E_map = 0.5 * (Phi(q0 + c0 * Hdelta) + Phi(q0 - c0 * Hdelta)) - Phi(q0)
    return dict(E_total=E_total, E_map=E_map, E_loss_residual=E_total - E_map)


def make_grad(H, T=None):
    """Gradient of L(x)=1/2 x^T H x [+ 1/6 T[x,x,x]] -> g(x)=H x [+ 1/2 T[x,x]]. T: (n,n,n) symmetric or None."""
    def g(x):
        out = H @ x
        if T is not None:
            out = out + 0.5 * torch.einsum('ijk,j,k->i', T, x, x)
        return out
    return g
