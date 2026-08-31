import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import BimaxwellMuon, NewtonBimaxwellMuon


def test_alpha_zero_is_exact_bypass():
    p_plain = torch.nn.Parameter(torch.zeros(4, 4))
    p_newton = torch.nn.Parameter(torch.zeros(4, 4))
    plain = BimaxwellMuon([p_plain], switch_step=2)
    newton = NewtonBimaxwellMuon([p_newton], switch_step=2, newton_alpha=0)
    for step in range(6):
        gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4) + step
        p_plain.grad = gradient.clone()
        p_newton.grad = gradient.clone()
        out_plain = plain.compute_polar_input(p_plain, plain.state[p_plain], plain.param_groups[0])
        out_newton = newton.compute_polar_input(
            p_newton, newton.state[p_newton], newton.param_groups[0])
        assert torch.equal(out_plain, out_newton)
        plain._muon_steps_seen += 1
        newton._muon_steps_seen += 1
    for key in plain.state[p_plain]:
        assert torch.equal(plain.state[p_plain][key], newton.state[p_newton][key])


def test_right_precondition_is_applied_before_kernel(monkeypatch):
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = NewtonBimaxwellMuon([p], switch_step=100, newton_alpha=0.5)
    p._newton_covariance_ref = {
        "blocks": 1, "inverse_power": torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])}
    p.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    seen = {}

    def capture(grad, momentum, mu=0.95, nesterov=True):
        seen["grad"] = grad.clone()
        return grad.clone()

    monkeypatch.setattr("optimizers.muon.muon_update", capture)
    raw = p.grad
    opt.compute_polar_input(p, opt.state[p], opt.param_groups[0])
    assert torch.equal(seen["grad"], torch.tensor([[2.0, 6.0], [6.0, 12.0]]))
    assert p.grad is raw
