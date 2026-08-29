import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.muon import (AnnealedDecayMuon, AnnealedWeightsMuon,
                             finite_history_power_law_weights,
                             least_squares_momentum_weights)


def _kernel(decays, coefs, K):
    betas = torch.tensor(decays, dtype=torch.float64)
    c = torch.tensor(coefs, dtype=torch.float64)
    k = torch.arange(1, K + 1, dtype=torch.float64)
    return ((1 - betas)[None, :] * betas[None, :] ** k[:, None]) @ c


def test_flat_kernel_is_near_exact_in_a_dense_basis():
    taus = torch.logspace(torch.log10(torch.tensor(4.0)),
                          torch.log10(torch.tensor(199.0)), 100)
    decays = (taus / (1 + taus)).tolist()
    c = least_squares_momentum_weights(decays, gamma=0.0, window=512)
    m = _kernel(decays, c, 512)
    t = torch.full((512,), 1.0 / 512, dtype=torch.float64)
    relerr = torch.sqrt((t * (m - t) ** 2).sum() / (t * t ** 2).sum())
    assert relerr < 0.06
    assert max(abs(x) for x in c) < 10
    assert any(x < 0 for x in c)  # the good fits are not convex


def test_small_basis_cannot_reach_past_its_range():
    decays = [0.8501, 0.9208, 0.9598, 0.98]      # ages 5.7-49
    c = least_squares_momentum_weights(decays, gamma=1.0, window=128)
    m = _kernel(decays, c, 128)
    t = torch.full((128,), 1.0 / 128, dtype=torch.float64)
    relerr = torch.sqrt((t * (m - t) ** 2).sum() / (t * t ** 2).sum())
    assert relerr > 0.05                          # documented limitation


def test_annealed_weights_endpoints_and_midpoint():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = AnnealedWeightsMuon([p], decays=[0.5, 0.9],
                              start_weights=[1.5, -0.5],
                              end_weights=[-0.25, 1.25],
                              switch_step=10, anneal_end_step=20)
    for step, expected in [(0, 0.0), (10, 0.0), (15, 0.5), (20, 1.0), (25, 1.0)]:
        opt._muon_steps_seen = step
        assert opt.interpolation_fraction() == expected


def test_annealed_single_decay_keeps_blend_fixed_and_interpolates_beta():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = AnnealedDecayMuon([p], mu=0.95, beta_start=0.9, beta_end=0.98,
                            switch_step=10, anneal_end_step=20)
    for step, expected in [(0, 0.9), (10, 0.9), (15, 0.94),
                           (20, 0.98), (25, 0.98)]:
        opt._muon_steps_seen = step
        assert abs(opt.current_beta() - expected) < 1e-12
    assert opt.param_groups[0]["mu"] == 0.95


def test_coefficient_interpolation_equals_kernel_interpolation():
    decays = [0.5, 0.8, 0.95]
    start = [1.2, -0.4, 0.2]
    end = [-0.1, 0.3, 0.8]
    alpha = 0.37
    mixed_coefficients = [(1 - alpha) * a + alpha * b for a, b in zip(start, end)]
    actual = _kernel(decays, mixed_coefficients, 64)
    expected = (1 - alpha) * _kernel(decays, start, 64) + alpha * _kernel(decays, end, 64)
    torch.testing.assert_close(actual, expected)


def test_finite_history_power_law_has_unit_dc_gain():
    decays = [0.5, 0.8, 0.95, 0.995]
    horizon = 1000
    coefficients = finite_history_power_law_weights(decays, gamma=0.75,
                                                     horizon=horizon)
    torch.testing.assert_close(torch.tensor(sum(coefficients), dtype=torch.float64),
                               torch.tensor(1.0, dtype=torch.float64))


def test_bias_corrected_power_law_has_unit_realized_gain():
    decays = torch.tensor([0.5, 0.8, 0.95, 0.995], dtype=torch.float64)
    horizon = 1000
    coefficients = torch.tensor(finite_history_power_law_weights(
        decays.tolist(), gamma=0.75, horizon=horizon), dtype=torch.float64)
    k = torch.arange(horizon + 1, dtype=torch.float64)
    design = ((1 - decays)[None, :] * decays[None, :] ** k[:, None]
              / (1 - decays ** (horizon + 1))[None, :])
    torch.testing.assert_close((design @ coefficients).sum(),
                               torch.tensor(1.0, dtype=torch.float64))


def test_finite_history_power_law_uses_current_gradient_as_age_zero():
    decays = [0.2, 0.5, 0.8, 0.95, 0.995]
    coefficients = finite_history_power_law_weights(decays, gamma=1.0,
                                                     horizon=64)
    taps = torch.cat((
        torch.tensor([sum(c * (1 - beta) for beta, c in zip(decays, coefficients))],
                     dtype=torch.float64),
        _kernel(decays, coefficients, 64),
    ))
    target = 1 / torch.arange(1, 66, dtype=torch.float64)
    target /= target.sum()
    assert torch.sqrt(((taps - target) ** 2).mean()) < 0.02


def test_annealed_switch_is_lazy_and_initializes_from_single_ema():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    opt = AnnealedWeightsMuon([p], decays=[0.5, 0.9],
                              start_weights=[0.4, 0.6],
                              end_weights=[0.7, 0.3],
                              switch_step=2, anneal_end_step=4)
    state = opt.state[p]
    group = opt.param_groups[0]
    for step in range(2):
        opt._muon_steps_seen = step
        p.grad = torch.full_like(p, float(step + 1))
        opt.compute_polar_input(p, state, group)
        assert "streams" not in state
    opt._muon_steps_seen = 2
    p.grad = torch.full_like(p, 3.0)
    opt.compute_polar_input(p, state, group)
    assert len(state["streams"]) == 2
    torch.testing.assert_close(state["streams"][0], state["momentum"])
    torch.testing.assert_close(state["streams"][1], state["momentum"])


def test_annealed_warm_streams_are_real_history_but_switch_update_is_baseline():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    p_control = torch.nn.Parameter(torch.zeros(2, 2))
    opt = AnnealedWeightsMuon([p], decays=[0.5, 0.9],
                              start_weights=[0.4, 0.6],
                              end_weights=[0.7, 0.3], switch_step=2,
                              anneal_end_step=4,
                              warm_streams_before_switch=True)
    control = AnnealedWeightsMuon([p_control], decays=[0.5, 0.9],
                                  start_weights=[0.4, 0.6],
                                  end_weights=[0.7, 0.3], switch_step=2,
                                  anneal_end_step=4,
                                  warm_streams_before_switch=False)
    state = opt.state[p]
    group = opt.param_groups[0]
    for step, value in enumerate((1.0, 2.0, 3.0)):
        opt._muon_steps_seen = step
        control._muon_steps_seen = step
        p.grad = torch.full_like(p, value)
        p_control.grad = torch.full_like(p_control, value)
        actual = opt.compute_polar_input(p, state, group)
        expected = control.compute_polar_input(
            p_control, control.state[p_control], control.param_groups[0])
        torch.testing.assert_close(actual, expected)
        if step == 2:
            torch.testing.assert_close(state["streams"][0],
                                       torch.full_like(p, 2.125))
            torch.testing.assert_close(state["streams"][1],
                                       torch.full_like(p, 0.561))
            assert not torch.equal(state["streams"][0], state["momentum"])
