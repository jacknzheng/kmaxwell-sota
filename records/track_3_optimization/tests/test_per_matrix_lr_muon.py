import ast
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import build_registered_optimizer
from optimizers.muon import BimaxwellMuon, PerMatrixLrMuon
from harness.hooks import log_learning_rates_at_steps


def make_params():
    # Deliberately unsorted: Muon's public multiplier convention is the
    # size-descending order, not caller order.
    return [
        torch.nn.Parameter(torch.full((1, 1), 2.0)),
        torch.nn.Parameter(torch.full((3, 3), 2.0)),
        torch.nn.Parameter(torch.full((2, 2), 2.0)),
    ]


def test_multiplier_count_must_match_sorted_matrices():
    with pytest.raises(ValueError, match="2 entries for 3 matrices"):
        PerMatrixLrMuon(make_params(), lr_multipliers=[0.6, 1.7])


def test_registry_constructs_per_matrix_optimizer():
    optimizer = build_registered_optimizer(
        "per_matrix_lr_muon", make_params(),
        lr=0.025, weight_decay=0.05,
        lr_multipliers=[0.6, 1.0, 1.7],
    )
    assert isinstance(optimizer, PerMatrixLrMuon)
    assert isinstance(optimizer, BimaxwellMuon)
    assert optimizer.lr_multipliers == (0.6, 1.0, 1.7)


def test_bimaxwell_state_loads_without_buffer_translation():
    source_params = make_params()
    source = BimaxwellMuon(source_params, lr=0.025, weight_decay=0.05)
    for index, param in enumerate(source.sorted_params()):
        source.state[param]["momentum"] = torch.full_like(param, index + 1.0)
        source.state[param]["m_fast"] = torch.full_like(param, index + 2.0)
        source.state[param]["m_slow"] = torch.full_like(param, index + 3.0)
    source._muon_steps_seen = 1500

    target = PerMatrixLrMuon(
        make_params(), lr=0.025, weight_decay=0.05,
        lr_multipliers=[0.6, 1.0, 1.7],
    )
    target.load_state_dict(source.state_dict())
    target._muon_steps_seen = source._muon_steps_seen

    for source_param, target_param in zip(source.sorted_params(),
                                          target.sorted_params()):
        assert set(target.state[target_param]) == {"momentum", "m_fast", "m_slow"}
        for key in ("momentum", "m_fast", "m_slow"):
            torch.testing.assert_close(target.state[target_param][key],
                                       source.state[source_param][key])


def test_step_scales_update_and_weight_decay_in_sorted_index_order():
    params = make_params()
    optimizer = PerMatrixLrMuon(
        params, lr=0.1, weight_decay=0.2,
        lr_multipliers=[0.5, 1.0, 2.0],
    )
    sorted_params = optimizer.sorted_params()
    assert [tuple(p.shape) for p in sorted_params] == [(3, 3), (2, 2), (1, 1)]

    # Isolate the application rule from the compiled momentum/polar kernel.
    optimizer.compute_polar_input = lambda p, state, group: torch.ones_like(p)
    for p in params:
        p.grad = torch.zeros_like(p)

    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.all_gather"):
        optimizer.step()

    for p, multiplier in zip(sorted_params, (0.5, 1.0, 2.0)):
        lr = 0.1 * multiplier
        expected = 2.0 * (1 - lr * 0.2) - lr
        torch.testing.assert_close(p, torch.full_like(p, expected))
    assert optimizer._muon_steps_seen == 1


def test_first_fork_trace_pairs_names_with_sorted_multipliers():
    model = torch.nn.Module()
    model.register_parameter("small", torch.nn.Parameter(torch.zeros(1, 1)))
    model.register_parameter("large", torch.nn.Parameter(torch.zeros(3, 3)))
    model.register_parameter("middle", torch.nn.Parameter(torch.zeros(2, 2)))
    optimizer = PerMatrixLrMuon(
        [model.small, model.large, model.middle], lr=0.1,
        lr_multipliers=[0.6, 1.0, 1.7],
    )
    grouped = SimpleNamespace(groups=[
        (SimpleNamespace(pattern="test"), optimizer),
    ])
    messages = []
    hook = log_learning_rates_at_steps(steps=[1500])
    hook({}, {"step": 1500, "optimizer": grouped, "model": model,
              "print_log": lambda message, console=False: messages.append(message)})

    assert len(messages) == 1
    payload = ast.literal_eval(messages[0].split(" ", 2)[2])
    trace = payload[0]["per_matrix"]
    assert [(row["sorted_index"], row["name"], row["multiplier"])
            for row in trace] == [
                (0, "large", 0.6),
                (1, "middle", 1.0),
                (2, "small", 1.7),
            ]
    assert [row["effective_lr"] for row in trace] == pytest.approx(
        [0.06, 0.1, 0.17])
