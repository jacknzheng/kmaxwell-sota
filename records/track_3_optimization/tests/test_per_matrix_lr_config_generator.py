import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_analysis.make_per_matrix_lr_configs import (
    MULTIPLIERS, make_assignments, matrix_type,
)


def synthetic_names():
    kinds = ("attn.k", "attn.q", "attn.v", "attn.proj", "mlp.fc", "mlp.proj")
    return [f"blocks.{depth}.{kind}.weight"
            for kind in kinds for depth in range(12)]


def test_assignments_are_balanced_within_type_and_across_replicates():
    names = synthetic_names()
    assignments = make_assignments(names)
    assert len(assignments) == 3
    for assignment in assignments:
        for kind in {matrix_type(name) for name in names}:
            values = [assignment[name] for name in names if matrix_type(name) == kind]
            assert {value: values.count(value) for value in MULTIPLIERS} == {
                0.6: 4, 1.0: 4, 1.7: 4,
            }
    for name in names:
        assert sorted(assignment[name] for assignment in assignments) == [0.6, 1.0, 1.7]
