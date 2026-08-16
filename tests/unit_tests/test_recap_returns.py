"""RECAP return-label contracts for LeRobot collection schemas."""

import pytest

from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    _success_scalar,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, False),
        (True, True),
        ([False], False),
        ([True], True),
    ],
)
def test_success_scalar_accepts_scalar_and_shape_one(value, expected):
    assert _success_scalar(value) is expected


def test_success_scalar_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="one is_success value"):
        _success_scalar([False, True])
