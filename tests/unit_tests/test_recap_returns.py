"""RECAP return-label contracts for LeRobot collection schemas."""

import pytest

from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    _success_scalar,
)
from rlinf.data.datasets.recap.utils import load_task_descriptions


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


def test_task_descriptions_accept_lerobot_v3_index(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.parquet").touch()
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=["pick", "place"])
    monkeypatch.setattr(pd, "read_parquet", lambda _: tasks)

    assert load_task_descriptions(tmp_path) == {0: "pick", 1: "place"}
