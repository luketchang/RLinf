"""Focused contracts for SO-101 RECAP data preparation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

openpi = pytest.importorskip("openpi")

from openpi.models import model as model_lib  # noqa: E402

from rlinf.models.embodiment.openpi.dataconfig import (  # noqa: E402
    get_openpi_config,
)
from rlinf.models.embodiment.openpi.policies.so101_vials_policy import (  # noqa: E402
    SO101VialsInputs,
    SO101VialsOutputs,
    SO101VialsRepack,
)


def _image(value: int = 0) -> np.ndarray:
    return np.full((8, 12, 3), value, dtype=np.uint8)


@pytest.mark.parametrize(
    ("record", "expected_external", "expected_wrist"),
    [
        (
            {
                "observation.images.external_D455": _image(11),
                "observation.images.ego": _image(22),
                "observation.state": np.arange(6, dtype=np.float32),
                "action": np.arange(12, dtype=np.float32).reshape(2, 6),
                "prompt": "place the vial",
            },
            11,
            22,
        ),
        (
            {
                "image": _image(33),
                "wrist_image": _image(44),
                "state": np.arange(6, dtype=np.float32),
                "actions": np.arange(12, dtype=np.float32).reshape(2, 6),
                "task": "place the vial",
            },
            33,
            44,
        ),
    ],
)
def test_so101_repack_accepts_teleop_and_rollout_schemas(
    record, expected_external, expected_wrist
):
    repacked = SO101VialsRepack()(record)

    assert repacked["observation/image"][0, 0, 0] == expected_external
    assert repacked["observation/wrist_image"][0, 0, 0] == expected_wrist
    assert repacked["observation/state"].shape == (6,)
    assert repacked["actions"].shape == (2, 6)
    assert repacked["prompt"] == "place the vial"


def test_so101_openpi_inputs_use_two_real_cameras_and_six_actions():
    repacked = SO101VialsRepack()(
        {
            "image": _image(1),
            "wrist_image": _image(2),
            "state": np.arange(6, dtype=np.float32),
            "actions": np.arange(18, dtype=np.float32).reshape(3, 6),
        }
    )
    transformed = SO101VialsInputs(model_type=model_lib.ModelType.PI05)(repacked)

    assert transformed["state"].shape == (6,)
    assert transformed["actions"].shape == (3, 6)
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert transformed["image"]["base_0_rgb"][0, 0, 0] == 1
    assert transformed["image"]["left_wrist_0_rgb"][0, 0, 0] == 2
    assert not transformed["image"]["right_wrist_0_rgb"].any()

    padded_actions = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    output = SO101VialsOutputs()({"actions": padded_actions})
    np.testing.assert_array_equal(output["actions"], padded_actions[:, :6])


def test_so101_openpi_inputs_omit_nullable_prompt():
    transformed = SO101VialsInputs(model_type=model_lib.ModelType.PI05)(
        {
            "observation/image": _image(1),
            "observation/wrist_image": _image(2),
            "observation/state": np.arange(6, dtype=np.float32),
            "prompt": None,
        }
    )

    assert "prompt" not in transformed


def test_so101_openpi_keeps_padded_internal_action_width():
    config = get_openpi_config("pi05_so101_vials")

    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50


def test_openpi_numpy_boundary_accepts_bfloat16():
    from rlinf.models.embodiment.openpi.openpi_action_model import _to_numpy

    converted = _to_numpy(torch.ones(2, dtype=torch.bfloat16))

    assert converted.dtype == np.float32
