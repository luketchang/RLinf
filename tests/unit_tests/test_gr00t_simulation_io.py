# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_SIMULATION_IO_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf/models/embodiment/gr00t/simulation_io.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gr00t_simulation_io_under_test", _SIMULATION_IO_PATH
)
_SIMULATION_IO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SIMULATION_IO)

ACTION_CONVERSION_N1D7 = _SIMULATION_IO.ACTION_CONVERSION_N1D7
OBS_CONVERSION = _SIMULATION_IO.OBS_CONVERSION
convert_so101_obs_to_gr00t_format = _SIMULATION_IO.convert_so101_obs_to_gr00t_format
convert_to_so101_action_n1d7 = _SIMULATION_IO.convert_to_so101_action_n1d7


def _make_so101_observation(batch_size: int = 2) -> dict:
    return {
        "main_images": torch.full((batch_size, 8, 10, 3), 64, dtype=torch.uint8),
        "wrist_images": torch.full((batch_size, 6, 7, 3), 128, dtype=torch.uint8),
        "states": torch.arange(batch_size * 6, dtype=torch.float32).reshape(
            batch_size, 6
        ),
        "task_descriptions": [f"place vial {index}" for index in range(batch_size)],
    }


def test_so101_observation_conversion_matches_nvidia_schema():
    env_obs = _make_so101_observation()

    converted = convert_so101_obs_to_gr00t_format(env_obs)

    assert set(converted) == {
        "video.front",
        "video.wrist",
        "state.single_arm",
        "state.gripper",
        "annotation.human.task_description",
    }
    assert converted["video.front"].shape == (2, 1, 8, 10, 3)
    assert converted["video.wrist"].shape == (2, 1, 6, 7, 3)
    assert converted["video.front"].dtype == np.uint8
    assert converted["video.wrist"].dtype == np.uint8
    assert converted["state.single_arm"].shape == (2, 1, 5)
    assert converted["state.gripper"].shape == (2, 1, 1)
    assert converted["state.single_arm"].dtype == np.float32
    assert converted["state.gripper"].dtype == np.float32
    np.testing.assert_array_equal(
        converted["state.single_arm"][:, 0], env_obs["states"].numpy()[:, :5]
    )
    np.testing.assert_array_equal(
        converted["state.gripper"][:, 0], env_obs["states"].numpy()[:, 5:6]
    )
    assert converted["annotation.human.task_description"] == [
        "place vial 0",
        "place vial 1",
    ]


def test_so101_observation_conversion_scales_unit_float_images():
    env_obs = _make_so101_observation(batch_size=1)
    env_obs["main_images"] = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32)
    env_obs["wrist_images"] = np.ones((1, 1, 1, 3), dtype=np.float32)
    env_obs["task_descriptions"] = "place the vial"

    converted = convert_so101_obs_to_gr00t_format(env_obs)

    np.testing.assert_array_equal(
        converted["video.front"][0, 0, 0, 0], np.array([0, 128, 255])
    )
    np.testing.assert_array_equal(
        converted["video.wrist"][0, 0, 0, 0], np.array([255, 255, 255])
    )
    assert converted["annotation.human.task_description"] == ["place the vial"]


def test_so101_observation_conversion_preserves_uint8_values():
    env_obs = _make_so101_observation(batch_size=1)
    env_obs["main_images"] = torch.tensor([[[[0, 1, 1]]]], dtype=torch.uint8)

    converted = convert_so101_obs_to_gr00t_format(env_obs)

    np.testing.assert_array_equal(
        converted["video.front"][0, 0, 0, 0], np.array([0, 1, 1])
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("states", torch.zeros(2, 7), r"shape \[B, 6\]"),
        ("main_images", torch.zeros(2, 8, 10, 1), r"shape \[B, H, W, 3\]"),
        ("wrist_images", torch.zeros(1, 8, 10, 3), "batch size 2"),
        ("task_descriptions", ["only one"], "one task description"),
    ],
)
def test_so101_observation_conversion_rejects_invalid_inputs(field, value, message):
    env_obs = _make_so101_observation()
    env_obs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        convert_so101_obs_to_gr00t_format(env_obs)


def test_so101_observation_conversion_reports_missing_fields():
    env_obs = _make_so101_observation()
    del env_obs["wrist_images"]

    with pytest.raises(KeyError, match="wrist_images"):
        convert_so101_obs_to_gr00t_format(env_obs)


def test_so101_action_conversion_concatenates_absolute_motor_targets():
    arm = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    gripper = np.arange(2 * 16, dtype=np.float32).reshape(2, 16, 1)

    converted = convert_to_so101_action_n1d7(
        {"single_arm": arm, "gripper": gripper}, chunk_size=8
    )

    assert converted.shape == (2, 8, 6)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted[..., :5], arm[:, :8])
    np.testing.assert_array_equal(converted[..., 5:], gripper[:, :8])


@pytest.mark.parametrize(
    ("action_chunk", "chunk_size", "message"),
    [
        (
            {"single_arm": np.zeros((2, 16, 4)), "gripper": np.zeros((2, 16, 1))},
            8,
            r"shape \[B, H, 5\]",
        ),
        (
            {"single_arm": np.zeros((2, 16, 5)), "gripper": np.zeros((2, 8, 1))},
            8,
            "share batch and horizon",
        ),
        (
            {"single_arm": np.zeros((2, 16, 5)), "gripper": np.zeros((2, 16, 1))},
            17,
            "predicted horizon",
        ),
    ],
)
def test_so101_action_conversion_rejects_invalid_inputs(
    action_chunk, chunk_size, message
):
    with pytest.raises(ValueError, match=message):
        convert_to_so101_action_n1d7(action_chunk, chunk_size=chunk_size)


def test_so101_converters_are_registered_for_n1d7():
    assert OBS_CONVERSION["so100"] is convert_so101_obs_to_gr00t_format
    assert OBS_CONVERSION["so101"] is convert_so101_obs_to_gr00t_format
    assert ACTION_CONVERSION_N1D7["so100"] is convert_to_so101_action_n1d7
    assert ACTION_CONVERSION_N1D7["so101"] is convert_to_so101_action_n1d7


def test_so101_model_config_matches_adapter_contract():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/model/gr00t_n1d7_so101.yaml"
    )

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    assert config["model_type"] == "gr00t_n1d7"
    assert config["obs_converter_type"] == "so101"
    assert config["embodiment_tag"] == "so101"
    assert config["action_dim"] == 6
    assert config["num_action_chunks"] == 16
    assert config["rl_head_config"]["noise_method"] == "flow_sde"
    assert config["rl_head_config"]["action_noise_scale"] == 0.0
