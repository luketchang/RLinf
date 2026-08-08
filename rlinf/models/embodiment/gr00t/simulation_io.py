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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value: Any, field_name: str) -> np.ndarray:
    """Convert an observation or action value to a CPU NumPy array.

    Args:
        value: NumPy array, PyTorch tensor, or array-like value.
        field_name: Name used in validation error messages.

    Returns:
        A NumPy representation of ``value``.

    Raises:
        TypeError: If the value cannot be represented as a NumPy array.
    """
    if torch.is_tensor(value):
        value = value.detach().cpu()
        # NumPy has no native bfloat16 representation.
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()

    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unable to convert '{field_name}' to a NumPy array.") from exc


def _to_uint8_rgb_images(value: Any, field_name: str) -> np.ndarray:
    """Validate an NHWC RGB image batch and convert it to uint8."""
    images = _to_numpy(value, field_name)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected '{field_name}' to have shape [B, H, W, 3], got {images.shape}."
        )
    if not np.issubdtype(images.dtype, np.number) or np.issubdtype(
        images.dtype, np.complexfloating
    ):
        raise TypeError(
            f"Expected '{field_name}' to contain real numeric RGB values, "
            f"got dtype {images.dtype}."
        )
    if not np.isfinite(images).all():
        raise ValueError(f"Expected '{field_name}' to contain only finite values.")

    scale_unit_range = np.issubdtype(images.dtype, np.floating)
    images = images.astype(np.float32, copy=False)
    min_value = float(images.min()) if images.size else 0.0
    max_value = float(images.max()) if images.size else 0.0
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError(
            f"Expected '{field_name}' values in [0, 1] or [0, 255], "
            f"got range [{min_value}, {max_value}]."
        )
    if scale_unit_range and max_value <= 1.0:
        images = images * 255.0
    return np.rint(images).astype(np.uint8)


def _normalize_task_descriptions(value: Any, batch_size: int) -> list[str]:
    """Normalize task prompts to one string per environment."""
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(
            "Expected 'task_descriptions' to be a string or a sequence of strings."
        )
    if len(value) != batch_size:
        raise ValueError(
            "Expected one task description per environment, "
            f"got {len(value)} descriptions for batch size {batch_size}."
        )
    if not all(isinstance(description, str) for description in value):
        raise TypeError("Expected every task description to be a string.")
    return list(value)


def convert_so101_obs_to_gr00t_format(
    env_obs: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert batched SO-101 observations to the GR00T N1.7 schema.

    This follows NVIDIA's official SO-100/SO-101 ``NEW_EMBODIMENT``
    configuration: two RGB views, five arm joints, one gripper joint, and one
    language instruction per environment. Simulator-specific joint conversion
    must happen before this function; ``states`` are LeRobot motor values.

    Args:
        env_obs: Mapping containing ``main_images`` and ``wrist_images`` in
            ``[B, H, W, 3]`` NHWC layout, ``states`` in ``[B, 6]`` layout, and
            ``task_descriptions``.

    Returns:
        Observation mapping accepted by the GR00T N1.7 modality processor.

    Raises:
        KeyError: If a required observation field is missing.
        TypeError: If a field has an unsupported type.
        ValueError: If field shapes, ranges, or batch dimensions are invalid.
    """
    required_fields = (
        "main_images",
        "wrist_images",
        "states",
        "task_descriptions",
    )
    missing_fields = [field for field in required_fields if field not in env_obs]
    if missing_fields:
        raise KeyError(f"Missing SO-101 observation fields: {missing_fields}.")

    front_images = _to_uint8_rgb_images(env_obs["main_images"], "main_images")
    wrist_images = _to_uint8_rgb_images(env_obs["wrist_images"], "wrist_images")
    states = _to_numpy(env_obs["states"], "states")

    if states.ndim != 2 or states.shape[-1] != 6:
        raise ValueError(f"Expected 'states' to have shape [B, 6], got {states.shape}.")
    if not np.issubdtype(states.dtype, np.number) or np.issubdtype(
        states.dtype, np.complexfloating
    ):
        raise TypeError(
            "Expected 'states' to contain real numeric motor values, "
            f"got {states.dtype}."
        )
    if not np.isfinite(states).all():
        raise ValueError("Expected 'states' to contain only finite values.")

    batch_size = states.shape[0]
    for field_name, images in (
        ("main_images", front_images),
        ("wrist_images", wrist_images),
    ):
        if images.shape[0] != batch_size:
            raise ValueError(
                f"Expected '{field_name}' batch size {batch_size}, "
                f"got {images.shape[0]}."
            )

    task_descriptions = _normalize_task_descriptions(
        env_obs["task_descriptions"], batch_size
    )
    states = states.astype(np.float32, copy=False)

    return {
        # GR00T expects an explicit observation-time dimension.
        "video.front": front_images[:, None],
        "video.wrist": wrist_images[:, None],
        "state.single_arm": states[:, None, :5],
        "state.gripper": states[:, None, 5:6],
        "annotation.human.task_description": task_descriptions,
    }


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t_1_7 pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action_n1d5(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T N1.5 action chunk to Libero format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d6(
    action_chunk: dict[str, np.array],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.6 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")

    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.7 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")
    # gripper conversion is handled by the shared
    # ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.

    if action_array.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim action, got {action_array.shape[-1]}")
    return action_array


def convert_to_so101_action_n1d7(
    action_chunk: Mapping[str, Any],
    chunk_size: int = 16,
) -> np.ndarray:
    """Convert decoded GR00T N1.7 actions to SO-101 motor targets.

    The N1.7 processor has already unnormalized the action and converted the
    five arm joints from relative deltas to absolute targets. This function
    preserves those values and appends the absolute gripper target, yielding
    the six-value LeRobot SO-101 action expected by an environment adapter.

    Args:
        action_chunk: Decoded action mapping with ``single_arm`` in
            ``[B, H, 5]`` and ``gripper`` in ``[B, H, 1]`` layout.
        chunk_size: Number of actions to return from the predicted horizon.

    Returns:
        Absolute motor targets in ``[B, chunk_size, 6]`` layout.

    Raises:
        KeyError: If either action component is missing.
        ValueError: If shapes are invalid or ``chunk_size`` exceeds the
            predicted horizon.
    """
    missing_fields = [
        field for field in ("single_arm", "gripper") if field not in action_chunk
    ]
    if missing_fields:
        raise KeyError(f"Missing decoded SO-101 action fields: {missing_fields}.")
    if chunk_size <= 0:
        raise ValueError(f"Expected a positive chunk_size, got {chunk_size}.")

    arm = _to_numpy(action_chunk["single_arm"], "single_arm")
    gripper = _to_numpy(action_chunk["gripper"], "gripper")
    if arm.ndim != 3 or arm.shape[-1] != 5:
        raise ValueError(
            f"Expected 'single_arm' to have shape [B, H, 5], got {arm.shape}."
        )
    if gripper.ndim != 3 or gripper.shape[-1] != 1:
        raise ValueError(
            f"Expected 'gripper' to have shape [B, H, 1], got {gripper.shape}."
        )
    if arm.shape[:2] != gripper.shape[:2]:
        raise ValueError(
            "Expected 'single_arm' and 'gripper' to share batch and horizon "
            f"dimensions, got {arm.shape[:2]} and {gripper.shape[:2]}."
        )
    if chunk_size > arm.shape[1]:
        raise ValueError(
            f"Requested chunk_size {chunk_size}, but the predicted horizon is "
            f"only {arm.shape[1]}."
        )

    action_array = np.concatenate(
        [arm[:, :chunk_size], gripper[:, :chunk_size]], axis=-1
    )
    if not np.issubdtype(action_array.dtype, np.number) or np.issubdtype(
        action_array.dtype, np.complexfloating
    ):
        raise TypeError(
            "Expected decoded SO-101 actions to contain real numeric motor targets."
        )
    if not np.isfinite(action_array).all():
        raise ValueError("Expected decoded SO-101 actions to contain finite values.")
    return action_array.astype(np.float32, copy=False)


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
    "so100": convert_so101_obs_to_gr00t_format,
    "so101": convert_so101_obs_to_gr00t_format,
}

ACTION_CONVERSION_N1D5 = {
    "libero": convert_to_libero_action_n1d5,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D6 = {
    "libero": convert_to_libero_action_n1d6,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D7 = {
    "libero": convert_to_libero_action_n1d7,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "so100": convert_to_so101_action_n1d7,
    "so101": convert_to_so101_action_n1d7,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Cut and resize the images to the crop size."""
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
