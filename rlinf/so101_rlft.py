"""Shared SO-101 semantic-action units for RLInf policy integrations."""

from __future__ import annotations

import numpy as np
import torch

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# This is the workshop's LeRobotSO101Interface mapping. Keep this module and
# the pinned workshop source in lockstep; verify with the unit round-trip test.
JOINT_MINS_DEG = (-110.0, -100.0, -100.0, -95.0, -160.0, -10.0)
JOINT_MAXS_DEG = (110.0, 100.0, 90.0, 95.0, 160.0, 100.0)


def _torch_limits(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.tensor(JOINT_MINS_DEG, device=values.device, dtype=values.dtype)
    maxs = torch.tensor(JOINT_MAXS_DEG, device=values.device, dtype=values.dtype)
    return mins, maxs


def isaac_radians_to_lerobot(values: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., 6]`` Isaac articulation radians to LeRobot values."""
    if values.shape[-1] != 6:
        raise ValueError(f"Expected six SO-101 joints, got {tuple(values.shape)}")
    mins, maxs = _torch_limits(values)
    normalized = (torch.rad2deg(values) - mins) / (maxs - mins)
    result = torch.empty_like(normalized)
    result[..., :5] = normalized[..., :5] * 200.0 - 100.0
    result[..., 5] = normalized[..., 5] * 100.0
    return result


def lerobot_actions_to_isaac_radians(values: np.ndarray) -> np.ndarray:
    """Convert ``[..., 6]`` LeRobot SO-101 action values to Isaac radians."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 6:
        raise ValueError(f"Expected six SO-101 joints, got {values.shape}")
    clipped = values.copy()
    clipped[..., :5] = np.clip(clipped[..., :5], -100.0, 100.0)
    clipped[..., 5] = np.clip(clipped[..., 5], 0.0, 100.0)
    normalized = np.empty_like(clipped)
    normalized[..., :5] = (clipped[..., :5] + 100.0) / 200.0
    normalized[..., 5] = clipped[..., 5] / 100.0
    mins = np.asarray(JOINT_MINS_DEG, dtype=np.float32)
    maxs = np.asarray(JOINT_MAXS_DEG, dtype=np.float32)
    return np.deg2rad(mins + normalized * (maxs - mins)).astype(np.float32)
