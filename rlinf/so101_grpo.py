"""Small, testable contracts used by the SO-101 GRPO environment overlay."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def per_environment_boolean(value: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    """Normalize an Isaac term to one Boolean per vectorized environment.

    IsaacLab observation terms are not required to squeeze singleton feature
    dimensions.  In particular, the workshop exposes ``vial_grasped`` as
    ``[num_envs, 1]`` while termination terms are ``[num_envs]``.  Letting
    those shapes broadcast silently creates an invalid ``[num_envs,
    num_envs]`` event matrix.
    """
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 0 or value.shape[0] != num_envs:
        raise ValueError(
            f"{name} must have leading environment dimension {num_envs}, "
            f"got {tuple(value.shape)}"
        )
    return value.to(dtype=torch.bool).reshape(num_envs, -1).any(dim=-1)


def repeat_group_leaders(value, group_size: int):
    """Repeat each contiguous group's first environment state across its group."""
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if isinstance(value, Mapping):
        return {key: repeat_group_leaders(item, group_size) for key, item in value.items()}
    if not torch.is_tensor(value):
        return value
    if value.ndim == 0:
        return value
    num_envs = value.shape[0]
    if num_envs % group_size:
        raise ValueError(
            f"State batch {num_envs} is not divisible by group_size={group_size}"
        )
    leader_ids = torch.arange(0, num_envs, group_size, device=value.device)
    source_ids = leader_ids.repeat_interleave(group_size)
    return value.index_select(0, source_ids).clone()


def group_max_difference(value, group_size: int) -> float:
    """Return the largest within-group tensor difference in a nested state."""
    if isinstance(value, Mapping):
        return max(
            (group_max_difference(item, group_size) for item in value.values()),
            default=0.0,
        )
    if not torch.is_tensor(value) or value.ndim == 0 or value.numel() == 0:
        return 0.0
    expected = repeat_group_leaders(value, group_size)
    if value.dtype == torch.bool:
        return float((value != expected).any().item())
    return float((value.to(torch.float32) - expected.to(torch.float32)).abs().max().item())


def group_image_difference(value: torch.Tensor, group_size: int) -> tuple[float, float, float]:
    """Return mean, maximum, and >10-level fraction versus each group leader."""
    expected = repeat_group_leaders(value, group_size)
    difference = (value.to(torch.float32) - expected.to(torch.float32)).abs()
    return (
        float(difference.mean().item()),
        float(difference.max().item()),
        float((difference > 10.0).to(torch.float32).mean().item()),
    )


def one_shot_event(event: torch.Tensor, already_seen: torch.Tensor):
    """Return an event's first pulse and its updated per-episode latch."""
    event = event.to(dtype=torch.bool)
    already_seen = already_seen.to(dtype=torch.bool)
    pulse = event & ~already_seen
    return pulse, already_seen | event


def one_shot_success(confirmed: torch.Tensor, already_rewarded: torch.Tensor):
    """Return the first confirmed-success pulse and the updated episode latch."""
    return one_shot_event(confirmed, already_rewarded)


def intentional_aligned_release(
    was_grasped: torch.Tensor,
    grasped_now: torch.Tensor,
    gripper_open_command: torch.Tensor,
    aligned_now: torch.Tensor,
    aligned_previously: torch.Tensor,
) -> torch.Tensor:
    """Detect an actual commanded release while a held vial is over a slot.

    A falling edge in the workshop's contact-based grasp predicate is not
    sufficient: contact flicker and accidental fumbles also create that edge.
    Require the commanded gripper to be open and accept alignment from either
    side of the transition so one simulator frame of falling motion does not
    erase an otherwise valid release.
    """
    tensors = (
        was_grasped,
        grasped_now,
        gripper_open_command,
        aligned_now,
        aligned_previously,
    )
    tensors = tuple(value.to(dtype=torch.bool) for value in tensors)
    was_grasped, grasped_now, gripper_open_command, aligned_now, aligned_previously = (
        tensors
    )
    return (
        was_grasped
        & ~grasped_now
        & gripper_open_command
        & (aligned_now | aligned_previously)
    )


def commanded_release(
    was_grasped: torch.Tensor,
    grasped_now: torch.Tensor,
    gripper_open_command: torch.Tensor,
) -> torch.Tensor:
    """Detect a commanded release, excluding contact flicker and fumbles.

    ``grasped`` is a contact predicate, so its falling edge alone only means
    that the vial stopped touching the jaws.  A release additionally requires
    the policy to command the gripper open on that transition.
    """
    tensors = (was_grasped, grasped_now, gripper_open_command)
    was_grasped, grasped_now, gripper_open_command = (
        value.to(dtype=torch.bool) for value in tensors
    )
    return was_grasped & ~grasped_now & gripper_open_command


def milestone_reward(
    grasp_pulse: torch.Tensor,
    aligned_pulse: torch.Tensor,
    release_pulse: torch.Tensor,
    success_pulse: torch.Tensor,
    current_step: torch.Tensor,
    max_steps: int,
    *,
    grasp_weight: float,
    alignment_weight: float,
    release_weight: float,
    success_weight: float,
    early_success_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose bounded one-shot shaping while preserving success dominance.

    The early bonus is paid only with strict success and decreases linearly to
    zero at the episode horizon.  Reject configurations where a failed
    trajectory could collect as much reward as a successful one.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    weights = (
        grasp_weight,
        alignment_weight,
        release_weight,
        success_weight,
        early_success_bonus,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Milestone rewards must be non-negative, got {weights}")
    if grasp_weight + alignment_weight + release_weight >= success_weight:
        raise ValueError(
            "Strict success must dominate every failed trajectory: "
            "grasp+alignment+release="
            f"{grasp_weight + alignment_weight + release_weight} >= "
            f"success={success_weight}"
        )
    dtype = torch.float32
    device = current_step.device
    success = success_pulse.to(device=device, dtype=dtype)
    remaining_fraction = (
        1.0 - current_step.to(dtype=dtype) / float(max_steps)
    ).clamp(min=0.0, max=1.0)
    timing_bonus = success * remaining_fraction * early_success_bonus
    reward = (
        grasp_pulse.to(device=device, dtype=dtype) * grasp_weight
        + aligned_pulse.to(device=device, dtype=dtype) * alignment_weight
        + release_pulse.to(device=device, dtype=dtype) * release_weight
        + success * success_weight
        + timing_bonus
    )
    return reward, timing_bonus
