"""OpenPI transforms for the six-axis SO-101 vial task.

The task has two LeRobot dataset producers:

* teleoperation data uses ``observation.images.external_D455`` and
  ``observation.images.ego``;
* RLinf's episode collector uses the compact ``image`` / ``wrist_image`` /
  ``state`` schema.

Keeping the aliases at the repack boundary lets online inference, value-model
training, advantage labeling, and CFG training share one canonical OpenPI
observation contract.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as model_lib

SO101_VIALS_PROMPT = "Pick up the vial and place it in the rack"


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, received shape {image.shape}")
    return image


def _lookup(data: dict[str, Any], paths: Iterable[str]) -> Any:
    """Read a flattened LeRobot key or a nested equivalent."""
    for path in paths:
        if path in data:
            return data[path]
        value: Any = data
        try:
            for part in path.split("."):
                value = value[part]
        except (KeyError, TypeError):
            continue
        return value
    raise KeyError(f"None of the SO-101 aliases are present: {tuple(paths)}")


@dataclasses.dataclass(frozen=True)
class SO101VialsRepack(transforms.DataTransformFn):
    """Normalize teleoperation, rollout, and live observations."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        packed = {
            "observation/image": _lookup(
                data,
                (
                    "observation/image",
                    "observation.images.external_D455",
                    "observation.images.image",
                    "image",
                ),
            ),
            "observation/wrist_image": _lookup(
                data,
                (
                    "observation/wrist_image",
                    "observation.images.ego",
                    "observation.images.wrist_image",
                    "wrist_image",
                ),
            ),
            "observation/state": _lookup(
                data,
                ("observation/state", "observation.state", "state"),
            ),
        }
        for action_key in ("actions", "action"):
            if action_key in data:
                packed["actions"] = data[action_key]
                break
        for prompt_key in ("prompt", "task"):
            if prompt_key in data:
                packed["prompt"] = data[prompt_key]
                break
        return packed


@dataclasses.dataclass(frozen=True)
class SO101VialsInputs(transforms.DataTransformFn):
    """Convert the canonical SO-101 record into OpenPI's observation schema."""

    model_type: model_lib.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        external = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        inputs: dict[str, Any] = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": external,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(external),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101VialsOutputs(transforms.DataTransformFn):
    """Return only the six semantic SO-101 joint commands."""

    def __call__(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"actions": np.asarray(data["actions"][:, :6])}
