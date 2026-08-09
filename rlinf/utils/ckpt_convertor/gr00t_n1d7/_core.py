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

"""Strict LeRobot <-> RLinf conversion helpers for GR00T N1.7."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from collections.abc import Mapping
from typing import Any

import torch

LEROBOT_PREFIX = "_groot_model."
TIED_EMBED_KEY = "backbone.model.model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "backbone.model.lm_head.weight"
RL_ONLY_PREFIXES = (
    "action_head.value_head.",
    "action_head.reinflow_explore_noise_net.",
)
WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
    "model.",
)
WEIGHT_GLOBS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)


def read_json(path: str | pathlib.Path) -> Any:
    """Read a JSON file."""
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | pathlib.Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_safetensors_file(path: pathlib.Path) -> dict[str, torch.Tensor]:
    import safetensors.torch

    return safetensors.torch.load_file(str(path), device="cpu")


def load_safetensors_checkpoint(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load a single-file or sharded safetensors checkpoint onto CPU."""
    path = pathlib.Path(path)
    if path.is_file() and path.suffix == ".safetensors":
        return _load_safetensors_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    single = path / "model.safetensors"
    if single.is_file():
        return _load_safetensors_file(single)

    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {path}"
        )
    index = read_json(index_path)
    shard_names = sorted(set(index["weight_map"].values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = _load_safetensors_file(path / shard_name)
        overlap = state_dict.keys() & shard.keys()
        if overlap:
            raise ValueError(
                f"Duplicate keys across safetensors shards: {sorted(overlap)}"
            )
        state_dict.update(shard)
    expected = set(index["weight_map"])
    if set(state_dict) != expected:
        missing = sorted(expected - set(state_dict))
        extra = sorted(set(state_dict) - expected)
        raise ValueError(f"Invalid safetensors index: missing={missing}, extra={extra}")
    return state_dict


def load_rlinf_state_dict(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load a consolidated RLinf state dict or native safetensors directory."""
    path = pathlib.Path(path)
    candidates = (
        path / "actor/model_state_dict/full_weights.pt",
        path / "model_state_dict/full_weights.pt",
        path / "full_weights.pt",
    )
    if path.is_file() and path.suffix == ".pt":
        candidates = (path,)
    for candidate in candidates:
        if candidate.is_file():
            loaded = torch.load(candidate, map_location="cpu", weights_only=True)
            return dict(_unwrap_state_dict(loaded))
    return load_safetensors_checkpoint(path)


def _unwrap_state_dict(value: Any) -> Mapping[str, torch.Tensor]:
    current = value
    for _ in range(5):
        if (
            isinstance(current, Mapping)
            and current
            and all(torch.is_tensor(item) for item in current.values())
        ):
            return current
        if not isinstance(current, Mapping):
            break
        for key in ("model", "state_dict", "module"):
            child = current.get(key)
            if isinstance(child, Mapping):
                current = child
                break
        else:
            break
    raise TypeError(f"Could not extract a tensor state dict from {type(value)!r}")


def strip_runtime_prefixes(key: str) -> str:
    """Remove only well-known FSDP/compile wrappers from a state-dict key."""
    bare = key
    while True:
        for prefix in WRAPPER_PREFIXES:
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
                break
        else:
            return bare


def lerobot_to_native_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Remove LeRobot's policy wrapper and restore the tied embedding alias."""
    native: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        if not source_key.startswith(LEROBOT_PREFIX):
            raise ValueError(
                f"Unexpected LeRobot key {source_key!r}; "
                f"expected prefix {LEROBOT_PREFIX!r}"
            )
        key = source_key[len(LEROBOT_PREFIX) :]
        if key in native:
            raise ValueError(f"Duplicate native key after prefix removal: {key}")
        tensor = tensor.detach().cpu()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        native[key] = tensor.contiguous()

    if TIED_EMBED_KEY not in native:
        if LM_HEAD_KEY not in native:
            raise ValueError(
                "Cannot restore tied embedding: checkpoint has neither "
                f"{TIED_EMBED_KEY} "
                f"nor {LM_HEAD_KEY}"
            )
        native[TIED_EMBED_KEY] = native[LM_HEAD_KEY].clone()
    return native


def native_to_lerobot_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    reference_state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Create an exact LeRobot state dict using a deployment checkpoint contract."""
    native: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        key = strip_runtime_prefixes(source_key)
        if key.startswith(LEROBOT_PREFIX):
            key = key[len(LEROBOT_PREFIX) :]
        if key in native:
            raise ValueError(f"Duplicate native key after wrapper removal: {key}")
        native[key] = tensor

    expected_native = {
        key[len(LEROBOT_PREFIX) :]: (key, tensor)
        for key, tensor in reference_state_dict.items()
        if key.startswith(LEROBOT_PREFIX)
    }
    if len(expected_native) != len(reference_state_dict):
        invalid = sorted(
            key for key in reference_state_dict if not key.startswith(LEROBOT_PREFIX)
        )
        raise ValueError(f"Reference is not a LeRobot GR00T checkpoint: {invalid}")

    missing = sorted(set(expected_native) - set(native))
    if missing:
        raise ValueError(f"RLinf checkpoint is missing deployment tensors: {missing}")

    allowed_extras = {TIED_EMBED_KEY}
    extras = sorted(
        key
        for key in set(native) - set(expected_native)
        if key not in allowed_extras
        and not any(key.startswith(prefix) for prefix in RL_ONLY_PREFIXES)
    )
    if extras:
        raise ValueError(
            f"Unexpected non-deployment tensors in RLinf checkpoint: {extras}"
        )

    converted: dict[str, torch.Tensor] = {}
    for native_key, (deployment_key, reference_tensor) in expected_native.items():
        tensor = native[native_key].detach().cpu()
        if tuple(tensor.shape) != tuple(reference_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {native_key}: checkpoint={tuple(tensor.shape)}, "
                f"reference={tuple(reference_tensor.shape)}"
            )
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        converted[deployment_key] = tensor.contiguous()
    return converted


def _find_processor_step(processor: Mapping[str, Any], registry_name: str) -> dict:
    matches = [
        step
        for step in processor.get("steps", [])
        if step.get("registry_name") == registry_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {registry_name!r} processor step, "
            f"found {len(matches)}"
        )
    return matches[0]["config"]


def build_native_processor_files(
    lerobot_dir: str | pathlib.Path,
    native_reference_dir: str | pathlib.Path,
) -> tuple[dict, dict, dict]:
    """Translate LeRobot processor state into official GR00T processor files."""
    lerobot_dir = pathlib.Path(lerobot_dir)
    native_reference_dir = pathlib.Path(native_reference_dir)
    preprocessor = read_json(lerobot_dir / "policy_preprocessor.json")
    pack = _find_processor_step(preprocessor, "groot_n1_7_pack_inputs_v1")
    vlm = _find_processor_step(preprocessor, "groot_n1_7_vlm_encode_v1")
    policy_config = read_json(lerobot_dir / "config.json")
    processor_config = read_json(native_reference_dir / "processor_config.json")
    kwargs = processor_config["processor_kwargs"]

    embodiment_tag = pack["embodiment_tag"]
    kwargs["modality_configs"] = {embodiment_tag: pack["modality_config"]}
    for key in (
        "max_state_dim",
        "max_action_dim",
        "formalize_language",
        "state_dropout_prob",
        "use_percentiles",
    ):
        kwargs[key] = pack[key]
    kwargs["use_mean_std"] = not pack["use_percentiles"]
    kwargs["max_action_horizon"] = pack["action_horizon"]
    kwargs["use_relative_action"] = bool(policy_config["use_relative_actions"])
    for key in (
        "image_crop_size",
        "image_target_size",
        "shortest_image_edge",
        "crop_fraction",
        "use_albumentations",
        "letter_box_transform",
    ):
        kwargs[key] = vlm[key]

    statistics = {embodiment_tag: pack["raw_stats"]}
    embodiment_ids = dict(pack["embodiment_mapping"])
    if embodiment_tag not in embodiment_ids:
        raise ValueError(f"No ID for embodiment {embodiment_tag!r}")
    return processor_config, statistics, embodiment_ids


def copy_metadata_tree(
    source: str | pathlib.Path, destination: str | pathlib.Path
) -> None:
    """Copy checkpoint metadata while excluding model weights and old manifests."""
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    excluded = {".cache", "conversion_manifest.json"}
    for item in source.iterdir():
        if item.name in excluded or any(
            item.match(pattern) for pattern in WEIGHT_GLOBS
        ):
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Return a stable key/dtype/shape digest for conversion manifests."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
    return digest.hexdigest()
