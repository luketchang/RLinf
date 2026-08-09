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

import json

import pytest
import torch
from rlinf.utils.ckpt_convertor.gr00t_n1d7._core import (
    LEROBOT_PREFIX,
    LM_HEAD_KEY,
    TIED_EMBED_KEY,
    build_native_processor_files,
    lerobot_to_native_state_dict,
    native_to_lerobot_state_dict,
)


def _deployment_state():
    return {
        f"{LEROBOT_PREFIX}{LM_HEAD_KEY}": torch.arange(6).reshape(2, 3).float(),
        f"{LEROBOT_PREFIX}action_head.weight": torch.ones(2, 2),
    }


def test_state_dict_round_trip_restores_tied_embedding_and_filters_rl_heads():
    deployment = _deployment_state()
    native = lerobot_to_native_state_dict(deployment, dtype=None)

    assert TIED_EMBED_KEY in native
    assert native[TIED_EMBED_KEY].data_ptr() != native[LM_HEAD_KEY].data_ptr()
    native["action_head.value_head.weight"] = torch.ones(1)
    native["action_head.reinflow_explore_noise_net.weight"] = torch.ones(1)

    restored = native_to_lerobot_state_dict(native, deployment)

    assert restored.keys() == deployment.keys()
    for key in deployment:
        torch.testing.assert_close(restored[key], deployment[key])


def test_import_rejects_non_lerobot_key():
    with pytest.raises(ValueError, match="expected prefix"):
        lerobot_to_native_state_dict({"backbone.weight": torch.ones(1)})


def test_export_rejects_missing_unexpected_and_shape_changed_tensors():
    reference = _deployment_state()
    native = lerobot_to_native_state_dict(reference, dtype=None)

    missing = dict(native)
    del missing["action_head.weight"]
    with pytest.raises(ValueError, match="missing deployment"):
        native_to_lerobot_state_dict(missing, reference)

    unexpected = dict(native)
    unexpected["unrecognized.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="Unexpected non-deployment"):
        native_to_lerobot_state_dict(unexpected, reference)

    wrong_shape = dict(native)
    wrong_shape["action_head.weight"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="Shape mismatch"):
        native_to_lerobot_state_dict(wrong_shape, reference)


def test_processor_conversion_preserves_sft_contract(tmp_path):
    lerobot = tmp_path / "lerobot"
    native = tmp_path / "native"
    lerobot.mkdir()
    native.mkdir()
    (lerobot / "config.json").write_text(json.dumps({"use_relative_actions": True}))
    pack = {
        "embodiment_tag": "new_embodiment",
        "embodiment_mapping": {"new_embodiment": 10},
        "modality_config": {
            "state": {"modality_keys": ["single_arm", "gripper"]},
            "action": {"modality_keys": ["single_arm", "gripper"]},
            "video": {"modality_keys": ["ego", "external_D455"]},
        },
        "raw_stats": {"state": {"single_arm": {"mean": [0]}}},
        "max_state_dim": 132,
        "max_action_dim": 132,
        "action_horizon": 40,
        "formalize_language": True,
        "state_dropout_prob": 0.0,
        "use_percentiles": True,
    }
    vlm = {
        "image_crop_size": [230, 230],
        "image_target_size": [256, 256],
        "shortest_image_edge": None,
        "crop_fraction": None,
        "use_albumentations": False,
        "letter_box_transform": False,
    }
    (lerobot / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"registry_name": "groot_n1_7_pack_inputs_v1", "config": pack},
                    {"registry_name": "groot_n1_7_vlm_encode_v1", "config": vlm},
                ]
            }
        )
    )
    (native / "processor_config.json").write_text(
        json.dumps({"processor_class": "Gr00tN1d7Processor", "processor_kwargs": {}})
    )

    processor, statistics, embodiment_ids = build_native_processor_files(
        lerobot, native
    )

    kwargs = processor["processor_kwargs"]
    assert kwargs["modality_configs"]["new_embodiment"]["video"]["modality_keys"] == [
        "ego",
        "external_D455",
    ]
    assert kwargs["max_action_horizon"] == 40
    assert kwargs["use_relative_action"] is True
    assert statistics == {"new_embodiment": pack["raw_stats"]}
    assert embodiment_ids == {"new_embodiment": 10}
