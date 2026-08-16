"""OpenPI data configuration for the SO-101 vial task."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as model_lib
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_vials_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSO101VialsDataConfig(DataConfigFactory):
    default_prompt: str | None = so101_vials_policy.SO101_VIALS_PROMPT

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: model_lib.BaseModelConfig,
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(
                inputs=[so101_vials_policy.SO101VialsRepack()]
            ),
            data_transforms=transforms.Group(
                inputs=[
                    so101_vials_policy.SO101VialsInputs(
                        model_type=model_config.model_type
                    )
                ],
                outputs=[so101_vials_policy.SO101VialsOutputs()],
            ),
            model_transforms=ModelTransformFactory(
                default_prompt=self.default_prompt
            )(model_config),
            action_sequence_keys=("action",),
        )
