# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression contract for critic-free GR00T GRPO configurations.

The full GR00T runtime is optional in the unit-test environment, so this test
checks the source-level contract that protects actor-only GRPO from invoking a
critic that the configuration deliberately omits.
"""

from pathlib import Path


def test_actor_only_rollout_and_replay_guard_value_head_access():
    source = (
        Path(__file__).parents[2]
        / "rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py"
    ).read_text()

    # Both the rollout sampler and replay forward pass must produce zero
    # placeholders when a GRPO config omits ``add_value_head``.
    assert source.count('if compute_values and hasattr(self, "value_head"):') == 2


def test_reference_kl_replay_does_not_require_rollout_logprobs():
    source = (
        Path(__file__).parents[2]
        / "rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py"
    ).read_text()

    # Reference-KL recomputation provides no ``prev_logprobs``. The current
    # policy-training path still gets the tensor and constructs PPO ratios.
    assert 'prev_logprobs = kwargs.get("prev_logprobs")' in source
    assert 'prev_logprobs.float() if prev_logprobs is not None else None' in source
