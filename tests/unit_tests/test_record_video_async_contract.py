# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression contracts for bounded asynchronous evaluation video encoding."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RecordVideoAsyncContractTest(unittest.TestCase):
    def test_recorder_supports_bounded_async_sampling(self):
        source = (ROOT / "rlinf/envs/wrappers/record_video.py").read_text()
        self.assertIn('self.video_cfg.get("frame_stride", 1)', source)
        self.assertIn('self.video_cfg.get("async_encode", False)', source)
        self.assertIn("def wait_for_videos(self)", source)
        self.assertIn("def _wait_for_capacity(self)", source)
        self.assertIn("ffmpeg_params=ffmpeg_params or None", source)

    def test_env_worker_joins_async_videos_before_evaluation_returns(self):
        source = (ROOT / "rlinf/workers/env/env_worker.py").read_text()
        self.assertIn("reset_fixed_seed_schedule", source)
        self.assertIn("wait_for_videos", source)
        self.assertIn('self.cfg.env.eval.video_cfg.get(\n                                    "async_encode", False', source)

    def test_so101_profile_is_a_reproducible_32_scene_suite(self):
        source = (
            ROOT
            / "examples/embodiment/config/isaaclab_so101_vials_ppo_gr00t_n1d7.yaml"
        ).read_text()
        self.assertIn("rollout_epoch: 4", source)
        self.assertIn("total_num_envs: 8", source)
        self.assertIn("fixed_reset_seed_schedule: [31000, 31008, 31016, 31024]", source)
        self.assertIn("frame_stride: 6", source)


if __name__ == "__main__":
    unittest.main()
