from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "rlinf/so101_grpo.py"
SPEC = importlib.util.spec_from_file_location("so101_grpo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SO101_GRPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SO101_GRPO)

commanded_release = SO101_GRPO.commanded_release
group_image_difference = SO101_GRPO.group_image_difference
group_max_difference = SO101_GRPO.group_max_difference
one_shot_success = SO101_GRPO.one_shot_success
per_environment_boolean = SO101_GRPO.per_environment_boolean
repeat_group_leaders = SO101_GRPO.repeat_group_leaders


class SO101GRPOContractsTest(unittest.TestCase):
    def test_repeats_contiguous_group_leaders(self):
        state = {"scene": {"pose": torch.arange(24).reshape(8, 3)}}
        grouped = repeat_group_leaders(state, 4)
        expected = torch.stack(
            [state["scene"]["pose"][0]] * 4
            + [state["scene"]["pose"][4]] * 4
        )
        torch.testing.assert_close(grouped["scene"]["pose"], expected)
        self.assertEqual(group_max_difference(grouped, 4), 0.0)

    def test_rejects_non_divisible_group_batch(self):
        with self.assertRaises(ValueError):
            repeat_group_leaders(torch.zeros(6, 2), 4)

    def test_group_image_difference_is_not_dominated_by_one_pixel(self):
        images = torch.zeros(4, 4, 4, 3)
        images[1, 0, 0, 0] = 255
        mean, maximum, fraction = group_image_difference(images, 4)
        self.assertLess(mean, 2.0)
        self.assertEqual(maximum, 255.0)
        self.assertLess(fraction, 0.05)

    def test_success_reward_is_one_shot(self):
        latch = torch.tensor([False, False])
        pulses = []
        for confirmed in (
            torch.tensor([False, False]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
            torch.tensor([True, True]),
        ):
            pulse, latch = one_shot_success(confirmed, latch)
            pulses.append(pulse)
        torch.testing.assert_close(
            torch.stack(pulses),
            torch.tensor(
                [[False, False], [True, False], [False, True], [False, False]]
            ),
        )

    def test_release_requires_open_command(self):
        release = commanded_release(
            was_grasped=torch.tensor([True, True, True, False]),
            grasped_now=torch.tensor([False, False, True, False]),
            gripper_open_command=torch.tensor([True, False, True, True]),
        )
        torch.testing.assert_close(
            release, torch.tensor([True, False, False, False])
        )

    def test_event_terms_collapse_to_one_boolean_per_environment(self):
        event = torch.tensor([[False, True], [False, False]])
        torch.testing.assert_close(
            per_environment_boolean(event, 2, "event"),
            torch.tensor([True, False]),
        )


if __name__ == "__main__":
    unittest.main()
