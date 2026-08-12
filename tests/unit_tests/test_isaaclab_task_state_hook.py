from __future__ import annotations

import unittest
from pathlib import Path


class IsaacLabTaskStateHookTest(unittest.TestCase):
    def test_subprocess_hook_is_generic_and_optional(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/isaaclab/venv.py"
        ).read_text()
        self.assertIn('getattr(isaac_env, "get_rlinf_task_state", None)', source)
        self.assertIn('info["_rlinf_task_state"]', source)
        self.assertNotIn("get_so101_task_state", source)

    def test_so101_uses_an_explicit_task_state_wrapper(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/isaaclab/tasks/so101_vials.py"
        ).read_text()
        self.assertIn("class _SO101TaskStateEnv", source)
        self.assertIn("env = _SO101TaskStateEnv(env)", source)
        self.assertNotIn("env.get_rlinf_task_state =", source)


if __name__ == "__main__":
    unittest.main()
