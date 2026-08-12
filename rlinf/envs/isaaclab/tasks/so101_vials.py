"""RLinf adapter for NVIDIA's existing SO-101 vials-to-rack task."""

from __future__ import annotations

import os

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.so101_grpo import (
    commanded_release,
    group_image_difference,
    group_max_difference,
    intentional_aligned_release,
    milestone_reward,
    one_shot_event,
    one_shot_success,
    per_environment_boolean,
    repeat_group_leaders,
)
from rlinf.so101_rlft import isaac_radians_to_lerobot, lerobot_actions_to_isaac_radians


def _quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by scalar-first quaternions without importing IsaacLab.

    Ray launches its environment worker with a Python path that can access the
    simulator runtime but not IsaacLab's ``pxr``-dependent utility package.
    These two tiny operations are all the reward adapter needs, so keeping
    them local makes reward computation independent of that import boundary.
    """
    q_vec = quat[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _quat_inv_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return the inverse of (normally unit) scalar-first quaternions."""
    conjugate = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    return conjugate / quat.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)


class _GroupedResetEnv:
    """Make each contiguous GRPO group start from the same physical scene.

    RLinf's GRPO advantage function reshapes contiguous trajectories into
    groups. IsaacLab normally randomizes every vectorized environment
    independently, which makes those comparisons invalid. This proxy uses
    IsaacLab's supported scene state API to copy each group leader to the rest
    of its group after the normal reset/event pipeline has run.
    """

    def __init__(self, env, group_size: int):
        self._env = env
        self._group_size = group_size
        self._validated = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *, seed=None, env_ids=None):
        if env_ids is not None:
            raise RuntimeError(
                "Grouped GRPO resets require full-batch resets; partial auto-reset "
                "would break the same-initial-state contract."
            )
        _, info = self._env.reset(seed=seed)
        # Workshop reset events write rack/vial poses directly to PhysX after
        # IsaacLab has populated its cached data tensors. Read the authoritative
        # PhysX views, clone leader-relative poses/velocities, and write them
        # back through the public asset API. Generic scene.get_state() would
        # otherwise return stale defaults for these four rigid objects.
        relative_rigid_state = {}
        env_origins = self._env.scene.env_origins
        for asset_name in ("vial_1", "vial_2", "vial_3", "rack_left"):
            asset = self._env.scene[asset_name]
            pose = asset.root_physx_view.get_transforms().clone()
            # PhysX returns xyzw; the public asset API expects wxyz.
            pose[:, 3:7] = pose[:, (6, 3, 4, 5)]
            pose[:, :3] -= env_origins
            velocity = asset.root_physx_view.get_velocities().clone()
            grouped_pose = repeat_group_leaders(pose, self._group_size)
            grouped_velocity = repeat_group_leaders(velocity, self._group_size)
            world_pose = grouped_pose.clone()
            world_pose[:, :3] += env_origins
            asset.write_root_pose_to_sim(world_pose)
            asset.write_root_velocity_to_sim(grouped_velocity)
            relative_rigid_state[asset_name] = {
                "root_pose": grouped_pose,
                "root_velocity": grouped_velocity,
            }

        self._env.sim.forward()
        if self._env.sim.has_rtx_sensors() and self._env.cfg.num_rerenders_on_reset > 0:
            for _ in range(self._env.cfg.num_rerenders_on_reset):
                self._env.sim.render()
        obs = self._env.observation_manager.compute(update_history=True)
        max_difference = group_max_difference(relative_rigid_state, self._group_size)
        if max_difference > 1e-5:
            raise RuntimeError(
                "Grouped reset failed same-state validation: "
                f"max physical-state difference={max_difference:.3e}"
            )
        proprio_difference = group_max_difference(
            obs["policy"]["joint_pos_obs"], self._group_size
        )
        image_stats = [
            group_image_difference(obs["visual"][name], self._group_size)
            for name in ("rgb_external_D455", "rgb_ego")
        ]
        visual_mean = max(item[0] for item in image_stats)
        visual_max = max(item[1] for item in image_stats)
        visual_fraction = max(item[2] for item in image_stats)
        if proprio_difference > 1e-5:
            debug_path = os.getenv("SO101_GRPO_DEBUG_PATH")
            if debug_path:
                torch.save(
                    {
                        "external": obs["visual"]["rgb_external_D455"].detach().cpu(),
                        "wrist": obs["visual"]["rgb_ego"].detach().cpu(),
                        "proprio": obs["policy"]["joint_pos_obs"].detach().cpu(),
                    },
                    debug_path,
                )
            raise RuntimeError(
                "Grouped reset produced non-equivalent proprioception: "
                f"max diff={proprio_difference:.3e}"
            )
        # Tiled RTX cameras render each otherwise-identical environment with
        # independent sampling/texture noise. GRPO requires the same initial
        # policy context, so return the leader's exact initial observation to
        # every member. Only reset observations are copied; later observations
        # remain independent as stochastic actions make trajectories diverge.
        obs = repeat_group_leaders(obs, self._group_size)
        exact_observation_difference = group_max_difference(obs, self._group_size)
        if exact_observation_difference != 0.0:
            raise RuntimeError(
                "Failed to broadcast exact grouped initial observations: "
                f"max diff={exact_observation_difference:.3e}"
            )
        if not self._validated:
            print(
                "[GRPO] validated same initial physical state and observation "
                f"within contiguous groups of {self._group_size} "
                f"(state {max_difference:.3e}; pre-broadcast RGB "
                f"mean/max/frac>10 {visual_mean:.3e}/{visual_max:.3e}/"
                f"{visual_fraction:.3e}; returned observation exact)"
            )
            self._validated = True
        return obs, info


class IsaaclabSO101VialsEnv(IsaaclabBaseEnv):
    """Expose the workshop task in RLInf's two-camera policy schema."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def reset(self, seed=None, env_ids=None):
        """Reset, optionally replaying the exact configured evaluation scenes."""
        if bool(self.cfg.get("fixed_reset_seed", False)):
            if env_ids is not None:
                raise RuntimeError(
                    "fixed_reset_seed requires full-batch evaluation resets"
                )
            seed = int(self.cfg.seed)
        return super().reset(seed=seed, env_ids=env_ids)

    def _make_env_function(self):
        def make_env_isaaclab():
            os.environ.pop("DISPLAY", None)
            from isaaclab.app import AppLauncher

            sim_app = AppLauncher({"headless": True, "enable_cameras": True}).app
            # The workshop registers its Gymnasium task on import. Import only
            # after AppLauncher initializes Isaac Sim.
            import sim_to_real_so101.tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg

            env_cfg = parse_env_cfg(
                self.isaaclab_env_id,
                device="cuda:0",
                num_envs=self.cfg.init_params.num_envs,
                # Preserve the prior working default. Some multi-GPU Isaac
                # hosts fail in PhysX Fabric's direct-GPU startup path; set
                # SO101_ISAAC_USE_FABRIC=0 to use the supported non-Fabric
                # transport without changing task physics or observations.
                use_fabric=os.getenv("SO101_ISAAC_USE_FABRIC", "1") == "1",
            )
            env_cfg.seed = self.seed
            env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            env_cfg.scene.camera_external_D455.height = self.cfg.init_params.external_cam.height
            env_cfg.scene.camera_external_D455.width = self.cfg.init_params.external_cam.width
            env_cfg.scene.camera_external_D455.data_types = ["rgb"]
            env_cfg.scene.camera_ego.height = self.cfg.init_params.wrist_cam.height
            env_cfg.scene.camera_ego.width = self.cfg.init_params.wrist_cam.width
            env_cfg.scene.camera_ego.data_types = ["rgb"]
            # Isaac Sim 5 rejects several RTX translucency carb settings that
            # were valid in the workshop's original simulator release. They
            # are cosmetic and this task uses opaque RGB cameras, so keep
            # the renderer compatible without altering scene physics/reward.
            render_cfg = env_cfg.sim.render
            render_cfg.enable_translucency = False
            carb_settings = getattr(render_cfg, "carb_settings", None)
            if isinstance(carb_settings, dict):
                for key in tuple(carb_settings):
                    if key.startswith("rtx.translucency."):
                        carb_settings.pop(key)
            # Policies consume only these two RGB streams. The workshop's base
            # observation group also declares depth and segmentation terms;
            # remove those terms whenever their camera outputs are omitted,
            # otherwise IsaacLab tries to resolve a non-existent ``depth``
            # tensor while constructing the observation manager.
            visual_cfg = env_cfg.observations.visual
            visual_cfg.depth_ego = None
            visual_cfg.instance_id_seg_ego = None
            visual_cfg.depth_external_D455 = None
            visual_cfg.instance_id_seg_external_D455 = None
            group_size = int(self.cfg.get("group_size", 1))
            if group_size > 1:
                # These reset events mutate USD camera/light/material state,
                # which IsaacLab's scene state snapshot intentionally does not
                # include. Leaving them enabled gives nominally grouped GRPO
                # trajectories different pixels. Keep physical rack/vial
                # randomization (then clone it per group), while holding visual
                # calibration fixed for valid within-group comparisons.
                for event_name in (
                    "reset_lightbox_light_exposure",
                    "reset_mat_rotation",
                    "reset_camera_ego_fov",
                    "reset_camera_external_pose",
                    "reset_set_robot_visual_material",
                    "reset_sky_light",
                ):
                    if hasattr(env_cfg.events, event_name):
                        setattr(env_cfg.events, event_name, None)
            env = gym.make(self.isaaclab_env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped

            def get_rlinf_task_state():
                """Return reward-only simulator state through RLinf's IPC boundary.

                ``IsaaclabBaseEnv`` lives in the parent process, while the
                native Isaac scene lives in ``SubProcIsaacLabEnv``.  Keep this
                callback on the native environment; ``venv.py`` attaches its
                result to each step's info dict through RLInf's optional hook.
                """
                contact_sensor = env.scene["contact_grasp"]
                contact_norm = torch.linalg.vector_norm(
                    contact_sensor.data.force_matrix_w, dim=-1
                ).sum(dim=1)
                rack = env.scene["rack_left"]
                return {
                    "contact_norm": contact_norm,
                    "rack_pos_w": rack.data.root_pos_w,
                    "rack_quat_w": rack.data.root_quat_w,
                    "vial_pos_w": torch.stack(
                        [env.scene[name].data.root_pos_w for name in ("vial_1", "vial_2", "vial_3")],
                        dim=1,
                    ),
                    "vial_quat_w": torch.stack(
                        [env.scene[name].data.root_quat_w for name in ("vial_1", "vial_2", "vial_3")],
                        dim=1,
                    ),
                }

            env.get_rlinf_task_state = get_rlinf_task_state
            if group_size > 1:
                if self.cfg.init_params.num_envs % group_size:
                    raise ValueError(
                        f"num_envs={self.cfg.init_params.num_envs} must be divisible "
                        f"by GRPO group_size={group_size}"
                    )
                env = _GroupedResetEnv(env, group_size)
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        return {
            "main_images": obs["visual"]["rgb_external_D455"],
            "wrist_images": obs["visual"]["rgb_ego"],
            "states": isaac_radians_to_lerobot(obs["policy"]["joint_pos_obs"]),
            "task_descriptions": [self.task_description] * self.num_envs,
        }

    def _target_vial_slot_state(self, grasped: torch.Tensor, task_state: dict):
        """Return target-contact and slot-alignment state from simulator truth.

        The target is the vial currently held with the largest jaw-contact
        force.  When contact disappears, retain its identity long enough to
        classify the release.  Alignment uses the workshop asset's actual
        rack-local hole centers and vial-bottom offset; it intentionally does
        not require release or settling, which are later milestones.
        """
        reward_cfg = self._reward_shaping_cfg
        if task_state is None:
            raise RuntimeError(
                "Missing native SO-101 task state. Ensure SubProcIsaacLabEnv "
                "forwards get_rlinf_task_state()."
            )
        contact_norm = task_state["contact_norm"]
        contact_threshold = float(reward_cfg.get("contact_force_threshold", 0.1))
        contact_now = contact_norm > contact_threshold
        strongest_contact = contact_norm.argmax(dim=1)
        active_contact = contact_now.any(dim=1) & grasped
        self._target_vial_idx = torch.where(
            active_contact, strongest_contact, self._target_vial_idx
        )

        target_valid = self._target_vial_idx >= 0
        target_index = self._target_vial_idx.clamp(min=0).unsqueeze(1)
        target_grasped = contact_now.gather(1, target_index).squeeze(1) & target_valid

        rack_quat_inv = _quat_inv_wxyz(task_state["rack_quat_w"])
        hole_centers = torch.tensor(
            reward_cfg.get(
                "hole_centers_xy",
                (
                    (0.0298317, 0.0298575),
                    (0.0901737, 0.0298575),
                    (0.0901737, 0.0900894),
                    (0.0308227, 0.0900894),
                ),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        bottom_offset = float(reward_cfg.get("vial_bottom_offset", -0.017))
        hole_tolerance = float(reward_cfg.get("hole_center_tolerance", 0.018))
        z_min = float(reward_cfg.get("alignment_bottom_z_min", 0.05))
        z_max = float(reward_cfg.get("alignment_bottom_z_max", 0.20))
        upright_threshold = float(reward_cfg.get("alignment_upright_threshold", 0.7))
        unit_z = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        unit_z[:, 2] = 1.0
        pose_aligned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rack_pos_w = task_state["rack_pos_w"]
        vial_positions = task_state["vial_pos_w"]
        vial_quaternions = task_state["vial_quat_w"]
        for vial_idx in range(3):
            target_mask = self._target_vial_idx == vial_idx
            if not target_mask.any():
                continue
            bottom_w = vial_positions[:, vial_idx] + _quat_apply_wxyz(
                vial_quaternions[:, vial_idx], unit_z * bottom_offset
            )
            bottom_local = _quat_apply_wxyz(
                rack_quat_inv, bottom_w - rack_pos_w
            )
            hole_distance = torch.linalg.vector_norm(
                bottom_local[:, None, :2] - hole_centers[None], dim=-1
            ).min(dim=1).values
            vial_up_w = _quat_apply_wxyz(vial_quaternions[:, vial_idx], unit_z)
            aligned = (
                (hole_distance <= hole_tolerance)
                & (bottom_local[:, 2] >= z_min)
                & (bottom_local[:, 2] <= z_max)
                & (torch.abs(vial_up_w[:, 2]) >= upright_threshold)
            )
            pose_aligned |= target_mask & aligned
        return target_grasped, pose_aligned

    def _init_metrics(self):
        super()._init_metrics()
        device = self.device
        self._reward_shaping_cfg = self.cfg.get("reward_shaping", {})
        self._reward_shaping_enabled = bool(
            self._reward_shaping_cfg.get("enabled", False)
        )
        self._success_rewarded = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._grasp_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._grasp_once = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._grasp_lost_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._release_once = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._commanded_open_while_grasped_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._max_gripper_command_while_grasped = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._target_vial_idx = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=device
        )
        self._target_grasp_previous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._slot_aligned_previous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._slot_aligned_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._aligned_release_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._placement_candidate_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._first_grasp_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_grasp_lost_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_release_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_slot_aligned_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_aligned_release_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_candidate_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._first_success_step = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=device
        )
        self._alignment_reward_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._grasp_reward_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._aligned_release_reward_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._strict_success_reward_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._early_success_bonus_total = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if env_idx is None:
            env_idx = slice(None)
        self._success_rewarded[env_idx] = False
        self._grasp_previous[env_idx] = False
        self._grasp_once[env_idx] = False
        self._grasp_lost_once[env_idx] = False
        self._release_once[env_idx] = False
        self._commanded_open_while_grasped_once[env_idx] = False
        self._max_gripper_command_while_grasped[env_idx] = 0.0
        self._target_vial_idx[env_idx] = -1
        self._target_grasp_previous[env_idx] = False
        self._slot_aligned_previous[env_idx] = False
        self._slot_aligned_once[env_idx] = False
        self._aligned_release_once[env_idx] = False
        self._placement_candidate_once[env_idx] = False
        self._first_grasp_step[env_idx] = -1
        self._first_grasp_lost_step[env_idx] = -1
        self._first_release_step[env_idx] = -1
        self._first_slot_aligned_step[env_idx] = -1
        self._first_aligned_release_step[env_idx] = -1
        self._first_candidate_step[env_idx] = -1
        self._first_success_step[env_idx] = -1
        self._alignment_reward_total[env_idx] = 0.0
        self._grasp_reward_total[env_idx] = 0.0
        self._aligned_release_reward_total[env_idx] = 0.0
        self._strict_success_reward_total[env_idx] = 0.0
        self._early_success_bonus_total[env_idx] = 0.0

    def step(self, actions=None, auto_reset=True):
        """Apply one-shot task milestones around the native placement terminal."""
        if actions is None:
            raise ValueError("SO-101 RLinf environment requires a six-joint action.")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.shape[-1] != 6:
            raise ValueError(f"Expected six-joint LeRobot action, got {tuple(actions.shape)}")
        semantic_actions = actions.detach().to(device=self.device, dtype=torch.float32)
        # Policies predict LeRobot SO-101 semantic values, whereas the workshop's
        # native Isaac environment expects articulation targets in radians.
        isaac_actions = lerobot_actions_to_isaac_radians(actions.detach().cpu().numpy())
        actions = torch.from_numpy(isaac_actions).to(device=self.device, dtype=torch.float32)
        obs, _native_reward, terminations, truncations, _native_infos = self.env.step(actions)
        terminations = terminations.clone()
        truncations = truncations.clone()
        # In this pinned Eval task, successful placement is the only native
        # termination; timeout is a truncation. The workshop confirmation term
        # stays true after its 25-frame debounce. Emit a single reward pulse so
        # success timing affects return without rewarding the remainder of an
        # already-completed action chunk multiple times.
        placement_terminations = per_environment_boolean(
            terminations, self.num_envs, "terminations"
        )
        new_success, self._success_rewarded = one_shot_success(
            placement_terminations, self._success_rewarded
        )

        subtask_terms = obs.get("subtask_terms", {})
        grasped = subtask_terms.get("vial_grasped")
        if grasped is None:
            grasped = torch.zeros_like(placement_terminations)
        grasped = per_environment_boolean(grasped, self.num_envs, "vial_grasped")
        placement_candidate = subtask_terms.get("vial_placed")
        if placement_candidate is None:
            placement_candidate = placement_terminations
        placement_candidate = per_environment_boolean(
            placement_candidate, self.num_envs, "vial_placed"
        )
        new_grasp = grasped & ~self._grasp_previous
        first_grasp, self._grasp_once = one_shot_event(grasped, self._grasp_once)
        # Contact loss is useful for diagnosing fumbles, but is not a release:
        # one-frame contact flicker and rack collisions also create this edge.
        new_grasp_lost = ~grasped & self._grasp_previous
        task_state = _native_infos.pop("_rlinf_task_state", None)
        target_grasped, slot_pose_aligned = self._target_vial_slot_state(grasped, task_state)
        slot_aligned = target_grasped & slot_pose_aligned
        new_slot_aligned, self._slot_aligned_once = one_shot_event(
            slot_aligned, self._slot_aligned_once
        )
        gripper_open = semantic_actions[:, 5] >= float(
            self._reward_shaping_cfg.get("gripper_open_threshold", 60.0)
        )
        new_release = commanded_release(
            self._target_grasp_previous,
            target_grasped,
            gripper_open,
        )
        aligned_release = intentional_aligned_release(
            self._target_grasp_previous,
            target_grasped,
            gripper_open,
            slot_pose_aligned,
            self._slot_aligned_previous,
        )
        new_aligned_release, self._aligned_release_once = one_shot_event(
            aligned_release, self._aligned_release_once
        )
        self._grasp_lost_once |= new_grasp_lost
        self._release_once |= new_release
        open_while_grasped = target_grasped & gripper_open
        self._commanded_open_while_grasped_once |= open_while_grasped
        self._max_gripper_command_while_grasped = torch.where(
            target_grasped | self._target_grasp_previous,
            torch.maximum(
                self._max_gripper_command_while_grasped,
                semantic_actions[:, 5],
            ),
            self._max_gripper_command_while_grasped,
        )
        self._placement_candidate_once |= placement_candidate
        obs = self._wrap_obs(obs)
        self._elapsed_steps += 1
        current_step = self._elapsed_steps.to(dtype=torch.int32)

        reward_cfg = self._reward_shaping_cfg
        if self._reward_shaping_enabled:
            step_reward, timing_bonus = milestone_reward(
                first_grasp,
                new_slot_aligned,
                new_aligned_release,
                new_success,
                current_step,
                int(self.cfg.max_episode_steps),
                grasp_weight=float(reward_cfg.get("grasp_reward", 0.0)),
                alignment_weight=float(reward_cfg.get("slot_alignment_reward", 0.05)),
                release_weight=float(reward_cfg.get("aligned_release_reward", 0.15)),
                success_weight=float(reward_cfg.get("strict_success_reward", 1.0)),
                early_success_bonus=float(reward_cfg.get("early_success_bonus", 0.20)),
            )
            grasp_component = first_grasp.to(torch.float32) * float(
                reward_cfg.get("grasp_reward", 0.0)
            )
            alignment_component = new_slot_aligned.to(torch.float32) * float(
                reward_cfg.get("slot_alignment_reward", 0.05)
            )
            release_component = new_aligned_release.to(torch.float32) * float(
                reward_cfg.get("aligned_release_reward", 0.15)
            )
            success_component = new_success.to(torch.float32) * float(
                reward_cfg.get("strict_success_reward", 1.0)
            )
        else:
            step_reward = new_success.to(dtype=torch.float32)
            timing_bonus = torch.zeros_like(step_reward)
            grasp_component = torch.zeros_like(step_reward)
            alignment_component = torch.zeros_like(step_reward)
            release_component = torch.zeros_like(step_reward)
            success_component = step_reward
        self._grasp_reward_total += grasp_component
        self._alignment_reward_total += alignment_component
        self._aligned_release_reward_total += release_component
        self._strict_success_reward_total += success_component
        self._early_success_bonus_total += timing_bonus

        for first_step, event in (
            (self._first_grasp_step, new_grasp),
            (self._first_grasp_lost_step, new_grasp_lost),
            (self._first_release_step, new_release),
            (self._first_slot_aligned_step, new_slot_aligned),
            (self._first_aligned_release_step, new_aligned_release),
            (self._first_candidate_step, placement_candidate),
            (self._first_success_step, new_success),
        ):
            first_step[(first_step < 0) & event] = current_step[(first_step < 0) & event]
        self._grasp_previous = grasped
        self._target_grasp_previous = target_grasped
        self._slot_aligned_previous = slot_pose_aligned
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        infos = self._record_metrics(step_reward, placement_terminations, {})
        # IsaaclabBaseEnv infers success_once from ``step_reward > 0``. That is
        # equivalent for the legacy sparse reward, but milestone shaping would
        # otherwise mislabel slot alignment as task success. Keep the public
        # task metric tied exclusively to the native strict-placement latch.
        infos["episode"]["success_once"] = self._success_rewarded.clone()
        infos["episode"]["success_at_end"] = placement_terminations
        infos["episode"]["grasp_once"] = self._grasp_once.clone()
        infos["episode"]["grasp_lost_once"] = self._grasp_lost_once.clone()
        # A release is target-contact loss while the policy commands the
        # gripper open.  It deliberately excludes uncommanded fumbles.
        infos["episode"]["release_once"] = self._release_once.clone()
        infos["episode"]["commanded_open_while_grasped_once"] = (
            self._commanded_open_while_grasped_once.clone()
        )
        infos["episode"]["max_gripper_command_while_grasped"] = (
            self._max_gripper_command_while_grasped.clone()
        )
        infos["episode"]["slot_aligned_once"] = self._slot_aligned_once.clone()
        infos["episode"]["intentional_aligned_release_once"] = (
            self._aligned_release_once.clone()
        )
        infos["episode"]["placement_candidate_once"] = self._placement_candidate_once.clone()
        infos["episode"]["first_grasp_step"] = self._first_grasp_step.clone()
        infos["episode"]["first_grasp_lost_step"] = (
            self._first_grasp_lost_step.clone()
        )
        infos["episode"]["first_release_step"] = self._first_release_step.clone()
        infos["episode"]["first_slot_aligned_step"] = (
            self._first_slot_aligned_step.clone()
        )
        infos["episode"]["first_intentional_aligned_release_step"] = (
            self._first_aligned_release_step.clone()
        )
        infos["episode"]["first_placement_candidate_step"] = (
            self._first_candidate_step.clone()
        )
        infos["episode"]["first_success_step"] = self._first_success_step.clone()
        infos["episode"]["slot_alignment_reward"] = (
            self._alignment_reward_total.clone()
        )
        infos["episode"]["grasp_reward"] = self._grasp_reward_total.clone()
        infos["episode"]["aligned_release_reward"] = (
            self._aligned_release_reward_total.clone()
        )
        infos["episode"]["strict_success_reward"] = (
            self._strict_success_reward_total.clone()
        )
        infos["episode"]["early_success_bonus"] = (
            self._early_success_bonus_total.clone()
        )
        if self.ignore_terminations:
            terminations[:] = False
        # Apply the evaluation mask before choosing which environments to reset.
        # Previously this used the pre-mask terminal and immediately reset a
        # successful eval episode despite ignore_terminations=True.
        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, step_reward, terminations, truncations, infos
