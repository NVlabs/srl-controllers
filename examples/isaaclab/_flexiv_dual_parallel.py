# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Parallel cooperative dual-Flexiv native controller rollouts."""

from __future__ import annotations

# Standard Library
import dataclasses
from typing import Any


def _make_scene(left_cfg: Any, right_cfg: Any, num_envs: int) -> Any:
    """Create two independently actuated Flexiv articulations in every environment."""
    # Third Party
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    @configclass
    class DualFlexivSceneCfg(InteractiveSceneCfg):
        ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        left = left_cfg
        right = right_cfg
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
        )

    return InteractiveScene(
        DualFlexivSceneCfg(num_envs=num_envs, env_spacing=2.0, lazy_sensor_update=True)
    )


def _circle_reference(elapsed_s: float) -> tuple[tuple[float, ...], ...]:
    """Sample the hardware arm-1-base circular position, velocity, and acceleration."""
    # Standard Library
    import math

    u = min(1.0, elapsed_s / 2.0)
    omega = 2.0 * math.pi * 0.2
    phase = omega * (2.0 * (u**3 - 0.5 * u**4) + max(0.0, elapsed_s - 2.0))
    speed = omega * (3.0 * u * u - 2.0 * u**3) if u < 1.0 else omega
    acceleration = omega / 2.0 * (6.0 * u - 6.0 * u * u) if u < 1.0 else 0.0
    sine, cosine = math.sin(phase), math.cos(phase)
    return (
        (0.0, 0.1 * sine, 0.1 * (cosine - 1.0)),
        (0.0, 0.1 * cosine * speed, -0.1 * sine * speed),
        (
            0.0,
            0.1 * (-sine * speed * speed + cosine * acceleration),
            -0.1 * (cosine * speed * speed + sine * acceleration),
        ),
    )


def _fill_pose(torch: Any, position: Any, quaternion: Any, output: Any) -> None:
    # Third Party
    from isaaclab.utils import math as math_utils

    output.zero_()
    output[:, :3, :3].copy_(math_utils.matrix_from_quat(quaternion))
    output[:, :3, 3].copy_(position)
    output[:, 3, 3] = 1.0


def _resolve_frames(
    api: Any,
    arm1_t_arm2: tuple[float, ...] | None,
    main_frame: str,
    reference_t_main: tuple[float, ...] | None,
) -> Any:
    """Validate the hardware frame options; rigid transforms use metres and row-major order."""
    # Third Party
    import numpy as np

    def transform(values: tuple[float, ...] | None, name: str) -> tuple[float, ...]:
        matrix = np.eye(4) if values is None else np.asarray(values, dtype=np.float64)
        if matrix.size != 16:
            raise ValueError(f"{name} must contain 16 row-major values")
        matrix = matrix.reshape(4, 4)
        if not (
            np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1], rtol=0.0, atol=1e-9)
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0.0, atol=1e-9)
            and np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(f"{name} must be a finite rigid transform")
        return tuple(float(value) for value in matrix.ravel())

    modes = {
        "arm1": (api.BimanualMainFrame.ARM1_BASE, api.BimanualBase.ARM1),
        "arm2": (api.BimanualMainFrame.ARM2_BASE, api.BimanualBase.ARM1),
        "explicit-arm1": (api.BimanualMainFrame.EXPLICIT, api.BimanualBase.ARM1),
        "explicit-arm2": (api.BimanualMainFrame.EXPLICIT, api.BimanualBase.ARM2),
    }
    if main_frame not in modes:
        raise ValueError("main_frame must be arm1, arm2, explicit-arm1, or explicit-arm2")
    if main_frame.startswith("explicit-") and reference_t_main is None:
        raise ValueError("an explicit main frame requires --reference-t-main")
    if arm1_t_arm2 is None:
        placement = np.eye(4)
        placement[1, 3] = 1.2
        arm1_t_arm2 = tuple(placement.ravel())
    mode, reference = modes[main_frame]
    return api.BimanualFrameConfig(
        arm1_t_arm2=transform(arm1_t_arm2, "--arm1-t-arm2"),
        main_frame=mode,
        explicit_reference=reference,
        reference_t_main=transform(reference_t_main, "--reference-t-main"),
    )


def _main_rotation(torch: Any, frames: Any, *, dtype: Any, device: Any) -> Any:
    """Return main_R_arm1, rotating arm-1-base vectors into the configured main axes."""
    arm1_r_main = torch.eye(3, dtype=dtype, device=device)
    if frames.main_frame.value == "arm2_base" or (
        frames.main_frame.value == "explicit" and frames.explicit_reference.value == "arm2"
    ):
        arm1_r_main = torch.tensor(frames.arm1_t_arm2, dtype=dtype, device=device).reshape(4, 4)[
            :3, :3
        ]
    if frames.main_frame.value == "explicit":
        reference_r_main = torch.tensor(
            frames.reference_t_main, dtype=dtype, device=device
        ).reshape(4, 4)[:3, :3]
        arm1_r_main = arm1_r_main @ reference_r_main
    return arm1_r_main.T.contiguous()


def _capture_reference_poses(
    torch: Any, math_utils: Any, arm1_pose: Any, arm2_pose: Any, arm1_t_arm2: Any
) -> tuple[Any, Any]:
    """Capture the Cartesian/SO(3) midpoint and decoupled displacement in arm-1-base axes."""
    right_pose_arm1 = arm1_t_arm2 @ arm2_pose
    relative_pose = torch.linalg.inv(arm1_pose) @ right_pose_arm1
    relative_rotation = math_utils.axis_angle_from_quat(
        math_utils.quat_from_matrix(relative_pose[:, :3, :3])
    )
    half_rotation = math_utils.matrix_from_quat(
        math_utils.quat_from_angle_axis(
            0.5 * torch.linalg.vector_norm(relative_rotation, dim=-1), relative_rotation
        )
    )
    absolute_pose = arm1_pose.clone()
    absolute_pose[:, :3, :3].copy_(arm1_pose[:, :3, :3] @ half_rotation)
    absolute_pose[:, :3, 3].copy_(0.5 * (arm1_pose[:, :3, 3] + right_pose_arm1[:, :3, 3]))
    relative_pose[:, :3, 3].copy_(right_pose_arm1[:, :3, 3] - arm1_pose[:, :3, 3])
    return absolute_pose, relative_pose


def run(
    simulation_app: Any,
    *,
    spec: Any,
    num_envs: int = 16,
    duration_s: float = 10.0,
    device: str = "cuda:0",
    validate: bool = False,
    left_robot_model: str = "rizon4s",
    right_robot_model: str = "rizon4s",
    arm1_t_arm2: tuple[float, ...] | None = None,
    torque_limits_nm: tuple[float, ...] | None = None,
    main_frame: str = "arm1",
    reference_t_main: tuple[float, ...] | None = None,
) -> Any:
    """Run the selected hardware profile with latched per-pair fault handling."""
    # Standard Library
    import math

    # Third Party
    import numpy as np

    if num_envs <= 0 or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("num_envs and finite duration_s must be positive")
    if not device.startswith("cuda"):
        raise ValueError("the dual Flexiv companion requires CUDA")
    # Third Party
    import isaaclab.sim as sim_utils
    import torch
    from _hardware_profiles import hardware_profile
    from _native_components import prepare_components
    from _parallel_common import (
        MotionMetrics,
        capture_buffer_pointers,
        current_execution_context,
        flagged_rows,
        require_controller_module,
        resolve_end_effector,
        resolve_exact_joint_ids,
        root_frame_kinematics,
        status_mask,
        unsafe_status_mask,
        update_motion_metrics,
        update_root_frame_kinematics,
        update_tracking_error,
        validate_motion_metrics,
    )
    from _single_arm_parallel import _make_robot_cfg, configure_hardware_plant
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import math as math_utils

    api = require_controller_module()
    if torque_limits_nm is None:
        raise ValueError("dual Flexiv requires captured RobotInfo.tau_max for both arms")
    profile = hardware_profile(
        spec.hardware_target, f"{left_robot_model},{right_robot_model}", torque_limits_nm
    )
    frames = _resolve_frames(api, arm1_t_arm2, main_frame, reference_t_main)
    transform = np.asarray(frames.arm1_t_arm2).reshape(4, 4)
    sim_utils.create_new_stage()
    left_cfg, left_names, left_candidates = _make_robot_cfg("rizon", left_robot_model)
    right_cfg, right_names, right_candidates = _make_robot_cfg("rizon", right_robot_model)
    left_cfg.prim_path, right_cfg.prim_path = (
        "{ENV_REGEX_NS}/LeftRobot",
        "{ENV_REGEX_NS}/RightRobot",
    )
    left_cfg.init_state.pos = (0.0, -0.6, 0.0)
    right_cfg.init_state.pos = tuple(np.array(left_cfg.init_state.pos) + transform[:3, 3])
    right_cfg.init_state.rot = tuple(
        math_utils.quat_from_matrix(torch.tensor(transform[:3, :3])).tolist()
    )
    for cfg, names, start in ((left_cfg, left_names, 0), (right_cfg, right_names, 7)):
        configure_hardware_plant(
            cfg,
            names,
            profile.initial_joint_position_rad[start : start + 7],
            profile.torque_limits_nm[start : start + 7],
        )
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=profile.period_s))
    try:
        scene = _make_scene(left_cfg, right_cfg, num_envs)
        sim.reset()
        robots = (scene["left"], scene["right"])
        for robot in robots:
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos, robot.data.default_joint_vel
            )
        scene.reset()
        sim.forward()
        scene.update(profile.period_s)
        ids = (
            resolve_exact_joint_ids(robots[0], left_names),
            resolve_exact_joint_ids(robots[1], right_names),
        )
        ends = (
            resolve_end_effector(robots[0], left_candidates),
            resolve_end_effector(robots[1], right_candidates),
        )
        obs = tuple(
            root_frame_kinematics(torch, math_utils, robot, joint_ids, end[0], end[1])
            for robot, joint_ids, end in zip(robots, ids, ends, strict=True)
        )
        cuda_device, dtype = obs[0]["joint_position"].device, torch.float64
        context = current_execution_context(api, torch)
        parts = prepare_components(
            api,
            profile,
            num_envs,
            api.Device.CUDA,
            lambda a: torch.as_tensor(a, device=cuda_device),
            context,
            bimanual_frames=frames,
        )
        controller = parts.controller
        storage = controller.storage

        def zeros(*shape: int) -> Any:
            return torch.zeros((num_envs, *shape), dtype=dtype, device=cuda_device)

        poses = (zeros(4, 4), zeros(4, 4))
        for observation, pose in zip(obs, poses, strict=True):
            _fill_pose(torch, observation["ee_position_b"], observation["ee_quat_b"], pose)
        base_transform = torch.tensor(transform, dtype=dtype, device=cuda_device)
        initial_absolute_pose, desired_relative_pose = _capture_reference_poses(
            torch, math_utils, poses[0], poses[1], base_transform
        )
        right_pose_arm1 = base_transform @ poses[1]
        initial_wrist_distance = torch.linalg.vector_norm(desired_relative_pose[:, :3, 3], dim=-1)
        maximum_wrist_distance_error = torch.zeros((), dtype=dtype, device=cuda_device)
        main_r_arm1 = _main_rotation(torch, frames, dtype=dtype, device=cuda_device)
        desired_absolute_pose = initial_absolute_pose.clone()
        initial_positions = tuple(o["ee_position_b"].clone() for o in obs)
        initial_joints = torch.cat(tuple(o["joint_position"] for o in obs), dim=-1)
        joint_position_error, joint_velocity, joint_bias = zeros(14), zeros(14), zeros(14)
        joint_mass = zeros(14, 14)
        desired_velocity, desired_acceleration = zeros(12), zeros(12)
        reference = torch.zeros((3, 3), dtype=dtype, device=cuda_device)
        reference_main = torch.zeros_like(reference)
        motor_torques = tuple(
            torch.zeros_like(o["joint_position"], dtype=r.data.joint_pos.dtype)
            for o, r in zip(obs, robots, strict=True)
        )
        dt = torch.full((num_envs,), profile.period_s, dtype=dtype, device=cuda_device)
        stopped = torch.zeros(num_envs, dtype=torch.bool, device=cuda_device)
        unsafe_count = torch.zeros((), dtype=torch.int64, device=cuda_device)
        fault_seen = torch.zeros((), dtype=torch.bool, device=cuda_device)
        saturation_count = torch.zeros((), dtype=torch.int64, device=cuda_device)
        saturation_run = torch.zeros_like(saturation_count)
        maximum_saturation_run = torch.zeros_like(saturation_count)
        saturation_mask = status_mask(api, ("STATUS_SATURATION",))
        displacement = tuple(zeros() for _ in range(2))
        spread = torch.zeros((), dtype=dtype, device=cuda_device)
        bad = torch.zeros_like(stopped)
        mask = unsafe_status_mask(api)
        component_statuses = (
            storage.kinematics.status,
            storage.mass.status,
            storage.task.output.status,
            storage.nullspace.status,
            controller.status,
        )
        roots = (
            parts.roots,
            obs,
            poses,
            desired_absolute_pose,
            desired_relative_pose,
            initial_absolute_pose,
            initial_positions,
            initial_joints,
            joint_position_error,
            joint_velocity,
            joint_mass,
            joint_bias,
            desired_velocity,
            desired_acceleration,
            reference,
            reference_main,
            main_r_arm1,
            right_pose_arm1,
            initial_wrist_distance,
            maximum_wrist_distance_error,
            motor_torques,
            dt,
            stopped,
        )
        pointers = capture_buffer_pointers(*roots)
        steps = 0
        fault_step = min(200, max(2, int(duration_s / profile.period_s) // 3)) if validate else -1
        tracking_error = torch.zeros((), dtype=torch.float64, device=cuda_device)
        while simulation_app.is_running() and steps < int(duration_s / profile.period_s):
            context = current_execution_context(api, torch)
            values = _circle_reference(
                0.0
                if spec.hardware_target.endswith("compliant_hold")
                else (steps + 1) * profile.period_s
            )
            for index, vector in enumerate(values):
                for axis, scalar in enumerate(vector):
                    reference[index, axis] = scalar
            for index, (robot, joint_ids, end, observation) in enumerate(
                zip(robots, ids, ends, obs, strict=True)
            ):
                update_root_frame_kinematics(
                    torch, math_utils, robot, joint_ids, end[0], end[1], observation
                )
                update_motion_metrics(
                    torch,
                    observation["ee_position_b"],
                    initial_positions[index],
                    displacement[index],
                    spread,
                    active_mask=~stopped,
                )
                _fill_pose(
                    torch, observation["ee_position_b"], observation["ee_quat_b"], poses[index]
                )
                start = index * 7
                joint_position_error[:, start : start + 7].copy_(
                    initial_joints[:, start : start + 7] - observation["joint_position"]
                )
                joint_velocity[:, start : start + 7].copy_(observation["joint_velocity"])
                joint_mass[:, start : start + 7, start : start + 7].copy_(
                    observation["joint_mass_matrix"]
                )
                joint_bias[:, start : start + 7].copy_(observation["coriolis"])
            torch.matmul(base_transform, poses[1], out=right_pose_arm1)
            distance = torch.linalg.vector_norm(
                right_pose_arm1[:, :3, 3] - poses[0][:, :3, 3], dim=-1
            )
            distance_error = (distance - initial_wrist_distance).abs()
            distance_error = torch.where(
                ~stopped & torch.isfinite(distance_error), distance_error, 0.0
            )
            maximum_wrist_distance_error.copy_(
                torch.maximum(maximum_wrist_distance_error, distance_error.amax())
            )
            torch.mm(reference, main_r_arm1.T, out=reference_main)
            desired_absolute_pose.copy_(initial_absolute_pose)
            desired_absolute_pose[:, :3, 3].add_(reference[0])
            desired_velocity.zero_()
            desired_velocity[:, :3].copy_(reference_main[1])
            desired_acceleration.zero_()
            desired_acceleration[:, :3].copy_(reference_main[2])
            if steps == fault_step:
                desired_velocity[0, 0] = float("nan")
            desired_velocity[stopped] = float("nan")
            for component_status in component_statuses:
                component_status.zero_()
            controller.step(
                desired_absolute_pose,
                poses[0],
                poses[1],
                desired_relative_pose,
                obs[0]["jacobian"],
                obs[1]["jacobian"],
                desired_velocity,
                desired_acceleration,
                joint_position_error,
                joint_velocity,
                joint_mass,
                joint_bias,
                dt,
                context,
            )
            bad.zero_()
            for component_status in component_statuses:
                bad |= flagged_rows(torch, component_status, mask)
            for start in (0, 6):
                update_tracking_error(
                    torch,
                    storage.kinematics.task_pose_error[:, start : start + 6],
                    stopped,
                    tracking_error,
                )
            saturated = flagged_rows(torch, controller.status, saturation_mask)
            saturation_count.add_(saturated.sum())
            saturation_run.copy_(torch.where(saturated.any(), saturation_run + 1, 0))
            maximum_saturation_run.copy_(torch.maximum(maximum_saturation_run, saturation_run))
            unsafe_count.add_((bad & ~stopped).sum())
            if steps == fault_step:
                fault_seen.copy_(bad[0] & ~bad[1:].any())
            stopped |= bad
            for index, (robot, joint_ids) in enumerate(zip(robots, ids, strict=True)):
                motor_torques[index].copy_(controller.output[:, index * 7 : index * 7 + 7])
                motor_torques[index][stopped] = 0.0
                robot.set_joint_effort_target(motor_torques[index], joint_ids=joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(profile.period_s)
            steps += 1
        metrics = MotionMetrics(
            maximum_tracking_error_m=float(tracking_error.item()),
            configuration=dataclasses.asdict(profile),
            simulation_metadata={
                "bimanual_frames": dataclasses.asdict(frames),
                "reference_frame": "arm1_base",
                "task_vector_order": "absolute_linear_angular_then_relative_linear_angular",
            },
            maximum_wrist_distance_error_m=float(maximum_wrist_distance_error.item()),
            final_native_command_first_env=controller.output[0].tolist(),
            robot=f"Flexiv {left_robot_model}/{right_robot_model}",
            controller=profile.kind,
            num_envs=num_envs,
            sim_steps=steps,
            minimum_env_displacement_m=float(torch.minimum(*displacement).min().item()),
            maximum_inter_env_spread_m=float(spread.item()),
            unsafe_status_rows=int(unsafe_count.item()),
            saturation_rows=int(saturation_count.item()),
            persistent_saturation_steps=int(maximum_saturation_run.item()),
            fault_isolation_verified=bool((fault_seen & stopped[0] & ~stopped[1:].any()).item()),
            buffers_stable=pointers == capture_buffer_pointers(*roots),
        )
        if validate:
            validate_motion_metrics(
                metrics,
                minimum_displacement_m=0.0,
                maximum_spread_m=0.001,
                maximum_tracking_error_m=0.02,
            )
        return metrics
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()
