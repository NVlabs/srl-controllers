# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Batched native pose controllers with an explicit Isaac DLS/servo adapter."""

from __future__ import annotations

# Standard Library
import dataclasses
import math
from typing import Any


def run(
    simulation_app: Any,
    *,
    spec: Any,
    num_envs: int = 16,
    duration_s: float = 10.0,
    device: str = "cuda:0",
    validate: bool = False,
    robot_model: str = "ur5e",
    torque_limits_nm: tuple[float, ...] | None = None,
    tcp_offset: tuple[float, ...] = (0.0,) * 6,
) -> Any:
    """Record base-frame pose commands before converting them for Isaac actuators."""
    if num_envs <= 0 or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("num_envs and finite duration_s must be positive")
    if not device.startswith("cuda"):
        raise ValueError("the pose companion requires CUDA")
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
        make_parallel_scene,
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
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import math as math_utils

    profile = hardware_profile(spec.hardware_target, robot_model, torque_limits_nm)
    if profile.command_mode != "pose":
        raise ValueError("the requested hardware profile does not output poses")
    api = require_controller_module()
    sim_utils.create_new_stage()
    cfg, joint_names, body_candidates = _make_robot_cfg(profile.family, robot_model)
    configure_hardware_plant(
        cfg,
        joint_names,
        profile.initial_joint_position_rad,
        profile.torque_limits_nm,
        pose_servo=True,
    )
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=profile.period_s))
    try:
        scene = make_parallel_scene(cfg, num_envs, contact_body_name=body_candidates[0])
        robot = scene["robot"]
        contact = scene["contact"]
        sim.reset()
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        scene.reset()
        sim.forward()
        scene.update(profile.period_s)
        ids = resolve_exact_joint_ids(robot, joint_names)
        body, jacobian, _ = resolve_end_effector(robot, body_candidates)
        frame_transforms = None
        simulation_metadata = None
        if profile.family == "universal_robots":
            # Third Party
            from _ur_adapter import ur_frame_transforms

            frame_transforms = ur_frame_transforms(robot_model, tcp_offset)
            simulation_metadata = {
                "controller_base_frame": "base",
                "tcp_offset_from_flange_m_rotvec": tcp_offset,
                "actuator_model": "isaac_implicit_pose_servo",
            }
        observations = root_frame_kinematics(
            torch, math_utils, robot, ids, body, jacobian, frame_transforms
        )
        dtype, cuda_device = (
            observations["joint_position"].dtype,
            observations["joint_position"].device,
        )
        context = current_execution_context(api, torch)
        parts = prepare_components(
            api,
            profile,
            num_envs,
            api.Device.CUDA,
            lambda a: torch.as_tensor(a, device=cuda_device),
            context,
        )
        controller = parts.controller
        state = (
            controller.storage.state.admittance
            if parts.shaping is not None
            else controller.storage.state
        )
        initial_position = observations["ee_position_b"].clone()
        initial_quaternion = observations["ee_quat_b"].clone()
        wrench = torch.zeros((num_envs, 6), dtype=dtype, device=cuda_device)
        filtered, delta, desired, feedforward = (torch.zeros_like(wrench) for _ in range(4))
        shaping_error, force_rotated = torch.zeros_like(wrench), torch.zeros_like(wrench)
        dt = torch.full((num_envs,), profile.period_s, dtype=dtype, device=cuda_device)
        last_joint_target = observations["_physics"]["joint_position"].clone()
        stopped = torch.zeros(num_envs, dtype=torch.bool, device=cuda_device)
        pose_command = torch.zeros((num_envs, 7), dtype=dtype, device=cuda_device)
        pose_command[:, :3].copy_(initial_position)
        pose_command[:, 3:].copy_(initial_quaternion)
        last_pose_command = pose_command.clone()
        actuator_pose = torch.empty_like(pose_command, dtype=robot.data.joint_pos.dtype)
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs,
            device=device,
        )
        unsafe_mask = unsafe_status_mask(api)
        displacement = torch.zeros(num_envs, dtype=dtype, device=cuda_device)
        spread = torch.zeros((), dtype=dtype, device=cuda_device)
        unsafe_count = torch.zeros((), dtype=torch.int64, device=cuda_device)
        fault_seen = torch.zeros((), dtype=torch.bool, device=cuda_device)
        saturation_count = torch.zeros((), dtype=torch.int64, device=cuda_device)
        saturation_run = torch.zeros_like(saturation_count)
        maximum_saturation_run = torch.zeros_like(saturation_count)
        saturation_mask = status_mask(api, ("STATUS_SATURATION",))
        matrix_diagonals = tuple(
            torch.tensor(values, dtype=dtype, device=cuda_device)
            for values in (
                profile.admittance_stiffness,
                profile.admittance_damping,
                profile.admittance_mass,
            )
        )
        matrix_scratch = torch.empty((num_envs, 3, 3), dtype=dtype, device=cuda_device)
        roots = (
            observations,
            parts.roots,
            wrench,
            filtered,
            delta,
            desired,
            feedforward,
            dt,
            stopped,
            pose_command,
            last_pose_command,
            actuator_pose,
            matrix_scratch,
        )
        pointers = capture_buffer_pointers(*roots)
        steps = 0
        fault_step = min(200, max(2, int(duration_s / profile.period_s) // 3)) if validate else -1
        tracking_error = torch.zeros((), dtype=torch.float64, device=cuda_device)
        while simulation_app.is_running() and steps < int(duration_s / profile.period_s):
            context = current_execution_context(api, torch)
            update_root_frame_kinematics(
                torch, math_utils, robot, ids, body, jacobian, observations
            )
            update_motion_metrics(
                torch,
                observations["ee_position_b"],
                initial_position,
                displacement,
                spread,
                active_mask=~stopped,
            )
            wrench.zero_()
            force_w = contact.data.net_forces_w
            if force_w.ndim == 3:
                force_w = force_w[:, 0]
            wrench[:, :3].copy_(
                math_utils.quat_rotate_inverse(observations["base_quat_w"], force_w.to(dtype=dtype))
            )
            alpha = profile.period_s / (
                1.0 / (2.0 * math.pi * profile.wrench_cutoff_hz) + profile.period_s
            )
            if steps == 0 and profile.wrench_seed_first_sample:
                filtered.copy_(wrench)
            else:
                delta.copy_(wrench).sub_(filtered)
                filtered.add_(delta, alpha=alpha)
            wrench.copy_(filtered)
            if profile.family == "franka":
                wrench.neg_()
            observation_bad = ~torch.isfinite(observations["ee_position_b"]).all(
                dim=1
            ) | ~torch.isfinite(observations["ee_quat_b"]).all(dim=1)
            wrench[stopped | observation_bad] = float("nan")
            controller.status.zero_()
            if parts.shaping is not None:
                rotation = math_utils.matrix_from_quat(observations["ee_quat_b"])
                desired.zero_()
                desired[:, 2] = profile.target_force_n * min(
                    (steps + 1) * profile.period_s / profile.force_ramp_s, 1.0
                )
                shaping_error.copy_(state.pose_offset).neg_()
                if profile.force_reference_frame == "end_effector":
                    for start in (0, 3):
                        torch.bmm(
                            rotation.transpose(1, 2),
                            filtered[:, start : start + 3, None],
                            out=force_rotated[:, start : start + 3, None],
                        )
                        torch.bmm(
                            rotation.transpose(1, 2),
                            -state.pose_offset[:, start : start + 3, None],
                            out=shaping_error[:, start : start + 3, None],
                        )
                    wrench.copy_(force_rotated)
                    for parameter, diagonal in zip(
                        (
                            controller.parameters.stiffness_matrix,
                            controller.parameters.damping_matrix,
                            controller.parameters.virtual_mass_matrix,
                        ),
                        matrix_diagonals,
                        strict=True,
                    ):
                        for start in (0, 3):
                            matrix_scratch.copy_(rotation).mul_(diagonal[start : start + 3])
                            torch.bmm(
                                matrix_scratch,
                                rotation.transpose(1, 2),
                                out=parameter.values[:, start : start + 3, start : start + 3],
                            )
                parts.shaping.status.zero_()
                parts.shaping.step(shaping_error, context)
                wrench.mul_(parts.shaping.output)
                desired.mul_(parts.shaping.output)
                if profile.force_reference_frame == "end_effector":
                    for vector in (wrench, desired):
                        for start in (0, 3):
                            torch.bmm(
                                rotation,
                                vector[:, start : start + 3, None],
                                out=force_rotated[:, start : start + 3, None],
                            )
                        vector.copy_(force_rotated)
                if steps == fault_step:
                    desired[0, 0] = float("nan")
                desired[stopped | observation_bad] = float("nan")
                controller.step(desired, wrench, feedforward, dt, context)
            else:
                if steps == fault_step:
                    wrench[0, 0] = float("nan")
                controller.step(wrench, feedforward, dt, context)
            saturated = flagged_rows(torch, controller.status, saturation_mask)
            saturation_count.add_(saturated.sum())
            saturation_run.copy_(torch.where(saturated.any(), saturation_run + 1, 0))
            maximum_saturation_run.copy_(torch.maximum(maximum_saturation_run, saturation_run))
            unsafe = flagged_rows(torch, controller.status, unsafe_mask)
            if parts.shaping is not None:
                unsafe |= flagged_rows(torch, parts.shaping.status, unsafe_mask)
            unsafe_count.add_((unsafe & ~stopped).sum())
            if steps == fault_step:
                fault_seen.copy_(unsafe[0] & ~unsafe[1:].any())
            stopped |= unsafe
            position, quaternion = math_utils.apply_delta_pose(
                initial_position, initial_quaternion, state.pose_offset
            )
            pose_command[:, :3].copy_(position)
            pose_command[:, 3:].copy_(quaternion)
            pose_command[stopped] = last_pose_command[stopped]
            last_pose_command.copy_(pose_command)
            # Native output remains float64; DLS and PhysX operate on their own dtype.
            actuator_pose.copy_(pose_command)
            ik.set_command(actuator_pose)
            physics = observations["_physics"]
            joint_target = ik.compute(
                physics["ee_position_b"],
                physics["ee_quat_b"],
                physics["jacobian"],
                physics["joint_position"],
            )
            actuator_bad = ~torch.isfinite(joint_target).all(dim=1)
            unsafe_count.add_((actuator_bad & ~stopped).sum())
            stopped |= actuator_bad
            joint_target[actuator_bad] = last_joint_target[actuator_bad]
            last_joint_target.copy_(joint_target)
            update_tracking_error(
                torch, pose_command[:, :3] - observations["ee_position_b"], stopped, tracking_error
            )
            robot.set_joint_position_target(joint_target, joint_ids=ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(profile.period_s)
            steps += 1
        metrics = MotionMetrics(
            maximum_tracking_error_m=float(tracking_error.item()),
            configuration=dataclasses.asdict(profile),
            final_native_command_first_env=pose_command[0].tolist(),
            simulation_metadata=simulation_metadata,
            robot=robot_model,
            controller=profile.kind,
            num_envs=num_envs,
            sim_steps=steps,
            minimum_env_displacement_m=float(displacement.min().item()),
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
                maximum_tracking_error_m=None,
            )
        return metrics
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()
