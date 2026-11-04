# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Run a canonical single-arm controller on parallel Isaac Lab robots.

Each launcher preserves its hardware counterpart's hold, stiffness, admittance,
or force-ramp reference. Translation is in meters, rotation
is an axis-angle error in radians, twists are per second, and Isaac Lab
quaternions use ``wxyz`` ordering. This is a simulator-only execution example:
it performs no motion planning, additional collision checking, hardware I/O,
or real-time qualification.
"""

from __future__ import annotations

# Standard Library
import dataclasses
import math
import os
import pathlib
import subprocess
from typing import Any

FR3_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
PANDA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
RIZON_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
FLEXIV_ASSET_COMMIT = "2d07d67887aff718ea93250e68e9e8eff4b0a62d"
FRANKA_MAXIMUM_JOINT_TORQUE_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
FRANKA_MAXIMUM_JOINT_TORQUE_RATE_NM_PER_S = (999.999,) * 7
RIZON_MAXIMUM_JOINT_TORQUE_NM = {
    "rizon4": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon4s": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon4m": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon10": (261.0, 261.0, 123.0, 123.0, 57.0, 57.0, 57.0),
    "rizon10s": (261.0, 261.0, 123.0, 123.0, 57.0, 57.0, 57.0),
}


def _maximum_joint_torques_nm(robot_family: str, robot_model: str) -> tuple[float, ...]:
    """Return manufacturer-published actuator torque maxima in Nm."""
    if robot_family == "franka" and robot_model in {"fr3", "panda"}:
        return FRANKA_MAXIMUM_JOINT_TORQUE_NM
    if robot_family == "rizon" and robot_model in RIZON_MAXIMUM_JOINT_TORQUE_NM:
        return RIZON_MAXIMUM_JOINT_TORQUE_NM[robot_model]
    raise ValueError(f"no official joint-torque limits for {robot_family}/{robot_model}")


def configure_hardware_plant(
    config: Any,
    joint_names: tuple[str, ...],
    positions: tuple[float, ...],
    limits: tuple[float, ...],
    *,
    pose_servo: bool = False,
) -> None:
    """Apply the hardware destination/limits to only the matching Isaac actuator groups."""
    # Standard Library
    import re

    config.init_state.joint_pos = {
        expression: position
        for expression, position in config.init_state.joint_pos.items()
        if not any(re.fullmatch(expression, name) for name in joint_names)
    }
    config.init_state.joint_pos.update(dict(zip(joint_names, positions, strict=True)))
    for actuator in config.actuators.values():
        selected = {
            name: limit
            for name, limit in zip(joint_names, limits, strict=True)
            if any(re.fullmatch(expression, name) for expression in actuator.joint_names_expr)
        }
        if selected:
            actuator.effort_limit_sim = selected
            if pose_servo:
                actuator.stiffness, actuator.damping = 2500.0, 125.0


def _verify_flexiv_asset_checkout(asset_root: pathlib.Path) -> None:
    """Require the reviewed official asset revision before loading any USD."""
    try:
        revision = subprocess.run(
            ("git", "-C", str(asset_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot verify Flexiv asset checkout {asset_root}: {exc}") from exc
    if revision != FLEXIV_ASSET_COMMIT:
        raise RuntimeError(
            f"Flexiv assets are at {revision}, expected pinned commit {FLEXIV_ASSET_COMMIT}"
        )


def _make_robot_cfg(
    robot_family: str, robot_model: str
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    """Create a repo-owned effort-control overlay without modifying vendor assets."""
    # Third Party
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    if robot_family == "franka" and robot_model == "panda":
        # Third Party
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

        config = FRANKA_PANDA_CFG.copy()
        config.prim_path = "{ENV_REGEX_NS}/Robot"
        config.spawn.activate_contact_sensors = True
        config.spawn.rigid_props.disable_gravity = True
        for actuator in config.actuators.values():
            if any("panda_joint" in expression for expression in actuator.joint_names_expr):
                actuator.stiffness = 0.0
                actuator.damping = 0.0
        return config, PANDA_JOINT_NAMES, ("panda_hand", "panda_link8", "panda_link7")

    if robot_family == "rizon":
        if robot_model == "rizon4m":
            raise RuntimeError(
                "The pinned official Flexiv Isaac Sim 5 asset commit does not contain Rizon4M; "
                "no unofficial asset is substituted."
            )
        asset_root_value = os.environ.get("FLEXIV_ISAAC_SIM_WS")
        if not asset_root_value:
            raise RuntimeError(
                "Rizon simulation requires FLEXIV_ISAAC_SIM_WS to point to the official "
                f"flexivrobotics/isaac_sim_ws checkout at {FLEXIV_ASSET_COMMIT}."
            )
        asset_root = pathlib.Path(asset_root_value).expanduser().resolve()
        _verify_flexiv_asset_checkout(asset_root)
        model_file = {
            "rizon4": "Rizon4.usd",
            "rizon4s": "Rizon4s.usd",
            "rizon10": "Rizon10.usd",
            "rizon10s": "Rizon10s.usd",
        }.get(robot_model)
        if model_file is None:
            raise ValueError(f"unsupported Rizon model: {robot_model}")
        maximum_joint_torque_nm = dict(
            zip(
                RIZON_JOINT_NAMES,
                _maximum_joint_torques_nm(robot_family, robot_model),
                strict=True,
            )
        )
        usd_path = (
            asset_root
            / "exts"
            / "isaacsim.robot.manipulators.examples"
            / "data"
            / "flexiv"
            / model_file
        )
        if not usd_path.is_file():
            raise RuntimeError(f"official Flexiv USD asset is missing: {usd_path}")
        config = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_vel={".*": 0.0}),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=maximum_joint_torque_nm,
                    stiffness=0.0,
                    damping=0.0,
                )
            },
            soft_joint_pos_limit_factor=1.0,
        )
        return config, RIZON_JOINT_NAMES, ("flange", "link7", "Link7")

    if robot_family == "universal_robots":
        asset_name = {"ur5e": "ur5e", "ur10e": "ur10e"}.get(robot_model)
        if asset_name is None:
            raise ValueError(f"unsupported Universal Robots model: {robot_model}")
        initial_positions = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
        config = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=(
                    f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/{asset_name}/{asset_name}.usd"
                ),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(zip(UR_JOINT_NAMES, initial_positions, strict=True)),
                joint_vel={".*": 0.0},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=list(UR_JOINT_NAMES),
                    effort_limit_sim=5.0,
                    velocity_limit_sim=None,
                    stiffness=0.0,
                    damping=0.0,
                )
            },
            soft_joint_pos_limit_factor=0.95,
        )
        return config, UR_JOINT_NAMES, ("wrist_3_link", "tool0", "flange", "ee_link")

    if robot_family != "franka" or robot_model != "fr3":
        raise ValueError(f"unsupported single-arm model: {robot_family}/{robot_model}")

    initial_positions = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    joint_positions = dict(zip(FR3_JOINT_NAMES, initial_positions, strict=True))
    joint_positions["fr3_finger_joint.*"] = 0.04
    effort_limits_nm = {
        name: limit
        for name, limit in zip(
            FR3_JOINT_NAMES,
            FRANKA_MAXIMUM_JOINT_TORQUE_NM,
            strict=True,
        )
    }
    config = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=joint_positions,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(FR3_JOINT_NAMES),
                effort_limit_sim=effort_limits_nm,
                stiffness=0.0,
                damping=0.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=500.0,
                damping=20.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    return config, FR3_JOINT_NAMES, ("fr3_hand", "fr3_link8", "fr3_link7")


def run(
    simulation_app: Any,
    *,
    spec: Any,
    num_envs: int = 16,
    duration_s: float = 10.0,
    device: str = "cuda:0",
    validate: bool = False,
    robot_model: str = "fr3",
    torque_limits_nm: tuple[float, ...] | None = None,
    tcp_offset: tuple[float, ...] = (0.0,) * 6,
) -> Any:
    """Execute one registered parallel single-arm rollout."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if not device.startswith("cuda"):
        raise ValueError("this Isaac Lab rollout requires a CUDA simulation device")
    if (
        spec.controller_kind
        in {
            "static_joint_impedance",
            "static_cartesian_impedance_torque",
        }
        and spec.robot_family != "universal_robots"
    ):
        raise ValueError("static impedance examples are restricted to Universal Robots")

    # Third Party
    from _hardware_profiles import hardware_profile
    from _native_components import prepare_components

    if spec.robot_family == "rizon" and torque_limits_nm is None:
        raise ValueError("Rizon requires captured RobotInfo.tau_max via torque_limits_nm")
    profile = hardware_profile(spec.hardware_target, robot_model, torque_limits_nm)
    if profile.command_mode == "pose":
        # Third Party
        from _pose_parallel import run as run_pose

        return run_pose(
            simulation_app,
            spec=spec,
            num_envs=num_envs,
            duration_s=duration_s,
            device=device,
            validate=validate,
            robot_model=robot_model,
            torque_limits_nm=torque_limits_nm,
            tcp_offset=tcp_offset,
        )

    # Third Party
    import isaaclab.sim as sim_utils
    import torch
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
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import math as math_utils

    controller_api = require_controller_module()
    execution_context = current_execution_context(controller_api, torch)
    # A fresh stage lets this run function be called after another rollout in
    # the same Isaac application (as done by the opt-in integration test).
    sim_utils.create_new_stage()
    sim_dt_s = float(spec.nominal_period_seconds)
    robot_cfg, arm_joint_names, ee_body_candidates = _make_robot_cfg(spec.robot_family, robot_model)
    field_kinds = {"cartesian_virtual_walls_torque", "cartesian_potential_field_torque"}
    configure_hardware_plant(
        robot_cfg, arm_joint_names, profile.initial_joint_position_rad, profile.torque_limits_nm
    )
    frame_transforms = None
    simulation_metadata = None
    if profile.family == "universal_robots":
        # Third Party
        from _ur_adapter import ur_frame_transforms

        frame_transforms = ur_frame_transforms(robot_model, tcp_offset)
        simulation_metadata = {
            "controller_base_frame": "base",
            "tcp_offset_from_flange_m_rotvec": tcp_offset,
            "actuator_model": "official_isaac_sim_asset",
        }
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=sim_dt_s))
    try:
        uses_contact_force = spec.controller_kind in {
            "cartesian_admittance_torque",
            "cartesian_unified_force_impedance_torque",
            "cartesian_force_impedance_with_tank_torque",
        }
        scene = make_parallel_scene(
            robot_cfg,
            num_envs,
            contact_body_name=(ee_body_candidates[0] if uses_contact_force else None),
        )
        sim.set_camera_view((3.5, 3.5, 2.5), (0.0, 0.0, 0.5))
        sim.reset()
        scene.update(sim_dt_s)
        robot = scene.articulations["robot"]
        contact_sensor = scene.sensors.get("contact") if uses_contact_force else None
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        )
        robot.reset()
        sim.forward()
        robot.update(sim_dt_s)
        arm_joint_ids = resolve_exact_joint_ids(robot, arm_joint_names)
        finger_joint_ids = [
            index for index, name in enumerate(robot.joint_names) if "finger_joint" in name
        ]
        ee_body_index, jacobian_index, ee_body_name = resolve_end_effector(
            robot, ee_body_candidates
        )
        print(
            f"[INFO] {robot_model}: {num_envs} environments, arm joints={list(arm_joint_names)}, "
            f"end effector={ee_body_name}, dt={sim_dt_s:.4f} s"
        )

        observations = root_frame_kinematics(
            torch,
            math_utils,
            robot,
            arm_joint_ids,
            ee_body_index,
            jacobian_index,
            frame_transforms,
        )
        dtype = observations["joint_position"].dtype
        torch_device = observations["joint_position"].device
        cartesian_dim = 6
        parts = prepare_components(
            controller_api,
            profile,
            num_envs,
            controller_api.Device.CUDA,
            lambda array: torch.as_tensor(array, device=torch_device),
            execution_context,
        )
        controller, mass_controller = parts.controller, parts.mass
        nullspace_controller, torque_filter, admittance = (
            parts.nullspace,
            parts.torque_filter,
            parts.admittance,
        )
        controller_kind = profile.kind
        unsafe_mask = unsafe_status_mask(controller_api)
        saturation_mask = status_mask(controller_api, ("STATUS_SATURATION",))
        cartesian_wrench = torch.zeros((num_envs, cartesian_dim), dtype=dtype, device=torch_device)
        cartesian_mass = (
            mass_controller.storage.cartesian_mass_matrix
            if mass_controller is not None
            else torch.zeros((num_envs, 6, 6), dtype=dtype, device=torch_device)
        )
        mass_step_status = (
            mass_controller.storage.status
            if mass_controller is not None
            else torch.zeros(num_envs, dtype=torch.uint64, device=torch_device)
        )
        impedance_step_status = controller.status
        feedback_status = (
            controller.status
            if controller_kind in {"joint_impedance", "static_joint_impedance"}
            else (
                nullspace_controller.status
                if nullspace_controller is not None
                else torch.zeros_like(mass_step_status)
            )
        )
        torque_output = torque_filter.storage.output
        joint_effort = torque_output.command
        pose_error, velocity_error, desired_acceleration = (
            torch.zeros_like(cartesian_wrench) for _ in range(3)
        )
        safe_wrench, external_wrench, desired_wrench, feedforward_wrench = (
            torch.zeros_like(cartesian_wrench) for _ in range(4)
        )
        filtered_wrench, filter_delta = torch.zeros_like(cartesian_wrench), torch.zeros_like(
            cartesian_wrench
        )
        unfiltered_joint_effort = torch.zeros_like(joint_effort)
        joint_bias_torque = torch.zeros_like(joint_effort)
        joint_velocity_error, joint_desired_acceleration = torch.zeros_like(
            joint_effort
        ), torch.zeros_like(joint_effort)
        dt = torch.full((num_envs,), sim_dt_s, dtype=dtype, device=torch_device)
        filter_dt = torch.full_like(dt, profile.torque_filter_period_s or sim_dt_s)
        if profile.family == "franka":
            torque_filter.storage.state.initialized.fill_(True)
        force_tank = None
        cartesian_field = None
        tank_twist = torch.zeros_like(cartesian_wrench)
        auxiliary_roots: tuple[Any, ...] = ()
        if controller_kind == "cartesian_force_impedance_with_tank_torque":
            # Third Party
            from _force_tank import ForceTank

            force_tank = ForceTank(
                controller_api, torch, parts.config, cartesian_wrench, execution_context
            )
            force_tank.reset(
                torch.ones(num_envs, dtype=torch.bool, device=torch_device), execution_context
            )
            parameter_statuses = (
                force_tank.force.storage.output.parameter_status,
                force_tank.tank.storage.parameter_status,
            )
            if any(
                torch.any(flagged_rows(torch, status, unsafe_mask)).item()
                for status in parameter_statuses
            ):
                raise RuntimeError("Force tank parameter validation failed")
            auxiliary_roots = (
                force_tank.force.storage,
                force_tank.tank.storage,
                force_tank.power,
                force_tank.additional_power,
                force_tank.power_terms,
                force_tank.scaled_wrench,
                force_tank.energy,
                force_tank.previous_energy,
                force_tank.work,
                force_tank.credit,
                force_tank.residual,
                tank_twist,
            )
        if controller_kind in field_kinds:
            # Third Party
            from _cartesian_fields import CartesianFields

            cartesian_field = CartesianFields(
                controller_api, torch, parts.config, observations["ee_position_b"], controller_kind
            )
            cartesian_field.markers = cartesian_field.visualize(robot, math_utils)
            auxiliary_roots += (cartesian_field.buffer_roots(),)
        controller_jacobian = torch.zeros_like(observations["jacobian"])
        effort_limits = torch.tensor(profile.torque_limits_nm, dtype=dtype, device=torch_device)
        requested_stiffness = (
            torch.diag(torch.tensor(profile.stiffness, dtype=dtype, device=torch_device))
            .unsqueeze(0)
            .repeat(num_envs, 1, 1)
            if controller_kind == "variable_cartesian_impedance_torque"
            else torch.empty(0, dtype=dtype, device=torch_device)
        )
        stopped_rows = torch.zeros(num_envs, dtype=torch.bool, device=torch_device)
        last_valid_effort = torch.zeros_like(joint_effort)
        motor_effort = torch.empty_like(joint_effort, dtype=robot.data.joint_pos.dtype)
        controller_roots = (
            observations,
            parts.roots,
            auxiliary_roots,
            pose_error,
            velocity_error,
            desired_acceleration,
            safe_wrench,
            joint_effort,
            unfiltered_joint_effort,
            joint_bias_torque,
            joint_velocity_error,
            joint_desired_acceleration,
            requested_stiffness,
            external_wrench,
            desired_wrench,
            feedforward_wrench,
            filtered_wrench,
            filter_delta,
            controller_jacobian,
            dt,
            filter_dt,
            stopped_rows,
            last_valid_effort,
            motor_effort,
        )
        pointer_snapshot = capture_buffer_pointers(*controller_roots)

        initial_position = observations["ee_position_b"].clone()
        initial_quaternion = observations["ee_quat_b"].clone()
        initial_joint_position = observations["joint_position"].clone()
        joint_position_error = torch.zeros_like(initial_joint_position)
        target_velocity = torch.zeros((num_envs, 3), dtype=dtype, device=torch_device)
        target_acceleration = torch.zeros_like(target_velocity)
        desired_position = initial_position.clone()
        maximum_displacement = torch.zeros(num_envs, dtype=dtype, device=torch_device)
        maximum_spread = torch.zeros((), dtype=dtype, device=torch_device)

        unsafe_row_count = torch.zeros((), dtype=torch.int64, device=torch_device)
        saturation_row_count = torch.zeros((), dtype=torch.int64, device=torch_device)
        saturation_run = torch.zeros((), dtype=torch.int64, device=torch_device)
        maximum_saturation_run = torch.zeros((), dtype=torch.int64, device=torch_device)
        zero_count = torch.zeros_like(saturation_run)
        fault_step = min(200, max(2, int(duration_s / sim_dt_s) // 3)) if validate else -1
        fault_seen = torch.zeros((), dtype=torch.bool, device=torch_device)
        sim_steps = 0

        tracking_error = torch.zeros((), dtype=torch.float64, device=torch_device)
        while simulation_app.is_running() and sim_steps < int(duration_s / sim_dt_s):
            execution_context = current_execution_context(controller_api, torch)
            update_root_frame_kinematics(
                torch,
                math_utils,
                robot,
                arm_joint_ids,
                ee_body_index,
                jacobian_index,
                observations,
            )
            update_motion_metrics(
                torch,
                observations["ee_position_b"],
                initial_position,
                maximum_displacement,
                maximum_spread,
                active_mask=~stopped_rows,
            )

            elapsed_s = sim_steps * sim_dt_s
            position_error, orientation_error = math_utils.compute_pose_error(
                observations["ee_position_b"],
                observations["ee_quat_b"],
                desired_position,
                initial_quaternion,
                rot_error_type="axis_angle",
            )
            pose_error[:, :3].copy_(position_error)
            pose_error[:, 3:].copy_(orientation_error)
            velocity_error[:, :3].copy_(target_velocity - observations["ee_velocity_b"][:, :3])
            velocity_error[:, 3:].copy_(-observations["ee_velocity_b"][:, 3:])
            desired_acceleration[:, :3].copy_(target_acceleration)
            desired_acceleration[:, 3:].zero_()
            if controller_kind == "variable_cartesian_impedance_torque":
                requested_stiffness[:, 2, 2] = profile.variable_z_stiffness(elapsed_s + sim_dt_s)
            controller_jacobian.copy_(observations["jacobian"])
            if spec.cartesian_frame is controller_api.CartesianFrame.END_EFFECTOR:
                ee_rotation_b = math_utils.matrix_from_quat(
                    math_utils.quat_inv(observations["ee_quat_b"])
                )
                pose_error[:, :3].copy_(
                    torch.bmm(ee_rotation_b, pose_error[:, :3].unsqueeze(-1)).squeeze(-1)
                )
                pose_error[:, 3:].copy_(
                    torch.bmm(ee_rotation_b, pose_error[:, 3:].unsqueeze(-1)).squeeze(-1)
                )
                velocity_error[:, :3].copy_(
                    torch.bmm(ee_rotation_b, velocity_error[:, :3].unsqueeze(-1)).squeeze(-1)
                )
                velocity_error[:, 3:].copy_(
                    torch.bmm(ee_rotation_b, velocity_error[:, 3:].unsqueeze(-1)).squeeze(-1)
                )
                desired_acceleration[:, :3].copy_(
                    torch.bmm(ee_rotation_b, desired_acceleration[:, :3].unsqueeze(-1)).squeeze(-1)
                )
                desired_acceleration[:, 3:].copy_(
                    torch.bmm(ee_rotation_b, desired_acceleration[:, 3:].unsqueeze(-1)).squeeze(-1)
                )
                torch.bmm(
                    ee_rotation_b,
                    observations["jacobian"][:, :3],
                    out=controller_jacobian[:, :3],
                )
                torch.bmm(
                    ee_rotation_b,
                    observations["jacobian"][:, 3:],
                    out=controller_jacobian[:, 3:],
                )
            if sim_steps == fault_step:
                pose_error[0, 0] = float("nan")

            joint_position_error.copy_(initial_joint_position).sub_(observations["joint_position"])
            if sim_steps == fault_step and controller_kind in {
                "joint_impedance",
                "static_joint_impedance",
            }:
                joint_position_error[0, 0] = float("nan")
            pose_error[stopped_rows] = float("nan")
            joint_position_error[stopped_rows] = float("nan")
            observations["joint_mass_matrix"][stopped_rows] = float("nan")
            mass_step_status.zero_()
            impedance_step_status.zero_()
            feedback_status.zero_()
            torque_output.status.zero_()
            joint_bias_torque.zero_()
            if (
                spec.robot_family != "universal_robots"
                and controller_kind != "cartesian_virtual_walls_torque"
            ):
                joint_bias_torque.copy_(observations["coriolis"])
            if mass_controller is not None:
                mass_controller.step(
                    observations["joint_mass_matrix"], controller_jacobian, execution_context
                )
            if nullspace_controller is not None:
                nullspace_controller.step(
                    joint_position_error,
                    observations["joint_velocity"],
                    observations["joint_mass_matrix"],
                    controller_jacobian,
                    execution_context,
                )
            if controller_kind == "static_joint_impedance":
                joint_velocity_error.copy_(observations["joint_velocity"]).neg_()
                controller.step(
                    joint_position_error,
                    joint_velocity_error,
                    execution_context,
                )
                unfiltered_joint_effort.copy_(controller.output).add_(joint_bias_torque)
                unfiltered_joint_effort[
                    stopped_rows | flagged_rows(torch, controller.status, unsafe_mask)
                ] = float("nan")
                torque_filter.step(unfiltered_joint_effort, filter_dt, execution_context)
            elif controller_kind == "joint_impedance":
                joint_velocity_error.copy_(observations["joint_velocity"]).neg_()
                controller.step(
                    joint_position_error,
                    joint_velocity_error,
                    joint_desired_acceleration,
                    observations["joint_mass_matrix"],
                    execution_context,
                )
                unfiltered_joint_effort.copy_(controller.output).add_(joint_bias_torque)
                unfiltered_joint_effort[
                    stopped_rows | flagged_rows(torch, controller.status, unsafe_mask)
                ] = float("nan")
                torque_filter.step(unfiltered_joint_effort, filter_dt, execution_context)
            elif controller_kind in {"static_cartesian_impedance_torque", *field_kinds}:
                controller.step(pose_error, velocity_error, execution_context)
                if cartesian_field is not None:
                    force = cartesian_field.step(
                        observations["ee_position_b"],
                        observations["ee_velocity_b"][:, :3],
                        elapsed_s,
                        sim_dt_s,
                        execution_context,
                    )
                    controller.output[:, :3].copy_(force)
                    impedance_step_status.view(torch.int64)[cartesian_field.unsafe] |= int(
                        controller_api.STATUS_INVALID_INPUT | controller_api.STATUS_SAFE_MODE
                    )
            elif controller_kind == "variable_cartesian_impedance_torque":
                controller.step(
                    pose_error,
                    requested_stiffness,
                    velocity_error,
                    desired_acceleration,
                    observations["joint_velocity"],
                    cartesian_mass,
                    controller_jacobian,
                    dt,
                    execution_context,
                )
            elif controller_kind == "cartesian_force_impedance_with_tank_torque":
                if contact_sensor is None:
                    raise RuntimeError("force-tank example requires the configured TCP sensor")
                torch.bmm(
                    controller_jacobian,
                    observations["joint_velocity"].unsqueeze(-1),
                    out=tank_twist.unsqueeze(-1),
                )
                velocity_error.copy_(tank_twist).neg_()
                controller.step(
                    pose_error,
                    velocity_error,
                    desired_acceleration,
                    observations["joint_velocity"],
                    cartesian_mass,
                    controller_jacobian,
                    dt,
                    execution_context,
                )
                external_wrench.zero_()
                contact_force_w = contact_sensor.data.net_forces_w
                if contact_force_w.ndim == 3:
                    contact_force_w = contact_force_w[:, 0]
                external_wrench[:, :3].copy_(
                    math_utils.quat_rotate_inverse(robot.data.root_quat_w, contact_force_w)
                )
                desired_wrench.zero_()
                desired_wrench[:, 2] = profile.target_force_n * min(
                    (elapsed_s + sim_dt_s) / profile.force_ramp_s, 1.0
                )
                upstream_unsafe = flagged_rows(torch, mass_step_status, unsafe_mask) | flagged_rows(
                    torch, controller.status, unsafe_mask
                )
                scaled_force = force_tank.step(
                    external_wrench,
                    desired_wrench,
                    feedforward_wrench,
                    pose_error,
                    tank_twist,
                    dt,
                    upstream_unsafe,
                    unsafe_mask,
                    execution_context,
                )
            elif controller_kind == "cartesian_unified_force_impedance_torque":
                if contact_sensor is None:
                    raise RuntimeError("force-control example requires the configured TCP sensor")
                external_wrench.zero_()
                contact_force_w = contact_sensor.data.net_forces_w
                if contact_force_w.ndim == 3:
                    contact_force_w = contact_force_w[:, 0]
                contact_force_b = math_utils.quat_rotate_inverse(
                    robot.data.root_quat_w, contact_force_w
                )
                if spec.cartesian_frame is controller_api.CartesianFrame.END_EFFECTOR:
                    contact_force_b = math_utils.quat_rotate_inverse(
                        observations["ee_quat_b"], contact_force_b.to(dtype=dtype)
                    )
                external_wrench[:, :3].copy_(contact_force_b)
                # ContactSensor supplies net normal-plus-friction force but no
                # qualified point-contact moment; torque channels stay zero.
                desired_wrench.zero_()
                desired_wrench[:, 2] = profile.target_force_n * min(
                    (elapsed_s + sim_dt_s) / profile.force_ramp_s, 1.0
                )
                controller.step(
                    pose_error,
                    velocity_error,
                    desired_acceleration,
                    observations["joint_velocity"],
                    cartesian_mass,
                    controller_jacobian,
                    external_wrench,
                    desired_wrench,
                    feedforward_wrench,
                    dt,
                    execution_context,
                )
            else:
                if admittance is not None:
                    if contact_sensor is None:
                        raise RuntimeError(
                            "admittance example requires the configured TCP contact sensor"
                        )
                    external_wrench.zero_()
                    contact_force_w = contact_sensor.data.net_forces_w
                    if contact_force_w.ndim == 3:
                        contact_force_w = contact_force_w[:, 0]
                    contact_force_b = math_utils.quat_rotate_inverse(
                        robot.data.root_quat_w, contact_force_w
                    )
                    external_wrench[:, :3].copy_(contact_force_b)
                    alpha = sim_dt_s / (1.0 / (2.0 * math.pi * profile.wrench_cutoff_hz) + sim_dt_s)
                    filter_delta.copy_(external_wrench).sub_(filtered_wrench)
                    filtered_wrench.add_(filter_delta, alpha=alpha)
                    external_wrench.copy_(filtered_wrench).neg_()
                    admittance.storage.output.status.zero_()
                    external_wrench[stopped_rows] = float("nan")
                    admittance.step(
                        external_wrench,
                        feedforward_wrench,
                        dt,
                        execution_context,
                    )
                    admittance_position, admittance_quaternion = math_utils.apply_delta_pose(
                        initial_position, initial_quaternion, admittance.storage.state.pose_offset
                    )
                    position_error, orientation_error = math_utils.compute_pose_error(
                        observations["ee_position_b"],
                        observations["ee_quat_b"],
                        admittance_position,
                        admittance_quaternion,
                        rot_error_type="axis_angle",
                    )
                    pose_error[:, :3].copy_(position_error)
                    pose_error[:, 3:].copy_(orientation_error)
                    if sim_steps == fault_step:
                        pose_error[0, 0] = float("nan")
                    pose_error[stopped_rows] = float("nan")
                    velocity_error.add_(admittance.storage.state.twist)
                    desired_acceleration.copy_(admittance.output)
                controller.step(
                    pose_error,
                    velocity_error,
                    desired_acceleration,
                    observations["joint_velocity"],
                    cartesian_mass,
                    controller_jacobian,
                    dt,
                    execution_context,
                )
            if controller_kind not in {"joint_impedance", "static_joint_impedance"}:
                safe_wrench.copy_(controller.output)
                if force_tank is not None:
                    safe_wrench.add_(scaled_force)
                upstream_unsafe = flagged_rows(torch, mass_step_status, unsafe_mask) | flagged_rows(
                    torch, controller.status, unsafe_mask
                )
                if nullspace_controller is not None:
                    upstream_unsafe |= flagged_rows(torch, nullspace_controller.status, unsafe_mask)
                safe_wrench[upstream_unsafe] = float("nan")
                torch.bmm(
                    controller_jacobian.transpose(1, 2),
                    safe_wrench.unsqueeze(-1),
                    out=unfiltered_joint_effort.unsqueeze(-1),
                )
                unfiltered_joint_effort.add_(joint_bias_torque)
                if nullspace_controller is not None:
                    unfiltered_joint_effort.add_(nullspace_controller.output)
                if profile.family == "franka":
                    torque_filter.storage.state.previous_command.copy_(last_valid_effort)
                unfiltered_joint_effort[
                    stopped_rows | flagged_rows(torch, controller.status, unsafe_mask)
                ] = float("nan")
                torque_filter.step(unfiltered_joint_effort, filter_dt, execution_context)
            if sim_steps == fault_step:
                pose_error[0, 0] = 0.0
                joint_position_error[0, 0] = 0.0

            update_tracking_error(torch, pose_error, stopped_rows, tracking_error)
            mass_unsafe = flagged_rows(torch, mass_step_status, unsafe_mask)
            impedance_unsafe = flagged_rows(torch, impedance_step_status, unsafe_mask)
            feedback_unsafe = flagged_rows(torch, feedback_status, unsafe_mask)
            torque_unsafe = flagged_rows(torch, torque_output.status, unsafe_mask)
            controller_unsafe = mass_unsafe | impedance_unsafe | feedback_unsafe | torque_unsafe
            if admittance is not None:
                controller_unsafe |= flagged_rows(
                    torch, admittance.storage.output.status, unsafe_mask
                )
            if force_tank is not None:
                controller_unsafe |= force_tank.unsafe_rows(unsafe_mask)
            fault_status = (
                feedback_unsafe
                if controller_kind in {"joint_impedance", "static_joint_impedance"}
                else impedance_unsafe
            )
            if force_tank is not None:
                fault_status |= force_tank.unsafe_rows(unsafe_mask)
            if sim_steps == fault_step:
                fault_seen.copy_(fault_status[0] & ~torch.any(fault_status[1:]))

            dynamics_finite = (
                torch.isfinite(observations["joint_mass_matrix"]).all(dim=(1, 2))
                & torch.isfinite(controller_jacobian).all(dim=(1, 2))
                & torch.isfinite(observations["coriolis"]).all(dim=1)
                & torch.isfinite(joint_bias_torque).all(dim=1)
            )
            effort_unsafe = (
                controller_unsafe | ~dynamics_finite | ~torch.isfinite(joint_effort).all(dim=1)
            )
            saturated = torch.any(torch.abs(joint_effort) > effort_limits + 1.0e-5, dim=1)
            saturated |= flagged_rows(torch, torque_output.status, saturation_mask)
            saturation_row_count.add_(saturated.sum())
            saturation_run.copy_(torch.where(torch.any(saturated), saturation_run + 1, zero_count))
            maximum_saturation_run.copy_(torch.maximum(maximum_saturation_run, saturation_run))
            unsafe_row_count.add_((effort_unsafe & ~stopped_rows).sum())
            stopped_rows |= effort_unsafe
            if profile.family == "franka":
                joint_effort[stopped_rows] = last_valid_effort[stopped_rows]
            else:
                joint_effort[stopped_rows] = 0.0

            last_valid_effort.copy_(joint_effort)
            motor_effort.copy_(joint_effort)
            robot.set_joint_effort_target(motor_effort, joint_ids=arm_joint_ids)
            if finger_joint_ids:
                finger_target = robot.data.default_joint_pos[:, finger_joint_ids]
                robot.set_joint_position_target(finger_target, joint_ids=finger_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt_s)
            sim_steps += 1

        buffers_stable = pointer_snapshot == capture_buffer_pointers(*controller_roots)
        metrics = MotionMetrics(
            maximum_tracking_error_m=float(tracking_error.item()),
            configuration=dataclasses.asdict(profile),
            final_native_command_first_env=joint_effort[0].tolist(),
            simulation_metadata=simulation_metadata,
            robot=robot_model,
            controller=spec.controller_kind,
            num_envs=num_envs,
            sim_steps=sim_steps,
            minimum_env_displacement_m=float(maximum_displacement.min().item()),
            maximum_inter_env_spread_m=float(maximum_spread.item()),
            unsafe_status_rows=int(unsafe_row_count.item()),
            saturation_rows=int(saturation_row_count.item()),
            persistent_saturation_steps=int(maximum_saturation_run.item()),
            fault_isolation_verified=bool(
                (fault_seen & stopped_rows[0] & ~torch.any(stopped_rows[1:])).item()
            ),
            buffers_stable=buffers_stable,
            energy_tank=force_tank.metrics() if force_tank is not None else None,
            cartesian_field=(
                cartesian_field.metrics(active_rows=~stopped_rows)
                if cartesian_field is not None
                else None
            ),
        )
        if validate:
            if (
                cartesian_field is not None
                and controller_kind == "cartesian_potential_field_torque"
            ):
                field_metrics = metrics.cartesian_field
                if (
                    not math.isfinite(field_metrics.minimum_clearance_m)
                    or field_metrics.repulsion_row_samples == 0
                    or not math.isfinite(field_metrics.maximum_goal_error_m)
                    or field_metrics.maximum_goal_error_m > 0.005
                    or field_metrics.minimum_settled_seconds < 0.5
                ):
                    raise RuntimeError(f"APF goal validation failed: {field_metrics}")
            if force_tank is not None:
                tank_metrics = metrics.energy_tank
                tolerance = 32 * torch.finfo(dtype).eps
                if (
                    not math.isfinite(tank_metrics.minimum_energy_j)
                    or not math.isfinite(tank_metrics.maximum_energy_j)
                    or not math.isfinite(tank_metrics.maximum_balance_residual_j)
                    or not 0.01 - tolerance <= tank_metrics.minimum_energy_j <= 1 + tolerance
                    or not 0.01 - tolerance <= tank_metrics.maximum_energy_j <= 1 + tolerance
                    or not 0 <= tank_metrics.minimum_scale <= 1
                    or tank_metrics.maximum_balance_residual_j > tolerance
                ):
                    raise RuntimeError("force tank energy accounting validation failed")
            validate_motion_metrics(
                metrics,
                minimum_displacement_m=0.0,
                maximum_spread_m=0.001,
                maximum_tracking_error_m=(
                    0.02
                    if profile.kind
                    in {
                        "joint_impedance",
                        "static_joint_impedance",
                        "cartesian_impedance_torque",
                        "static_cartesian_impedance_torque",
                        "variable_cartesian_impedance_torque",
                    }
                    else None
                ),
            )
        return metrics
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()
