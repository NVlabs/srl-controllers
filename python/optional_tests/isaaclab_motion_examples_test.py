# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Opt-in nominal/fault matrix for all canonical Isaac companions.

Set NVIDIA_SRL_CONTROLLERS_ENABLE_ISAACLAB_MOTION_TEST=1 in the simulator's
ABI-matching interpreter. NVIDIA_SRL_CONTROLLERS_ISAACLAB_ENV_COUNTS defaults
to 1,16,256; NVIDIA_SRL_CONTROLLERS_ISAACLAB_TARGETS optionally selects a
comma-separated target subset; NVIDIA_SRL_CONTROLLERS_ISAACLAB_DURATION
overrides rollout duration. The dual-Rizon circle defaults to 10 seconds;
other targets default to 1 second.
Each target/batch/fault case runs in a fresh process because replaced PhysX
scenes retain GPU allocations in Isaac Sim 5.1. The full matrix is intentionally slow.
Rizon cases require FLEXIV_ISAAC_SIM_WS. Test-only torque limits are nominal
model fixtures, not measurements from connected robots. No hardware is used.
"""

from __future__ import annotations

# Standard Library
import atexit
import dataclasses
import importlib
import json
import os
import pathlib
import subprocess
import sys

# Third Party
import pytest

ENABLED = os.environ.get("NVIDIA_SRL_CONTROLLERS_ENABLE_ISAACLAB_MOTION_TEST", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
COUNTS = tuple(
    int(n)
    for n in os.environ.get("NVIDIA_SRL_CONTROLLERS_ISAACLAB_ENV_COUNTS", "1,16,256").split(",")
)
pytestmark = [
    pytest.mark.isaaclab,
    pytest.mark.skipif(not ENABLED, reason="opt-in Isaac matrix disabled"),
]


@pytest.fixture(scope="module")
def simulator():
    """Load one Isaac application and defer teardown until pytest has reported results."""
    example_dir = pathlib.Path(__file__).resolve().parents[2] / "examples/isaaclab"
    sys.path.insert(0, str(example_dir))
    if os.environ.get("CONTROLLERS_ISAAC_MATRIX_CHILD") != "1":
        return None
    # Third Party
    from isaaclab.app import AppLauncher

    saved = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        launcher = AppLauncher(headless=True)
    finally:
        sys.argv = saved
    atexit.register(launcher.app.close)
    return launcher.app


@pytest.mark.parametrize("batch_size", COUNTS)
@pytest.mark.parametrize("inject_fault", [False, True])
def test_canonical_matrix(simulator, batch_size, inject_fault, request, tmp_path):
    """Check registry dispatch, raw native output and separate nominal/fault outcomes."""
    # Third Party
    from _hardware_profiles import hardware_profile
    from _parallel_common import require_controller_module

    api = require_controller_module()
    selected = set(
        filter(None, os.environ.get("NVIDIA_SRL_CONTROLLERS_ISAACLAB_TARGETS", "").split(","))
    )
    if os.environ.get("CONTROLLERS_ISAAC_MATRIX_CHILD") != "1":
        failures = []
        for spec in api.controller_example_specs():
            if selected and spec.hardware_target not in selected:
                continue
            env = os.environ.copy()
            env["CONTROLLERS_ISAAC_MATRIX_CHILD"] = "1"
            env["NVIDIA_SRL_CONTROLLERS_ISAACLAB_TARGETS"] = spec.hardware_target
            log = tmp_path / f"{spec.hardware_target}.log"
            with log.open("w") as output:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-s",
                        f"{pathlib.Path(__file__).resolve()}::{request.node.name}",
                    ],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                failures.append(f"{spec.hardware_target}: {log}\n{log.read_text()[-6000:]}")
        assert not failures, "\n".join(failures)
        return
    failures = []
    for spec in api.controller_example_specs():
        if selected and spec.hardware_target not in selected:
            continue
        profile = hardware_profile(spec.hardware_target, spec.default_robot_model)
        is_circle = spec.hardware_target == "flexiv_dual_bimanual_cartesian_impedance"
        duration = float(
            os.environ.get(
                "NVIDIA_SRL_CONTROLLERS_ISAACLAB_DURATION", "10.0" if is_circle else "1.0"
            )
        )
        kwargs = dict(
            num_envs=batch_size, duration_s=duration, validate=inject_fault, device="cuda:0"
        )
        if spec.robot_family == "flexiv_dual":
            module = "_flexiv_dual_parallel"
            kwargs.update(spec=spec, torque_limits_nm=profile.torque_limits_nm)
        else:
            module = "_single_arm_parallel"
            kwargs.update(spec=spec, robot_model=spec.default_robot_model)
            if spec.robot_family == "rizon":
                kwargs["torque_limits_nm"] = profile.torque_limits_nm
        try:
            metrics = importlib.import_module(module).run(simulator, **kwargs)
            print(json.dumps(dataclasses.asdict(metrics), sort_keys=True), flush=True)
            assert metrics.controller == profile.kind
            assert metrics.configuration["target"] == spec.hardware_target
            assert metrics.configuration["model"] == spec.default_robot_model
            assert metrics.num_envs == batch_size
            assert metrics.sim_steps == int(duration / profile.period_s)
            assert metrics.buffers_stable
            assert metrics.unsafe_status_rows == (1 if inject_fault else 0)
            assert metrics.final_native_command_first_env is not None
            if inject_fault:
                assert metrics.fault_isolation_verified
            else:
                assert metrics.persistent_saturation_steps <= 10
                if batch_size > 1:
                    limit = 0.001
                    assert metrics.maximum_inter_env_spread_m <= limit
                if metrics.maximum_wrist_distance_error_m is not None:
                    assert metrics.maximum_wrist_distance_error_m <= 0.02
                if is_circle:
                    assert metrics.minimum_env_displacement_m >= 0.002
                if spec.robot_family == "flexiv_dual":
                    assert metrics.maximum_wrist_distance_error_m is not None
                if profile.kind in {
                    "joint_impedance",
                    "static_joint_impedance",
                    "cartesian_impedance_torque",
                    "static_cartesian_impedance_torque",
                    "bimanual_cartesian_impedance_torque",
                }:
                    assert metrics.maximum_tracking_error_m <= 0.02
        except Exception as exc:
            failures.append(f"{spec.hardware_target}, B={batch_size}, fault={inject_fault}: {exc}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("robot_model", ["ur5e", "ur10e"])
def test_ur_simulator_frames_match_hardware_urdf(simulator, robot_model, request):
    """Check 16 poses, rotated roots and nonzero TCPs against independent URDF kinematics."""
    if os.environ.get("CONTROLLERS_ISAAC_MATRIX_CHILD") != "1":
        env = os.environ.copy()
        env["CONTROLLERS_ISAAC_MATRIX_CHILD"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", f"{__file__}::{request.node.name}"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 passed" in result.stdout and "FAILED" not in result.stdout, (
            result.stdout + result.stderr
        )
        return
    # Third Party
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from _parallel_common import (
        capture_buffer_pointers,
        make_parallel_scene,
        resolve_end_effector,
        resolve_exact_joint_ids,
        root_frame_kinematics,
        update_root_frame_kinematics,
    )
    from _single_arm_parallel import _make_robot_cfg
    from _ur_adapter import ur_frame_transforms
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import math as math_utils

    test_dir = pathlib.Path(__file__).resolve().parents[1] / "tests/controllers"
    sys.path.insert(0, str(test_dir))
    try:
        oracle = importlib.import_module("isaac_adapters_test")
    finally:
        sys.path.pop(0)
    sim_utils.create_new_stage()
    cfg, names, candidates = _make_robot_cfg("universal_robots", robot_model)
    cfg.init_state.rot = (0.9238795325, 0.0, 0.0, 0.3826834324)
    sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=0.002))
    try:
        scene = make_parallel_scene(cfg, 16)
        sim.reset()
        robot = scene["robot"]
        ids = resolve_exact_joint_ids(robot, names)
        body, jacobian, _ = resolve_end_effector(robot, candidates)
        positions = robot.data.default_joint_pos.clone()
        offsets = torch.linspace(-0.15, 0.15, 16, device="cuda:0")
        positions[:, ids] += offsets[:, None]
        robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
        scene.reset()
        sim.forward()
        scene.update(0.002)
        for tcp_offset in ((0.0,) * 6, (0.02, -0.01, 0.15, 0.1, -0.05, 0.02)):
            observations = root_frame_kinematics(
                torch,
                math_utils,
                robot,
                ids,
                body,
                jacobian,
                ur_frame_transforms(robot_model, tcp_offset),
            )
            pointers = capture_buffer_pointers(observations)
            update_root_frame_kinematics(
                torch, math_utils, robot, ids, body, jacobian, observations
            )
            assert pointers == capture_buffer_pointers(observations)
            poses = observations["current_pose"].cpu().numpy()
            jacobians = observations["jacobian"].cpu().numpy()
            joint_positions = observations["joint_position"].cpu().numpy()
            for row in range(16):
                expected_pose, expected_jacobian, _ = oracle.hardware_urdf_pose_and_jacobian(
                    robot_model, joint_positions[row], tcp_offset
                )
                np.testing.assert_allclose(poses[row], expected_pose, atol=3e-6, rtol=1e-5)
                np.testing.assert_allclose(jacobians[row], expected_jacobian, atol=3e-6, rtol=1e-5)
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()
