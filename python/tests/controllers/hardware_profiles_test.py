# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Hardware source audits and independently checked native CPU/CUDA companions."""

from __future__ import annotations

# Standard Library
import ast
import dataclasses
import hashlib
import importlib
import itertools
import json
import math
import os
import pathlib
import re
import subprocess
import sys

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "isaaclab"


@pytest.fixture(scope="module")
def companion_modules():
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield (
            importlib.import_module("_hardware_profiles"),
            importlib.import_module("_native_components"),
        )
    finally:
        sys.path.remove(str(EXAMPLES))


def _literal_array(source, name, occurrence=0):
    matches = re.findall(r"\b" + re.escape(name) + r"\s*(?:=\s*)?\{([^{}]*)\}", source)
    assert len(matches) > occurrence, f"missing audited array {name}"
    return tuple(
        float(ast.literal_eval(token.strip()))
        for token in matches[occurrence].split(",")
        if token.strip()
    )


def test_hardware_sources_match_the_reviewed_working_tree():
    contract = json.loads((EXAMPLES / "hardware_profile_contract.json").read_text())
    for relative, expected in contract["hardware_files"].items():
        assert (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        ), f"hardware source changed: {relative}; review profiles before updating the audit"


@pytest.mark.parametrize("model", ["fr3", "panda"])
def test_wall_guiding_has_zero_pose_and_posture_torque(companion_modules, model):
    profiles, native = companion_modules
    profile = profiles.hardware_profile("franka_virtual_walls_controller", model)
    parts = native.prepare_components(ctl, profile, 2, ctl.Device.CPU, np.copy)
    assert parts.nullspace is None
    assert parts.mass is None
    parts.controller.step(np.ones((2, 6)), np.full((2, 6), -2.0))
    np.testing.assert_array_equal(parts.controller.output, np.zeros((2, 6)))
    assert not np.any(parts.controller.status)


@pytest.mark.parametrize("family", ["franka", "rizon"])
def test_impedance_values_are_read_independently_from_hardware(companion_modules, family):
    profiles, _ = companion_modules
    if family == "franka":
        source = (ROOT / "examples/franka/franka_cartesian_impedance_controller.cpp").read_text()
        k, ns, nd = (
            "cartesian_diagonal",
            "nullspace_stiffness_diagonal",
            "nullspace_damping_diagonal",
        )
        models = ("fr3", "panda")
    else:
        source = (
            (ROOT / "examples/rizon/rizon_workspaces.hpp")
            .read_text()
            .split("class CartesianImpedanceExample", 1)[1]
            .split("class VariableCartesian", 1)[0]
        )
        k, ns, nd = "kCartesianStiffness", "kNullspaceStiffness", "kNullspaceDamping"
        models = tuple(profiles.RIZON_LIMITS)
    for model in models:
        p = profiles.hardware_profile(f"{family}_cartesian_impedance_controller", model)
        assert p.stiffness == _literal_array(source, k)
        assert p.nullspace_stiffness == _literal_array(source, ns)
        assert p.nullspace_damping == _literal_array(source, nd)
        assert p.damping_factor == 0.7


@pytest.mark.parametrize("target", ["joint_impedance_controller", "cartesian_impedance_controller"])
def test_ur_static_gains_and_limits_come_from_hardware(companion_modules, target):
    profiles, _ = companion_modules
    source = (ROOT / f"examples/ur/{target}.cpp").read_text()
    common = (ROOT / "examples/ur/impedance_common.hpp").read_text()
    for model, symbol in (("ur5e", "kUr5eTorqueLimitsNm"), ("ur10e", "kUr10eTorqueLimitsNm")):
        p = profiles.hardware_profile(target, model)
        assert p.stiffness == _literal_array(source, "kStiffness")
        assert p.damping == _literal_array(source, "kDamping")
        assert p.torque_limits_nm == _literal_array(common, symbol)
        rate = float(re.search(r"kTorqueRateLimitNmPerSecond\s*=\s*([0-9.]+)", common)[1])
        assert p.torque_rate_nm_per_s == (rate,) * 6


def test_franka_admittance_model_branches_and_filter_are_exact(companion_modules):
    profiles, _ = companion_modules
    source = (ROOT / "examples/franka/franka_admittance.hpp").read_text()
    for model, occurrence in (("panda", 0), ("fr3", 1)):
        p = profiles.hardware_profile("franka_cartesian_admittance_controller", model)
        k, m = _literal_array(source, "admittance_stiffness"), _literal_array(
            source, "virtual_mass", occurrence
        )
        assert p.admittance_stiffness == k
        assert p.admittance_mass == m
        assert p.admittance_damping == tuple(4.0 * math.sqrt(a * b) for a, b in zip(k, m))
        assert p.wrench_deadzone == (0.0,) * 6
        assert p.wrench_cutoff_hz == 7.0 and not p.wrench_seed_first_sample
        assert "external_wrench_[axis] = -filtered_wrench" in source
        assert p.command_mode == ("pose" if model == "panda" else "torque")


def test_force_and_variable_profiles_match_the_hardware_branches(companion_modules):
    profiles, _ = companion_modules
    for ee in (False, True):
        suffix = "_in_ee" if ee else ""
        target = "franka_cartesian_force_impedance_controller" + suffix
        source = (ROOT / f"examples/franka/{target}.cpp").read_text()
        p = profiles.hardware_profile(target, "fr3")
        assert p.stiffness == _literal_array(source, "stiffness")
        assert p.target_force_n == float(
            re.search(r"kTargetForceNewtons\s*=\s*([-0-9.]+)", source)[1]
        )
        assert p.force_ramp_s == float(re.search(r"kForceRampSeconds\s*=\s*([0-9.]+)", source)[1])
        rizon = profiles.hardware_profile(
            "rizon_cartesian_force_impedance_controller" + suffix, "rizon4s"
        )
        source = (
            (ROOT / "examples/rizon/rizon_workspaces.hpp")
            .read_text()
            .split("class ForceImpedanceExample", 1)[1]
        )
        assert rizon.stiffness == _literal_array(source, "kStiffness")
        assert rizon.target_force_n == (10.0 if ee else -15.0)
    franka = profiles.hardware_profile("franka_variable_cartesian_impedance_controller", "fr3")
    rizon = profiles.hardware_profile("rizon_variable_cartesian_impedance_controller", "rizon4s")
    source = (
        ROOT / "examples/franka/franka_variable_cartesian_impedance_controller.cpp"
    ).read_text()
    assert franka.variable_frequency_hz == float(
        re.search(r"std::sin\(2.0 \* kPi \* ([0-9.]+) \*", source)[1]
    )
    assert franka.stiffness == _literal_array(source, "initial")
    assert franka.variable_maximum == _literal_array(source, "maximum_stiffness_")
    source = (
        (ROOT / "examples/rizon/rizon_workspaces.hpp")
        .read_text()
        .split("class VariableCartesianImpedanceExample", 1)[1]
        .split("class CartesianAdmittanceExample", 1)[0]
    )
    assert rizon.stiffness == _literal_array(source, "kInitial")
    assert rizon.variable_maximum == _literal_array(source, "maximum_stiffness_")
    for time in (0.0, 0.5, 2.5, 5.0, 10.0):
        assert franka.variable_z_stiffness(time) == 1000.0 * (
            math.sin(2.0 * math.pi * 0.1 * time) + 1.001
        )
        assert rizon.variable_z_stiffness(time) == 1000.5 + 999.5 * math.sin(
            2.0 * math.pi * 0.1 * time
        )


@pytest.mark.parametrize(
    "suffix, gains", [("cartesian_impedance", "Circle"), ("compliant_hold", "Hold")]
)
def test_dual_controller_topology(companion_modules, suffix, gains):
    profiles, native = companion_modules
    source = (ROOT / "examples/flexiv_dual/consumer/flexiv_dual_controller_example.cpp").read_text()
    source = source.split("class CooperativeBimanualController", 1)[1]
    target = f"flexiv_dual_bimanual_{suffix}"
    p = profiles.hardware_profile(target, "rizon4s,rizon10s")
    assert p.stiffness == (
        _literal_array(source, f"k{gains}AbsoluteStiffness")
        + _literal_array(source, f"k{gains}RelativeStiffness")
    )
    assert p.nullspace_stiffness == _literal_array(source, "kNullspaceStiffness") * 2
    assert p.nullspace_damping == _literal_array(source, "kNullspaceDamping") * 2
    assert p.frame == "main_frame_orientation"
    assert p.torque_rate_nm_per_s == (1000.0,) * 14
    assert not p.torque_filter_seed_zero
    parts = native.prepare_components(ctl, p, 1, ctl.Device.CPU, np.copy)
    assert isinstance(parts.controller, ctl.BimanualCartesianImpedanceTorqueController)
    assert parts.config.coordination_mode == ctl.BimanualCoordinationMode.DECOUPLED_COOPERATIVE
    assert parts.controller.storage.task_velocity_error.shape == (1, 12)
    assert not parts.arms
    aliased = dataclasses.replace(
        parts.controller.storage,
        task_velocity_error=parts.controller.storage.kinematics.task_pose_error,
    )
    with pytest.raises(ValueError, match="overlap"):
        ctl.BimanualCartesianImpedanceTorqueController(
            parts.config, parts.controller.parameters, aliased
        )


def _exercise(parts, runtime, tick=0, bad_row=None):
    """Supply model fixtures; composition reads native results only at test boundaries."""
    p, n = parts.profile, parts.config.batch_size
    j, t = p.joint_dim, p.task_dim or 6

    def array(value):
        uploaded = runtime.array(value)
        if runtime.device == ctl.Device.CUDA:
            # Pageable fixture uploads use the default stream; make the
            # test boundary explicit before launching on our nonblocking stream.
            runtime.check(runtime.library.cudaDeviceSynchronize())
        return uploaded

    if p.family == "flexiv_dual":
        pose = np.tile(np.eye(4), (n, 1, 1))
        right_pose = pose.copy()
        right_pose[:, 1, 3] = 1.0
        desired_absolute = pose.copy()
        desired_absolute[:, 1, 3] = 0.5
        desired_absolute[:, 0, 3] = 0.001 + tick * 0.0001
        desired_relative = right_pose.copy()
        jacobian = np.zeros((n, 6, 7))
        jacobian[:, :, :6] = np.eye(6)
        velocity = np.full((n, 12), -0.002)
        if bad_row is not None:
            velocity[bad_row, 0] = np.nan
        parts.controller.step(
            array(desired_absolute),
            array(pose),
            array(right_pose),
            array(desired_relative),
            array(jacobian),
            array(jacobian),
            array(velocity),
            array(np.zeros((n, 12))),
            array(np.full((n, 14), 0.004)),
            array(np.full((n, 14), 0.003)),
            array(np.tile(np.diag(np.linspace(1.0, 2.0, 14)), (n, 1, 1))),
            array(np.full((n, 14), 0.1)),
            array(np.full(n, p.period_s)),
            runtime.context,
        )
        return runtime.numpy(parts.controller.output)
    if parts.arms:
        outputs = [
            _exercise(arm, runtime, tick, bad_row if i == 0 else None)
            for i, arm in enumerate(parts.arms)
        ]
        command = array(np.concatenate(outputs, axis=1))
        parts.torque_filter.step(command, array(np.full(n, p.period_s)), runtime.context)
        return runtime.numpy(parts.torque_filter.output)
    error = np.full((n, t), 0.001 + tick * 0.0001)
    if bad_row is not None:
        error[bad_row, 0] = np.nan
    velocity, acceleration = np.full((n, t), -0.002), np.zeros((n, t))
    mass = np.tile(np.diag(np.linspace(1.0, 2.0, j)), (n, 1, 1))
    jacobian = np.zeros((n, t, j))
    for axis in range(min(t, j)):
        jacobian[:, axis, axis] = 1.0
    qdot = np.full((n, j), 0.003)
    dt = array(np.full(n, p.period_s))
    if parts.mass is not None:
        parts.mass.step(array(mass), array(jacobian), runtime.context)
    kind = p.kind
    if kind in {"joint_impedance", "static_joint_impedance"}:
        error = np.full((n, j), 0.002 + tick * 0.0001)
        if bad_row is not None:
            error[bad_row, 0] = np.nan
        args = [array(error), array(-qdot)]
        if kind == "joint_impedance":
            args += [array(np.zeros_like(qdot)), array(mass)]
    elif p.command_mode == "pose" or kind == "cartesian_admittance_torque":
        force = array(error)
        zeros = array(np.zeros((n, 6)))
        if parts.shaping is not None:
            parts.shaping.step(array(np.zeros((n, 6))), runtime.context)
            parts.controller.step(force, zeros, zeros, dt, runtime.context)
        else:
            parts.admittance.step(force, zeros, dt, runtime.context)
        if p.command_mode == "pose":
            return runtime.numpy(parts.controller.output)
        args = [
            array(error),
            array(velocity),
            array(acceleration),
            array(qdot),
            parts.mass.storage.cartesian_mass_matrix,
            array(jacobian),
            dt,
        ]
    elif kind in {
        "static_cartesian_impedance_torque",
        "cartesian_virtual_walls_torque",
        "cartesian_potential_field_torque",
    }:
        args = [array(error), array(velocity)]
    else:
        args = [
            array(error),
            array(velocity),
            array(acceleration),
            array(qdot),
            parts.mass.storage.cartesian_mass_matrix,
            array(jacobian),
        ]
        if kind == "variable_cartesian_impedance_torque":
            requested = np.tile(np.diag(p.stiffness), (n, 1, 1))
            requested[:, 2, 2] = p.variable_z_stiffness(tick * p.period_s)
            args.insert(1, array(requested))
        elif kind == "cartesian_unified_force_impedance_torque":
            args += [array(np.zeros((n, t)))] * 3
        args += [dt]
    parts.controller.step(*args, runtime.context)
    command = runtime.numpy(parts.controller.output)
    if command.shape[1] != j or kind not in {"joint_impedance", "static_joint_impedance"}:
        command = np.einsum("ntj,nt->nj", jacobian, command)
    if parts.nullspace is not None:
        parts.nullspace.step(
            array(np.zeros((n, j))), array(qdot), array(mass), array(jacobian), runtime.context
        )
        command += runtime.numpy(parts.nullspace.output)
    if parts.torque_filter is not None:
        parts.torque_filter.step(array(command), dt, runtime.context)
        command = runtime.numpy(parts.torque_filter.output)
    return command


@pytest.mark.parametrize("batch_size", [1, 16, 256, 1024])
def test_all_native_companions_match_independent_single_rows(
    companion_modules, runtime, batch_size
):
    # Local Folder
    from .conftest import ArrayRuntime

    profiles, native = companion_modules
    variants = []
    for spec in ctl.controller_example_specs():
        models = tuple(spec.supported_robot_models)
        if spec.robot_family == "flexiv_dual":
            models = tuple(",".join(pair) for pair in itertools.product(models, repeat=2))
        variants.extend((spec.hardware_target, model) for model in models)
    for target, model in variants:
        p = profiles.hardware_profile(target, model)
        reference_runtime = ArrayRuntime(ctl.Device.CPU)
        try:
            reference = native.prepare_components(
                ctl, p, 1, ctl.Device.CPU, reference_runtime.array
            )

            def allocate(value):
                uploaded = runtime.array(value)
                if runtime.device == ctl.Device.CUDA:
                    # Preparation also consumes parameters on the nonblocking caller
                    # stream. Order default-stream uploads before prepare/reset kernels.
                    runtime.check(runtime.library.cudaDeviceSynchronize())
                return uploaded

            candidate = native.prepare_components(
                ctl, p, batch_size, runtime.device, allocate, runtime.context
            )
            for tick in range(3):
                expected = _exercise(reference, reference_runtime, tick)
                actual = _exercise(candidate, runtime, tick)
                np.testing.assert_allclose(
                    actual,
                    np.repeat(expected, batch_size, axis=0),
                    atol=1e-9,
                    rtol=1e-8,
                    err_msg=f"{p.target}/{p.model}, batch={batch_size}, tick={tick}",
                )
            if batch_size > 1:
                expected = _exercise(reference, reference_runtime, 3)
                actual = _exercise(candidate, runtime, 3, bad_row=0)
                np.testing.assert_allclose(
                    actual[1:], np.repeat(expected, batch_size - 1, axis=0), atol=1e-9, rtol=1e-8
                )
        finally:
            reference_runtime.close()


@pytest.mark.parametrize("model", ["fr3", "panda"])
def test_admittance_trace_matches_unchanged_hardware_helper(companion_modules, model):
    """Compare native inputs, outputs and integration history with the actual C++ helper."""
    build = os.environ.get("CONTROLLERS_ORACLE_BUILD")
    if not build:
        pytest.skip("set CONTROLLERS_ORACLE_BUILD to an Eigen-enabled CMake test build")
    executable = (
        pathlib.Path(build) / "tests/nvidia/srl/controllers" / f"franka_admittance_oracle_{model}"
    )
    oracle = np.loadtxt(subprocess.check_output([str(executable)], text=True).splitlines())
    profiles, factory = companion_modules
    # Local Folder
    from .conftest import ArrayRuntime

    runtime = ArrayRuntime(ctl.Device.CPU)
    profile = profiles.hardware_profile("franka_cartesian_admittance_controller", model)
    parts = factory.prepare_components(
        ctl, profile, 1, runtime.device, runtime.array, runtime.context
    )
    controller = parts.admittance if parts.admittance is not None else parts.controller
    filtered = np.zeros((1, 6), dtype=np.float64)
    wrench, dt = runtime.zeros((1, 6), np.float64), runtime.zeros((1,), np.float64)
    zero = runtime.zeros((1, 6), np.float64)
    for tick, expected in enumerate(oracle):
        period = 0.001 + tick * 0.00001
        measured = np.arange(1, 7) * (tick + 1) * 0.125
        alpha = period / (1 / (2 * math.pi * 7) + period)
        filtered += alpha * (measured - filtered)
        np.copyto(wrench, -filtered)
        np.copyto(dt, np.array([period]))
        controller.status.fill(0)
        controller.step(wrench, zero, dt, runtime.context)
        state = controller.storage.state
        vectors = (
            wrench,
            controller.output,
            state.twist,
            state.pose_offset,
            state.pose_previous_signal,
            state.twist_previous_signal,
            controller.storage.output.safety.last_valid_output,
        )
        actual = np.concatenate([runtime.numpy(v).ravel() for v in vectors])
        np.testing.assert_allclose(actual, expected[:42], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(runtime.numpy(controller.status), expected[42:43])


@pytest.mark.parametrize(
    "target",
    [
        "cartesian_admittance_controller",
        "cartesian_unified_force_admittance_controller",
        "cartesian_unified_force_admittance_controller_in_ee",
    ],
)
def test_ur_admittance_values_and_branch_damping_from_source(companion_modules, target):
    profiles, factory = companion_modules
    source = (ROOT / f"examples/ur/{target.removesuffix('_in_ee')}.cpp").read_text()
    p = profiles.hardware_profile(target, "ur5e")

    def constant(name):
        return float(re.search(r"\b" + name + r"\s*=\s*([0-9.]+)", source)[1])

    stiffness = _literal_array(source, "stiffness_diagonal")
    mass = tuple(constant("kVirtualMassScale") * m for m in _literal_array(source, "mass_diagonal"))
    assert p.admittance_stiffness == stiffness
    assert p.admittance_mass == mass
    unified = "unified" in target
    ee = target.endswith("_in_ee")
    damping = []
    for axis, (k, m) in enumerate(zip(stiffness, mass, strict=True)):
        scale = constant("kDampingScale")
        if unified and axis == 2:
            scale = constant(
                "kEndEffectorForceAxisDampingScale" if ee else "kForceAxisDampingScale"
            )
            if ee:
                k += constant("kTargetForceMagnitude") / (
                    constant("kMaximumShapingDistance") - constant("kFullForceDistance")
                )
        damping.append(scale * 2.0 * math.sqrt(k * m))
    np.testing.assert_array_equal(p.admittance_damping, damping)
    assert p.wrench_deadzone == _literal_array(
        source, "error_deadzone" if unified else "force_filter_threshold"
    )
    parts = factory.prepare_components(ctl, p, 1, ctl.Device.CPU, lambda a: a.copy())
    for name in ("acceleration_limit", "twist_limit", "pose_limit"):
        assert getattr(parts.controller.parameters, name) is None
    if unified:
        assert p.force_proportional == _literal_array(source, "proportional_gain")
        assert parts.controller.parameters.force.integral_lower_limit is None
        assert parts.controller.parameters.force.integral_upper_limit is None
        assert not parts.config.use_shaping_function and parts.shaping is not None
        assert p.frame == "robot_base"


def test_rizon_admittance_and_runtime_limits(companion_modules):
    profiles, factory = companion_modules
    source = (
        (ROOT / "examples/rizon/rizon_workspaces.hpp")
        .read_text()
        .split("class CartesianAdmittanceExample", 1)[1]
        .split("class ForceImpedanceExample", 1)[0]
    )
    p = profiles.hardware_profile("rizon_cartesian_admittance_controller", "rizon4s")
    assert p.admittance_stiffness == _literal_array(source, "kAdmittanceStiffness")
    assert p.admittance_damping == _literal_array(source, "kAdmittanceDamping")
    assert p.admittance_mass == (1.0,) * 6
    assert p.command_mode == "pose" and p.wrench_cutoff_hz == 7.0
    supplied = (42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0)
    p = profiles.hardware_profile("rizon_joint_impedance_controller", "rizon4s", supplied)
    parts = factory.prepare_components(ctl, p, 1, ctl.Device.CPU, lambda a: a.copy())
    np.testing.assert_array_equal(parts.torque_filter.parameters.upper_limit.values, supplied)
    assert p.torque_rate_nm_per_s == (1000.0,) * 7
    for invalid in ((1.0,), (float("nan"),) * 7, (0.0,) * 7):
        with pytest.raises(ValueError):
            profiles.hardware_profile(p.target, p.model, invalid)


def test_actuator_groups_preserve_joint_order_and_do_not_overlap(companion_modules):
    """Panda splits arm actuators; Rizon's default regex must not overlap literal destinations."""
    # Standard Library
    from types import SimpleNamespace

    sys.path.insert(0, str(EXAMPLES))
    try:
        helper = importlib.import_module("_single_arm_parallel").configure_hardware_plant
        cfg = SimpleNamespace(
            init_state=SimpleNamespace(joint_pos={"joint.*": 0.0}),
            actuators={
                "upper": SimpleNamespace(joint_names_expr=["joint[1-2]"]),
                "lower": SimpleNamespace(joint_names_expr=["joint[3-4]"]),
            },
        )
        helper(
            cfg, ("joint1", "joint2", "joint3", "joint4"), (0.1, 0.2, 0.3, 0.4), (10, 20, 30, 40)
        )
        assert cfg.init_state.joint_pos == dict(joint1=0.1, joint2=0.2, joint3=0.3, joint4=0.4)
        assert cfg.actuators["upper"].effort_limit_sim == dict(joint1=10, joint2=20)
        assert cfg.actuators["lower"].effort_limit_sim == dict(joint3=30, joint4=40)
    finally:
        sys.path.remove(str(EXAMPLES))
