# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Downstream tests: public storage discovery, numerical primitives and custom composition."""

# Standard Library
import dataclasses
import inspect
from typing import get_type_hints

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl

NEW_BLOCKS = (
    "ProportionalBlock",
    "IntegratorBlock",
    "DifferentiatorBlock",
    "JointStiffness",
    "JointDamping",
    "JointDampingDoubleDiagonalization",
    "CartesianStiffness",
    "VariableCartesianStiffness",
    "CartesianDamping",
    "CartesianDampingDoubleDiagonalization",
    "CartesianDampingFactorizationDesign",
    "CartesianVirtualMass",
    "ShapingFunction",
    "CommandObserver",
    "FlowFunnel",
    "InverseAPF",
    "InertialCoupling",
    "EnergyTank",
    "VariableCartesianUnifiedForceImpedanceController",
)


def storage_for(name, config, array=np.zeros):
    """Allocate a typed storage tree exclusively from public requirement descriptors."""
    tree = {}
    requirements = getattr(ctl, name).requirements(config)
    for item in (*requirements.outputs, *requirements.state, *requirements.scratch):
        node = tree
        parts = item.name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = array(item.shape, dtype=item.dtype)

    def build(cls, values):
        types = get_type_hints(cls)
        kwargs = {}
        for field in dataclasses.fields(cls):
            if dataclasses.is_dataclass(types[field.name]):
                kwargs[field.name] = build(types[field.name], values.get(field.name, {}))
            elif field.name in values:
                kwargs[field.name] = values[field.name]
        return cls(**kwargs)

    return build(getattr(ctl, name + "Storage"), tree)


def config_for(name, batch, dtype, mode=ctl.ComponentMode.DENSE):
    kind = ctl.DType.FLOAT64 if dtype == np.float64 else ctl.DType.FLOAT32
    if name in (
        "ProportionalBlock",
        "IntegratorBlock",
        "DifferentiatorBlock",
        "ShapingFunction",
        "CommandObserver",
        "EnergyTank",
    ):
        return ctl.SignalBlockConfig(ctl.Device.CPU, kind, batch, 1 if name == "EnergyTank" else 17)
    config = dict(
        device=ctl.Device.CPU,
        dtype=kind,
        batch_size=batch,
        joint_dim=7,
        cartesian_frame=ctl.CartesianFrame.ROBOT_BASE,
    )
    return (
        ctl.JointBlockConfig(**config, mode=mode)
        if name in ("JointStiffness", "JointDamping")
        else ctl.ControllerConfig(**config)
    )


def parameters_for(name, config, dtype, per_batch=False):
    n = getattr(config, "signal_dim", 6)
    joint = getattr(config, "joint_dim", 7)
    values = {
        "gain": np.full(n, 2.0),
        "initial_output": np.zeros(n),
        "lower_limit": np.full(n, -10.0),
        "upper_limit": np.full(n, 10.0),
        "stiffness": np.eye(joint) * 2,
        "damping": np.eye(joint) * 2,
        "stiffness_matrix": np.eye(6) * 2,
        "initial_stiffness_matrix": np.eye(6) * 2,
        "damping_matrix": np.eye(6) * 2,
        "virtual_mass_matrix": np.eye(6) * 2,
        "damping_factors": np.ones(joint if name.startswith("Joint") else 6),
        "maximum_stiffness": np.full(6, 10),
        "minimum_stiffness": np.zeros(6),
        "maximum_stiffness_rate": np.full(6, 10),
        "maximum_distance": np.full(n, 10),
        "shaping_distance": np.full(n, 5),
        "proportional_gain": np.ones(6),
        "differentiator_gain": np.zeros(6),
        "integral_gain": np.zeros(6),
        "funnel_radius": 1,
        "max_distance": 2,
        "funnel_velocity": 1,
        "funnel_start_point": [0, 0, 0],
        "funnel_end_point": [1, 0, 0],
        "alpha": 1,
        "gamma": 1,
        "d_min": 1,
        "d_max": 2,
        "deadzone": 0.01,
        "upper_energy_limit": 10,
        "lower_energy_limit": 1,
        "initial_energy": 5,
        "empty_threshold": 1,
        "power_noise": 0,
    }
    if getattr(config, "mode", None) == ctl.ComponentMode.DIAGONAL:
        values["stiffness"] = values["damping"] = np.full(joint, 2)
    parameter_type = getattr(ctl, name + "Parameters")
    parameters = {}
    for field in dataclasses.fields(parameter_type):
        value = np.asarray(values[field.name], dtype=dtype)
        if per_batch:
            value = np.broadcast_to(value, (config.batch_size, *value.shape)).copy()
        parameters[field.name] = (ctl.per_batch if per_batch else ctl.shared)(value)
    return parameter_type(**parameters)


def inputs_for(name, config, dtype):
    batch, joints, n = (
        config.batch_size,
        getattr(config, "joint_dim", 7),
        getattr(config, "signal_dim", 6),
    )
    names = inspect.signature(getattr(ctl, name).step).parameters
    result = {}
    for field in names:
        if field in ("self", "context"):
            continue
        if field == "dt":
            value = np.full(batch, 0.01)
        elif field in ("power_signal", "additional_power"):
            value = np.zeros(batch)
        elif field in ("current_pose", "goal_pose", "cartesian_pose"):
            value = np.tile(np.eye(4), (batch, 1, 1))
        elif field in ("joint_mass_matrix", "cartesian_mass_matrix", "stiffness_matrix"):
            size = joints if field == "joint_mass_matrix" or name.startswith("Joint") else 6
            value = np.tile(np.eye(size), (batch, 1, 1))
        elif field == "jacobian":
            value = np.tile(np.eye(6, joints), (batch, 1, 1))
        elif field.startswith("joint_"):
            value = np.zeros((batch, joints))
        else:
            value = np.zeros((batch, n if field in ("input_signal", "error", "command") else 6))
        result[field] = value.astype(dtype)
    return result


@pytest.mark.parametrize("name", NEW_BLOCKS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("per_batch", [False, True])
def test_every_new_block_uses_public_metadata_and_lifecycle(runtime, name, dtype, per_batch):
    config = dataclasses.replace(config_for(name, 2, dtype), device=runtime.device)
    storage = storage_for(name, config, runtime.zeros)
    parameters = runtime.tree(parameters_for(name, config, dtype, per_batch))
    block = getattr(ctl, name)(config, parameters, storage, context=runtime.context)
    block.reset(context=runtime.context)
    inputs = {key: runtime.array(value) for key, value in inputs_for(name, config, dtype).items()}
    runtime.captured_step(lambda: block.step(**inputs, context=runtime.context))
    assert np.isfinite(runtime.numpy(block.output)).all()
    assert not np.any(runtime.numpy(block.status) & ctl.STATUS_SAFE_MODE), name
    block.reset(runtime.array(np.array([True, False])), context=runtime.context)
    runtime.captured_step(lambda: block.step(**inputs, context=runtime.context))
    assert not np.any(runtime.numpy(block.status) & ctl.STATUS_SAFE_MODE), name


@pytest.mark.parametrize("mode", [ctl.ComponentMode.DENSE, ctl.ComponentMode.DIAGONAL])
def test_joint_feedback_matches_independent_matrix_equation(mode):
    config = config_for("JointStiffness", 2, np.float64, mode)
    block = ctl.JointStiffness(
        config,
        parameters_for("JointStiffness", config, np.float64),
        storage_for("JointStiffness", config),
    )
    block.reset()
    error = np.arange(14, dtype=np.float64).reshape(2, 7)
    block.step(error)
    np.testing.assert_allclose(block.output, 2 * error)


def test_pid_primitives_keep_independent_history_and_support_custom_nonlinearity():
    config = config_for("ProportionalBlock", 2, np.float64)
    blocks = [
        getattr(ctl, name)(
            config, parameters_for(name, config, np.float64), storage_for(name, config)
        )
        for name in ("ProportionalBlock", "IntegratorBlock", "DifferentiatorBlock")
    ]
    for block in blocks:
        block.reset()
    first, second = np.ones((2, 17)), np.full((2, 17), 3.0)
    dt = np.array(0.1)
    blocks[0].step(first)
    blocks[1].step(first, dt)
    blocks[2].step(first, dt)
    np.testing.assert_allclose(blocks[2].output, 0)
    previous_integral = blocks[1].output.copy()
    blocks[0].step(second)
    blocks[1].step(second, dt)
    blocks[2].step(second, dt)
    np.testing.assert_allclose(blocks[0].output, 6)
    np.testing.assert_allclose(blocks[1].output, previous_integral + 0.2)
    np.testing.assert_allclose(blocks[2].output, 20)
    custom_command = np.tanh(blocks[0].output + blocks[1].output + blocks[2].output)
    assert np.isfinite(custom_command).all()


def test_storage_aliases_and_invalid_parameters_are_rejected():
    config = config_for("IntegratorBlock", 1, np.float64)
    storage = storage_for("IntegratorBlock", config)
    parameters = parameters_for("IntegratorBlock", config, np.float64)
    overlapping = dataclasses.replace(
        storage, state=dataclasses.replace(storage.state, integrated_signal=storage.output.command)
    )
    with pytest.raises(ValueError, match="overlap"):
        ctl.IntegratorBlock(config, parameters, overlapping)
    block = ctl.IntegratorBlock(config, parameters, storage)
    block.reset()
    with pytest.raises(ValueError, match="overlap"):
        block.step(storage.output.command, np.array(0.1))
    with pytest.raises(ValueError):
        block.apply_parameters(dataclasses.replace(parameters, lower_limit=ctl.shared(np.ones(1))))
    with pytest.raises(ValueError, match="not prepared"):
        block.step(np.ones((1, 17)), np.array(0.1))


def test_signal_dimensions_are_independent_of_joint_solver_limits():
    config = ctl.SignalBlockConfig(ctl.Device.CPU, ctl.DType.FLOAT64, 1, 97)
    block = ctl.ProportionalBlock(
        config,
        ctl.ProportionalBlockParameters(ctl.shared(np.ones(97))),
        storage_for("ProportionalBlock", config),
    )
    block.reset()
    block.step(np.full((1, 97), 3.0))
    np.testing.assert_array_equal(block.output, 3.0)


def test_signal_requirements_reject_overflow_before_emitting_descriptors():
    config = ctl.SignalBlockConfig(ctl.Device.CPU, ctl.DType.FLOAT64, 1, 2**63)
    with pytest.raises(ValueError, match="overflow"):
        ctl.DifferentiatorBlock.requirements(config)
