# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Analytical CPU/CUDA examples for every checked canonical equation."""

# Standard Library
import inspect

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl
from nvidia.srl.controllers import equations as eq

CASES = {
    "proportional_value": ([2, 3], 6),
    "differentiator_value": ([3, 1, 0.5], 4),
    "clamp_value": ([4, -2, 2], (2, True)),
    "integrator_trapezoid_value": ([1, 3, 1, 0.5, -10, 10], (2, False)),
    "joint_diagonal_feedback_value": ([2, 3], 6),
    "shaping_factor_value": ([3, 4, 2], 0.5),
    "cartesian_force_command_value": ([1, 2, 3, 4, 5, 0.5], 7.5),
    "validate_energy_tank_parameters": ([10, 1, 5, 1], True),
    "energy_tank_step": ([2, 2, 0, 0.1, 10, 1, 1, 0], (1, 2, np.sqrt(4.4), False)),
    "rotation_tolerance": ([], None),
    "vector_norm": ([[3, 4]], 5),
    "rotation_is_valid": (["IDENTITY"], True),
    "so3_log_error": (["IDENTITY", "IDENTITY"], (True, [0, 0, 0])),
    "dense_feedback_row": ([[[2, 1], [0, 3]], [2, 4], 0], 8),
    "cartesian_stiffness_value": ([[[2, 1], [0, 3]], [2, 4], 0], 8),
    "cartesian_damping_value": ([[[2, 1], [0, 3]], [2, 4], 0], 8),
    "joint_stiffness_value": ([[[2, 1], [0, 3]], [2, 4], 0], 8),
    "joint_damping_value": ([[[2, 1], [0, 3]], [2, 4], 0], 8),
    "variable_stiffness_value": ([1, 5, 0, 10, 2, 0.5], 2),
    "deadband_value": ([0.1, 0.2], 0),
    "force_error_value": ([5, 2], 3),
    "matrix_derivative_value": ([5, 2, 0.5], 6),
    "flow_funnel_geometry": (
        [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
        (2, 1, 1, True, [1, 0, 0], [0, -1, 0]),
    ),
    "flow_funnel_weight": ([2, 1, 3], 0),
    "flow_funnel_direction_value": ([2, 1, 3, 0, 1, 2], 2),
    "virtual_wall_force_value": ([-0.01, 0, 500, 20, 0.005], 5),
    "spherical_apf_repulsion_value": ([0.02, 0.04, 0.0001], 6.25),
    "validate_inverse_apf_parameters": ([1, 1, 0, 2, 0], True),
    "inverse_apf_damping": ([1, 2, 4, 1], 3),
    "inverse_apf_wrench_value": ([1, 2, 1, 1, 1, 0, 3], -6.5),
    "damping_factorization_value": ([2, 3], 5),
    "damping_double_diagonalization_value": ([3], 6),
    "command_observer_accepts": ([2, 3], True),
    "inertial_coupling_value": ([6, 1], 2),
    "cartesian_impedance_value": ([1, 2, 3, 4], 10),
    "joint_impedance_value": ([1, 2, 3], 6),
    "nullspace_feedback_value": ([1, 2], 3),
    "nullspace_projector_value": ([1, 1, 0.3], 0.7),
    "nullspace_projected_torque_value": ([5, 2], 3),
    "unified_force_impedance_value": ([2, 3], 5),
    "admittance_force_balance_value": ([7, 5, 3, 2], 7),
    "operational_space_acceleration_value": ([1, 2, 3], 6),
    "operational_space_mass_parameters_valid": ([0, 0.1], True),
    "symmetric_average": ([2, 4], 3),
    "symmetric_pair_is_valid": ([2, 2, 0.01], True),
    "operational_space_inverse_mass_value": ([2, 4, 0.5, True], 3.5),
    "torque_filter_value": ([10, 1, -20, 20, 2, 0.5, True], (2, True)),
    "safe_output_value": ([True, True, True, 5, 2], 2),
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_checked_equations_match_analytical_examples(runtime, name, dtype, batched):
    function = getattr(eq, name)
    values, expected = CASES[name]
    parameters = inspect.signature(function).parameters
    arguments = {}
    numeric = [
        p
        for p in parameters.values()
        if p.name not in ("config", "output", "outputs", "status", "context")
    ]
    for parameter, value in zip(numeric, values, strict=True):
        if parameter.annotation is int:
            arguments[parameter.name] = value
            continue
        array = (
            np.eye(4, dtype=dtype)
            if isinstance(value, str)
            else np.asarray(value, dtype=bool if isinstance(value, bool) else dtype)
        )
        if batched:
            array = np.broadcast_to(array, (2, *array.shape)).copy()
        arguments[parameter.name] = (ctl.per_batch if batched else ctl.shared)(runtime.array(array))
    results = []
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    for value in expected_values:
        shape = (2, len(value)) if isinstance(value, list) else (2,)
        results.append(runtime.zeros(shape, bool if isinstance(value, bool) else dtype))
    if "outputs" in parameters:
        arguments["outputs"] = parameters["outputs"].annotation(*results)
    else:
        arguments["output"] = results[0]
    status = runtime.zeros(2, np.uint64)
    config = eq.EquationConfig(
        runtime.device, ctl.DType.FLOAT32 if dtype == np.float32 else ctl.DType.FLOAT64, 2
    )
    runtime.captured_step(
        lambda: function(config, **arguments, status=status, context=runtime.context)
    )
    assert not np.any(runtime.numpy(status) & ctl.STATUS_SAFE_MODE)
    for actual, expected in zip(results, expected_values, strict=True):
        if expected is None:
            assert np.all(runtime.numpy(actual) > 0)
        else:
            target = np.broadcast_to(expected, runtime.numpy(actual).shape)
            np.testing.assert_allclose(runtime.numpy(actual), target, rtol=2e-6, atol=2e-6)


def test_bad_dt_and_nonfinite_are_row_local_faults(runtime):
    config = eq.EquationConfig(runtime.device, ctl.DType.FLOAT64, 3)
    output, status = runtime.zeros(3, np.float64), runtime.zeros(3, np.uint64)
    eq.differentiator_value(
        config,
        input_signal=ctl.per_batch(runtime.array(np.array([2.0, 3.0, np.nan]))),
        previous_signal=ctl.shared(runtime.array(np.array(0.0))),
        t_delta=ctl.per_batch(runtime.array(np.array([0.5, 0.0, 0.5]))),
        output=output,
        status=status,
        context=runtime.context,
    )
    np.testing.assert_allclose(runtime.numpy(output), [4, 0, 0])
    flags = runtime.numpy(status)
    assert flags[0] == 0
    assert int(flags[1]) & ctl.STATUS_INVALID_DT
    assert int(flags[2]) & ctl.STATUS_NON_FINITE_INPUT
    assert np.all(flags[1:] & ctl.STATUS_SAFE_MODE)


def test_structural_failure_does_not_write_output():
    config = eq.EquationConfig(ctl.Device.CPU, ctl.DType.FLOAT64, 2)
    output, status = np.full(2, 123.0), np.zeros(2, np.uint64)
    with pytest.raises(ValueError):
        eq.proportional_value(
            config,
            gain=ctl.shared(np.ones(1)),
            input_signal=ctl.shared(np.array(2.0)),
            output=output,
            status=status,
        )
    np.testing.assert_array_equal(output, [123, 123])
    with pytest.raises(ValueError, match="overlap"):
        eq.proportional_value(
            config,
            gain=ctl.per_batch(output),
            input_signal=ctl.shared(np.array(2.0)),
            output=output,
            status=status,
        )
