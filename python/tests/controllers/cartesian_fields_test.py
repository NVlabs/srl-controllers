# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Independent domain and wrench oracles for point fields and legacy InverseAPF."""

# Standard Library
import dataclasses

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl
from nvidia.srl.controllers import equations as eq

# Local Folder
from .public_blocks_test import config_for, storage_for


def make_inverse(runtime, dtype, overrides=None, per_batch=True):
    config = dataclasses.replace(config_for("InverseAPF", 2, dtype), device=runtime.device)
    values = dict(alpha=2.0, gamma=3.0, d_min=1.0, d_max=4.0, deadzone=0.01)
    values.update(overrides or {})
    parameters = ctl.InverseAPFParameters(
        **{
            key: (ctl.per_batch if per_batch else ctl.shared)(
                runtime.array(np.asarray([value, value] if per_batch else value, dtype=dtype))
            )
            for key, value in values.items()
        }
    )
    block = ctl.InverseAPF(
        config, parameters, storage_for("InverseAPF", config, runtime.zeros), runtime.context
    )
    return block


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("per_batch", [False, True])
@pytest.mark.parametrize(
    "parameter,value",
    [
        ("alpha", -1),
        ("gamma", -1),
        ("d_min", -1),
        ("d_max", 0),
        ("deadzone", -1),
        ("alpha", np.nan),
        ("gamma", np.inf),
    ],
)
def test_legacy_apf_prepare_and_apply_reject_invalid_domains(
    runtime, dtype, per_batch, parameter, value
):
    block = make_inverse(runtime, dtype, {parameter: value}, per_batch)
    assert np.all(
        runtime.numpy(block.storage.output.parameter_status) & ctl.STATUS_INVALID_PARAMETER
    )
    good = make_inverse(runtime, dtype, per_batch=per_batch)
    block.apply_parameters(good.parameters, runtime.context)
    assert not np.any(runtime.numpy(block.storage.output.parameter_status))
    # Reapply a distinct invalid prepared parameter set; status must change before step.
    invalid = make_inverse(runtime, dtype, {parameter: value}, per_batch)
    good.apply_parameters(invalid.parameters, runtime.context)
    assert np.all(
        runtime.numpy(good.storage.output.parameter_status) & ctl.STATUS_INVALID_PARAMETER
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_apf_repulsive_pose_deadzone_and_damping_match_independent_oracle(runtime, dtype):
    block = make_inverse(runtime, dtype)
    current = np.tile(np.eye(4), (2, 1, 1)).astype(dtype)
    goal = current.copy()
    goal[0, 0, 3] = 0.1
    goal[1, 0, 3] = 0.01  # boundary: damping only
    angle = 0.2
    goal[0, :3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    velocity = np.full((2, 6), 0.03, dtype=dtype)
    inputs = [runtime.array(value) for value in (current, goal, velocity)]
    runtime.captured_step(lambda: block.step(*inputs, context=runtime.context))
    error = np.zeros((2, 6))
    error[:, 0] = goal[:, 0, 3]
    error[0, 5] = angle
    squared = np.sum(error**2, axis=1)
    damping = 1 + 3 / (1 + 3 * squared)
    expected = -damping[:, None] * velocity.astype(np.float64)
    expected[0] -= 2 * error[0] / (1 + 2 * (squared[0] - 0.01**2))
    np.testing.assert_allclose(runtime.numpy(block.output), expected, atol=2e-7, rtol=2e-6)
    assert not np.any(runtime.numpy(block.status))
    assert runtime.numpy(block.output)[0, 0] < 0  # repulsive from the supplied pose


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legacy_apf_invalid_row_isolated_before_and_during_step(runtime, dtype):
    config = dataclasses.replace(config_for("InverseAPF", 2, dtype), device=runtime.device)
    parameters = ctl.InverseAPFParameters(
        *(
            ctl.per_batch(runtime.array(np.array(value, dtype=dtype)))
            for value in ([-1, 2], [3, 3], [1, 1], [4, 4], [0.01, 0.01])
        )
    )
    block = ctl.InverseAPF(
        config, parameters, storage_for("InverseAPF", config, runtime.zeros), runtime.context
    )
    np.testing.assert_array_equal(
        runtime.numpy(block.storage.output.parameter_status), [ctl.STATUS_INVALID_PARAMETER, 0]
    )
    pose = np.tile(np.eye(4), (2, 1, 1)).astype(dtype)
    goal = pose.copy()
    goal[:, 0, 3] = 0.1
    inputs = [runtime.array(value) for value in (pose, goal, np.zeros((2, 6), dtype))]
    runtime.captured_step(lambda: block.step(*inputs, context=runtime.context))
    np.testing.assert_array_equal(runtime.numpy(block.output)[0], np.zeros(6))
    assert int(runtime.numpy(block.status)[0]) & ctl.STATUS_SAFE_MODE
    assert not runtime.numpy(block.status)[1]
    assert runtime.numpy(block.output)[1, 0] < 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("clearance", [0.0, -0.1, np.nan])
def test_checked_sphere_rejects_contact_before_division(runtime, dtype, clearance):
    config = eq.EquationConfig(
        runtime.device, ctl.DType.FLOAT32 if dtype == np.float32 else ctl.DType.FLOAT64, 2
    )
    output, status = runtime.zeros(2, dtype), runtime.zeros(2, np.uint64)
    parameters = {
        key: ctl.shared(runtime.array(np.array(value, dtype=dtype)))
        for key, value in dict(
            surface_clearance=clearance, influence_distance=0.035, gain=0.00005
        ).items()
    }
    runtime.captured_step(
        lambda: eq.spherical_apf_repulsion_value(
            config, **parameters, output=output, status=status, context=runtime.context
        )
    )
    np.testing.assert_array_equal(runtime.numpy(output), [0, 0])
    assert np.all(runtime.numpy(status) & (ctl.STATUS_INVALID_INPUT | ctl.STATUS_NON_FINITE_INPUT))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_checked_wall_boundary_unilateral_and_normal_damping(runtime, dtype):
    distances = np.array([0, 0.01, -0.01, -0.01], dtype=dtype)
    velocities = np.array([-1, -1, -1, 1], dtype=dtype)
    config = eq.EquationConfig(
        runtime.device, ctl.DType.FLOAT32 if dtype == np.float32 else ctl.DType.FLOAT64, 4
    )
    parameters = {
        key: ctl.shared(runtime.array(np.array(value, dtype=dtype)))
        for key, value in dict(stiffness=500, damping=20, damping_activation_distance=0.005).items()
    }
    parameters.update(
        signed_distance=ctl.per_batch(runtime.array(distances)),
        normal_velocity=ctl.per_batch(runtime.array(velocities)),
    )
    output, status = runtime.zeros(4, dtype), runtime.zeros(4, np.uint64)
    runtime.captured_step(
        lambda: eq.virtual_wall_force_value(
            config, **parameters, output=output, status=status, context=runtime.context
        )
    )
    np.testing.assert_allclose(runtime.numpy(output), [0, 0, 25, 0], atol=1e-5)
    assert not np.any(runtime.numpy(status))
