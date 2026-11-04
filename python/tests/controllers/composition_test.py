# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Fault propagation and bounded final-output history across custom control stages."""

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl


@pytest.mark.parametrize(
    "fault",
    [
        ctl.STATUS_UNSUPPORTED_MODE,
        ctl.STATUS_INVALID_PARAMETER,
        ctl.STATUS_INVALID_INPUT,
        ctl.STATUS_NON_FINITE_INPUT,
        ctl.STATUS_INVALID_DT,
        ctl.STATUS_NO_VALID_OUTPUT,
        ctl.STATUS_HOLD_EXPIRED,
    ],
)
def test_faults_without_safe_mode_cannot_be_treated_as_healthy(runtime, fault):
    first = runtime.array(np.array([fault, ctl.STATUS_SATURATION], np.uint64))
    second = runtime.zeros(2, np.uint64)
    merged = runtime.zeros(2, np.uint64)
    ctl.merge_status(first, second, output=merged, context=runtime.context)
    expected = [fault | ctl.STATUS_SAFE_MODE, ctl.STATUS_SATURATION]
    np.testing.assert_array_equal(runtime.numpy(merged), expected)
    ctl.merge_status(first, second, output=first, context=runtime.context)
    np.testing.assert_array_equal(runtime.numpy(first), expected)


def test_final_history_advances_once_and_expires(runtime):
    shape = (2, 3)
    state = ctl.SafetyState(
        runtime.zeros(shape, np.float64), runtime.zeros(2, bool), runtime.zeros(2, np.uint64)
    )
    status = runtime.zeros(2, np.uint64)
    ctl.reset_safety_state(state, status=status, context=runtime.context)
    command = runtime.array(np.full(shape, 4.0))
    healthy = runtime.zeros(2, np.uint64)
    ctl.apply_safe_output_policy(command, healthy, state, context=runtime.context)
    for tick in (1, 2, 3):
        faults = runtime.array(np.array([ctl.STATUS_INVALID_INPUT, 0], np.uint64))
        command = runtime.array(np.full(shape, 9.0))
        ctl.apply_safe_output_policy(
            command,
            faults,
            state,
            policy=ctl.SafeOutputPolicy.HOLD_LAST_VALID,
            max_hold_steps=2,
            context=runtime.context,
        )
        np.testing.assert_array_equal(runtime.numpy(command)[0], 4 if tick <= 2 else 0)
        np.testing.assert_array_equal(runtime.numpy(command)[1], 9)
        if tick <= 2:
            assert runtime.numpy(state.hold_count)[0] == tick
            assert runtime.numpy(faults)[0] & ctl.STATUS_OUTPUT_HELD
        else:
            assert runtime.numpy(faults)[0] & ctl.STATUS_HOLD_EXPIRED


def test_merge_rejects_alias_of_later_input_before_mutation():
    flags = np.array([ctl.STATUS_INVALID_INPUT], np.uint64)
    with pytest.raises(ValueError, match="first"):
        ctl.merge_status(np.zeros(1, np.uint64), flags, output=flags)
    assert flags[0] == ctl.STATUS_INVALID_INPUT


def test_component_requirements_cover_all_pipeline_stages():
    config = ctl.ControllerConfig(
        ctl.Device.CPU, ctl.DType.FLOAT64, 2, 7, cartesian_frame=ctl.CartesianFrame.ROBOT_BASE
    )
    for name in (
        "CartesianImpedanceTorqueController",
        "VariableCartesianImpedanceTorqueController",
    ):
        controller = getattr(ctl, name)
        complete = controller.component_requirements(config)
        assert complete.output.filter.outputs == controller.requirements(config).outputs
        assert complete.output.unfiltered_joint_torque.shape == (2, 7)
        assert complete.mass.outputs and complete.task.outputs and complete.nullspace.outputs
        assert len(complete.task.state) > 0


def test_final_policy_rejects_overlapping_arrays():
    command = np.ones((2, 3))
    state = ctl.SafetyState(command, np.zeros(2, bool), np.zeros(2, np.uint64))
    with pytest.raises(ValueError, match="overlap"):
        ctl.apply_safe_output_policy(command, np.zeros(2, np.uint64), state)
    state = ctl.SafetyState(np.zeros((2, 3)), np.zeros(2, bool), np.zeros(2, np.uint64))
    with pytest.raises(ValueError, match="overlap"):
        ctl.apply_safe_output_policy(command, state.hold_count, state)
