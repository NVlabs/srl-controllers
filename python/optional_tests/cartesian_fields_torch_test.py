# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Simulator field composition without Isaac startup; opt-in Torch CPU/CUDA checks."""

# Standard Library
import importlib
import os
import pathlib
import sys

# Third Party
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTROLLERS_TEST_FIELD_TORCH") != "1",
    reason="set CONTROLLERS_TEST_FIELD_TORCH=1 with Torch and this checkout's extension",
)


@pytest.fixture(
    params=[("cpu", "float32"), ("cpu", "float64"), ("cuda:0", "float32"), ("cuda:0", "float64")]
)
def field_runtime(request):
    """Create a six-row caller-owned CPU or CUDA field workspace."""
    torch = pytest.importorskip("torch")
    examples = pathlib.Path(__file__).resolve().parents[2] / "examples" / "isaaclab"
    sys.path.insert(0, str(examples))
    common = importlib.import_module("_parallel_common")
    api = common.require_controller_module()
    module = importlib.import_module("_cartesian_fields")
    device, dtype_name = request.param
    dtype = getattr(torch, dtype_name)
    config = api.ControllerConfig(
        device=api.Device.CPU if device == "cpu" else api.Device.CUDA,
        dtype=api.DType.FLOAT32 if dtype_name == "float32" else api.DType.FLOAT64,
        batch_size=6,
        joint_dim=7,
        cartesian_frame=api.CartesianFrame.ROBOT_BASE,
    )
    context = (
        api.ExecutionContext() if device == "cpu" else common.current_execution_context(api, torch)
    )
    reference = torch.zeros((6, 3), dtype=dtype, device=device)
    return torch, api, module, common, config, context, reference


def test_all_faces_uncapped_corner_and_buffer_addresses(field_runtime):
    """Probe every wall normal, uncapped corner force and free guiding."""
    torch, api, module, common, config, context, reference = field_runtime
    fields = module.CartesianFields(api, torch, config, reference, "cartesian_virtual_walls_torque")
    pointers = common.capture_buffer_pointers(fields.buffer_roots())
    position, velocity = reference.clone(), reference.clone()
    for axis in range(3):
        position[2 * axis, axis], position[2 * axis + 1, axis] = -0.11, 0.11
    force = fields.step(position, velocity, 1.0, 0.001, context)
    torch.testing.assert_close(force, -position / 0.11 * 10.0, atol=1e-5, rtol=1e-5)
    position.fill_(0.2)
    force = fields.step(position, velocity, 1.0, 0.001, context)
    torch.testing.assert_close(force, reference.new_full((6, 3), -100))
    position.zero_()
    velocity.fill_(1.0)
    torch.testing.assert_close(fields.step(position, velocity, 1.0, 0.001, context), reference)
    position[:, 0] = 0.11
    expected = reference.clone()
    expected[:, 0] = -50.0
    torch.testing.assert_close(
        fields.step(position, velocity, 1.0, 0.001, context), expected, atol=1e-5, rtol=1e-5
    )
    assert pointers == common.capture_buffer_pointers(fields.buffer_roots())


def test_apf_gradient_ramp_uncapped_repulsion_and_invalid_row_isolation(field_runtime):
    """Compare uncapped force values and reject only invalid equation inputs."""
    torch, api, module, common, config, context, reference = field_runtime
    fields = module.CartesianFields(
        api, torch, config, reference, "cartesian_potential_field_torque"
    )
    pointers = common.capture_buffer_pointers(fields.buffer_roots())
    velocity = reference.clone()
    torch.testing.assert_close(fields.step(reference, velocity, 0.0, 0.001, context), reference)
    force = fields.step(reference, velocity, 0.5, 0.001, context)
    expected = reference.clone()
    expected[:, 1] = 3
    torch.testing.assert_close(force, expected)
    position = reference.clone()
    position[0] = fields.obstacle_offset
    position[1] = fields.obstacle_offset + reference.new_tensor((0, 0.01, 0))
    position[2] = fields.obstacle_offset + reference.new_tensor((0, 0.04, 0))
    position[3] = fields.obstacle_offset + reference.new_tensor((0, 0.024, 0))
    fields.step(position, velocity, 1, 0.001, context)
    assert fields.unsafe[:2].all() and not fields.unsafe[2:].any()
    assert torch.linalg.vector_norm(fields.force[3]) > 10.0
    torch.testing.assert_close(fields.force[:2], reference[:2])
    # At 20 mm clearance: 50*(goal-p) + outward negative gradient of
    # U = eta/2*(1/d - 1/d0)^2.
    expected = 50 * (fields.goal_offset - position[2])
    expected[1] += 0.00005 * (1 / 0.02 - 1 / 0.035) / 0.02**2
    torch.testing.assert_close(fields.force[2], expected, atol=1e-5, rtol=1e-5)
    fields.step(reference, velocity, 1, 0.001, context)
    assert not fields.unsafe.any()  # the simulator runner owns stopped-row latching
    assert pointers == common.capture_buffer_pointers(fields.buffer_roots())


def test_apf_goal_diagnostics_exclude_stopped_rows_without_hiding_clearance(field_runtime):
    """A stopped row must not prevent healthy-row goal checks or hide measured clearance."""
    torch, api, module, common, config, context, reference = field_runtime
    fields = module.CartesianFields(
        api, torch, config, reference, "cartesian_potential_field_torque"
    )
    position, velocity = reference.clone(), reference.clone()
    position[1:] = fields.goal_offset
    fields.step(position, velocity, 1.0, 0.5, context)
    active = torch.ones(6, dtype=torch.bool, device=reference.device)
    active[0] = False
    all_rows = fields.metrics()
    healthy = fields.metrics(active_rows=active)
    assert all_rows.maximum_goal_error_m > 0.005 and all_rows.minimum_settled_seconds == 0.0
    assert healthy.maximum_goal_error_m == 0.0 and healthy.minimum_settled_seconds == 0.5
    position[0] = fields.obstacle_offset
    fields.step(position, velocity, 1.0, 0.001, context)
    healthy = fields.metrics(active_rows=active)
    assert fields.unsafe[0] and healthy.minimum_clearance_m < 0.0
    active.zero_()
    empty = fields.metrics(active_rows=active)
    assert empty.minimum_settled_seconds == 0.0
    assert empty.maximum_goal_error_m != empty.maximum_goal_error_m
