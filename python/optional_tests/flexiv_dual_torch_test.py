# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Opt-in dual-Rizon reference, frame, and complete CPU/CUDA controller checks without Kit."""

from __future__ import annotations

# Standard Library
import importlib
import importlib.util
import math
import os
import pathlib
import sys

# Third Party
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTROLLERS_TEST_BIMANUAL_TORCH") != "1",
    reason="set CONTROLLERS_TEST_BIMANUAL_TORCH=1 in the ABI-matching Torch/Isaac environment",
)


@pytest.fixture(scope="module")
def companion():
    """Import the example and Isaac's standalone math utilities without launching Kit."""
    examples = pathlib.Path(__file__).resolve().parents[2] / "examples/isaaclab"
    sys.path.insert(0, str(examples))
    common = importlib.import_module("_parallel_common")
    api = common.require_controller_module()
    runner = importlib.import_module("_flexiv_dual_parallel")
    factory = importlib.import_module("_native_components")
    profiles = importlib.import_module("_hardware_profiles")
    spec = importlib.util.find_spec("isaaclab")
    assert spec is not None and spec.origin is not None
    math_path = pathlib.Path(spec.origin).parent / "utils/math.py"
    math_spec = importlib.util.spec_from_file_location("_dual_rizon_test_math", math_path)
    assert math_spec is not None and math_spec.loader is not None
    math_utils = importlib.util.module_from_spec(math_spec)
    sys.modules[math_spec.name] = math_utils
    math_spec.loader.exec_module(math_utils)
    try:
        yield runner, math_utils, api, factory, profiles, common
    finally:
        sys.path.remove(str(examples))
        sys.modules.pop(math_spec.name, None)


@pytest.fixture(params=["cpu", "cuda:0"])
def runtime(request):
    """Use real CUDA on a non-default caller stream when selected; unavailable CUDA fails."""
    # Third Party
    import torch

    device = request.param
    if device == "cpu":
        yield torch, device, None
    else:
        assert torch.cuda.is_available(), "CUDA qualification requires a functioning GPU"
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            yield torch, device, stream
        stream.synchronize()


def _native(array):
    """Expose NumPy CPU views or CUDA array interfaces without copying storage."""
    return array.numpy() if array.device.type == "cpu" else array


def _rotation(torch, angle, device):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64, device=device
    )


@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi - 1e-7, math.pi])
def test_capture_cartesian_midpoint_and_relative_displacement(companion, runtime, angle):
    """Rotated bases and large relative rotations retain zero initial native task error."""
    runner, math_utils, api, _, _, common = companion
    torch, device, stream = runtime
    left = torch.eye(4, dtype=torch.float64, device=device).repeat(16, 1, 1)
    left[:, :3, :3] = _rotation(torch, 0.4, device)
    left[:, :3, 3] = left.new_tensor([0.2, -0.3, 0.5])
    right_arm1 = left.clone()
    right_arm1[:, :3, :3] = left[:, :3, :3] @ _rotation(torch, angle, device)
    right_arm1[:, :3, 3] = left.new_tensor([0.8, 0.6, 0.7])
    base = torch.eye(4, dtype=torch.float64, device=device)
    base[:3, :3] = _rotation(torch, -0.6, device)
    base[:3, 3] = base.new_tensor([0.7, 0.1, 0.0])
    right_local = torch.linalg.inv(base) @ right_arm1
    absolute, relative = runner._capture_reference_poses(torch, math_utils, left, right_local, base)
    torch.testing.assert_close(
        absolute[:, :3, 3], left.new_tensor([[0.5, 0.15, 0.6]]).expand(16, 3)
    )
    torch.testing.assert_close(relative[:, :3, 3], left.new_tensor([[0.6, 0.9, 0.2]]).expand(16, 3))
    expected_rotation = left[:, :3, :3] @ _rotation(torch, angle / 2.0, device)
    torch.testing.assert_close(absolute[:, :3, :3], expected_rotation, rtol=1e-8, atol=1e-8)
    context = common.current_execution_context(api, torch) if stream is not None else None
    config = api.BimanualTaskConfig(
        api.Device.CUDA if stream is not None else api.Device.CPU,
        api.DType.FLOAT64,
        16,
        14,
        cartesian_frame=api.CartesianFrame.MAIN_FRAME_ORIENTATION,
        task_dim=12,
        frames=api.BimanualFrameConfig(arm1_t_arm2=tuple(base.cpu().flatten().tolist())),
    )
    errors = left.new_zeros((16, 12))
    jacobian = left.new_zeros((16, 12, 14))
    status = torch.zeros(16, dtype=torch.uint64, device=device)
    block = api.BimanualTaskKinematics(
        config, api.BimanualTaskStorage(_native(errors), _native(jacobian), _native(status))
    )
    jac = torch.eye(6, 7, dtype=torch.float64, device=device).repeat(16, 1, 1)
    block.step(*map(_native, (absolute, left, right_local, relative, jac, jac)), context)
    torch.testing.assert_close(errors, torch.zeros_like(errors), atol=1e-8, rtol=0.0)
    assert not torch.any(status.to(torch.int64)).item()


@pytest.mark.parametrize("mode", ["arm1", "arm2", "explicit-arm1", "explicit-arm2"])
def test_main_frame_rotates_all_reference_derivatives(companion, runtime, mode):
    """Use the hardware reference axes for position, velocity and acceleration vectors."""
    runner, _, api, _, _, _ = companion
    torch, device, _ = runtime
    base = torch.eye(4, dtype=torch.float64, device=device)
    base[:3, :3] = _rotation(torch, 0.8, device)
    reference = torch.eye(4, dtype=torch.float64, device=device)
    reference[:3, :3] = _rotation(torch, -0.3, device)
    frames = runner._resolve_frames(
        api,
        tuple(base.cpu().flatten().tolist()),
        mode,
        tuple(reference.cpu().flatten().tolist()) if mode.startswith("explicit-") else None,
    )
    main_r_arm1 = runner._main_rotation(torch, frames, dtype=torch.float64, device=device)
    angle = {"arm1": 0.0, "arm2": 0.8, "explicit-arm1": -0.3, "explicit-arm2": 0.5}[mode]
    torch.testing.assert_close(main_r_arm1, _rotation(torch, -angle, device))
    sampled = torch.tensor(runner._circle_reference(1.0), dtype=torch.float64, device=device)
    expected = torch.stack([main_r_arm1 @ vector for vector in sampled])
    torch.testing.assert_close(sampled @ main_r_arm1.T, expected)


@pytest.mark.parametrize("suffix", ["cartesian_impedance", "compliant_hold"])
def test_complete_profile_posture_fault_isolation_and_graph(companion, runtime, suffix):
    """Exercise the complete controller, both independent posture torques, and the caller stream."""
    _, _, api, factory, profiles, common = companion
    torch, device, stream = runtime
    context = common.current_execution_context(api, torch) if stream is not None else None
    profile = profiles.hardware_profile(f"flexiv_dual_bimanual_{suffix}", "rizon4s,rizon10s")
    parts = factory.prepare_components(
        api,
        profile,
        16,
        api.Device.CUDA if stream is not None else api.Device.CPU,
        lambda a: torch.as_tensor(a, device=device) if stream is not None else a.copy(),
        context,
    )
    pose = torch.eye(4, dtype=torch.float64, device=device).repeat(16, 1, 1)
    right = pose.clone()
    right[:, 1, 3] = 1.0
    absolute = pose.clone()
    absolute[:, 1, 3] = 0.5
    relative = right.clone()
    jac = torch.eye(6, 7, dtype=torch.float64, device=device).repeat(16, 1, 1)
    desired_velocity = torch.zeros((16, 12), dtype=torch.float64, device=device)
    acceleration = torch.zeros_like(desired_velocity)
    error = torch.zeros((16, 14), dtype=torch.float64, device=device)
    error[:, 6], error[:, 13] = 0.2, -0.4
    joint_velocity = torch.zeros_like(error)
    mass = torch.eye(14, dtype=torch.float64, device=device).repeat(16, 1, 1)
    bias = torch.zeros_like(error)
    dt = torch.full((16,), 0.001, dtype=torch.float64, device=device)
    roots = (
        parts.roots,
        pose,
        right,
        absolute,
        relative,
        jac,
        desired_velocity,
        acceleration,
        error,
        joint_velocity,
        mass,
        bias,
        dt,
    )
    pointers = common.capture_buffer_pointers(*roots)

    def step():
        parts.controller.step(
            *map(
                _native,
                (
                    absolute,
                    pose,
                    right,
                    relative,
                    jac,
                    jac,
                    desired_velocity,
                    acceleration,
                    error,
                    joint_velocity,
                    mass,
                    bias,
                    dt,
                ),
            ),
            context,
        )

    def output():
        return torch.as_tensor(parts.controller.output, device=device)

    step()
    expected = torch.zeros_like(error)
    expected[:, 6], expected[:, 13] = 0.4, -0.8
    torch.testing.assert_close(output(), expected, rtol=1e-8, atol=1e-8)
    if stream is not None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
        graph.replay()
    desired_velocity[:, 0] = 0.001
    if stream is not None:
        graph.replay()
    else:
        step()
    torch.testing.assert_close(
        torch.as_tensor(parts.controller.storage.task_velocity_error, device=device),
        desired_velocity,
    )
    assert (output()[:, 0].abs() > 0.0).all().item()
    desired_velocity.zero_()
    desired_velocity[0, 0] = float("nan")
    step()
    torch.testing.assert_close(output()[0], torch.zeros_like(error[0]))
    torch.testing.assert_close(output()[1:], expected[1:], rtol=1e-8, atol=1e-8)
    assert common.flagged_rows(
        torch, torch.as_tensor(parts.controller.status, device=device), api.STATUS_SAFE_MODE
    )[0].item()
    assert (
        not common.flagged_rows(
            torch, torch.as_tensor(parts.controller.status, device=device), api.STATUS_SAFE_MODE
        )[1:]
        .any()
        .item()
    )
    assert pointers == common.capture_buffer_pointers(*roots)
