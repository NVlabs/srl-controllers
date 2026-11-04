# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Independent UR frame calculations against the hardware URDF."""

from __future__ import annotations

# Standard Library
import importlib
import pathlib
import sys
from xml.etree import ElementTree

# Third Party
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def adapter():
    sys.path.insert(0, str(ROOT / "examples/isaaclab"))
    try:
        yield importlib.import_module("_ur_adapter")
    finally:
        sys.path.pop(0)


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def _origin(joint):
    transform = np.eye(4)
    origin = joint.find("origin")
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        transform[:3, :3] = (
            _axis_rotation([0, 0, 1], yaw)
            @ _axis_rotation([0, 1, 0], pitch)
            @ _axis_rotation([1, 0, 0], roll)
        )
    return transform


def quaternion_pose(position, quaternion):
    """Build an independent matrix from a translation and unit wxyz quaternion."""
    quaternion = np.asarray(quaternion)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
    w, x, y, z = quaternion
    transform = np.eye(4)
    transform[:3, 3] = position
    transform[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return transform


def hardware_urdf_pose_and_jacobian(robot_model, joint_positions, tcp_offset):
    """Compute base-to-TCP pose and spatial Jacobian independently from the URDF."""
    model = ElementTree.parse(ROOT / f"examples/ur/models/{robot_model}.urdf").getroot()
    joint_names = tuple(
        joint.attrib["name"]
        for joint in model.findall("joint")
        if joint.attrib["type"] in {"revolute", "continuous"}
    )
    angles = dict(zip(joint_names, joint_positions, strict=True))
    poses = {"world": np.eye(4)}
    joints = list(model.findall("joint"))
    axes = {}
    while joints:
        unresolved = len(joints)
        for joint in joints[:]:
            parent = joint.find("parent").attrib["link"]
            if parent not in poses:
                continue
            transform = poses[parent] @ _origin(joint)
            if joint.attrib["type"] in {"revolute", "continuous"}:
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                axes[joint.attrib["name"]] = (transform[:3, 3], transform[:3, :3] @ axis)
                rotation = np.eye(4)
                rotation[:3, :3] = _axis_rotation(axis, angles[joint.attrib["name"]])
                transform = transform @ rotation
            poses[joint.find("child").attrib["link"]] = transform
            joints.remove(joint)
        assert len(joints) < unresolved, "disconnected UR model"
    offset = np.eye(4)
    offset[:3, 3] = tcp_offset[:3]
    angle = np.linalg.norm(tcp_offset[3:])
    if angle:
        offset[:3, :3] = _axis_rotation(tcp_offset[3:], angle)
    tcp = poses["flange"] @ offset
    jacobian = np.empty((6, 6))
    for column, name in enumerate(joint_names):
        position, axis = axes[name]
        jacobian[:3, column] = np.cross(axis, tcp[:3, 3] - position)
        jacobian[3:, column] = axis
    base_rotation = poses["base"][:3, :3].T
    jacobian[:3] = base_rotation @ jacobian[:3]
    jacobian[3:] = base_rotation @ jacobian[3:]
    return np.linalg.inv(poses["base"]) @ tcp, jacobian, poses


@pytest.mark.parametrize("robot_model", ["ur5e", "ur10e"])
@pytest.mark.parametrize("tcp_offset", [(0.0,) * 6, (0.02, -0.01, 0.15, 0.1, -0.05, 0.02)])
def test_ur_fixed_frames_match_hardware_model(adapter, robot_model, tcp_offset):
    transforms = adapter.ur_frame_transforms(robot_model, tcp_offset)
    base = quaternion_pose(
        transforms.base_position_in_root_m, transforms.base_quaternion_in_root_wxyz
    )
    tcp = quaternion_pose(transforms.tcp_position_in_body_m, transforms.tcp_quaternion_in_body_wxyz)
    q = np.array([0.2, -1.3, -1.1, -1.6, 1.7, 0.3])
    expected, _, poses = hardware_urdf_pose_and_jacobian(robot_model, q, tcp_offset)
    observed = np.linalg.inv(base) @ poses["wrist_3_link"] @ tcp
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    np.testing.assert_allclose(base[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-12)


@pytest.mark.parametrize("tcp_offset", [(0.0,), (0.0,) * 5 + (float("nan"),)])
def test_ur_tcp_offset_rejects_invalid_values(adapter, tcp_offset):
    with pytest.raises(ValueError, match="six finite"):
        adapter.ur_frame_transforms("ur5e", tcp_offset)


@pytest.fixture(scope="module")
def bimanual_adapter():
    sys.path.insert(0, str(ROOT / "examples/isaaclab"))
    try:
        yield importlib.import_module("_flexiv_dual_parallel")
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize(
    "mode, expected, reference",
    [
        ("arm1", "arm1_base", "arm1"),
        ("arm2", "arm2_base", "arm1"),
        ("explicit-arm1", "explicit", "arm1"),
        ("explicit-arm2", "explicit", "arm2"),
    ],
)
def test_bimanual_frame_options_match_hardware(bimanual_adapter, mode, expected, reference):
    # NVIDIA
    import nvidia.srl.controllers as ctl

    identity = tuple(np.eye(4).ravel())
    frames = bimanual_adapter._resolve_frames(
        ctl, identity, mode, identity if expected == "explicit" else None
    )
    assert frames.main_frame.value == expected
    assert frames.explicit_reference.value == reference
    assert frames.arm1_t_arm2 == identity
    assert frames.reference_t_main == identity


@pytest.mark.parametrize(
    "fault", ["shape", "nan", "reflection", "scale", "affine", "main", "explicit"]
)
def test_bimanual_frames_reject_invalid_configuration(bimanual_adapter, fault):
    # NVIDIA
    import nvidia.srl.controllers as ctl

    base = np.eye(4)
    mode = "arm1"
    if fault == "shape":
        base = np.ones(15)
    elif fault == "nan":
        base[0, 3] = np.nan
    elif fault == "reflection":
        base[0, 0] = -1.0
    elif fault == "scale":
        base[0, 0] = 1.000001
    elif fault == "affine":
        base[3, 0] = 0.1
    elif fault == "main":
        mode = "unsupported"
    else:
        mode = "explicit-arm1"
    with pytest.raises(ValueError):
        bimanual_adapter._resolve_frames(ctl, tuple(base.ravel()), mode, None)


def test_bimanual_circle_velocity_and_acceleration_match_reference(bimanual_adapter):
    """Check derivatives independently at ramp, cruise, and both sides of the ramp boundary."""
    for time in (0.3, 1.0, 1.999, 2.001, 3.0):
        step = 1e-5
        position, velocity, acceleration = map(np.array, bimanual_adapter._circle_reference(time))
        before = tuple(map(np.array, bimanual_adapter._circle_reference(time - step)))
        after = tuple(map(np.array, bimanual_adapter._circle_reference(time + step)))
        np.testing.assert_allclose(velocity, (after[0] - before[0]) / (2 * step), atol=1e-9)
        np.testing.assert_allclose(acceleration, (after[1] - before[1]) / (2 * step), atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(position + [0.0, 0.0, 0.1]), 0.1, atol=1e-12)
    np.testing.assert_array_equal(bimanual_adapter._circle_reference(0.0), np.zeros((3, 3)))
