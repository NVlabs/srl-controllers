# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""UR controller frame transforms read from the hardware URDF."""

from __future__ import annotations

# Standard Library
import math
import pathlib
from dataclasses import dataclass
from xml.etree import ElementTree

# Third Party
import numpy as np


@dataclass(frozen=True)
class CartesianFrameTransforms:
    """Fixed translations in metres and rotations as unit wxyz quaternions."""

    base_position_in_root_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    base_quaternion_in_root_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    tcp_position_in_body_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    tcp_quaternion_in_body_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            [left[0] * right[0] - np.dot(left[1:], right[1:])],
            left[0] * right[1:] + right[0] * left[1:] + np.cross(left[1:], right[1:]),
        )
    )


def _fixed_pose(
    model: ElementTree.Element, parent: str, child: str
) -> tuple[np.ndarray, np.ndarray]:
    for joint in model.findall("joint"):
        if (
            joint.attrib["type"] == "fixed"
            and joint.find("parent").attrib["link"] == parent
            and joint.find("child").attrib["link"] == child
        ):
            origin = joint.find("origin")
            position = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") / 2
            qx = np.array([math.cos(roll), math.sin(roll), 0.0, 0.0])
            qy = np.array([math.cos(pitch), 0.0, math.sin(pitch), 0.0])
            qz = np.array([math.cos(yaw), 0.0, 0.0, math.sin(yaw)])
            return position, _quaternion_product(_quaternion_product(qz, qy), qx)
    raise ValueError(f"UR model has no fixed transform {parent} -> {child}")


def ur_frame_transforms(
    robot_model: str, tcp_offset: tuple[float, ...] = (0.0,) * 6
) -> CartesianFrameTransforms:
    """Read the hardware URDF base/flange conventions and configured TCP offset.

    TCP offset is translation in metres followed by a rotation vector in radians,
    expressed relative to flange, exactly as in UrKinematics.configure_tcp().
    """
    if robot_model not in {"ur5e", "ur10e"}:
        raise ValueError(f"unsupported UR model: {robot_model}")
    if len(tcp_offset) != 6 or not all(math.isfinite(v) for v in tcp_offset):
        raise ValueError("UR TCP offset must contain six finite m/rotation-vector-rad values")
    model = ElementTree.parse(
        pathlib.Path(__file__).resolve().parents[1] / "ur/models" / f"{robot_model}.urdf"
    ).getroot()
    base_position, base_quaternion = _fixed_pose(model, "base_link", "base")
    flange_position, flange_quaternion = _fixed_pose(model, "wrist_3_link", "flange")
    translation = np.asarray(tcp_offset[:3])
    vector = np.asarray(tcp_offset[3:])
    angle = math.hypot(*vector)
    rotation = np.concatenate(
        ([math.cos(angle / 2)], vector * (math.sin(angle / 2) / angle if angle else 0.5))
    )
    rotated = _quaternion_product(
        _quaternion_product(flange_quaternion, np.concatenate(([0.0], translation))),
        flange_quaternion * np.array([1.0, -1.0, -1.0, -1.0]),
    )[1:]
    return CartesianFrameTransforms(
        tuple(base_position),
        tuple(base_quaternion),
        tuple(flange_position + rotated),
        tuple(_quaternion_product(flange_quaternion, rotation)),
    )
