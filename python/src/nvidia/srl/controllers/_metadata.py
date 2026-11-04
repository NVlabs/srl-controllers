# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Typed storage discovery without allocating controller arrays."""

# Standard Library
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class TensorRequirement:
    """Named array allocation, including dtype, extents, units and frame."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    units: str
    frame: str


@dataclass(frozen=True)
class Requirements:
    """Output, persistent-state and scratch allocations for a prepared component."""

    batch_size: int
    joint_dim: int
    outputs: tuple[TensorRequirement, ...]
    state: tuple[TensorRequirement, ...]
    scratch: tuple[TensorRequirement, ...]


def describe_requirements(name: str, config: Any) -> Requirements:
    """Describe public storage using the canonical C++ lifecycle query."""
    # NVIDIA
    from nvidia.srl._controllers.nvidia_srl_controllers_cpp import _controller_api

    batch, joints, outputs, state, scratch = _controller_api.requirements(name, config)
    return Requirements(
        batch,
        joints,
        *(tuple(TensorRequirement(*item) for item in group) for group in (outputs, state, scratch)),
    )


@dataclass(frozen=True)
class TaskWrenchToJointTorqueRequirements:
    """Allocations matching the raw command and nested filter storage."""

    unfiltered_joint_torque: TensorRequirement
    filter: Requirements


def describe_torque_output(config: Any) -> TaskWrenchToJointTorqueRequirements:
    """Expand the legacy flat query into its public storage hierarchy."""
    flat = describe_requirements("TaskWrenchToJointTorqueController", config)
    return TaskWrenchToJointTorqueRequirements(flat.scratch[0], replace(flat, scratch=()))


@dataclass(frozen=True)
class CartesianTorqueRequirements:
    """Allocations for every stage of a Cartesian torque controller."""

    mass: Requirements
    task: Requirements
    nullspace: Requirements
    output: TaskWrenchToJointTorqueRequirements


@dataclass(frozen=True)
class BimanualCartesianTorqueRequirements(CartesianTorqueRequirements):
    """Complete bimanual allocations, including task kinematics."""

    kinematics: Requirements
    velocity: Requirements


def describe_components(
    task: str, config: Any, *, bimanual: bool = False
) -> CartesianTorqueRequirements:
    """Resolve each component's canonical storage requirements."""
    mass = describe_requirements("OperationalSpaceMass", config)
    task_requirements = describe_requirements(task, config)
    nullspace = describe_requirements("NullspaceController", config)
    output = describe_torque_output(config)
    if bimanual:
        return BimanualCartesianTorqueRequirements(
            mass,
            task_requirements,
            nullspace,
            output,
            describe_requirements("BimanualTaskKinematics", config),
            Requirements(
                config.batch_size,
                config.joint_dim,
                (),
                (),
                (
                    TensorRequirement(
                        "task_velocity_error",
                        config.dtype.value,
                        (config.batch_size, 12),
                        "meter_per_second_and_radian_per_second",
                        "main_frame_orientation_at_task_origins",
                    ),
                ),
            ),
        )
    return CartesianTorqueRequirements(mass, task_requirements, nullspace, output)
