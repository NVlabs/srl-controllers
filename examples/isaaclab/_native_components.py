# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Prepare the hardware-configured native blocks on CPU or batched CUDA storage."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Callable

# Third Party
import numpy as np
from _hardware_profiles import HardwareProfile


@dataclass(frozen=True)
class NativeComponents:
    """Prepared native blocks and their caller-owned storage, with no plant adapter."""

    profile: HardwareProfile
    config: Any
    controller: Any
    mass: Any = None
    nullspace: Any = None
    nullspace_storage: Any = None
    torque_filter: Any = None
    admittance: Any = None
    shaping: Any = None
    arms: tuple[NativeComponents, ...] = ()
    roots: tuple[Any, ...] = ()


def prepare_components(
    api: Any,
    profile: HardwareProfile,
    batch_size: int,
    device: Any,
    allocate: Callable[[np.ndarray], Any],
    context: Any = None,
    *,
    bimanual_frames: Any = None,
) -> NativeComponents:
    """Use the same profiles and native classes for offline CPU and CUDA execution."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n, j, t = batch_size, profile.joint_dim, profile.task_dim or 6
    roots: list[Any] = []

    def zeros(shape: Any, dtype: Any = np.float64) -> Any:
        return allocate(np.zeros(shape, dtype=dtype))

    def parameter(values: Any, diagonal: bool = False) -> Any:
        array = np.asarray(values, dtype=np.float64)
        if diagonal:
            array = np.diag(array)
        return api.shared(allocate(array))

    def output(dim: int) -> Any:
        storage = api.OutputStorage(
            command=zeros((n, dim)),
            parameter_status=zeros(n, np.uint64),
            status=zeros(n, np.uint64),
            safety=api.SafetyState(zeros((n, dim)), zeros(n, np.bool_), zeros(n, np.uint64)),
        )
        roots.append(storage)
        return storage

    def inertial() -> Any:
        return api.InertialState(zeros((n, t, t)), zeros((n, t, j)), zeros(n, np.bool_))

    def admittance_state() -> Any:
        return api.AdmittanceState(
            zeros((n, 6)),
            zeros((n, 6)),
            zeros((n, 6)),
            zeros((n, 6)),
            zeros(n, np.bool_),
            zeros(n, np.bool_),
        )

    def native(cls: Any, parameters: Any, storage: Any, cfg: Any = None) -> Any:
        roots.extend((parameters, storage))
        instance = cls(cfg or config, parameters, storage, context)
        instance.reset(allocate(np.ones(n, dtype=np.bool_)), context)
        return instance

    frame = None if profile.frame == "unspecified" else api.CartesianFrame(profile.frame)
    config = api.ControllerConfig(
        device=device,
        dtype=api.DType.FLOAT64,
        batch_size=n,
        joint_dim=(
            0 if profile.family == "universal_robots" and profile.command_mode == "pose" else j
        ),
        task_dim=t,
        cartesian_frame=frame,
        use_shaping_function=profile.kind == "cartesian_unified_force_impedance_torque",
    )
    mass = nullspace = nullspace_storage = admittance = controller = torque_filter = shaping = None
    arms: tuple[NativeComponents, ...] = ()

    if profile.family == "flexiv_dual":
        config = api.BimanualTaskConfig(
            device=device,
            dtype=api.DType.FLOAT64,
            batch_size=n,
            joint_dim=j,
            task_dim=t,
            cartesian_frame=frame,
            frames=bimanual_frames if bimanual_frames is not None else api.BimanualFrameConfig(),
            coordination_mode=api.BimanualCoordinationMode.DECOUPLED_COOPERATIVE,
        )
        parameters = api.CartesianTorqueParameters(
            mass=api.OperationalSpaceMassParameters(
                parameter([profile.regularization]), parameter([profile.minimum_pivot])
            ),
            task=api.CartesianImpedanceParameters(
                parameter(profile.stiffness, True), parameter((profile.damping_factor,) * t)
            ),
            nullspace=api.NullspaceParameters(
                parameter(profile.nullspace_stiffness, True),
                parameter(profile.nullspace_damping, True),
            ),
            output=api.TorqueFilterParameters(
                parameter(tuple(-v for v in profile.torque_limits_nm)),
                parameter(profile.torque_limits_nm),
                parameter(profile.torque_rate_nm_per_s),
            ),
        )
        storage = api.BimanualCartesianImpedanceTorqueStorage(
            kinematics=api.BimanualTaskStorage(
                zeros((n, t)), zeros((n, t, j)), zeros(n, np.uint64)
            ),
            mass=api.OperationalSpaceMassStorage(
                zeros((n, t, t)),
                zeros(n, np.uint64),
                zeros(n, np.uint64),
                api.OperationalSpaceMassScratch(
                    zeros((n, j, j)), zeros((n, j, t)), zeros((n, t, t))
                ),
            ),
            task=api.CartesianImpedanceStorage(output(t), inertial()),
            nullspace=output(j),
            output=api.TaskWrenchToJointTorqueStorage(
                zeros((n, j)),
                api.TorqueFilterStorage(
                    output(j), api.TorqueFilterState(zeros((n, j)), zeros(n, np.bool_))
                ),
            ),
            task_velocity_error=zeros((n, t)),
        )
        controller = native(api.BimanualCartesianImpedanceTorqueController, parameters, storage)
        return NativeComponents(profile, config, controller, roots=tuple(roots))
    if profile.kind in {"joint_impedance", "static_joint_impedance"}:
        if profile.kind == "joint_impedance":
            controller = native(
                api.JointImpedanceController,
                api.JointImpedanceParameters(
                    parameter(profile.stiffness, True),
                    parameter((profile.damping_factor,) * j),
                ),
                output(j),
            )
        else:
            controller = native(
                api.StaticJointImpedanceController,
                api.StaticJointImpedanceParameters(
                    parameter(profile.stiffness, True), parameter(profile.damping, True)
                ),
                output(j),
                api.JointBlockConfig(device, api.DType.FLOAT64, n, j, mode=api.ComponentMode.DENSE),
            )
    elif profile.command_mode == "pose" or profile.kind == "cartesian_admittance_torque":
        state = admittance_state()
        if profile.kind == "cartesian_unified_force_admittance_pose":
            force = api.ForceErrorPidParameters(
                parameter(profile.force_proportional),
                parameter(profile.force_differentiator),
                parameter(profile.force_integral),
                parameter((0.0,) * 6),
                None,
                None,
                parameter(profile.wrench_deadzone),
            )

            def task_matrix(values: tuple[float, ...]) -> Any:
                matrix = np.diag(np.asarray(values, dtype=np.float64))
                if profile.force_reference_frame == "end_effector":
                    return api.per_batch(allocate(np.tile(matrix, (n, 1, 1))))
                return api.shared(allocate(matrix))

            parameters = api.CartesianUnifiedForceAdmittanceParameters(
                task_matrix(profile.admittance_stiffness),
                task_matrix(profile.admittance_damping),
                task_matrix(profile.admittance_mass),
                None,
                None,
                None,
                force,
            )
            shaping = native(
                api.ShapingFunction,
                api.ShapingFunctionParameters(
                    parameter(profile.force_maximum_distance),
                    parameter(profile.force_shaping_distance),
                ),
                api.ShapingFunctionStorage(output(6)),
                api.SignalBlockConfig(device, api.DType.FLOAT64, n, 6),
            )
            pid = api.ForceControllerState(
                zeros((n, 6)),
                zeros(n, np.bool_),
                zeros((n, 6)),
                zeros((n, 6)),
                zeros(n, np.bool_),
            )
            controller = native(
                api.CartesianUnifiedForceAdmittanceController,
                parameters,
                api.CartesianUnifiedForceAdmittanceStorage(
                    output(6), api.UnifiedForceAdmittanceState(pid, state)
                ),
            )
        else:
            admittance = native(
                api.CartesianAdmittancePoseController,
                api.CartesianAdmittanceParameters(
                    parameter(profile.admittance_stiffness, True),
                    parameter(profile.admittance_damping, True),
                    parameter(profile.admittance_mass, True),
                    parameter(profile.wrench_deadzone),
                    None,
                    None,
                    None,
                ),
                api.CartesianAdmittanceStorage(output(6), state),
            )
            if profile.command_mode == "pose":
                controller = admittance
    elif profile.kind in {
        "static_cartesian_impedance_torque",
        "cartesian_virtual_walls_torque",
        "cartesian_potential_field_torque",
    }:
        controller = native(
            api.StaticCartesianImpedanceController,
            api.StaticCartesianImpedanceParameters(
                parameter(profile.stiffness, True), parameter(profile.damping, True)
            ),
            output(6),
        )

        if profile.nullspace_stiffness:
            nullspace_storage = output(j)
            nullspace = native(
                api.NullspaceController,
                api.NullspaceParameters(
                    parameter(profile.nullspace_stiffness, True),
                    parameter(profile.nullspace_damping, True),
                ),
                nullspace_storage,
            )

    if controller is None and not arms:
        storage = api.OperationalSpaceMassStorage(
            zeros((n, t, t)),
            zeros(n, np.uint64),
            zeros(n, np.uint64),
            api.OperationalSpaceMassScratch(zeros((n, j, j)), zeros((n, j, t)), zeros((n, t, t))),
        )
        mass = native(
            api.OperationalSpaceMass,
            api.OperationalSpaceMassParameters(
                parameter([profile.regularization]),
                parameter([profile.minimum_pivot]) if profile.minimum_pivot is not None else None,
            ),
            storage,
        )
        parameters = api.CartesianImpedanceParameters(
            parameter(profile.stiffness, True), parameter((profile.damping_factor,) * t)
        )
        storage = api.CartesianImpedanceStorage(output(t), inertial())
        if profile.kind == "variable_cartesian_impedance_torque":
            parameters = api.VariableCartesianImpedanceParameters(
                parameter(profile.stiffness, True),
                parameter(profile.variable_maximum),
                parameter((0.0,) * t),
                parameter(profile.variable_maximum),
                parameter((profile.damping_factor,) * t),
            )
            storage = api.VariableImpedanceStorage(
                output(t),
                api.VariableImpedanceState(
                    zeros((n, t, t)), zeros((n, t, t)), zeros(n, np.bool_), inertial()
                ),
                api.VariableImpedanceScratch(
                    zeros((n, t)),
                    zeros((n, t)),
                    zeros((n, t)),
                    zeros(n, np.uint64),
                    zeros(n, np.uint64),
                    zeros(n, np.uint64),
                ),
            )
            controller = native(api.VariableCartesianImpedanceController, parameters, storage)
        elif profile.kind == "cartesian_unified_force_impedance_torque":
            parameters = api.CartesianUnifiedForceImpedanceParameters(
                parameters.stiffness_matrix,
                parameters.damping_factors,
                api.ForceControllerParameters(
                    parameter(profile.force_proportional),
                    parameter(profile.force_differentiator),
                    parameter(profile.force_integral),
                    parameter((0.0,) * 6),
                    parameter(profile.force_maximum_distance),
                    parameter(profile.force_shaping_distance),
                ),
            )
            storage = api.UnifiedControllerStorage(
                output(t),
                api.UnifiedControllerState(
                    inertial(),
                    api.ForceControllerState(
                        zeros((n, t)),
                        zeros(n, np.bool_),
                        zeros((n, t)),
                        zeros((n, t)),
                        zeros(n, np.bool_),
                    ),
                ),
                api.UnifiedControllerScratch(
                    zeros((n, t)), zeros((n, t)), zeros(n, np.uint64), zeros(n, np.uint64)
                ),
            )
            controller = native(api.CartesianUnifiedForceImpedanceController, parameters, storage)
        else:
            cls = (
                api.BimanualCartesianImpedanceController
                if t == 12
                else api.CartesianImpedanceController
            )
            controller = native(cls, parameters, storage)
        if profile.nullspace_stiffness:
            nullspace_storage = output(j)
            nullspace = native(
                api.NullspaceController,
                api.NullspaceParameters(
                    parameter(profile.nullspace_stiffness, True),
                    parameter(profile.nullspace_damping, True),
                ),
                nullspace_storage,
            )

    if profile.command_mode == "torque":
        torque_filter = native(
            api.TorqueFilter,
            api.TorqueFilterParameters(
                parameter(tuple(-v for v in profile.torque_limits_nm)),
                parameter(profile.torque_limits_nm),
                parameter(profile.torque_rate_nm_per_s),
            ),
            api.TorqueFilterStorage(
                output(j), api.TorqueFilterState(zeros((n, j)), zeros(n, np.bool_))
            ),
        )
        if profile.torque_filter_seed_zero:
            torque_filter.step(zeros((n, j)), allocate(np.full(n, profile.period_s)), context)
    return NativeComponents(
        profile=profile,
        config=config,
        controller=controller,
        mass=mass,
        nullspace=nullspace,
        nullspace_storage=nullspace_storage,
        torque_filter=torque_filter,
        admittance=admittance,
        shaping=shaping,
        arms=arms,
        roots=tuple(roots),
    )
