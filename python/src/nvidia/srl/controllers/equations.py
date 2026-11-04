# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Checked stateless CPU/CUDA equation arrays, backed by the canonical C++ equations.

Numeric arguments use explicit shared/per_batch wrappers. Scalar outputs have shape [B];
vector outputs have shape [B,N]. Every output and status array is caller-owned. CUDA execution
is asynchronous: retain all allocations until the supplied stream has completed.
"""
# Standard Library
from dataclasses import dataclass
from typing import Any

# Local Folder
from ._api import Device, DType, ExecutionContext, Parameter, _backend


@dataclass(frozen=True)
class EquationConfig:
    """Execution device, scalar precision and number of independent rows."""

    device: Device
    dtype: DType
    batch_size: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def proportional_value(
    config: EquationConfig,
    *,
    gain: Parameter,
    input_signal: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "proportional_value",
        config,
        (
            gain,
            input_signal,
        ),
        (),
        (output,),
        status,
        context,
    )


def differentiator_value(
    config: EquationConfig,
    *,
    input_signal: Parameter,
    previous_signal: Parameter,
    t_delta: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "differentiator_value",
        config,
        (
            input_signal,
            previous_signal,
            t_delta,
        ),
        (),
        (output,),
        status,
        context,
    )


@dataclass(frozen=True)
class ClampValueOutputs:
    """Caller-owned named results of clamp_value."""

    value: Any
    saturated: Any


def clamp_value(
    config: EquationConfig,
    *,
    value: Parameter,
    lower_limit: Parameter,
    upper_limit: Parameter,
    outputs: ClampValueOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "clamp_value",
        config,
        (
            value,
            lower_limit,
            upper_limit,
        ),
        (),
        (
            outputs.value,
            outputs.saturated,
        ),
        status,
        context,
    )


@dataclass(frozen=True)
class IntegratorTrapezoidValueOutputs:
    """Caller-owned named results of integrator_trapezoid_value."""

    value: Any
    saturated: Any


def integrator_trapezoid_value(
    config: EquationConfig,
    *,
    integrated_signal: Parameter,
    input_signal: Parameter,
    previous_signal: Parameter,
    t_delta: Parameter,
    lower_limit: Parameter,
    upper_limit: Parameter,
    outputs: IntegratorTrapezoidValueOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "integrator_trapezoid_value",
        config,
        (
            integrated_signal,
            input_signal,
            previous_signal,
            t_delta,
            lower_limit,
            upper_limit,
        ),
        (),
        (
            outputs.value,
            outputs.saturated,
        ),
        status,
        context,
    )


def joint_diagonal_feedback_value(
    config: EquationConfig,
    *,
    diagonal_gain: Parameter,
    joint_error: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "joint_diagonal_feedback_value",
        config,
        (
            diagonal_gain,
            joint_error,
        ),
        (),
        (output,),
        status,
        context,
    )


def shaping_factor_value(
    config: EquationConfig,
    *,
    error: Parameter,
    maximum_distance: Parameter,
    shaping_distance: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "shaping_factor_value",
        config,
        (
            error,
            maximum_distance,
            shaping_distance,
        ),
        (),
        (output,),
        status,
        context,
    )


def cartesian_force_command_value(
    config: EquationConfig,
    *,
    proportional_term: Parameter,
    differentiator_term: Parameter,
    integral_term: Parameter,
    desired_wrench: Parameter,
    feedforward_wrench: Parameter,
    shaping_factor: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "cartesian_force_command_value",
        config,
        (
            proportional_term,
            differentiator_term,
            integral_term,
            desired_wrench,
            feedforward_wrench,
            shaping_factor,
        ),
        (),
        (output,),
        status,
        context,
    )


def validate_energy_tank_parameters(
    config: EquationConfig,
    *,
    upper_energy_limit: Parameter,
    lower_energy_limit: Parameter,
    initial_energy: Parameter,
    empty_threshold: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "validate_energy_tank_parameters",
        config,
        (
            upper_energy_limit,
            lower_energy_limit,
            initial_energy,
            empty_threshold,
        ),
        (),
        (output,),
        status,
        context,
    )


@dataclass(frozen=True)
class EnergyTankStepOutputs:
    """Caller-owned named results of energy_tank_step."""

    scale: Any
    filtered_power: Any
    state_variable: Any
    saturated: Any


def energy_tank_step(
    config: EquationConfig,
    *,
    state_variable: Parameter,
    power_signal: Parameter,
    additional_power: Parameter,
    t_delta: Parameter,
    upper_energy_limit: Parameter,
    lower_energy_limit: Parameter,
    empty_threshold: Parameter,
    power_noise: Parameter,
    outputs: EnergyTankStepOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "energy_tank_step",
        config,
        (
            state_variable,
            power_signal,
            additional_power,
            t_delta,
            upper_energy_limit,
            lower_energy_limit,
            empty_threshold,
            power_noise,
        ),
        (),
        (
            outputs.scale,
            outputs.filtered_power,
            outputs.state_variable,
            outputs.saturated,
        ),
        status,
        context,
    )


def rotation_tolerance(
    config: EquationConfig, *, output: Any, status: Any, context: ExecutionContext | None = None
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation("rotation_tolerance", config, (), (), (output,), status, context)


def vector_norm(
    config: EquationConfig,
    *,
    vector: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation("vector_norm", config, (vector,), (), (output,), status, context)


def rotation_is_valid(
    config: EquationConfig,
    *,
    pose: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation("rotation_is_valid", config, (pose,), (), (output,), status, context)


@dataclass(frozen=True)
class So3LogErrorOutputs:
    """Caller-owned named results of so3_log_error."""

    output: Any
    rotation_error: Any


def so3_log_error(
    config: EquationConfig,
    *,
    current_pose: Parameter,
    goal_pose: Parameter,
    outputs: So3LogErrorOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "so3_log_error",
        config,
        (
            current_pose,
            goal_pose,
        ),
        (),
        (
            outputs.output,
            outputs.rotation_error,
        ),
        status,
        context,
    )


def dense_feedback_row(
    config: EquationConfig,
    *,
    gain: Parameter,
    error: Parameter,
    row: int,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "dense_feedback_row",
        config,
        (
            gain,
            error,
        ),
        (row,),
        (output,),
        status,
        context,
    )


def cartesian_stiffness_value(
    config: EquationConfig,
    *,
    stiffness: Parameter,
    pose_error: Parameter,
    row: int,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "cartesian_stiffness_value",
        config,
        (
            stiffness,
            pose_error,
        ),
        (row,),
        (output,),
        status,
        context,
    )


def cartesian_damping_value(
    config: EquationConfig,
    *,
    damping: Parameter,
    velocity_error: Parameter,
    row: int,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "cartesian_damping_value",
        config,
        (
            damping,
            velocity_error,
        ),
        (row,),
        (output,),
        status,
        context,
    )


def joint_stiffness_value(
    config: EquationConfig,
    *,
    stiffness: Parameter,
    position_error: Parameter,
    row: int,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "joint_stiffness_value",
        config,
        (
            stiffness,
            position_error,
        ),
        (row,),
        (output,),
        status,
        context,
    )


def joint_damping_value(
    config: EquationConfig,
    *,
    damping: Parameter,
    velocity_error: Parameter,
    row: int,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "joint_damping_value",
        config,
        (
            damping,
            velocity_error,
        ),
        (row,),
        (output,),
        status,
        context,
    )


def variable_stiffness_value(
    config: EquationConfig,
    *,
    previous: Parameter,
    requested: Parameter,
    lower: Parameter,
    upper: Parameter,
    maximum_rate: Parameter,
    dt: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "variable_stiffness_value",
        config,
        (
            previous,
            requested,
            lower,
            upper,
            maximum_rate,
            dt,
        ),
        (),
        (output,),
        status,
        context,
    )


def deadband_value(
    config: EquationConfig,
    *,
    value: Parameter,
    threshold: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "deadband_value",
        config,
        (
            value,
            threshold,
        ),
        (),
        (output,),
        status,
        context,
    )


def force_error_value(
    config: EquationConfig,
    *,
    desired: Parameter,
    measured: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "force_error_value",
        config,
        (
            desired,
            measured,
        ),
        (),
        (output,),
        status,
        context,
    )


def matrix_derivative_value(
    config: EquationConfig,
    *,
    current: Parameter,
    previous: Parameter,
    dt: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "matrix_derivative_value",
        config,
        (
            current,
            previous,
            dt,
        ),
        (),
        (output,),
        status,
        context,
    )


@dataclass(frozen=True)
class FlowFunnelGeometryOutputs:
    """Caller-owned named results of flow_funnel_geometry."""

    length: Any
    projection_distance: Any
    radius: Any
    valid: Any
    unit_direction: Any
    unit_direction_to_projection: Any


def flow_funnel_geometry(
    config: EquationConfig,
    *,
    start: Parameter,
    end: Parameter,
    position: Parameter,
    outputs: FlowFunnelGeometryOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "flow_funnel_geometry",
        config,
        (
            start,
            end,
            position,
        ),
        (),
        (
            outputs.length,
            outputs.projection_distance,
            outputs.radius,
            outputs.valid,
            outputs.unit_direction,
            outputs.unit_direction_to_projection,
        ),
        status,
        context,
    )


def flow_funnel_weight(
    config: EquationConfig,
    *,
    radius: Parameter,
    funnel_radius: Parameter,
    maximum_distance: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "flow_funnel_weight",
        config,
        (
            radius,
            funnel_radius,
            maximum_distance,
        ),
        (),
        (output,),
        status,
        context,
    )


def flow_funnel_direction_value(
    config: EquationConfig,
    *,
    radius: Parameter,
    funnel_radius: Parameter,
    maximum_distance: Parameter,
    weight: Parameter,
    funnel_direction: Parameter,
    projection_direction: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "flow_funnel_direction_value",
        config,
        (
            radius,
            funnel_radius,
            maximum_distance,
            weight,
            funnel_direction,
            projection_direction,
        ),
        (),
        (output,),
        status,
        context,
    )


def virtual_wall_force_value(
    config: EquationConfig,
    *,
    signed_distance: Parameter,
    normal_velocity: Parameter,
    stiffness: Parameter,
    damping: Parameter,
    damping_activation_distance: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "virtual_wall_force_value",
        config,
        (
            signed_distance,
            normal_velocity,
            stiffness,
            damping,
            damping_activation_distance,
        ),
        (),
        (output,),
        status,
        context,
    )


def spherical_apf_repulsion_value(
    config: EquationConfig,
    *,
    surface_clearance: Parameter,
    influence_distance: Parameter,
    gain: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "spherical_apf_repulsion_value",
        config,
        (
            surface_clearance,
            influence_distance,
            gain,
        ),
        (),
        (output,),
        status,
        context,
    )


def validate_inverse_apf_parameters(
    config: EquationConfig,
    *,
    alpha: Parameter,
    gamma: Parameter,
    minimum_damping: Parameter,
    maximum_damping: Parameter,
    deadzone: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "validate_inverse_apf_parameters",
        config,
        (
            alpha,
            gamma,
            minimum_damping,
            maximum_damping,
            deadzone,
        ),
        (),
        (output,),
        status,
        context,
    )


def inverse_apf_damping(
    config: EquationConfig,
    *,
    error_norm_squared: Parameter,
    minimum_damping: Parameter,
    maximum_damping: Parameter,
    gamma: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "inverse_apf_damping",
        config,
        (
            error_norm_squared,
            minimum_damping,
            maximum_damping,
            gamma,
        ),
        (),
        (output,),
        status,
        context,
    )


def inverse_apf_wrench_value(
    config: EquationConfig,
    *,
    error: Parameter,
    velocity: Parameter,
    error_norm: Parameter,
    error_norm_squared: Parameter,
    alpha: Parameter,
    deadzone: Parameter,
    damping: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "inverse_apf_wrench_value",
        config,
        (
            error,
            velocity,
            error_norm,
            error_norm_squared,
            alpha,
            deadzone,
            damping,
        ),
        (),
        (output,),
        status,
        context,
    )


def damping_factorization_value(
    config: EquationConfig,
    *,
    mass_times_damping_times_stiffness: Parameter,
    stiffness_times_damping_times_mass: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "damping_factorization_value",
        config,
        (
            mass_times_damping_times_stiffness,
            stiffness_times_damping_times_mass,
        ),
        (),
        (output,),
        status,
        context,
    )


def damping_double_diagonalization_value(
    config: EquationConfig,
    *,
    mass_times_damping_times_stiffness_times_mass_transpose: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "damping_double_diagonalization_value",
        config,
        (mass_times_damping_times_stiffness_times_mass_transpose,),
        (),
        (output,),
        status,
        context,
    )


def command_observer_accepts(
    config: EquationConfig,
    *,
    command: Parameter,
    absolute_limit: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "command_observer_accepts",
        config,
        (
            command,
            absolute_limit,
        ),
        (),
        (output,),
        status,
        context,
    )


def inertial_coupling_value(
    config: EquationConfig,
    *,
    mass_derivative_times_velocity: Parameter,
    mass_times_jacobian_derivative_velocity: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "inertial_coupling_value",
        config,
        (
            mass_derivative_times_velocity,
            mass_times_jacobian_derivative_velocity,
        ),
        (),
        (output,),
        status,
        context,
    )


def cartesian_impedance_value(
    config: EquationConfig,
    *,
    stiffness_wrench: Parameter,
    damping_wrench: Parameter,
    inertial_wrench: Parameter,
    acceleration_wrench: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "cartesian_impedance_value",
        config,
        (
            stiffness_wrench,
            damping_wrench,
            inertial_wrench,
            acceleration_wrench,
        ),
        (),
        (output,),
        status,
        context,
    )


def joint_impedance_value(
    config: EquationConfig,
    *,
    stiffness_torque: Parameter,
    damping_torque: Parameter,
    acceleration_torque: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "joint_impedance_value",
        config,
        (
            stiffness_torque,
            damping_torque,
            acceleration_torque,
        ),
        (),
        (output,),
        status,
        context,
    )


def nullspace_feedback_value(
    config: EquationConfig,
    *,
    stiffness_torque: Parameter,
    damping_torque: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "nullspace_feedback_value",
        config,
        (
            stiffness_torque,
            damping_torque,
        ),
        (),
        (output,),
        status,
        context,
    )


def nullspace_projector_value(
    config: EquationConfig,
    *,
    row: int,
    col: int,
    dynamically_consistent_inverse_term: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "nullspace_projector_value",
        config,
        (dynamically_consistent_inverse_term,),
        (
            row,
            col,
        ),
        (output,),
        status,
        context,
    )


def nullspace_projected_torque_value(
    config: EquationConfig,
    *,
    desired_torque: Parameter,
    task_space_correction: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "nullspace_projected_torque_value",
        config,
        (
            desired_torque,
            task_space_correction,
        ),
        (),
        (output,),
        status,
        context,
    )


def unified_force_impedance_value(
    config: EquationConfig,
    *,
    impedance_wrench: Parameter,
    force_wrench: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "unified_force_impedance_value",
        config,
        (
            impedance_wrench,
            force_wrench,
        ),
        (),
        (output,),
        status,
        context,
    )


def admittance_force_balance_value(
    config: EquationConfig,
    *,
    external_wrench: Parameter,
    feedforward_wrench: Parameter,
    stiffness_wrench: Parameter,
    damping_wrench: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "admittance_force_balance_value",
        config,
        (
            external_wrench,
            feedforward_wrench,
            stiffness_wrench,
            damping_wrench,
        ),
        (),
        (output,),
        status,
        context,
    )


def operational_space_acceleration_value(
    config: EquationConfig,
    *,
    desired_acceleration: Parameter,
    position_feedback: Parameter,
    velocity_feedback: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "operational_space_acceleration_value",
        config,
        (
            desired_acceleration,
            position_feedback,
            velocity_feedback,
        ),
        (),
        (output,),
        status,
        context,
    )


def operational_space_mass_parameters_valid(
    config: EquationConfig,
    *,
    regularization: Parameter,
    minimum_pivot: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "operational_space_mass_parameters_valid",
        config,
        (
            regularization,
            minimum_pivot,
        ),
        (),
        (output,),
        status,
        context,
    )


def symmetric_average(
    config: EquationConfig,
    *,
    forward: Parameter,
    transpose: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "symmetric_average",
        config,
        (
            forward,
            transpose,
        ),
        (),
        (output,),
        status,
        context,
    )


def symmetric_pair_is_valid(
    config: EquationConfig,
    *,
    forward: Parameter,
    transpose: Parameter,
    relative_tolerance: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "symmetric_pair_is_valid",
        config,
        (
            forward,
            transpose,
            relative_tolerance,
        ),
        (),
        (output,),
        status,
        context,
    )


def operational_space_inverse_mass_value(
    config: EquationConfig,
    *,
    forward: Parameter,
    transpose: Parameter,
    regularization: Parameter,
    diagonal: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "operational_space_inverse_mass_value",
        config,
        (
            forward,
            transpose,
            regularization,
            diagonal,
        ),
        (),
        (output,),
        status,
        context,
    )


@dataclass(frozen=True)
class TorqueFilterValueOutputs:
    """Caller-owned named results of torque_filter_value."""

    output: Any
    saturated: Any


def torque_filter_value(
    config: EquationConfig,
    *,
    requested: Parameter,
    previous: Parameter,
    lower_limit: Parameter,
    upper_limit: Parameter,
    maximum_rate: Parameter,
    dt: Parameter,
    initialized: Parameter,
    outputs: TorqueFilterValueOutputs,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "torque_filter_value",
        config,
        (
            requested,
            previous,
            lower_limit,
            upper_limit,
            maximum_rate,
            dt,
            initialized,
        ),
        (),
        (
            outputs.output,
            outputs.saturated,
        ),
        status,
        context,
    )


def safe_output_value(
    config: EquationConfig,
    *,
    safe_mode: Parameter,
    hold_last_valid: Parameter,
    has_last_valid: Parameter,
    current: Parameter,
    last_valid: Parameter,
    output: Any,
    status: Any,
    context: ExecutionContext | None = None,
) -> None:
    """Evaluate independently per batch row; inspect status before reuse."""
    _backend.equation(
        "safe_output_value",
        config,
        (
            safe_mode,
            hold_last_valid,
            has_last_valid,
            current,
            last_valid,
        ),
        (),
        (output,),
        status,
        context,
    )


__all__ = [
    "EquationConfig",
    "proportional_value",
    "differentiator_value",
    "ClampValueOutputs",
    "clamp_value",
    "IntegratorTrapezoidValueOutputs",
    "integrator_trapezoid_value",
    "joint_diagonal_feedback_value",
    "shaping_factor_value",
    "cartesian_force_command_value",
    "validate_energy_tank_parameters",
    "EnergyTankStepOutputs",
    "energy_tank_step",
    "rotation_tolerance",
    "vector_norm",
    "rotation_is_valid",
    "So3LogErrorOutputs",
    "so3_log_error",
    "dense_feedback_row",
    "cartesian_stiffness_value",
    "cartesian_damping_value",
    "joint_stiffness_value",
    "joint_damping_value",
    "variable_stiffness_value",
    "deadband_value",
    "force_error_value",
    "matrix_derivative_value",
    "FlowFunnelGeometryOutputs",
    "flow_funnel_geometry",
    "flow_funnel_weight",
    "flow_funnel_direction_value",
    "virtual_wall_force_value",
    "spherical_apf_repulsion_value",
    "validate_inverse_apf_parameters",
    "inverse_apf_damping",
    "inverse_apf_wrench_value",
    "damping_factorization_value",
    "damping_double_diagonalization_value",
    "command_observer_accepts",
    "inertial_coupling_value",
    "cartesian_impedance_value",
    "joint_impedance_value",
    "nullspace_feedback_value",
    "nullspace_projector_value",
    "nullspace_projected_torque_value",
    "unified_force_impedance_value",
    "admittance_force_balance_value",
    "operational_space_acceleration_value",
    "operational_space_mass_parameters_valid",
    "symmetric_average",
    "symmetric_pair_is_valid",
    "operational_space_inverse_mass_value",
    "TorqueFilterValueOutputs",
    "torque_filter_value",
    "safe_output_value",
]
