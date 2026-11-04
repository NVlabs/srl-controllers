// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#pragma once
#include "nvidia/srl/controllers/equations.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
// Private compatibility names share the installed canonical equations.
using ::nvidia::srl::controllers::equations::absolute_value;
using ::nvidia::srl::controllers::equations::admittance_force_balance_value;
using ::nvidia::srl::controllers::equations::arc_tangent2;
using ::nvidia::srl::controllers::equations::cartesian_damping_value;
using ::nvidia::srl::controllers::equations::cartesian_force_command_value;
using ::nvidia::srl::controllers::equations::cartesian_impedance_value;
using ::nvidia::srl::controllers::equations::cartesian_stiffness_value;
using ::nvidia::srl::controllers::equations::clamp_value;
using ::nvidia::srl::controllers::equations::command_observer_accepts;
using ::nvidia::srl::controllers::equations::ConstMatrixView;
using ::nvidia::srl::controllers::equations::ConstVectorView;
using ::nvidia::srl::controllers::equations::cosine;
using ::nvidia::srl::controllers::equations::damping_double_diagonalization_value;
using ::nvidia::srl::controllers::equations::damping_factorization_value;
using ::nvidia::srl::controllers::equations::deadband_value;
using ::nvidia::srl::controllers::equations::dense_feedback_row;
using ::nvidia::srl::controllers::equations::differentiator_value;
using ::nvidia::srl::controllers::equations::energy_tank_step;
using ::nvidia::srl::controllers::equations::EnergyTankStepResult;
using ::nvidia::srl::controllers::equations::finite_value;
using ::nvidia::srl::controllers::equations::flow_funnel_direction_value;
using ::nvidia::srl::controllers::equations::flow_funnel_geometry;
using ::nvidia::srl::controllers::equations::flow_funnel_weight;
using ::nvidia::srl::controllers::equations::FlowFunnelGeometry;
using ::nvidia::srl::controllers::equations::force_error_value;
using ::nvidia::srl::controllers::equations::inertial_coupling_value;
using ::nvidia::srl::controllers::equations::integrator_trapezoid_value;
using ::nvidia::srl::controllers::equations::inverse_apf_damping;
using ::nvidia::srl::controllers::equations::inverse_apf_wrench_value;
using ::nvidia::srl::controllers::equations::joint_damping_value;
using ::nvidia::srl::controllers::equations::joint_diagonal_feedback_value;
using ::nvidia::srl::controllers::equations::joint_impedance_value;
using ::nvidia::srl::controllers::equations::joint_stiffness_value;
using ::nvidia::srl::controllers::equations::kPi;
using ::nvidia::srl::controllers::equations::LimitResult;
using ::nvidia::srl::controllers::equations::matrix_derivative_value;
using ::nvidia::srl::controllers::equations::MutableVectorView;
using ::nvidia::srl::controllers::equations::nullspace_feedback_value;
using ::nvidia::srl::controllers::equations::nullspace_projected_torque_value;
using ::nvidia::srl::controllers::equations::nullspace_projector_value;
using ::nvidia::srl::controllers::equations::operational_space_acceleration_value;
using ::nvidia::srl::controllers::equations::operational_space_inverse_mass_value;
using ::nvidia::srl::controllers::equations::operational_space_mass_parameters_valid;
using ::nvidia::srl::controllers::equations::proportional_value;
using ::nvidia::srl::controllers::equations::rotation_is_valid;
using ::nvidia::srl::controllers::equations::rotation_tolerance;
using ::nvidia::srl::controllers::equations::safe_output_value;
using ::nvidia::srl::controllers::equations::shaping_factor_value;
using ::nvidia::srl::controllers::equations::so3_log_error;
using ::nvidia::srl::controllers::equations::square_root;
using ::nvidia::srl::controllers::equations::symmetric_average;
using ::nvidia::srl::controllers::equations::symmetric_pair_is_valid;
using ::nvidia::srl::controllers::equations::torque_filter_value;
using ::nvidia::srl::controllers::equations::unified_force_impedance_value;
using ::nvidia::srl::controllers::equations::validate_energy_tank_parameters;
using ::nvidia::srl::controllers::equations::variable_stiffness_value;
using ::nvidia::srl::controllers::equations::vector_norm;
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
