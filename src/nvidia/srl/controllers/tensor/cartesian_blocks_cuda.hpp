/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_blocks_cuda.hpp
 * @brief Private CUDA tensor backend declarations for Cartesian controller blocks.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_cartesian_matrix6_parameters_cuda(const ParameterSetView& parameters,
                                               const char* parameter_name,
                                               std::size_t num_envs,
                                               MutableStatusView parameter_status);
Result reset_cartesian_stateless_cuda(std::size_t num_envs,
                                      const TensorView& reset_mask,
                                      MutableStatusView status);
Result step_cartesian_matrix6_cuda(const TensorView& input,
                                   const ParameterSetView& parameters,
                                   const char* parameter_name,
                                   StatusView parameter_status,
                                   const MutableTensorView& output,
                                   MutableStatusView status);
Result step_cartesian_virtual_mass_cuda(const TensorView& force,
                                        const ParameterSetView& parameters,
                                        StatusView parameter_status,
                                        const MutableTensorView& cartesian_acceleration,
                                        MutableStatusView status);
Result apply_cartesian_damping_design_parameters_cuda(const ParameterSetView& parameters,
                                                      std::size_t num_envs,
                                                      MutableStatusView parameter_status);
Result step_cartesian_damping_design_cuda(bool factorization_design,
                                          const TensorView& cartesian_velocity_error,
                                          const TensorView& cartesian_mass_matrix,
                                          const TensorView& stiffness_matrix,
                                          const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          const MutableTensorView& cartesian_wrench,
                                          MutableStatusView status);
Result apply_shaping_function_parameters_cuda(const ParameterSetView& parameters,
                                              std::size_t num_envs,
                                              std::size_t signal_dim,
                                              MutableStatusView parameter_status);
Result step_shaping_function_cuda(const TensorView& error,
                                  const ParameterSetView& parameters,
                                  StatusView parameter_status,
                                  const MutableTensorView& shaping_factor,
                                  MutableStatusView status);
Result apply_command_observer_block_parameters_cuda(std::size_t num_envs,
                                                    MutableStatusView parameter_status);
Result step_command_observer_block_cuda(const TensorView& command,
                                        StatusView parameter_status,
                                        const MutableTensorView& command_is_valid,
                                        MutableStatusView status);
Result apply_flow_funnel_parameters_cuda(const ParameterSetView& parameters,
                                         std::size_t num_envs,
                                         MutableStatusView parameter_status);
Result step_flow_funnel_cuda(const TensorView& cartesian_pose,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& cartesian_flow,
                             MutableStatusView status);
Result apply_inverse_apf_parameters_cuda(const ParameterSetView& parameters,
                                         std::size_t num_envs,
                                         MutableStatusView parameter_status);
Result step_inverse_apf_cuda(const TensorView& current_pose,
                             const TensorView& goal_pose,
                             const TensorView& cartesian_twist,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& cartesian_wrench,
                             MutableStatusView status);
Result apply_inertial_coupling_parameters_cuda(std::size_t num_envs,
                                               MutableStatusView parameter_status);
Result reset_inertial_coupling_cuda(std::size_t num_envs,
                                    std::size_t joint_dim,
                                    StateView& state,
                                    const TensorView& reset_mask,
                                    MutableStatusView status);
Result step_inertial_coupling_cuda(const TensorView& cartesian_velocity_error,
                                   const TensorView& joint_velocity,
                                   const TensorView& cartesian_mass_matrix,
                                   const TensorView& jacobian,
                                   const TensorView& dt,
                                   StatusView parameter_status,
                                   StateView& state,
                                   const MutableTensorView& cartesian_wrench,
                                   MutableStatusView status);
Result apply_cartesian_impedance_controller_parameters_cuda(const ParameterSetView& parameters,
                                                            std::size_t num_envs,
                                                            MutableStatusView parameter_status);
Result step_cartesian_impedance_controller_cuda(const TensorView& cartesian_pose_error,
                                                const TensorView& cartesian_velocity_error,
                                                const TensorView& desired_acceleration,
                                                const TensorView& joint_velocity,
                                                const TensorView& cartesian_mass_matrix,
                                                const TensorView& jacobian,
                                                const TensorView& dt,
                                                const ParameterSetView& parameters,
                                                StatusView parameter_status,
                                                StateView& state,
                                                const MutableTensorView& cartesian_wrench,
                                                MutableStatusView status);
Result apply_operational_space_controller_parameters_cuda(const ParameterSetView& parameters,
                                                          std::size_t num_envs,
                                                          MutableStatusView parameter_status);
Result step_operational_space_controller_cuda(const TensorView& cartesian_pose_error,
                                              const TensorView& cartesian_velocity_error,
                                              const TensorView& desired_acceleration,
                                              const TensorView& jacobian_dot_joint_velocity,
                                              const TensorView& cartesian_mass_matrix,
                                              const TensorView& jacobian,
                                              const TensorView& joint_bias_torque,
                                              const ParameterSetView& parameters,
                                              StatusView parameter_status,
                                              const MutableTensorView& joint_torque,
                                              MutableStatusView status);
Result apply_cartesian_admittance_controller_parameters_cuda(const ParameterSetView& parameters,
                                                             std::size_t num_envs,
                                                             MutableStatusView parameter_status);
Result reset_cartesian_admittance_controller_cuda(std::size_t num_envs,
                                                  StateView& state,
                                                  const TensorView& reset_mask,
                                                  MutableStatusView status);
Result step_cartesian_admittance_controller_cuda(const TensorView& external_wrench,
                                                 const TensorView& feedforward_wrench,
                                                 const TensorView& dt,
                                                 const ParameterSetView& parameters,
                                                 StatusView parameter_status,
                                                 StateView& state,
                                                 const MutableTensorView& commanded_acceleration,
                                                 MutableStatusView status);
Result apply_variable_cartesian_stiffness_parameters_cuda(const ParameterSetView& parameters,
                                                          std::size_t num_envs,
                                                          MutableStatusView parameter_status);
Result reset_variable_cartesian_stiffness_cuda(const ParameterSetView& parameters,
                                               StatusView parameter_status,
                                               StateView& state,
                                               const TensorView& reset_mask,
                                               MutableStatusView status);
Result step_variable_cartesian_stiffness_cuda(const TensorView& cartesian_pose_error,
                                              const TensorView& stiffness_matrix,
                                              const TensorView& dt,
                                              const ParameterSetView& parameters,
                                              StatusView parameter_status,
                                              StateView& state,
                                              const MutableTensorView& cartesian_wrench,
                                              MutableStatusView status);
Result apply_cartesian_force_controller_parameters_cuda(const ParameterSetView& parameters,
                                                        std::size_t num_envs,
                                                        bool use_shaping_function,
                                                        MutableStatusView parameter_status);
Result apply_cartesian_unified_force_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters,
    std::size_t num_envs,
    bool use_shaping_function,
    MutableStatusView parameter_status);
Result apply_variable_cartesian_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters, std::size_t num_envs, MutableStatusView parameter_status);
Result apply_variable_cartesian_unified_force_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters,
    std::size_t num_envs,
    bool use_shaping_function,
    MutableStatusView parameter_status);
Result reset_cartesian_force_controller_cuda(std::size_t num_envs,
                                             StateView& state,
                                             const TensorView& reset_mask,
                                             MutableStatusView status);
Result step_cartesian_force_controller_cuda(const TensorView& external_wrench,
                                            const TensorView& desired_wrench,
                                            const TensorView& feedforward_wrench,
                                            const TensorView& pose_error,
                                            const TensorView& dt,
                                            bool use_shaping_function,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            StateView& state,
                                            const MutableTensorView& commanded_wrench,
                                            MutableStatusView status);
Result combine_cartesian_wrench_sum_cuda(const TensorView& left_wrench,
                                         StatusView left_status,
                                         const TensorView& right_wrench,
                                         StatusView right_status,
                                         const MutableTensorView& cartesian_wrench,
                                         MutableStatusView status);
Result combine_variable_cartesian_impedance_cuda(const TensorView& stiffness_wrench,
                                                 StatusView stiffness_status,
                                                 const TensorView& damping_wrench,
                                                 StatusView damping_status,
                                                 const TensorView& inertial_wrench,
                                                 StatusView inertial_status,
                                                 const TensorView& desired_acceleration,
                                                 const TensorView& cartesian_mass_matrix,
                                                 const MutableTensorView& cartesian_wrench,
                                                 MutableStatusView status);

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
