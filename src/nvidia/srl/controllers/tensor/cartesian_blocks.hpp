/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_blocks.hpp
 * @brief Batched tensor APIs for Cartesian and utility controller blocks.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

PrepareResult prepare_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                          const ParameterSetView& parameters,
                                          const MutableTensorView& cartesian_wrench);
Result apply_cartesian_stiffness_parameters(const ParameterSetView& parameters,
                                            std::size_t num_envs,
                                            MutableStatusView parameter_status);
Result reset_cartesian_stiffness(std::size_t num_envs,
                                 const TensorView& reset_mask,
                                 MutableStatusView status);
Result step_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                const ParameterSetView& parameters,
                                StatusView parameter_status,
                                const MutableTensorView& cartesian_wrench,
                                MutableStatusView status);

PrepareResult prepare_variable_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                                   const TensorView& stiffness_matrix,
                                                   const TensorView& dt,
                                                   const ParameterSetView& parameters,
                                                   StateView& state,
                                                   const MutableTensorView& cartesian_wrench);
Result apply_variable_cartesian_stiffness_parameters(const ParameterSetView& parameters,
                                                     std::size_t num_envs,
                                                     MutableStatusView parameter_status);
Result reset_variable_cartesian_stiffness(const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          StateView& state,
                                          const TensorView& reset_mask,
                                          MutableStatusView status);
Result step_variable_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                         const TensorView& stiffness_matrix,
                                         const TensorView& dt,
                                         const ParameterSetView& parameters,
                                         StatusView parameter_status,
                                         StateView& state,
                                         const MutableTensorView& cartesian_wrench,
                                         MutableStatusView status);

PrepareResult prepare_cartesian_damping(const TensorView& cartesian_velocity_error,
                                        const ParameterSetView& parameters,
                                        const MutableTensorView& cartesian_wrench);
Result apply_cartesian_damping_parameters(const ParameterSetView& parameters,
                                          std::size_t num_envs,
                                          MutableStatusView parameter_status);
Result reset_cartesian_damping(std::size_t num_envs,
                               const TensorView& reset_mask,
                               MutableStatusView status);
Result step_cartesian_damping(const TensorView& cartesian_velocity_error,
                              const ParameterSetView& parameters,
                              StatusView parameter_status,
                              const MutableTensorView& cartesian_wrench,
                              MutableStatusView status);

PrepareResult prepare_cartesian_damping_double_diagonalization(
    const TensorView& cartesian_velocity_error,
    const TensorView& cartesian_mass_matrix,
    const TensorView& stiffness_matrix,
    const ParameterSetView& parameters,
    const MutableTensorView& cartesian_wrench);
Result apply_cartesian_damping_double_diagonalization_parameters(
    const ParameterSetView& parameters, std::size_t num_envs, MutableStatusView parameter_status);
Result reset_cartesian_damping_double_diagonalization(std::size_t num_envs,
                                                      const TensorView& reset_mask,
                                                      MutableStatusView status);
Result step_cartesian_damping_double_diagonalization(const TensorView& cartesian_velocity_error,
                                                     const TensorView& cartesian_mass_matrix,
                                                     const TensorView& stiffness_matrix,
                                                     const ParameterSetView& parameters,
                                                     StatusView parameter_status,
                                                     const MutableTensorView& cartesian_wrench,
                                                     MutableStatusView status);

PrepareResult prepare_cartesian_damping_factorization_design(
    const TensorView& cartesian_velocity_error,
    const TensorView& cartesian_mass_matrix,
    const TensorView& stiffness_matrix,
    const ParameterSetView& parameters,
    const MutableTensorView& cartesian_wrench);
Result apply_cartesian_damping_factorization_design_parameters(const ParameterSetView& parameters,
                                                               std::size_t num_envs,
                                                               MutableStatusView parameter_status);
Result reset_cartesian_damping_factorization_design(std::size_t num_envs,
                                                    const TensorView& reset_mask,
                                                    MutableStatusView status);
Result step_cartesian_damping_factorization_design(const TensorView& cartesian_velocity_error,
                                                   const TensorView& cartesian_mass_matrix,
                                                   const TensorView& stiffness_matrix,
                                                   const ParameterSetView& parameters,
                                                   StatusView parameter_status,
                                                   const MutableTensorView& cartesian_wrench,
                                                   MutableStatusView status);

PrepareResult prepare_cartesian_virtual_mass(const TensorView& force,
                                             const ParameterSetView& parameters,
                                             const MutableTensorView& cartesian_acceleration);
Result apply_cartesian_virtual_mass_parameters(const ParameterSetView& parameters,
                                               std::size_t num_envs,
                                               MutableStatusView parameter_status);
Result reset_cartesian_virtual_mass(std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    MutableStatusView status);
Result step_cartesian_virtual_mass(const TensorView& force,
                                   const ParameterSetView& parameters,
                                   StatusView parameter_status,
                                   const MutableTensorView& cartesian_acceleration,
                                   MutableStatusView status);

PrepareResult prepare_shaping_function(const TensorView& error,
                                       const ParameterSetView& parameters,
                                       const MutableTensorView& shaping_factor);
Result apply_shaping_function_parameters(const ParameterSetView& parameters,
                                         std::size_t num_envs,
                                         std::size_t signal_dim,
                                         MutableStatusView parameter_status);
Result reset_shaping_function(std::size_t num_envs,
                              const TensorView& reset_mask,
                              MutableStatusView status);
Result step_shaping_function(const TensorView& error,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& shaping_factor,
                             MutableStatusView status);

PrepareResult prepare_command_observer_block(const TensorView& command,
                                             const MutableTensorView& command_is_valid);
Result apply_command_observer_block_parameters(std::size_t num_envs,
                                               MutableStatusView parameter_status);
Result reset_command_observer_block(std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    MutableStatusView status);
Result step_command_observer_block(const TensorView& command,
                                   StatusView parameter_status,
                                   const MutableTensorView& command_is_valid,
                                   MutableStatusView status);

PrepareResult prepare_flow_funnel(const TensorView& cartesian_pose,
                                  const ParameterSetView& parameters,
                                  const MutableTensorView& cartesian_flow);
Result apply_flow_funnel_parameters(const ParameterSetView& parameters,
                                    std::size_t num_envs,
                                    MutableStatusView parameter_status);
Result reset_flow_funnel(std::size_t num_envs,
                         const TensorView& reset_mask,
                         MutableStatusView status);
Result step_flow_funnel(const TensorView& cartesian_pose,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        const MutableTensorView& cartesian_flow,
                        MutableStatusView status);

PrepareResult prepare_inverse_apf(const TensorView& current_pose,
                                  const TensorView& goal_pose,
                                  const TensorView& cartesian_twist,
                                  const ParameterSetView& parameters,
                                  const MutableTensorView& cartesian_wrench);
Result apply_inverse_apf_parameters(const ParameterSetView& parameters,
                                    std::size_t num_envs,
                                    MutableStatusView parameter_status);
Result reset_inverse_apf(std::size_t num_envs,
                         const TensorView& reset_mask,
                         MutableStatusView status);
Result step_inverse_apf(const TensorView& current_pose,
                        const TensorView& goal_pose,
                        const TensorView& cartesian_twist,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        const MutableTensorView& cartesian_wrench,
                        MutableStatusView status);

PrepareResult prepare_inertial_coupling(const TensorView& cartesian_velocity_error,
                                        const TensorView& joint_velocity,
                                        const TensorView& cartesian_mass_matrix,
                                        const TensorView& jacobian,
                                        const TensorView& dt,
                                        StateView& state,
                                        const MutableTensorView& cartesian_wrench);
Result apply_inertial_coupling_parameters(std::size_t num_envs, MutableStatusView parameter_status);
Result reset_inertial_coupling(std::size_t num_envs,
                               std::size_t joint_dim,
                               StateView& state,
                               const TensorView& reset_mask,
                               MutableStatusView status);
Result step_inertial_coupling(const TensorView& cartesian_velocity_error,
                              const TensorView& joint_velocity,
                              const TensorView& cartesian_mass_matrix,
                              const TensorView& jacobian,
                              const TensorView& dt,
                              StatusView parameter_status,
                              StateView& state,
                              const MutableTensorView& cartesian_wrench,
                              MutableStatusView status);

PrepareResult prepare_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
                                                     const TensorView& cartesian_velocity_error,
                                                     const TensorView& desired_acceleration,
                                                     const TensorView& joint_velocity,
                                                     const TensorView& cartesian_mass_matrix,
                                                     const TensorView& jacobian,
                                                     const TensorView& dt,
                                                     const ParameterSetView& parameters,
                                                     StateView& state,
                                                     const MutableTensorView& cartesian_wrench);
Result apply_cartesian_impedance_controller_parameters(const ParameterSetView& parameters,
                                                       std::size_t num_envs,
                                                       MutableStatusView parameter_status);
Result reset_cartesian_impedance_controller(std::size_t num_envs,
                                            std::size_t joint_dim,
                                            StateView& state,
                                            const TensorView& reset_mask,
                                            MutableStatusView status);
Result step_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
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

PrepareResult prepare_cartesian_admittance_controller(
    const TensorView& external_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    const ParameterSetView& parameters,
    StateView& state,
    const MutableTensorView& commanded_acceleration);
Result apply_cartesian_admittance_controller_parameters(const ParameterSetView& parameters,
                                                        std::size_t num_envs,
                                                        MutableStatusView parameter_status);
Result reset_cartesian_admittance_controller(std::size_t num_envs,
                                             StateView& state,
                                             const TensorView& reset_mask,
                                             MutableStatusView status);
Result step_cartesian_admittance_controller(const TensorView& external_wrench,
                                            const TensorView& feedforward_wrench,
                                            const TensorView& dt,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            StateView& state,
                                            const MutableTensorView& commanded_acceleration,
                                            MutableStatusView status);

PrepareResult prepare_operational_space_controller(const TensorView& cartesian_pose_error,
                                                   const TensorView& cartesian_velocity_error,
                                                   const TensorView& desired_acceleration,
                                                   const TensorView& jacobian_dot_joint_velocity,
                                                   const TensorView& cartesian_mass_matrix,
                                                   const TensorView& jacobian,
                                                   const TensorView& joint_bias_torque,
                                                   const ParameterSetView& parameters,
                                                   const MutableTensorView& joint_torque);
Result apply_operational_space_controller_parameters(const ParameterSetView& parameters,
                                                     std::size_t num_envs,
                                                     MutableStatusView parameter_status);
Result reset_operational_space_controller(std::size_t num_envs,
                                          const TensorView& reset_mask,
                                          MutableStatusView status);
Result step_operational_space_controller(const TensorView& cartesian_pose_error,
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

PrepareResult prepare_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    bool use_shaping_function,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench);
Result apply_cartesian_unified_force_impedance_controller_parameters(
    const ParameterSetView& parameters,
    std::size_t num_envs,
    bool use_shaping_function,
    MutableStatusView parameter_status);
Result reset_cartesian_unified_force_impedance_controller(std::size_t num_envs,
                                                          std::size_t joint_dim,
                                                          StateView& state,
                                                          const TensorView& reset_mask,
                                                          MutableStatusView status);
Result step_cartesian_unified_force_impedance_controller(const TensorView& cartesian_pose_error,
                                                         const TensorView& cartesian_velocity_error,
                                                         const TensorView& desired_acceleration,
                                                         const TensorView& joint_velocity,
                                                         const TensorView& cartesian_mass_matrix,
                                                         const TensorView& jacobian,
                                                         const TensorView& external_wrench,
                                                         const TensorView& desired_wrench,
                                                         const TensorView& feedforward_wrench,
                                                         const TensorView& dt,
                                                         bool use_shaping_function,
                                                         const ParameterSetView& parameters,
                                                         StatusView parameter_status,
                                                         StateView& state,
                                                         ScratchView& scratch,
                                                         const MutableTensorView& cartesian_wrench,
                                                         MutableStatusView status);

PrepareResult prepare_variable_cartesian_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& dt,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench);
Result apply_variable_cartesian_impedance_controller_parameters(const ParameterSetView& parameters,
                                                                std::size_t num_envs,
                                                                MutableStatusView parameter_status);
Result reset_variable_cartesian_impedance_controller(const ParameterSetView& parameters,
                                                     StatusView parameter_status,
                                                     std::size_t num_envs,
                                                     std::size_t joint_dim,
                                                     StateView& state,
                                                     const TensorView& reset_mask,
                                                     MutableStatusView status);
Result step_variable_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
                                                    const TensorView& stiffness_matrix,
                                                    const TensorView& cartesian_velocity_error,
                                                    const TensorView& desired_acceleration,
                                                    const TensorView& joint_velocity,
                                                    const TensorView& cartesian_mass_matrix,
                                                    const TensorView& jacobian,
                                                    const TensorView& dt,
                                                    const ParameterSetView& parameters,
                                                    StatusView parameter_status,
                                                    StateView& state,
                                                    ScratchView& scratch,
                                                    const MutableTensorView& cartesian_wrench,
                                                    MutableStatusView status);

PrepareResult prepare_variable_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    bool use_shaping_function,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench);
Result apply_variable_cartesian_unified_force_impedance_controller_parameters(
    const ParameterSetView& parameters,
    std::size_t num_envs,
    bool use_shaping_function,
    MutableStatusView parameter_status);
Result reset_variable_cartesian_unified_force_impedance_controller(
    const ParameterSetView& parameters,
    StatusView parameter_status,
    std::size_t num_envs,
    std::size_t joint_dim,
    StateView& state,
    const TensorView& reset_mask,
    MutableStatusView status);
Result step_variable_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    bool use_shaping_function,
    const ParameterSetView& parameters,
    StatusView parameter_status,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench,
    MutableStatusView status);

PrepareResult prepare_cartesian_force_controller(const TensorView& external_wrench,
                                                 const TensorView& desired_wrench,
                                                 const TensorView& feedforward_wrench,
                                                 const TensorView& pose_error,
                                                 const TensorView& dt,
                                                 bool use_shaping_function,
                                                 const ParameterSetView& parameters,
                                                 StateView& state,
                                                 const MutableTensorView& commanded_wrench);
Result apply_cartesian_force_controller_parameters(const ParameterSetView& parameters,
                                                   std::size_t num_envs,
                                                   bool use_shaping_function,
                                                   MutableStatusView parameter_status);
Result reset_cartesian_force_controller(std::size_t num_envs,
                                        StateView& state,
                                        const TensorView& reset_mask,
                                        MutableStatusView status);
Result step_cartesian_force_controller(const TensorView& external_wrench,
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

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
