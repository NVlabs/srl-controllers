/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_blocks.hpp
 * @brief Batched tensor APIs for diagonal joint stiffness and damping blocks.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

PrepareResult prepare_joint_stiffness(ComponentMode mode,
                                      const TensorView& joint_position_error,
                                      const ParameterSetView& parameters,
                                      const MutableTensorView& joint_torque);
Result apply_joint_stiffness_parameters(ComponentMode mode,
                                        const ParameterSetView& parameters,
                                        std::size_t num_envs,
                                        std::size_t joint_dim,
                                        MutableStatusView parameter_status);
Result reset_joint_stiffness(ComponentMode mode,
                             std::size_t num_envs,
                             const TensorView& reset_mask,
                             MutableStatusView status);
Result step_joint_stiffness(ComponentMode mode,
                            const TensorView& joint_position_error,
                            const ParameterSetView& parameters,
                            StatusView parameter_status,
                            const MutableTensorView& joint_torque,
                            MutableStatusView status);

PrepareResult prepare_joint_damping(ComponentMode mode,
                                    const TensorView& joint_velocity_error,
                                    const ParameterSetView& parameters,
                                    const MutableTensorView& joint_torque);
Result apply_joint_damping_parameters(ComponentMode mode,
                                      const ParameterSetView& parameters,
                                      std::size_t num_envs,
                                      std::size_t joint_dim,
                                      MutableStatusView parameter_status);
Result reset_joint_damping(ComponentMode mode,
                           std::size_t num_envs,
                           const TensorView& reset_mask,
                           MutableStatusView status);
Result step_joint_damping(ComponentMode mode,
                          const TensorView& joint_velocity_error,
                          const ParameterSetView& parameters,
                          StatusView parameter_status,
                          const MutableTensorView& joint_torque,
                          MutableStatusView status);

PrepareResult prepare_joint_damping_double_diagonalization(const TensorView& joint_velocity_error,
                                                           const TensorView& joint_mass_matrix,
                                                           const TensorView& stiffness_matrix,
                                                           const ParameterSetView& parameters,
                                                           const MutableTensorView& joint_torque);
Result apply_joint_damping_double_diagonalization_parameters(const ParameterSetView& parameters,
                                                             std::size_t num_envs,
                                                             std::size_t joint_dim,
                                                             MutableStatusView parameter_status);
Result reset_joint_damping_double_diagonalization(std::size_t num_envs,
                                                  const TensorView& reset_mask,
                                                  MutableStatusView status);
Result step_joint_damping_double_diagonalization(const TensorView& joint_velocity_error,
                                                 const TensorView& joint_mass_matrix,
                                                 const TensorView& stiffness_matrix,
                                                 const ParameterSetView& parameters,
                                                 StatusView parameter_status,
                                                 const MutableTensorView& joint_torque,
                                                 MutableStatusView status);

PrepareResult prepare_joint_impedance_controller(const TensorView& joint_position_error,
                                                 const TensorView& joint_velocity_error,
                                                 const TensorView& desired_acceleration,
                                                 const TensorView& joint_mass_matrix,
                                                 const ParameterSetView& parameters,
                                                 const MutableTensorView& joint_torque);
Result apply_joint_impedance_controller_parameters(const ParameterSetView& parameters,
                                                   std::size_t num_envs,
                                                   std::size_t joint_dim,
                                                   MutableStatusView parameter_status);
Result reset_joint_impedance_controller(std::size_t num_envs,
                                        const TensorView& reset_mask,
                                        MutableStatusView status);
Result step_joint_impedance_controller(const TensorView& joint_position_error,
                                       const TensorView& joint_velocity_error,
                                       const TensorView& desired_acceleration,
                                       const TensorView& joint_mass_matrix,
                                       const ParameterSetView& parameters,
                                       StatusView parameter_status,
                                       const MutableTensorView& joint_torque,
                                       MutableStatusView status);

PrepareResult prepare_nullspace_controller(const TensorView& joint_position_error,
                                           const TensorView& joint_velocity,
                                           const TensorView& joint_mass_matrix,
                                           const TensorView& jacobian,
                                           const ParameterSetView& parameters,
                                           const MutableTensorView& joint_torque);
Result apply_nullspace_controller_parameters(const ParameterSetView& parameters,
                                             std::size_t num_envs,
                                             std::size_t joint_dim,
                                             MutableStatusView parameter_status);
Result reset_nullspace_controller(std::size_t num_envs,
                                  const TensorView& reset_mask,
                                  MutableStatusView status);
Result step_nullspace_controller(const TensorView& joint_position_error,
                                 const TensorView& joint_velocity,
                                 const TensorView& joint_mass_matrix,
                                 const TensorView& jacobian,
                                 const ParameterSetView& parameters,
                                 StatusView parameter_status,
                                 const MutableTensorView& joint_torque,
                                 MutableStatusView status);
Result step_bimanual_independent_nullspace_controller(const TensorView& joint_position_error,
                                                      const TensorView& joint_velocity,
                                                      const TensorView& joint_mass_matrix,
                                                      const TensorView& arm1_jacobian,
                                                      const TensorView& arm2_jacobian,
                                                      const ParameterSetView& parameters,
                                                      StatusView parameter_status,
                                                      const MutableTensorView& joint_torque,
                                                      MutableStatusView status);

Result mark_unsupported(std::size_t num_envs, MutableStatusView status);

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
