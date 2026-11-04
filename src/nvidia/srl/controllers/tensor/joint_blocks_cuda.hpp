/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_blocks_cuda.hpp
 * @brief Private CUDA tensor backend declarations for joint controller blocks.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_joint_diagonal_parameters_cuda(const ParameterSetView& parameters,
                                            const char* parameter_name,
                                            std::size_t num_envs,
                                            std::size_t joint_dim,
                                            MutableStatusView parameter_status);
Result apply_joint_dense_parameters_cuda(const ParameterSetView& parameters,
                                         const char* parameter_name,
                                         std::size_t num_envs,
                                         std::size_t joint_dim,
                                         MutableStatusView parameter_status);
Result apply_joint_damping_factors_parameters_cuda(const ParameterSetView& parameters,
                                                   std::size_t num_envs,
                                                   std::size_t joint_dim,
                                                   MutableStatusView parameter_status);
Result apply_joint_impedance_parameters_cuda(const ParameterSetView& parameters,
                                             std::size_t num_envs,
                                             std::size_t joint_dim,
                                             MutableStatusView parameter_status);
Result apply_nullspace_parameters_cuda(const ParameterSetView& parameters,
                                       std::size_t num_envs,
                                       std::size_t joint_dim,
                                       MutableStatusView parameter_status);
Result reset_stateless_joint_cuda(std::size_t num_envs,
                                  const TensorView& reset_mask,
                                  MutableStatusView status);
Result step_joint_diagonal_cuda(const TensorView& joint_error,
                                const ParameterSetView& parameters,
                                const char* parameter_name,
                                StatusView parameter_status,
                                const MutableTensorView& joint_torque,
                                MutableStatusView status);
Result step_joint_dense_cuda(const TensorView& joint_error,
                             const ParameterSetView& parameters,
                             const char* parameter_name,
                             StatusView parameter_status,
                             const MutableTensorView& joint_torque,
                             MutableStatusView status);
Result step_joint_damping_double_diagonalization_cuda(const TensorView& joint_velocity_error,
                                                      const TensorView& joint_mass_matrix,
                                                      const TensorView& stiffness_matrix,
                                                      const ParameterSetView& parameters,
                                                      StatusView parameter_status,
                                                      const MutableTensorView& joint_torque,
                                                      MutableStatusView status);
Result step_joint_impedance_controller_cuda(const TensorView& joint_position_error,
                                            const TensorView& joint_velocity_error,
                                            const TensorView& desired_acceleration,
                                            const TensorView& joint_mass_matrix,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            const MutableTensorView& joint_torque,
                                            MutableStatusView status);
Result step_nullspace_controller_cuda(const TensorView& joint_position_error,
                                      const TensorView& joint_velocity,
                                      const TensorView& joint_mass_matrix,
                                      const TensorView& jacobian,
                                      const ParameterSetView& parameters,
                                      StatusView parameter_status,
                                      const MutableTensorView& joint_torque,
                                      MutableStatusView status);
Result step_bimanual_independent_nullspace_controller_cuda(const TensorView& joint_position_error,
                                                           const TensorView& joint_velocity,
                                                           const TensorView& joint_mass_matrix,
                                                           const TensorView& arm1_jacobian,
                                                           const TensorView& arm2_jacobian,
                                                           const ParameterSetView& parameters,
                                                           StatusView parameter_status,
                                                           const MutableTensorView& joint_torque,
                                                           MutableStatusView status);

namespace internal {

enum class JointImpedanceCudaRoute {
  kAuto,
  kGeneric,
  kOptimized,
};

struct JointImpedanceCudaLaunchPlan {
  JointImpedanceCudaRoute route{JointImpedanceCudaRoute::kGeneric};
  std::size_t capacity{0};
  int warps_per_block{0};
};

int joint_impedance_cuda_build_sm() noexcept;
JointImpedanceCudaLaunchPlan joint_impedance_cuda_launch_plan(DType dtype,
                                                              std::size_t joint_dim) noexcept;
Result step_joint_impedance_controller_cuda_with_route(const TensorView& joint_position_error,
                                                       const TensorView& joint_velocity_error,
                                                       const TensorView& desired_acceleration,
                                                       const TensorView& joint_mass_matrix,
                                                       const ParameterSetView& parameters,
                                                       StatusView parameter_status,
                                                       const MutableTensorView& joint_torque,
                                                       MutableStatusView status,
                                                       JointImpedanceCudaRoute route);
Result step_joint_impedance_controller_cuda_with_launch_plan(
    const TensorView& joint_position_error,
    const TensorView& joint_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_mass_matrix,
    const ParameterSetView& parameters,
    StatusView parameter_status,
    const MutableTensorView& joint_torque,
    MutableStatusView status,
    JointImpedanceCudaLaunchPlan launch_plan);

}  // namespace internal

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
