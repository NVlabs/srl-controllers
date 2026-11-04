/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_impedance_cuda.hpp
 * @brief Private CUDA declarations for bounded variable-dimension impedance.
 */

#pragma once

#include "tensor/task_impedance.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_task_impedance_parameters_cuda(const ParameterSetView& parameters,
                                            std::size_t num_envs,
                                            std::size_t task_dim,
                                            MutableStatusView parameter_status) noexcept;
Result reset_task_impedance_controller_cuda(std::size_t num_envs,
                                            std::size_t task_dim,
                                            std::size_t joint_dim,
                                            StateView& state,
                                            const TensorView& reset_mask,
                                            MutableStatusView status) noexcept;
Result step_task_impedance_controller_cuda(const TensorView& task_pose_error,
                                           const TensorView& task_velocity_error,
                                           const TensorView& desired_acceleration,
                                           const TensorView& joint_velocity,
                                           const TensorView& task_mass_matrix,
                                           const TensorView& jacobian,
                                           const TensorView& dt,
                                           const ParameterSetView& parameters,
                                           StatusView parameter_status,
                                           StateView& state,
                                           const MutableTensorView& task_wrench,
                                           MutableStatusView status) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
