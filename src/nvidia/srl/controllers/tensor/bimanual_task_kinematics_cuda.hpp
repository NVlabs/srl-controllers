/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file bimanual_task_kinematics_cuda.hpp
 * @brief Private CUDA declarations for bimanual task kinematics.
 */

#pragma once

#include "tensor/bimanual_task_kinematics.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result reset_bimanual_task_kinematics_cuda(std::size_t num_envs,
                                           const TensorView& reset_mask,
                                           MutableTensorView task_pose_error,
                                           MutableTensorView task_jacobian,
                                           MutableStatusView status) noexcept;

Result step_bimanual_task_kinematics_cuda(BimanualCoordinationMode coordination_mode,
                                          const BimanualFrameConfig& config,
                                          const TensorView& desired_absolute_pose,
                                          const TensorView& arm1_tcp_pose,
                                          const TensorView& arm2_tcp_pose,
                                          const TensorView& desired_relative_pose,
                                          const TensorView& arm1_jacobian,
                                          const TensorView& arm2_jacobian,
                                          const MutableTensorView& task_pose_error,
                                          const MutableTensorView& task_jacobian,
                                          MutableStatusView status) noexcept;

Result step_bimanual_task_velocity_error_cuda(
    const TensorView& desired_task_velocity,
    const TensorView& joint_velocity,
    const TensorView& task_jacobian,
    const MutableTensorView& task_velocity_error) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
