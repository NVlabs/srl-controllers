/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_wrench_to_joint_torque_cuda.hpp
 * @brief CUDA task-wrench to joint-torque composition.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result step_task_wrench_to_joint_torque_cuda(const TensorView& task_wrench,
                                             const TensorView& task_jacobian,
                                             const TensorView& joint_bias_torque,
                                             const TensorView& joint_feedback_torque,
                                             StatusView task_status,
                                             StatusView feedback_status,
                                             const MutableTensorView& joint_torque) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
