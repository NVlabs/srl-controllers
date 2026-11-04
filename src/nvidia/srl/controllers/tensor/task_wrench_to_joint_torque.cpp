/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_wrench_to_joint_torque.cpp
 * @brief CPU task-wrench to joint-torque composition and CUDA dispatch.
 */

#include "tensor/task_wrench_to_joint_torque.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/task_wrench_to_joint_torque_cuda.hpp"
#endif

#include "nvidia/srl/controllers/composition.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

Result validate(const TensorView& task_wrench,
                const TensorView& task_jacobian,
                const TensorView& joint_bias_torque,
                const TensorView& joint_feedback_torque,
                const StatusView task_status,
                const StatusView feedback_status,
                const MutableTensorView& joint_torque) noexcept {
  if (task_wrench.data == nullptr || !internal::is_float_dtype(task_wrench.dtype) ||
      !internal::is_row_major_matrix(task_wrench) || task_wrench.shape[0] <= 0 ||
      task_wrench.shape[1] <= 0) {
    return internal::invalid_view("task_wrench must be row-major [batch_size,task_dim]");
  }
  const std::int64_t batch_size = task_wrench.shape[0];
  const std::int64_t task_dim = task_wrench.shape[1];
  if (task_jacobian.data == nullptr || task_jacobian.dtype != task_wrench.dtype ||
      task_jacobian.device != task_wrench.device ||
      !internal::is_row_major_tensor3(task_jacobian) || task_jacobian.shape[0] != batch_size ||
      task_jacobian.shape[1] != task_dim || task_jacobian.shape[2] <= 0) {
    return internal::invalid_view(
        "task_jacobian must be row-major [batch_size,task_dim,joint_dim]");
  }
  const std::int64_t joint_dim = task_jacobian.shape[2];
  const TensorView joint_inputs[]{joint_bias_torque, joint_feedback_torque};
  for (const TensorView& input : joint_inputs) {
    if (input.data == nullptr || input.dtype != task_wrench.dtype ||
        input.device != task_wrench.device || !internal::is_row_major_matrix(input) ||
        input.shape[0] != batch_size || input.shape[1] != joint_dim) {
      return internal::invalid_view("joint torque inputs must be row-major [batch_size,joint_dim]");
    }
  }
  if (joint_torque.data == nullptr || joint_torque.dtype != task_wrench.dtype ||
      joint_torque.device != task_wrench.device || !internal::is_row_major_matrix(joint_torque) ||
      joint_torque.shape[0] != batch_size || joint_torque.shape[1] != joint_dim) {
    return internal::invalid_view("joint_torque must be row-major [batch_size,joint_dim]");
  }
  const StatusView statuses[]{task_status, feedback_status};
  for (const StatusView status : statuses) {
    if (status.flags == nullptr || status.device != task_wrench.device ||
        status.size < static_cast<std::size_t>(batch_size)) {
      return internal::invalid_view("component status must match the controller batch and device");
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_cpu(const TensorView& task_wrench,
                const TensorView& task_jacobian,
                const TensorView& joint_bias_torque,
                const TensorView& joint_feedback_torque,
                const StatusView task_status,
                const StatusView feedback_status,
                const MutableTensorView& joint_torque) noexcept {
  const std::size_t batch_size = static_cast<std::size_t>(task_wrench.shape[0]);
  const std::size_t task_dim = static_cast<std::size_t>(task_wrench.shape[1]);
  const std::size_t joint_dim = static_cast<std::size_t>(task_jacobian.shape[2]);
  const std::uint64_t unsafe_mask = status_bit(StatusFlag::kSafeMode);
  for (std::size_t env = 0; env < batch_size; ++env) {
    const bool upstream_unsafe =
        (normalize_status(task_status.flags[env] | feedback_status.flags[env]) & unsafe_mask) != 0U;
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      Scalar torque = internal::read_matrix_value<Scalar>(joint_bias_torque, env, joint) +
                      internal::read_matrix_value<Scalar>(joint_feedback_torque, env, joint);
      for (std::size_t task = 0; task < task_dim; ++task) {
        torque += internal::read_tensor3_value<Scalar>(task_jacobian, env, task, joint) *
                  internal::read_matrix_value<Scalar>(task_wrench, env, task);
      }
      if (upstream_unsafe || !std::isfinite(torque)) {
        torque = std::numeric_limits<Scalar>::quiet_NaN();
      }
      internal::write_matrix_value(joint_torque, env, joint, torque);
    }
  }
  return internal::ok();
}

}  // namespace

Result step_task_wrench_to_joint_torque(const TensorView& task_wrench,
                                        const TensorView& task_jacobian,
                                        const TensorView& joint_bias_torque,
                                        const TensorView& joint_feedback_torque,
                                        const StatusView task_status,
                                        const StatusView feedback_status,
                                        const MutableTensorView& joint_torque) noexcept {
  const Result result = validate(task_wrench,
                                 task_jacobian,
                                 joint_bias_torque,
                                 joint_feedback_torque,
                                 task_status,
                                 feedback_status,
                                 joint_torque);
  if (!result.ok()) {
    return result;
  }
  if (task_wrench.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_task_wrench_to_joint_torque_cuda(task_wrench,
                                                 task_jacobian,
                                                 joint_bias_torque,
                                                 joint_feedback_torque,
                                                 task_status,
                                                 feedback_status,
                                                 joint_torque);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (task_wrench.device != Device::kCpu) {
    return internal::invalid_view("controller device is unsupported");
  }
  return task_wrench.dtype == DType::kFloat32 ? step_cpu<float>(task_wrench,
                                                                task_jacobian,
                                                                joint_bias_torque,
                                                                joint_feedback_torque,
                                                                task_status,
                                                                feedback_status,
                                                                joint_torque)
                                              : step_cpu<double>(task_wrench,
                                                                 task_jacobian,
                                                                 joint_bias_torque,
                                                                 joint_feedback_torque,
                                                                 task_status,
                                                                 feedback_status,
                                                                 joint_torque);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
