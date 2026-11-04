/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_wrench_to_joint_torque_cuda.cu
 * @brief CUDA task-wrench to joint-torque composition.
 */

#include "tensor/task_wrench_to_joint_torque_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "tensor/cuda_stream.hpp"

#include "nvidia/srl/controllers/composition.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 128;

template <typename Scalar>
__global__ void compose_kernel(const Scalar* task_wrench,
                               std::int64_t wrench_env_stride,
                               const Scalar* task_jacobian,
                               std::int64_t jacobian_env_stride,
                               std::int64_t jacobian_task_stride,
                               const Scalar* joint_bias_torque,
                               std::int64_t bias_env_stride,
                               const Scalar* joint_feedback_torque,
                               std::int64_t feedback_env_stride,
                               const std::uint64_t* task_status,
                               const std::uint64_t* feedback_status,
                               Scalar* joint_torque,
                               std::int64_t output_env_stride,
                               std::size_t batch_size,
                               std::size_t task_dim,
                               std::size_t joint_dim) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = batch_size * joint_dim;
  if (index >= total) {
    return;
  }
  const std::size_t env = index / joint_dim;
  const std::size_t joint = index % joint_dim;
  Scalar value = joint_bias_torque[env * static_cast<std::size_t>(bias_env_stride) + joint] +
                 joint_feedback_torque[env * static_cast<std::size_t>(feedback_env_stride) + joint];
  for (std::size_t task = 0; task < task_dim; ++task) {
    value += task_jacobian[env * static_cast<std::size_t>(jacobian_env_stride) +
                           task * static_cast<std::size_t>(jacobian_task_stride) + joint] *
             task_wrench[env * static_cast<std::size_t>(wrench_env_stride) + task];
  }
  constexpr std::uint64_t unsafe_mask = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
  if ((normalize_status(task_status[env] | feedback_status[env]) & unsafe_mask) != 0U ||
      !isfinite(value)) {
    value = nan("");
  }
  joint_torque[env * static_cast<std::size_t>(output_env_stride) + joint] = value;
}

template <typename Scalar>
Result launch(const TensorView& task_wrench,
              const TensorView& task_jacobian,
              const TensorView& joint_bias_torque,
              const TensorView& joint_feedback_torque,
              const StatusView task_status,
              const StatusView feedback_status,
              const MutableTensorView& joint_torque) noexcept {
  const std::size_t batch_size = static_cast<std::size_t>(task_wrench.shape[0]);
  const std::size_t task_dim = static_cast<std::size_t>(task_wrench.shape[1]);
  const std::size_t joint_dim = static_cast<std::size_t>(task_jacobian.shape[2]);
  const std::size_t total = batch_size * joint_dim;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  compose_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      static_cast<const Scalar*>(task_wrench.data),
      task_wrench.strides[0],
      static_cast<const Scalar*>(task_jacobian.data),
      task_jacobian.strides[0],
      task_jacobian.strides[1],
      static_cast<const Scalar*>(joint_bias_torque.data),
      joint_bias_torque.strides[0],
      static_cast<const Scalar*>(joint_feedback_torque.data),
      joint_feedback_torque.strides[0],
      task_status.flags,
      feedback_status.flags,
      static_cast<Scalar*>(joint_torque.data),
      joint_torque.strides[0],
      batch_size,
      task_dim,
      joint_dim);
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{ResultCode::kOk, "ok"}
             : Result{ResultCode::kInvalidArgument,
                      "CUDA task-wrench composition kernel launch failed"};
}

}  // namespace

Result step_task_wrench_to_joint_torque_cuda(const TensorView& task_wrench,
                                             const TensorView& task_jacobian,
                                             const TensorView& joint_bias_torque,
                                             const TensorView& joint_feedback_torque,
                                             const StatusView task_status,
                                             const StatusView feedback_status,
                                             const MutableTensorView& joint_torque) noexcept {
  return task_wrench.dtype == DType::kFloat32 ? launch<float>(task_wrench,
                                                              task_jacobian,
                                                              joint_bias_torque,
                                                              joint_feedback_torque,
                                                              task_status,
                                                              feedback_status,
                                                              joint_torque)
                                              : launch<double>(task_wrench,
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
