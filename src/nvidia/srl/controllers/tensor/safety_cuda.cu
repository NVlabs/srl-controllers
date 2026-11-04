/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file safety_cuda.cu
 * @brief CUDA kernels for bounded tensor safe-output handling.
 */

#include "tensor/safety_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/cuda_stream.hpp"

#include "nvidia/srl/controllers/composition.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 128;
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kOutputHeldFlag = static_cast<std::uint64_t>(StatusFlag::kOutputHeld);
constexpr std::uint64_t kNoValidOutputFlag = static_cast<std::uint64_t>(StatusFlag::kNoValidOutput);
constexpr std::uint64_t kHoldExpiredFlag = static_cast<std::uint64_t>(StatusFlag::kHoldExpired);
constexpr std::uint64_t kMaximumHoldCount = ~std::uint64_t{0};

struct CudaTensorMetadata {
  int rank{0};
  std::int64_t shape[kMaxTensorRank]{};
  std::int64_t strides[kMaxTensorRank]{};
};

CudaTensorMetadata metadata(const MutableTensorView& view) noexcept {
  CudaTensorMetadata result{};
  result.rank = view.rank;
  for (int dim = 0; dim < view.rank; ++dim) {
    result.shape[dim] = view.shape[dim];
    result.strides[dim] = view.strides[dim];
  }
  return result;
}

__device__ std::size_t tensor_offset(const CudaTensorMetadata view,
                                     const std::size_t env,
                                     std::size_t flat_tail) {
  std::size_t offset = env * static_cast<std::size_t>(view.strides[0]);
  for (int dim = view.rank - 1; dim >= 1; --dim) {
    const std::size_t extent = static_cast<std::size_t>(view.shape[dim]);
    const std::size_t coordinate = flat_tail % extent;
    flat_tail /= extent;
    offset += coordinate * static_cast<std::size_t>(view.strides[dim]);
  }
  return offset;
}

template <typename Scalar>
__global__ void reset_safety_kernel(Scalar* last_valid_output,
                                    CudaTensorMetadata output_metadata,
                                    std::size_t num_envs,
                                    std::size_t values_per_env,
                                    std::uint8_t* valid_mask,
                                    std::int64_t valid_stride,
                                    std::uint64_t* hold_count,
                                    std::int64_t hold_stride,
                                    const std::uint8_t* reset_mask,
                                    std::int64_t reset_stride,
                                    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool selected =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_stride)] != 0U;
  if (!selected) {
    return;
  }
  for (std::size_t value = 0; value < values_per_env; ++value) {
    last_valid_output[tensor_offset(output_metadata, env, value)] = static_cast<Scalar>(0);
  }
  valid_mask[env * static_cast<std::size_t>(valid_stride)] = 0U;
  hold_count[env * static_cast<std::size_t>(hold_stride)] = 0U;
  status[env] |= kResetAppliedFlag;
}

template <typename Scalar>
__global__ void apply_safety_kernel(Scalar* output,
                                    Scalar* last_valid_output,
                                    CudaTensorMetadata output_metadata,
                                    std::size_t num_envs,
                                    std::size_t values_per_env,
                                    std::uint8_t* valid_mask,
                                    std::int64_t valid_stride,
                                    std::uint64_t* hold_count,
                                    std::int64_t hold_stride,
                                    bool hold_last,
                                    std::uint64_t max_hold_steps,
                                    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = normalize_status(status[env]);
  if ((flags & kSafeModeFlag) == 0U) {
    for (std::size_t value = 0; value < values_per_env; ++value) {
      if (!isfinite(output[tensor_offset(output_metadata, env, value)])) {
        flags |= kNonFiniteInputFlag | kSafeModeFlag;
        break;
      }
    }
  }
  std::uint8_t& valid = valid_mask[env * static_cast<std::size_t>(valid_stride)];
  std::uint64_t& count = hold_count[env * static_cast<std::size_t>(hold_stride)];
  if ((flags & kSafeModeFlag) == 0U) {
    for (std::size_t value = 0; value < values_per_env; ++value) {
      const std::size_t offset = tensor_offset(output_metadata, env, value);
      last_valid_output[offset] = output[offset];
    }
    valid = 1U;
    count = 0U;
    status[env] = flags;
    return;
  }
  const bool use_held_output = hold_last && valid != 0U && count < max_hold_steps;
  for (std::size_t value = 0; value < values_per_env; ++value) {
    const std::size_t offset = tensor_offset(output_metadata, env, value);
    output[offset] = detail::safe_output_value(
        true, use_held_output, valid != 0U, output[offset], last_valid_output[offset]);
  }
  if (use_held_output) {
    ++count;
    status[env] = flags | kOutputHeldFlag;
    return;
  }
  if (valid != 0U && count != kMaximumHoldCount) {
    ++count;
  }
  status[env] = flags | (valid == 0U ? kNoValidOutputFlag : (hold_last ? kHoldExpiredFlag : 0U));
}

Result launch_result() noexcept {
  if (cudaPeekAtLastError() != cudaSuccess) {
    return Result{ResultCode::kInvalidArgument, "CUDA safety kernel launch failed"};
  }
  return Result{ResultCode::kOk, "ok"};
}

std::size_t values_per_environment(const MutableTensorView& output) noexcept {
  std::size_t values = 1;
  for (int dim = 1; dim < output.rank; ++dim) {
    values *= static_cast<std::size_t>(output.shape[dim]);
  }
  return values;
}

template <typename Scalar>
Result launch_reset(const std::size_t num_envs,
                    const SafetyStateView safety_state,
                    const TensorView& reset_mask,
                    const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  reset_safety_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      static_cast<Scalar*>(safety_state.last_valid_output.data),
      metadata(safety_state.last_valid_output),
      num_envs,
      values_per_environment(safety_state.last_valid_output),
      static_cast<std::uint8_t*>(safety_state.valid_mask.data),
      safety_state.valid_mask.strides[0],
      static_cast<std::uint64_t*>(safety_state.hold_count.data),
      safety_state.hold_count.strides[0],
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      status.flags);
  return launch_result();
}

template <typename Scalar>
Result launch_apply(const MutableTensorView& output,
                    const MutableStatusView status,
                    const SafetyStateView safety_state,
                    const StepOptions options) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(output.shape[0]);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  apply_safety_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      static_cast<Scalar*>(output.data),
      static_cast<Scalar*>(safety_state.last_valid_output.data),
      metadata(output),
      num_envs,
      values_per_environment(output),
      static_cast<std::uint8_t*>(safety_state.valid_mask.data),
      safety_state.valid_mask.strides[0],
      static_cast<std::uint64_t*>(safety_state.hold_count.data),
      safety_state.hold_count.strides[0],
      options.safe_output_policy == SafeOutputPolicy::kHoldLastValid,
      options.max_hold_steps,
      status.flags);
  return launch_result();
}

}  // namespace

Result reset_safety_state_cuda(const std::size_t num_envs,
                               const SafetyStateView safety_state,
                               const TensorView& reset_mask,
                               const MutableStatusView status) noexcept {
  return safety_state.last_valid_output.dtype == DType::kFloat32
             ? launch_reset<float>(num_envs, safety_state, reset_mask, status)
             : launch_reset<double>(num_envs, safety_state, reset_mask, status);
}

Result apply_safe_output_policy_cuda(const MutableTensorView& output,
                                     const MutableStatusView status,
                                     const SafetyStateView safety_state,
                                     const StepOptions options) noexcept {
  return output.dtype == DType::kFloat32
             ? launch_apply<float>(output, status, safety_state, options)
             : launch_apply<double>(output, status, safety_state, options);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
