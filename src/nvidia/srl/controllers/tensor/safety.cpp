/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file safety.cpp
 * @brief CPU dispatch and validation for tensor safe-output handling.
 */

#include "nvidia/srl/controllers/tensor/safety.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/safety_cuda.hpp"
#endif

#include "nvidia/srl/controllers/composition.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr std::uint64_t kMaximumHoldCount = ~std::uint64_t{0};

bool is_supported_output(const MutableTensorView& view) noexcept {
  if (view.data == nullptr || !internal::is_float_dtype(view.dtype) ||
      view.layout != Layout::kRowMajor || view.rank < 1 || view.rank > kMaxTensorRank) {
    return false;
  }
  for (int dim = 0; dim < view.rank; ++dim) {
    if (view.shape[dim] <= 0 || view.strides[dim] <= 0) {
      return false;
    }
  }
  if (view.strides[view.rank - 1] != 1) {
    return false;
  }
  for (int dim = view.rank - 2; dim >= 0; --dim) {
    if (view.strides[dim] < view.shape[dim + 1] * view.strides[dim + 1]) {
      return false;
    }
  }
  return true;
}

bool matching_output(const MutableTensorView& left, const MutableTensorView& right) noexcept {
  if (!is_supported_output(right) || left.dtype != right.dtype || left.device != right.device ||
      left.layout != right.layout || left.rank != right.rank) {
    return false;
  }
  for (int dim = 0; dim < left.rank; ++dim) {
    if (left.shape[dim] != right.shape[dim] || left.strides[dim] != right.strides[dim]) {
      return false;
    }
  }
  return true;
}

bool is_environment_vector(const MutableTensorView& view,
                           const DType dtype,
                           const Device device,
                           const std::size_t num_envs) noexcept {
  return view.data != nullptr && view.dtype == dtype && view.device == device &&
         view.layout == Layout::kRowMajor && view.rank == 1 &&
         view.shape[0] == static_cast<std::int64_t>(num_envs) && view.strides[0] >= 1;
}

Result validate_views(const MutableTensorView& output,
                      const MutableStatusView status,
                      const SafetyStateView safety_state,
                      const StepOptions options,
                      std::size_t* num_envs,
                      std::size_t* values_per_env) noexcept {
  if (!is_supported_output(output)) {
    return internal::invalid_view("output must be a row-major float tensor with rank 1..4");
  }
  *num_envs = static_cast<std::size_t>(output.shape[0]);
  *values_per_env = 1;
  for (int dim = 1; dim < output.rank; ++dim) {
    *values_per_env *= static_cast<std::size_t>(output.shape[dim]);
  }
  if (status.flags == nullptr || status.size < *num_envs || status.device != output.device) {
    return internal::invalid_view("status must match the output device and num_envs");
  }
  if (!matching_output(output, safety_state.last_valid_output)) {
    return internal::invalid_view("last_valid_output must exactly match the output view");
  }
  if (!is_environment_vector(safety_state.valid_mask, DType::kBool, output.device, *num_envs)) {
    return internal::invalid_view("safety valid_mask must be bool [num_envs]");
  }
  if (!is_environment_vector(safety_state.hold_count, DType::kUInt64, output.device, *num_envs)) {
    return internal::invalid_view("safety hold_count must be uint64 [num_envs]");
  }
  if (options.safe_output_policy == SafeOutputPolicy::kHoldLastValid &&
      options.max_hold_steps == 0) {
    return internal::invalid_argument("hold-last policy requires max_hold_steps > 0");
  }
  return internal::ok();
}

std::size_t tensor_offset(const MutableTensorView& view,
                          const std::size_t env,
                          std::size_t flat_tail) noexcept {
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
bool output_is_finite(const MutableTensorView& output,
                      const std::size_t env,
                      const std::size_t values_per_env) noexcept {
  const auto* values = static_cast<const Scalar*>(output.data);
  for (std::size_t value = 0; value < values_per_env; ++value) {
    if (!std::isfinite(values[tensor_offset(output, env, value)])) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void copy_environment(const MutableTensorView& source,
                      const MutableTensorView& destination,
                      const std::size_t env,
                      const std::size_t values_per_env) noexcept {
  const auto* source_values = static_cast<const Scalar*>(source.data);
  auto* destination_values = static_cast<Scalar*>(destination.data);
  for (std::size_t value = 0; value < values_per_env; ++value) {
    destination_values[tensor_offset(destination, env, value)] =
        source_values[tensor_offset(source, env, value)];
  }
}

template <typename Scalar>
void zero_environment(const MutableTensorView& output,
                      const std::size_t env,
                      const std::size_t values_per_env) noexcept {
  auto* values = static_cast<Scalar*>(output.data);
  for (std::size_t value = 0; value < values_per_env; ++value) {
    values[tensor_offset(output, env, value)] = static_cast<Scalar>(0);
  }
}

template <typename Scalar>
void apply_cpu(const MutableTensorView& output,
               const MutableStatusView status,
               const SafetyStateView safety_state,
               const StepOptions options,
               const std::size_t num_envs,
               const std::size_t values_per_env) noexcept {
  auto* valid = static_cast<std::uint8_t*>(safety_state.valid_mask.data);
  auto* hold_count = static_cast<std::uint64_t*>(safety_state.hold_count.data);
  const std::size_t valid_stride = static_cast<std::size_t>(safety_state.valid_mask.strides[0]);
  const std::size_t hold_stride = static_cast<std::size_t>(safety_state.hold_count.strides[0]);
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t& flags = status.flags[env];
    flags = normalize_status(flags);
    if (!has_status_flag(flags, StatusFlag::kSafeMode) &&
        !output_is_finite<Scalar>(output, env, values_per_env)) {
      flags |= status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kSafeMode);
    }
    std::uint8_t& env_valid = valid[env * valid_stride];
    std::uint64_t& env_hold_count = hold_count[env * hold_stride];
    if (!has_status_flag(flags, StatusFlag::kSafeMode)) {
      copy_environment<Scalar>(output, safety_state.last_valid_output, env, values_per_env);
      env_valid = 1U;
      env_hold_count = 0U;
      continue;
    }
    const bool hold_last = options.safe_output_policy == SafeOutputPolicy::kHoldLastValid &&
                           env_valid != 0U && env_hold_count < options.max_hold_steps;
    auto* output_values = static_cast<Scalar*>(output.data);
    const auto* last_valid_values = static_cast<const Scalar*>(safety_state.last_valid_output.data);
    for (std::size_t value = 0; value < values_per_env; ++value) {
      const std::size_t output_offset = tensor_offset(output, env, value);
      const std::size_t last_valid_offset =
          tensor_offset(safety_state.last_valid_output, env, value);
      output_values[output_offset] =
          detail::safe_output_value(true,
                                    hold_last,
                                    env_valid != 0U,
                                    output_values[output_offset],
                                    last_valid_values[last_valid_offset]);
    }
    if (hold_last) {
      ++env_hold_count;
      flags |= status_bit(StatusFlag::kOutputHeld);
      continue;
    }
    if (env_valid == 0U) {
      flags |= status_bit(StatusFlag::kNoValidOutput);
    } else {
      if (env_hold_count != kMaximumHoldCount) {
        ++env_hold_count;
      }
      if (options.safe_output_policy == SafeOutputPolicy::kHoldLastValid) {
        flags |= status_bit(StatusFlag::kHoldExpired);
      }
    }
  }
}

}  // namespace

Result reset_safety_state(const std::size_t num_envs,
                          const SafetyStateView safety_state,
                          const TensorView& reset_mask,
                          const MutableStatusView status) noexcept {
  MutableTensorView output = safety_state.last_valid_output;
  std::size_t view_num_envs = 0;
  std::size_t values_per_env = 0;
  Result result =
      validate_views(output, status, safety_state, StepOptions{}, &view_num_envs, &values_per_env);
  if (!result.ok()) {
    return result;
  }
  if (num_envs != view_num_envs) {
    return internal::invalid_view("num_envs must match the safety-state tensors");
  }
  if (reset_mask.data != nullptr) {
    if (reset_mask.device != output.device || reset_mask.dtype != DType::kBool ||
        reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs) ||
        reset_mask.strides[0] < 1) {
      return internal::invalid_view("reset_mask must be bool [num_envs] on the output device");
    }
  }
  if (output.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_safety_state_cuda(num_envs, safety_state, reset_mask, status);
#else
    return internal::unsupported("CUDA safety support is not enabled in this build");
#endif
  }
  if (output.device != Device::kCpu) {
    return internal::invalid_view("safety-state device is unsupported");
  }
  auto* valid = static_cast<std::uint8_t*>(safety_state.valid_mask.data);
  auto* hold_count = static_cast<std::uint64_t*>(safety_state.hold_count.data);
  const auto* reset = static_cast<const std::uint8_t*>(reset_mask.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    const bool selected =
        reset == nullptr || reset[env * static_cast<std::size_t>(reset_mask.strides[0])] != 0U;
    if (!selected) {
      continue;
    }
    if (output.dtype == DType::kFloat32) {
      zero_environment<float>(output, env, values_per_env);
    } else {
      zero_environment<double>(output, env, values_per_env);
    }
    valid[env * static_cast<std::size_t>(safety_state.valid_mask.strides[0])] = 0U;
    hold_count[env * static_cast<std::size_t>(safety_state.hold_count.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

Result apply_safe_output_policy(const MutableTensorView& output,
                                const MutableStatusView status,
                                const SafetyStateView safety_state,
                                const StepOptions options) noexcept {
  std::size_t num_envs = 0;
  std::size_t values_per_env = 0;
  const Result result =
      validate_views(output, status, safety_state, options, &num_envs, &values_per_env);
  if (!result.ok()) {
    return result;
  }
  const Result overlap_result =
      internal::validate_disjoint(output.as_const(),
                                  safety_state.last_valid_output.as_const(),
                                  "last_valid_output must not alias output");
  if (!overlap_result.ok()) {
    return overlap_result;
  }
  if (output.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_safe_output_policy_cuda(output, status, safety_state, options);
#else
    return internal::unsupported("CUDA safety support is not enabled in this build");
#endif
  }
  if (output.device != Device::kCpu) {
    return internal::invalid_view("output device is unsupported");
  }
  if (output.dtype == DType::kFloat32) {
    apply_cpu<float>(output, status, safety_state, options, num_envs, values_per_env);
  } else {
    apply_cpu<double>(output, status, safety_state, options, num_envs, values_per_env);
  }
  return internal::ok();
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
