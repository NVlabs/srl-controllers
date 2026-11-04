/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file torque_filter.cpp
 * @brief CPU tensor torque filter and CUDA dispatch.
 */

#include "nvidia/srl/controllers/tensor/torque_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/torque_filter_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kLowerLimit = "lower_limit";
constexpr const char* kUpperLimit = "upper_limit";
constexpr const char* kMaximumRate = "maximum_rate";
constexpr const char* kPreviousCommand = "previous_command";
constexpr const char* kInitialized = "initialized";

bool set_has_cuda(const ParameterSetView& parameters) noexcept {
  for (std::size_t index = 0; index < parameters.parameter_count; ++index) {
    if (parameters.parameters[index].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

bool state_has_cuda(const StateView& state) noexcept {
  for (std::size_t index = 0; index < state.entry_count; ++index) {
    if (state.entries[index].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

Result validate_signal_views(const TensorView& command,
                             const TensorView& dt,
                             const MutableTensorView& output,
                             std::size_t* num_envs,
                             std::size_t* signal_dim) noexcept {
  if (command.data == nullptr || !internal::is_float_dtype(command.dtype) ||
      !internal::is_row_major_matrix(command) || command.shape[0] <= 0 || command.shape[1] <= 0) {
    return internal::invalid_view("command must be row-major [num_envs, signal_dim]");
  }
  *num_envs = static_cast<std::size_t>(command.shape[0]);
  *signal_dim = static_cast<std::size_t>(command.shape[1]);
  if (*signal_dim > kMaxTorqueFilterSignalDimension) {
    return internal::invalid_argument("command signal_dim exceeds the bounded filter limit");
  }
  if (command.device != Device::kCpu && command.device != Device::kCuda) {
    return internal::invalid_view("command device is unsupported");
  }
  if (dt.data == nullptr || dt.dtype != command.dtype || dt.device != command.device ||
      !(dt.rank == 0 || (dt.rank == 1 && dt.shape[0] >= command.shape[0] && dt.strides[0] >= 1))) {
    return internal::invalid_view("dt must be a matching scalar or [num_envs]");
  }
  if (output.data == nullptr || output.dtype != command.dtype || output.device != command.device ||
      !internal::is_row_major_matrix(output) || output.shape[0] != command.shape[0] ||
      output.shape[1] != command.shape[1]) {
    return internal::invalid_view("filtered_command must exactly match command");
  }
  return internal::ok();
}

Result validate_vector_parameter(const ParameterView* parameter,
                                 const char* missing_message,
                                 const DType dtype,
                                 const Device device,
                                 const std::size_t num_envs,
                                 const std::size_t signal_dim) noexcept {
  if (parameter == nullptr) {
    return internal::invalid_view(missing_message);
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr || values.dtype != dtype || values.device != device ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view("filter parameters must match command dtype and device");
  }
  if (parameter->shared) {
    return values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
                   values.strides[0] >= 1
               ? internal::ok()
               : internal::invalid_view("shared filter parameters must have shape [signal_dim]");
  }
  return values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
                 values.shape[1] == static_cast<std::int64_t>(signal_dim) &&
                 values.strides[1] == 1 && values.strides[0] >= values.shape[1]
             ? internal::ok()
             : internal::invalid_view(
                   "batched filter parameters must have shape [num_envs, signal_dim]");
}

Result validate_parameters(const ParameterSetView& parameters,
                           const DType dtype,
                           const Device device,
                           const std::size_t num_envs,
                           const std::size_t signal_dim) noexcept {
  Result result = validate_vector_parameter(find_parameter(parameters, kLowerLimit),
                                            "lower_limit parameter is required",
                                            dtype,
                                            device,
                                            num_envs,
                                            signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_vector_parameter(find_parameter(parameters, kUpperLimit),
                                     "upper_limit parameter is required",
                                     dtype,
                                     device,
                                     num_envs,
                                     signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_vector_parameter(find_parameter(parameters, kMaximumRate),
                                   "maximum_rate parameter is required",
                                   dtype,
                                   device,
                                   num_envs,
                                   signal_dim);
}

Result validate_state(StateView& state,
                      const DType dtype,
                      const Device device,
                      const std::size_t num_envs,
                      const std::size_t signal_dim) noexcept {
  StateEntryView* previous = find_state(state, kPreviousCommand);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous == nullptr || initialized == nullptr) {
    return internal::invalid_view("torque filter state entries are missing");
  }
  const MutableTensorView& previous_view = previous->values;
  if (previous_view.data == nullptr || previous_view.dtype != dtype ||
      previous_view.device != device || !internal::is_row_major_matrix(previous_view) ||
      previous_view.shape[0] != static_cast<std::int64_t>(num_envs) ||
      previous_view.shape[1] != static_cast<std::int64_t>(signal_dim)) {
    return internal::invalid_view("previous_command state has an invalid view");
  }
  const MutableTensorView& initialized_view = initialized->values;
  if (initialized_view.data == nullptr || initialized_view.dtype != DType::kBool ||
      initialized_view.device != device || initialized_view.layout != Layout::kRowMajor ||
      initialized_view.rank != 1 ||
      initialized_view.shape[0] != static_cast<std::int64_t>(num_envs) ||
      initialized_view.strides[0] < 1) {
    return internal::invalid_view("initialized state must be bool [num_envs]");
  }
  return internal::ok();
}

Result validate_status(const MutableStatusView status,
                       const Device device,
                       const std::size_t num_envs) noexcept {
  return status.flags != nullptr && status.device == device && status.size >= num_envs
             ? internal::ok()
             : internal::invalid_view("status must match command device and num_envs");
}

template <typename Scalar>
bool parameters_valid_for_env(const ParameterSetView& parameters,
                              const std::size_t env,
                              const std::size_t signal_dim) noexcept {
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* rate = find_parameter(parameters, kMaximumRate);
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar lower_value =
        internal::read_vector_parameter<Scalar>(lower, env, dim, static_cast<Scalar>(0));
    const Scalar upper_value =
        internal::read_vector_parameter<Scalar>(upper, env, dim, static_cast<Scalar>(0));
    const Scalar rate_value =
        internal::read_vector_parameter<Scalar>(rate, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(lower_value) || !std::isfinite(upper_value) || !std::isfinite(rate_value) ||
        lower_value > upper_value || rate_value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_cpu(const ParameterSetView& parameters,
                 const std::size_t num_envs,
                 const std::size_t signal_dim,
                 const MutableStatusView status) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] = parameters_valid_for_env<Scalar>(parameters, env, signal_dim)
                            ? 0U
                            : status_bit(StatusFlag::kInvalidParameter);
  }
  return internal::ok();
}

template <typename Scalar>
Result reset_cpu(const std::size_t num_envs,
                 StateView& state,
                 const TensorView& reset_mask,
                 const MutableStatusView status,
                 const std::size_t signal_dim) noexcept {
  StateEntryView* previous = find_state(state, kPreviousCommand);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      status.flags[env] = 0U;
      continue;
    }
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      internal::write_matrix_value(previous->values, env, dim, static_cast<Scalar>(0));
    }
    initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    status.flags[env] = status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_cpu(const TensorView& command,
                const TensorView& dt,
                const ParameterSetView& parameters,
                const StatusView parameter_status,
                StateView& state,
                const MutableTensorView& output,
                const MutableStatusView status,
                const std::size_t num_envs,
                const std::size_t signal_dim) noexcept {
  StateEntryView* previous = find_state(state, kPreviousCommand);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* maximum_rate = find_parameter(parameters, kMaximumRate);
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags == nullptr ? 0U : parameter_status.flags[env];
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!parameters_valid_for_env<Scalar>(parameters, env, signal_dim)) {
      flags |= status_bit(StatusFlag::kInvalidParameter);
    }
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      flags |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(command, env, signal_dim)) {
      flags |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if (flags != 0U) {
      internal::zero_output_env<Scalar>(output, env, signal_dim);
      status.flags[env] = flags | status_bit(StatusFlag::kSafeMode);
      continue;
    }
    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    const bool has_previous = initialized_values[initialized_offset] != 0U;
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      const Scalar lower_value =
          internal::read_vector_parameter<Scalar>(lower, env, dim, static_cast<Scalar>(0));
      const Scalar upper_value =
          internal::read_vector_parameter<Scalar>(upper, env, dim, static_cast<Scalar>(0));
      const Scalar rate_value =
          internal::read_vector_parameter<Scalar>(maximum_rate, env, dim, static_cast<Scalar>(0));
      const Scalar raw = internal::read_matrix_value<Scalar>(command, env, dim);
      const Scalar old = has_previous
                             ? internal::read_matrix_value<Scalar>(previous->values, env, dim)
                             : static_cast<Scalar>(0);
      bool saturated = false;
      const Scalar filtered = detail::torque_filter_value(
          raw, old, lower_value, upper_value, rate_value, dt_value, has_previous, &saturated);
      if (saturated) {
        flags |= status_bit(StatusFlag::kSaturation);
      }
      internal::write_matrix_value(output, env, dim, filtered);
      internal::write_matrix_value(previous->values, env, dim, filtered);
    }
    initialized_values[initialized_offset] = 1U;
    status.flags[env] = flags;
  }
  return internal::ok();
}

}  // namespace

PrepareResult prepare_torque_filter(const TensorView& command,
                                    const TensorView& dt,
                                    const ParameterSetView& parameters,
                                    StateView& state,
                                    const MutableTensorView& filtered_command) {
  (void)state;
  PrepareResult result;
  result.result =
      validate_signal_views(command, dt, filtered_command, &result.num_envs, &result.signal_dim);
  if (!result.result.ok()) {
    return result;
  }
  result.result = validate_parameters(
      parameters, command.dtype, command.device, result.num_envs, result.signal_dim);
  if (!result.result.ok()) {
    return result;
  }
  result.state.push_back(internal::matrix_descriptor("previous_command",
                                                     command.dtype,
                                                     result.num_envs,
                                                     result.signal_dim,
                                                     "caller_defined",
                                                     "caller_defined",
                                                     "previous filtered command"));
  result.state.push_back(internal::vector_descriptor("initialized",
                                                     DType::kBool,
                                                     result.num_envs,
                                                     "unitless",
                                                     "none",
                                                     "per-environment initialized flag"));
  result.outputs.push_back(internal::matrix_descriptor("filtered_command",
                                                       command.dtype,
                                                       result.num_envs,
                                                       result.signal_dim,
                                                       "caller_defined",
                                                       "caller_defined",
                                                       "magnitude and rate limited command"));
  for (TensorDescriptor* descriptor : {&result.state[0], &result.state[1], &result.outputs[0]}) {
    descriptor->device = command.device;
  }
  return result;
}

Result apply_torque_filter_parameters(const ParameterSetView& parameters,
                                      const std::size_t num_envs,
                                      const std::size_t signal_dim,
                                      const MutableStatusView parameter_status) noexcept {
  if (num_envs == 0 || signal_dim == 0 || signal_dim > kMaxTorqueFilterSignalDimension) {
    return internal::invalid_argument("num_envs and bounded signal_dim must be positive");
  }
  if (parameter_status.flags == nullptr || parameter_status.size < num_envs ||
      (parameter_status.device != Device::kCpu && parameter_status.device != Device::kCuda)) {
    return internal::invalid_view("parameter_status is smaller than num_envs");
  }
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  if (lower == nullptr || !internal::is_float_dtype(lower->values.dtype)) {
    return internal::invalid_view("lower_limit parameter is required");
  }
  Result result = validate_parameters(
      parameters, lower->values.dtype, parameter_status.device, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.device == Device::kCuda || set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_torque_filter_parameters_cuda(parameters, num_envs, signal_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (parameter_status.device != Device::kCpu) {
    return internal::invalid_view("parameter_status device is unsupported");
  }
  return lower->values.dtype == DType::kFloat32
             ? apply_cpu<float>(parameters, num_envs, signal_dim, parameter_status)
             : apply_cpu<double>(parameters, num_envs, signal_dim, parameter_status);
}

Result reset_torque_filter(const std::size_t num_envs,
                           StateView& state,
                           const TensorView& reset_mask,
                           const MutableStatusView status) noexcept {
  if (num_envs == 0) {
    return internal::invalid_argument("num_envs must be positive");
  }
  StateEntryView* previous = find_state(state, kPreviousCommand);
  if (previous == nullptr || previous->values.rank != 2 || previous->values.shape[1] <= 0) {
    return internal::invalid_view("previous_command state is missing");
  }
  const std::size_t signal_dim = static_cast<std::size_t>(previous->values.shape[1]);
  Result result =
      validate_state(state, previous->values.dtype, status.device, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_status(status, previous->values.device, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (reset_mask.data != nullptr &&
      (reset_mask.dtype != DType::kBool || reset_mask.device != status.device ||
       reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs) ||
       reset_mask.strides[0] < 1)) {
    return internal::invalid_view("reset_mask must be bool [num_envs] on the state device");
  }
  if (status.device == Device::kCuda || state_has_cuda(state)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_torque_filter_cuda(num_envs, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return previous->values.dtype == DType::kFloat32
             ? reset_cpu<float>(num_envs, state, reset_mask, status, signal_dim)
             : reset_cpu<double>(num_envs, state, reset_mask, status, signal_dim);
}

Result step_torque_filter(const TensorView& command,
                          const TensorView& dt,
                          const ParameterSetView& parameters,
                          const StatusView parameter_status,
                          StateView& state,
                          const MutableTensorView& filtered_command,
                          const MutableStatusView status) noexcept {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = validate_signal_views(command, dt, filtered_command, &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_parameters(parameters, command.dtype, command.device, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_state(state, command.dtype, command.device, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  const StateEntryView* previous = find_state(state, kPreviousCommand);
  const TensorView protected_views[] = {command, filtered_command.as_const()};
  for (const TensorView& view : protected_views) {
    result =
        internal::validate_disjoint(previous->values.as_const(),
                                    view,
                                    "previous_command must not alias command or filtered_command");
    if (!result.ok()) {
      return result;
    }
  }
  result = internal::validate_disjoint(command,
                                       filtered_command.as_const(),
                                       "command and filtered_command must not partially overlap",
                                       true);
  if (!result.ok()) {
    return result;
  }
  result = validate_status(status, command.device, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr &&
      (parameter_status.device != command.device || parameter_status.size < num_envs)) {
    return internal::invalid_view("parameter_status must match command device and num_envs");
  }
  if (command.device == Device::kCuda || set_has_cuda(parameters) || state_has_cuda(state)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_torque_filter_cuda(
        command, dt, parameters, parameter_status, state, filtered_command, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return command.dtype == DType::kFloat32 ? step_cpu<float>(command,
                                                            dt,
                                                            parameters,
                                                            parameter_status,
                                                            state,
                                                            filtered_command,
                                                            status,
                                                            num_envs,
                                                            signal_dim)
                                          : step_cpu<double>(command,
                                                             dt,
                                                             parameters,
                                                             parameter_status,
                                                             state,
                                                             filtered_command,
                                                             status,
                                                             num_envs,
                                                             signal_dim);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
