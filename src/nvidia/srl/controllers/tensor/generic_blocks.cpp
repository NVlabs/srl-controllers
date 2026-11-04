/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file generic_blocks.cpp
 * @brief CPU tensor kernels for generic controller blocks.
 */

#include "nvidia/srl/controllers/tensor/generic_blocks.hpp"

#include <cmath>
#include <limits>

#include "nvidia/srl/controllers/detail/math_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/generic_blocks_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kGain = "gain";
constexpr const char* kU0 = "u_0";
constexpr const char* kUpperLimit = "upper_limit";
constexpr const char* kLowerLimit = "lower_limit";
constexpr const char* kPreviousSignal = "previous_signal";
constexpr const char* kIntegratedSignal = "integrated_signal";
constexpr const char* kInitialized = "initialized";

bool parameter_set_has_cuda(const ParameterSetView& parameters) {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

bool state_has_cuda(const StateView& state) {
  for (std::size_t i = 0; i < state.entry_count; ++i) {
    if (state.entries[i].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

bool stateful_uses_cuda(const TensorView& input_signal,
                        const TensorView& dt,
                        const ParameterSetView& parameters,
                        const MutableTensorView& output_signal) {
  return input_signal.device == Device::kCuda || dt.device == Device::kCuda ||
         output_signal.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool stateful_uses_cuda(const TensorView& reset_mask,
                        const StateView& state,
                        const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda ||
         state_has_cuda(state);
}

bool stateful_uses_cuda(const TensorView& input_signal,
                        const TensorView& dt,
                        const ParameterSetView& parameters,
                        const StatusView& parameter_status,
                        const StateView& state,
                        const MutableTensorView& output_signal,
                        const MutableStatusView& status) {
  return input_signal.device == Device::kCuda || dt.device == Device::kCuda ||
         output_signal.device == Device::kCuda || status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters) || state_has_cuda(state);
}

bool proportional_uses_cuda(const TensorView& input_signal,
                            const ParameterSetView& parameters,
                            const MutableTensorView& output_signal) {
  return input_signal.device == Device::kCuda || output_signal.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool proportional_uses_cuda(const ParameterSetView& parameters, const MutableStatusView& status) {
  return status.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool proportional_uses_cuda(const TensorView& reset_mask, const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda;
}

bool proportional_uses_cuda(const TensorView& input_signal,
                            const ParameterSetView& parameters,
                            const StatusView& parameter_status,
                            const MutableTensorView& output_signal,
                            const MutableStatusView& status) {
  return input_signal.device == Device::kCuda || output_signal.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
Result validate_cuda_prepare_common(const TensorView& input_signal,
                                    const TensorView& dt,
                                    const MutableTensorView& output_signal,
                                    std::size_t* num_envs,
                                    std::size_t* signal_dim) {
  if (input_signal.data == nullptr || input_signal.device != Device::kCuda ||
      !internal::is_float_dtype(input_signal.dtype) ||
      !internal::is_row_major_matrix(input_signal) || input_signal.shape[0] <= 0 ||
      input_signal.shape[1] <= 0) {
    return internal::invalid_view("input tensor must be CUDA row-major [num_envs, signal_dim]");
  }
  *num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  *signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  if (dt.data != nullptr) {
    if (dt.device != Device::kCuda || dt.dtype != input_signal.dtype) {
      return internal::invalid_view("dt must be a CUDA scalar batch matching input dtype");
    }
    if (!(dt.rank == 0 || (dt.rank == 1 && dt.shape[0] >= static_cast<std::int64_t>(*num_envs) &&
                           dt.strides[0] >= 1))) {
      return internal::invalid_view("dt must be scalar or [num_envs]");
    }
  }
  if (output_signal.data == nullptr || output_signal.device != Device::kCuda ||
      output_signal.dtype != input_signal.dtype || !internal::is_row_major_matrix(output_signal) ||
      output_signal.shape[0] != input_signal.shape[0] ||
      output_signal.shape[1] != input_signal.shape[1]) {
    return internal::invalid_view("output tensor must be CUDA row-major [num_envs, signal_dim]");
  }
  return internal::ok();
}

Result validate_cuda_vector_parameter_metadata(const ParameterView* parameter,
                                               const char* missing_message,
                                               const DType dtype,
                                               const std::size_t num_envs,
                                               const std::size_t signal_dim,
                                               const bool required) {
  if (parameter == nullptr) {
    return required ? internal::invalid_view(missing_message) : internal::ok();
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr) {
    return internal::invalid_view("parameter data is null");
  }
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view("parameter must be a CUDA tensor matching input dtype");
  }
  if (parameter->shared) {
    if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
        values.strides[0] >= 1) {
      return internal::ok();
    }
    return internal::invalid_view("shared vector parameter must have shape [signal_dim]");
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(signal_dim) && values.strides[1] == 1 &&
      values.strides[0] >= values.shape[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched vector parameter must have shape [num_envs, signal_dim]");
}

Result validate_cuda_proportional_prepare(const TensorView& input_signal,
                                          const ParameterSetView& parameters,
                                          const MutableTensorView& output_signal,
                                          std::size_t* num_envs,
                                          std::size_t* signal_dim) {
  const ParameterView* gain = find_parameter(parameters, kGain);
  Result result =
      validate_cuda_prepare_common(input_signal, TensorView{}, output_signal, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter_metadata(gain,
                                                 "proportional gain parameter is missing",
                                                 input_signal.dtype,
                                                 *num_envs,
                                                 *signal_dim,
                                                 true);
}

Result validate_cuda_integrator_prepare(const TensorView& input_signal,
                                        const TensorView& dt,
                                        const ParameterSetView& parameters,
                                        const MutableTensorView& output_signal,
                                        std::size_t* num_envs,
                                        std::size_t* signal_dim) {
  Result result =
      validate_cuda_prepare_common(input_signal, dt, output_signal, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter_metadata(find_parameter(parameters, kU0),
                                                   "u_0 parameter is missing",
                                                   input_signal.dtype,
                                                   *num_envs,
                                                   *signal_dim,
                                                   false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter_metadata(find_parameter(parameters, kUpperLimit),
                                                   "upper_limit parameter is missing",
                                                   input_signal.dtype,
                                                   *num_envs,
                                                   *signal_dim,
                                                   false);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter_metadata(find_parameter(parameters, kLowerLimit),
                                                 "lower_limit parameter is missing",
                                                 input_signal.dtype,
                                                 *num_envs,
                                                 *signal_dim,
                                                 false);
}
#endif

Result validate_initialized_state(const MutableTensorView& initialized,
                                  const std::size_t num_envs) {
  if (initialized.data == nullptr) {
    return internal::invalid_view("initialized state is missing");
  }
  if (internal::is_cuda(initialized)) {
    return internal::unsupported("CUDA state tensors require CUDA tensor kernels");
  }
  if (!internal::is_cpu(initialized)) {
    return internal::invalid_view("initialized state device is unsupported");
  }
  if (initialized.dtype != DType::kBool || initialized.rank != 1 ||
      initialized.shape[0] != static_cast<std::int64_t>(num_envs) || initialized.strides[0] < 1) {
    return internal::invalid_view("initialized state must be bool [num_envs]");
  }
  return internal::ok();
}

Result validate_integrator_state(StateView& state,
                                 const DType dtype,
                                 const std::size_t num_envs,
                                 const std::size_t signal_dim) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous == nullptr || integrated == nullptr || initialized == nullptr) {
    return internal::invalid_view(
        "integrator state requires previous_signal, integrated_signal, and initialized");
  }
  Result result = internal::validate_matching_output(previous->values, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(integrated->values, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(initialized->values, num_envs);
}

Result validate_differentiator_state(StateView& state,
                                     const DType dtype,
                                     const std::size_t num_envs,
                                     const std::size_t signal_dim) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous == nullptr || initialized == nullptr) {
    return internal::invalid_view("differentiator state requires previous_signal and initialized");
  }
  Result result = internal::validate_matching_output(previous->values, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(initialized->values, num_envs);
}

Result validate_common_step(const TensorView& input_signal,
                            const MutableTensorView& output_signal,
                            MutableStatusView status,
                            std::size_t* num_envs,
                            std::size_t* signal_dim) {
  Result result = internal::validate_batched_signal(
      input_signal, "input signal tensor data is null", num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matching_output(output_signal, input_signal.dtype, *num_envs, *signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, *num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, *num_envs);
  return internal::ok();
}

Result validate_dt(const TensorView& dt, const DType dtype, const std::size_t num_envs) {
  return internal::validate_scalar_batch(dt, "dt tensor data is null", dtype, num_envs);
}

template <typename Scalar>
bool gain_is_valid(const ParameterSetView& parameters,
                   const std::size_t env,
                   const std::size_t signal_dim) {
  const ParameterView* gain = find_parameter(parameters, kGain);
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!std::isfinite(
            internal::read_vector_parameter<Scalar>(gain, env, dim, static_cast<Scalar>(0)))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_proportional_parameters(const ParameterSetView& parameters,
                                     const std::size_t num_envs,
                                     const std::size_t signal_dim,
                                     MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);

  const ParameterView* gain = find_parameter(parameters, kGain);
  result = internal::validate_vector_parameter_metadata(gain,
                                                        "proportional gain parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        signal_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }

  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!gain_is_valid<Scalar>(parameters, env, signal_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_proportional(const TensorView& input_signal,
                         const ParameterSetView& parameters,
                         StatusView parameter_status,
                         const MutableTensorView& output_signal,
                         MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = validate_common_step(input_signal, output_signal, status, &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* gain = find_parameter(parameters, kGain);
  result = internal::validate_vector_parameter_metadata(gain,
                                                        "proportional gain parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        signal_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }

  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
    if (!gain_is_valid<Scalar>(parameters, env, signal_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    if (!internal::matrix_env_is_finite<Scalar>(input_signal, env, signal_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kInvalidParameter) |
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(output_signal, env, signal_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      internal::write_matrix_value(
          output_signal,
          env,
          dim,
          detail::proportional_value(
              internal::read_vector_parameter<Scalar>(gain, env, dim, static_cast<Scalar>(0)),
              internal::read_matrix_value<Scalar>(input_signal, env, dim)));
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool integrator_parameters_are_valid(const ParameterSetView& parameters,
                                     const std::size_t env,
                                     const std::size_t signal_dim) {
  const ParameterView* u0 = find_parameter(parameters, kU0);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar u0_value =
        internal::read_vector_parameter<Scalar>(u0, env, dim, static_cast<Scalar>(0));
    const Scalar upper_value = internal::read_vector_parameter<Scalar>(
        upper, env, dim, std::numeric_limits<Scalar>::max());
    const Scalar lower_value = internal::read_vector_parameter<Scalar>(
        lower, env, dim, -std::numeric_limits<Scalar>::max());
    if (!std::isfinite(u0_value) || !std::isfinite(upper_value) || !std::isfinite(lower_value) ||
        upper_value < lower_value) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_integrator_parameters(const ParameterSetView& parameters,
                                   const std::size_t num_envs,
                                   const std::size_t signal_dim,
                                   MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);

  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kU0),
                                                        "u_0 parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        signal_dim,
                                                        false);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kUpperLimit),
                                                        "upper_limit parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        signal_dim,
                                                        false);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kLowerLimit),
                                                        "lower_limit parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        signal_dim,
                                                        false);
  if (!result.ok()) {
    return result;
  }

  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!integrator_parameters_are_valid<Scalar>(parameters, env, signal_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result reset_integrator(const std::size_t num_envs,
                        StateView& state,
                        const TensorView& reset_mask,
                        MutableStatusView status) {
  const std::size_t signal_dim =
      find_state(state, kPreviousSignal) == nullptr
          ? 0
          : static_cast<std::size_t>(find_state(state, kPreviousSignal)->values.shape[1]);
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_integrator_state(state, dtype_for<Scalar>(), num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }

  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      internal::write_matrix_value(previous->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(integrated->values, env, dim, static_cast<Scalar>(0));
    }
    initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_integrator(const TensorView& input_signal,
                       const TensorView& dt,
                       const ParameterSetView& parameters,
                       StatusView parameter_status,
                       StateView& state,
                       const MutableTensorView& output_signal,
                       MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = validate_common_step(input_signal, output_signal, status, &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_dt(dt, input_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_integrator_state(state, dtype_for<Scalar>(), num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = apply_integrator_parameters<Scalar>(
      parameters,
      num_envs,
      signal_dim,
      MutableStatusView{status.flags, status.size, status.device});
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  const ParameterView* u0 = find_parameter(parameters, kU0);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);

  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(input_signal, env, signal_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kInvalidParameter) |
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kNonFiniteInput) |
          status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(output_signal, env, signal_dim);
      continue;
    }

    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    if (initialized_values[initialized_offset] == 0U) {
      for (std::size_t dim = 0; dim < signal_dim; ++dim) {
        const Scalar input_value = internal::read_matrix_value<Scalar>(input_signal, env, dim);
        internal::write_matrix_value(previous->values, env, dim, input_value);
        internal::write_matrix_value(
            integrated->values,
            env,
            dim,
            internal::read_vector_parameter<Scalar>(u0, env, dim, static_cast<Scalar>(0)));
        internal::write_matrix_value(output_signal, env, dim, static_cast<Scalar>(0));
      }
      initialized_values[initialized_offset] = 1U;
      continue;
    }

    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      const Scalar upper_value = internal::read_vector_parameter<Scalar>(
          upper, env, dim, std::numeric_limits<Scalar>::max());
      const Scalar lower_value = internal::read_vector_parameter<Scalar>(
          lower, env, dim, -std::numeric_limits<Scalar>::max());
      const auto update = detail::integrator_trapezoid_value(
          internal::read_matrix_value<Scalar>(integrated->values, env, dim),
          internal::read_matrix_value<Scalar>(input_signal, env, dim),
          internal::read_matrix_value<Scalar>(previous->values, env, dim),
          dt_value,
          lower_value,
          upper_value);
      if (update.saturated) {
        status.flags[env] |= status_bit(StatusFlag::kSaturation);
      }
      internal::write_matrix_value(integrated->values, env, dim, update.value);
      internal::write_matrix_value(output_signal, env, dim, update.value);
      internal::write_matrix_value(
          previous->values, env, dim, internal::read_matrix_value<Scalar>(input_signal, env, dim));
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result reset_differentiator(const std::size_t num_envs,
                            StateView& state,
                            const TensorView& reset_mask,
                            MutableStatusView status) {
  const std::size_t signal_dim =
      find_state(state, kPreviousSignal) == nullptr
          ? 0
          : static_cast<std::size_t>(find_state(state, kPreviousSignal)->values.shape[1]);
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_differentiator_state(state, dtype_for<Scalar>(), num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }

  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      internal::write_matrix_value(previous->values, env, dim, static_cast<Scalar>(0));
    }
    initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_differentiator(const TensorView& input_signal,
                           const TensorView& dt,
                           StatusView parameter_status,
                           StateView& state,
                           const MutableTensorView& output_signal,
                           MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = validate_common_step(input_signal, output_signal, status, &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_dt(dt, input_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_differentiator_state(state, dtype_for<Scalar>(), num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }

  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);

  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(input_signal, env, signal_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kInvalidParameter) |
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kNonFiniteInput) |
          status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(output_signal, env, signal_dim);
      continue;
    }

    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    if (initialized_values[initialized_offset] == 0U) {
      for (std::size_t dim = 0; dim < signal_dim; ++dim) {
        internal::write_matrix_value(previous->values,
                                     env,
                                     dim,
                                     internal::read_matrix_value<Scalar>(input_signal, env, dim));
        internal::write_matrix_value(output_signal, env, dim, static_cast<Scalar>(0));
      }
      initialized_values[initialized_offset] = 1U;
      continue;
    }

    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      const Scalar input_value = internal::read_matrix_value<Scalar>(input_signal, env, dim);
      internal::write_matrix_value(output_signal,
                                   env,
                                   dim,
                                   detail::differentiator_value(input_value,
                                                                internal::read_matrix_value<Scalar>(
                                                                    previous->values, env, dim),
                                                                dt_value));
      internal::write_matrix_value(previous->values, env, dim, input_value);
    }
  }
  return internal::ok();
}

PrepareResult generic_prepare_stateless(const TensorView& input_signal,
                                        const ParameterSetView& parameters,
                                        const MutableTensorView& output_signal,
                                        const char* output_name,
                                        const char* units,
                                        const char* semantics) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  if (proportional_uses_cuda(input_signal, parameters, output_signal)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_proportional_prepare(
        input_signal, parameters, output_signal, &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = signal_dim;
    prepare_result.outputs.push_back(internal::matrix_descriptor(
        output_name, input_signal.dtype, num_envs, signal_dim, units, "caller_defined", semantics));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      input_signal, "input signal tensor data is null", &num_envs, &signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matching_output(output_signal, input_signal.dtype, num_envs, signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  const ParameterView* gain = find_parameter(parameters, kGain);
  prepare_result.result =
      internal::validate_vector_parameter_metadata(gain,
                                                   "proportional gain parameter is missing",
                                                   input_signal.dtype,
                                                   num_envs,
                                                   signal_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = signal_dim;
  prepare_result.outputs.push_back(internal::matrix_descriptor(
      output_name, input_signal.dtype, num_envs, signal_dim, units, "caller_defined", semantics));
  prepare_result.outputs.back().device = output_signal.device;
  return prepare_result;
}

PrepareResult generic_prepare_stateful(const TensorView& input_signal,
                                       const TensorView& dt,
                                       const MutableTensorView& output_signal,
                                       const bool needs_integral,
                                       const char* output_name,
                                       const char* units,
                                       const char* semantics) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  if (stateful_uses_cuda(input_signal, dt, ParameterSetView{}, output_signal)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    if (dt.data == nullptr) {
      prepare_result.result = internal::invalid_view("dt tensor data is null");
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_prepare_common(input_signal, dt, output_signal, &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = signal_dim;
    prepare_result.state.push_back(internal::matrix_descriptor("previous_signal",
                                                               input_signal.dtype,
                                                               num_envs,
                                                               signal_dim,
                                                               units,
                                                               "caller_defined",
                                                               "previous valid input signal"));
    prepare_result.state.back().device = Device::kCuda;
    if (needs_integral) {
      prepare_result.state.push_back(internal::matrix_descriptor("integrated_signal",
                                                                 input_signal.dtype,
                                                                 num_envs,
                                                                 signal_dim,
                                                                 units,
                                                                 "caller_defined",
                                                                 "persistent trapezoid integral"));
      prepare_result.state.back().device = Device::kCuda;
    }
    prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "per-environment initialized flag"));
    prepare_result.state.back().device = Device::kCuda;
    prepare_result.outputs.push_back(internal::matrix_descriptor(
        output_name, input_signal.dtype, num_envs, signal_dim, units, "caller_defined", semantics));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      input_signal, "input signal tensor data is null", &num_envs, &signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_dt(dt, input_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matching_output(output_signal, input_signal.dtype, num_envs, signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = signal_dim;
  prepare_result.state.push_back(internal::matrix_descriptor("previous_signal",
                                                             input_signal.dtype,
                                                             num_envs,
                                                             signal_dim,
                                                             units,
                                                             "caller_defined",
                                                             "previous valid input signal"));
  if (needs_integral) {
    prepare_result.state.push_back(internal::matrix_descriptor("integrated_signal",
                                                               input_signal.dtype,
                                                               num_envs,
                                                               signal_dim,
                                                               units,
                                                               "caller_defined",
                                                               "persistent trapezoid integral"));
  }
  prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "per-environment initialized flag"));
  prepare_result.outputs.push_back(internal::matrix_descriptor(
      output_name, input_signal.dtype, num_envs, signal_dim, units, "caller_defined", semantics));
  return prepare_result;
}

}  // namespace

PrepareResult prepare_generic_proportional_block(const TensorView& input_signal,
                                                 const ParameterSetView& parameters,
                                                 const MutableTensorView& output_signal) {
  return generic_prepare_stateless(input_signal,
                                   parameters,
                                   output_signal,
                                   "output_signal",
                                   "same_as_input_signal",
                                   "elementwise gain multiplied by input_signal");
}

Result apply_generic_proportional_block_parameters(const ParameterSetView& parameters,
                                                   const std::size_t num_envs,
                                                   const std::size_t signal_dim,
                                                   MutableStatusView parameter_status) {
  if (proportional_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_generic_proportional_block_parameters_cuda(
        parameters, num_envs, signal_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (parameters.parameter_count == 0) {
    return internal::invalid_view("proportional gain parameter is missing");
  }
  const ParameterView* gain = find_parameter(parameters, kGain);
  if (gain == nullptr) {
    return internal::invalid_view("proportional gain parameter is missing");
  }
  return gain->values.dtype == DType::kFloat32
             ? apply_proportional_parameters<float>(
                   parameters, num_envs, signal_dim, parameter_status)
             : apply_proportional_parameters<double>(
                   parameters, num_envs, signal_dim, parameter_status);
}

Result reset_generic_proportional_block(const std::size_t num_envs,
                                        const TensorView& reset_mask,
                                        MutableStatusView status) {
  if (proportional_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_generic_proportional_block_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (internal::reset_mask_value(reset_mask, env)) {
      status.flags[env] |= status_bit(StatusFlag::kResetApplied);
    }
  }
  return internal::ok();
}

Result step_generic_proportional_block(const TensorView& input_signal,
                                       const ParameterSetView& parameters,
                                       StatusView parameter_status,
                                       const MutableTensorView& output_signal,
                                       MutableStatusView status) {
  if (proportional_uses_cuda(input_signal, parameters, parameter_status, output_signal, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_generic_proportional_block_cuda(
        input_signal, parameters, parameter_status, output_signal, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (input_signal.dtype == DType::kFloat32) {
    return step_proportional<float>(
        input_signal, parameters, parameter_status, output_signal, status);
  }
  if (input_signal.dtype == DType::kFloat64) {
    return step_proportional<double>(
        input_signal, parameters, parameter_status, output_signal, status);
  }
  return internal::invalid_view("input tensor dtype must be float32 or float64");
}

PrepareResult prepare_generic_integrator_block(const TensorView& input_signal,
                                               const TensorView& dt,
                                               const ParameterSetView& parameters,
                                               StateView& state,
                                               const MutableTensorView& output_signal) {
  (void)state;
  PrepareResult prepare_result = generic_prepare_stateful(input_signal,
                                                          dt,
                                                          output_signal,
                                                          true,
                                                          "integrated_signal",
                                                          "input_units_seconds",
                                                          "trapezoid integral output");
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (stateful_uses_cuda(input_signal, dt, parameters, output_signal)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_integrator_prepare(input_signal,
                                                             dt,
                                                             parameters,
                                                             output_signal,
                                                             &prepare_result.num_envs,
                                                             &prepare_result.signal_dim);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kU0),
                                                   "u_0 parameter is missing",
                                                   input_signal.dtype,
                                                   prepare_result.num_envs,
                                                   prepare_result.signal_dim,
                                                   false);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kUpperLimit),
                                                   "upper_limit parameter is missing",
                                                   input_signal.dtype,
                                                   prepare_result.num_envs,
                                                   prepare_result.signal_dim,
                                                   false);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kLowerLimit),
                                                   "lower_limit parameter is missing",
                                                   input_signal.dtype,
                                                   prepare_result.num_envs,
                                                   prepare_result.signal_dim,
                                                   false);
  return prepare_result;
}

Result apply_generic_integrator_block_parameters(const ParameterSetView& parameters,
                                                 const std::size_t num_envs,
                                                 const std::size_t signal_dim,
                                                 MutableStatusView parameter_status) {
  if (proportional_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_generic_integrator_block_parameters_cuda(
        parameters, num_envs, signal_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  DType dtype = DType::kFloat64;
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.dtype == DType::kFloat32 ||
        parameters.parameters[i].values.dtype == DType::kFloat64) {
      dtype = parameters.parameters[i].values.dtype;
      break;
    }
  }
  return dtype == DType::kFloat32 ? apply_integrator_parameters<float>(
                                        parameters, num_envs, signal_dim, parameter_status)
                                  : apply_integrator_parameters<double>(
                                        parameters, num_envs, signal_dim, parameter_status);
}

Result reset_generic_integrator_block(const std::size_t num_envs,
                                      StateView& state,
                                      const TensorView& reset_mask,
                                      MutableStatusView status) {
  if (stateful_uses_cuda(reset_mask, state, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_generic_integrator_block_cuda(num_envs, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* previous = find_state(state, kPreviousSignal);
  if (previous == nullptr) {
    return internal::invalid_view("integrator state requires previous_signal");
  }
  return previous->values.dtype == DType::kFloat32
             ? reset_integrator<float>(num_envs, state, reset_mask, status)
             : reset_integrator<double>(num_envs, state, reset_mask, status);
}

Result step_generic_integrator_block(const TensorView& input_signal,
                                     const TensorView& dt,
                                     const ParameterSetView& parameters,
                                     StatusView parameter_status,
                                     StateView& state,
                                     const MutableTensorView& output_signal,
                                     MutableStatusView status) {
  if (stateful_uses_cuda(
          input_signal, dt, parameters, parameter_status, state, output_signal, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_generic_integrator_block_cuda(
        input_signal, dt, parameters, parameter_status, state, output_signal, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (input_signal.dtype == DType::kFloat32) {
    return step_integrator<float>(
        input_signal, dt, parameters, parameter_status, state, output_signal, status);
  }
  if (input_signal.dtype == DType::kFloat64) {
    return step_integrator<double>(
        input_signal, dt, parameters, parameter_status, state, output_signal, status);
  }
  return internal::invalid_view("input tensor dtype must be float32 or float64");
}

PrepareResult prepare_generic_differentiator_block(const TensorView& input_signal,
                                                   const TensorView& dt,
                                                   StateView& state,
                                                   const MutableTensorView& output_signal) {
  (void)state;
  return generic_prepare_stateful(input_signal,
                                  dt,
                                  output_signal,
                                  false,
                                  "derivative_signal",
                                  "input_units_per_second",
                                  "finite-difference derivative output");
}

Result apply_generic_differentiator_block_parameters(const std::size_t num_envs,
                                                     MutableStatusView parameter_status) {
  if (parameter_status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_generic_differentiator_block_parameters_cuda(num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  return internal::ok();
}

Result reset_generic_differentiator_block(const std::size_t num_envs,
                                          StateView& state,
                                          const TensorView& reset_mask,
                                          MutableStatusView status) {
  if (stateful_uses_cuda(reset_mask, state, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_generic_differentiator_block_cuda(num_envs, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* previous = find_state(state, kPreviousSignal);
  if (previous == nullptr) {
    return internal::invalid_view("differentiator state requires previous_signal");
  }
  return previous->values.dtype == DType::kFloat32
             ? reset_differentiator<float>(num_envs, state, reset_mask, status)
             : reset_differentiator<double>(num_envs, state, reset_mask, status);
}

Result step_generic_differentiator_block(const TensorView& input_signal,
                                         const TensorView& dt,
                                         StatusView parameter_status,
                                         StateView& state,
                                         const MutableTensorView& output_signal,
                                         MutableStatusView status) {
  if (stateful_uses_cuda(
          input_signal, dt, ParameterSetView{}, parameter_status, state, output_signal, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_generic_differentiator_block_cuda(
        input_signal, dt, parameter_status, state, output_signal, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (input_signal.dtype == DType::kFloat32) {
    return step_differentiator<float>(
        input_signal, dt, parameter_status, state, output_signal, status);
  }
  if (input_signal.dtype == DType::kFloat64) {
    return step_differentiator<double>(
        input_signal, dt, parameter_status, state, output_signal, status);
  }
  return internal::invalid_view("input tensor dtype must be float32 or float64");
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
