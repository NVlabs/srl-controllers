/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file energy_tank.cpp
 * @brief CPU tensor kernel for EnergyTank.
 */

#include "nvidia/srl/controllers/tensor/energy_tank.hpp"

#include <cmath>
#include <limits>

#include "nvidia/srl/controllers/detail/math_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/energy_tank_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kUpperEnergyLimit = "upper_energy_limit";
constexpr const char* kLowerEnergyLimit = "lower_energy_limit";
constexpr const char* kEmptyThreshold = "empty_threshold";
constexpr const char* kInitialEnergy = "initial_energy";
constexpr const char* kPowerNoise = "power_noise";
constexpr const char* kStateVariable = "state_variable";

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

bool energy_lifecycle_uses_cuda(const TensorView& power_signal,
                                const TensorView& additional_power,
                                const TensorView& dt,
                                const ParameterSetView& parameters,
                                const StateView& state,
                                const MutableTensorView& scale,
                                const MutableTensorView& filtered_power) {
  return power_signal.device == Device::kCuda || additional_power.device == Device::kCuda ||
         dt.device == Device::kCuda || scale.device == Device::kCuda ||
         filtered_power.device == Device::kCuda || parameter_set_has_cuda(parameters) ||
         state_has_cuda(state);
}

bool energy_lifecycle_uses_cuda(const ParameterSetView& parameters,
                                const MutableStatusView& status) {
  return status.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool energy_lifecycle_uses_cuda(const ParameterSetView& parameters,
                                const StatusView& parameter_status,
                                const StateView& state,
                                const TensorView& reset_mask,
                                const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters) || state_has_cuda(state);
}

bool energy_lifecycle_uses_cuda(const TensorView& power_signal,
                                const TensorView& additional_power,
                                const TensorView& dt,
                                const ParameterSetView& parameters,
                                const StatusView& parameter_status,
                                const StateView& state,
                                const MutableTensorView& scale,
                                const MutableTensorView& filtered_power,
                                const MutableStatusView& status) {
  return power_signal.device == Device::kCuda || additional_power.device == Device::kCuda ||
         dt.device == Device::kCuda || scale.device == Device::kCuda ||
         filtered_power.device == Device::kCuda || status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters) || state_has_cuda(state);
}

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
Result validate_cuda_scalar_batch_metadata(const TensorView& input,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const bool allow_scalar) {
  if (input.data == nullptr) {
    return internal::invalid_view(name);
  }
  if (input.device != Device::kCuda || input.dtype != dtype) {
    return internal::invalid_view("scalar batch tensor must be CUDA and match dtype");
  }
  if (allow_scalar && input.rank == 0) {
    return internal::ok();
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return internal::ok();
  }
  return internal::invalid_view("scalar batch tensor must have shape [num_envs]");
}

Result validate_cuda_scalar_output_metadata(const MutableTensorView& output,
                                            const DType dtype,
                                            const std::size_t num_envs,
                                            const char* name) {
  if (output.data == nullptr) {
    return internal::invalid_view(name);
  }
  if (output.device != Device::kCuda || output.dtype != dtype || output.rank != 1 ||
      output.shape[0] < static_cast<std::int64_t>(num_envs) || output.strides[0] < 1) {
    return internal::invalid_view("scalar output must be CUDA [num_envs] matching dtype");
  }
  return internal::ok();
}

Result validate_cuda_scalar_parameter_metadata(const ParameterView* parameter,
                                               const char* missing_message,
                                               const DType dtype,
                                               const std::size_t num_envs,
                                               const bool required) {
  if (parameter == nullptr) {
    return required ? internal::invalid_view(missing_message) : internal::ok();
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr) {
    return internal::invalid_view("EnergyTank parameter data is null");
  }
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view(
        "EnergyTank parameter must be a CUDA tensor matching component dtype");
  }
  if (parameter->shared) {
    if (values.rank == 0 || (values.rank == 1 && values.shape[0] == 1 && values.strides[0] >= 1)) {
      return internal::ok();
    }
    return internal::invalid_view("shared EnergyTank parameter must be scalar or [1]");
  }
  if (values.rank == 1 && values.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      values.strides[0] >= 1) {
    return internal::ok();
  }
  return internal::invalid_view("batched EnergyTank parameter must be [num_envs]");
}

Result validate_cuda_energy_parameter_metadata(const ParameterSetView& parameters,
                                               const DType dtype,
                                               const std::size_t num_envs) {
  const char* required_names[] = {
      kUpperEnergyLimit, kLowerEnergyLimit, kEmptyThreshold, kInitialEnergy};
  for (const char* name : required_names) {
    Result result = validate_cuda_scalar_parameter_metadata(find_parameter(parameters, name),
                                                            "required EnergyTank parameter missing",
                                                            dtype,
                                                            num_envs,
                                                            true);
    if (!result.ok()) {
      return result;
    }
  }
  return validate_cuda_scalar_parameter_metadata(find_parameter(parameters, kPowerNoise),
                                                 "power_noise parameter missing",
                                                 dtype,
                                                 num_envs,
                                                 false);
}

Result validate_cuda_prepare(const TensorView& power_signal,
                             const TensorView& additional_power,
                             const TensorView& dt,
                             const ParameterSetView& parameters,
                             const MutableTensorView& scale,
                             const MutableTensorView& filtered_power,
                             std::size_t* num_envs) {
  if (power_signal.data == nullptr || power_signal.device != Device::kCuda ||
      !internal::is_float_dtype(power_signal.dtype) || power_signal.rank != 1 ||
      power_signal.shape[0] <= 0 || power_signal.strides[0] < 1) {
    return internal::invalid_view("power_signal must be CUDA [num_envs]");
  }
  *num_envs = static_cast<std::size_t>(power_signal.shape[0]);
  Result result = validate_cuda_scalar_batch_metadata(additional_power,
                                                      "additional_power tensor data is null",
                                                      power_signal.dtype,
                                                      *num_envs,
                                                      true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch_metadata(
      dt, "dt tensor data is null", power_signal.dtype, *num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_output_metadata(
      scale, power_signal.dtype, *num_envs, "scale output data is null");
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_output_metadata(
      filtered_power, power_signal.dtype, *num_envs, "filtered_power output data is null");
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_energy_parameter_metadata(parameters, power_signal.dtype, *num_envs);
}
#endif

Result validate_energy_state(StateView& state, const DType dtype, const std::size_t num_envs) {
  StateEntryView* state_variable = find_state(state, kStateVariable);
  if (state_variable == nullptr) {
    return internal::invalid_view("EnergyTank state requires state_variable");
  }
  MutableTensorView& values = state_variable->values;
  if (values.data == nullptr) {
    return internal::invalid_view("EnergyTank state_variable data is null");
  }
  if (internal::is_cuda(values)) {
    return internal::unsupported("CUDA state tensors require CUDA tensor kernels");
  }
  if (!internal::is_cpu(values)) {
    return internal::invalid_view("EnergyTank state device is unsupported");
  }
  if (values.dtype != dtype || values.rank != 1 ||
      values.shape[0] != static_cast<std::int64_t>(num_envs) || values.strides[0] < 1) {
    return internal::invalid_view("EnergyTank state_variable must be [num_envs]");
  }
  return internal::ok();
}

Result validate_energy_parameter_metadata(const ParameterSetView& parameters,
                                          const DType dtype,
                                          const std::size_t num_envs) {
  const char* required_names[] = {
      kUpperEnergyLimit, kLowerEnergyLimit, kEmptyThreshold, kInitialEnergy};
  for (const char* name : required_names) {
    Result result =
        internal::validate_scalar_parameter_metadata(find_parameter(parameters, name),
                                                     "required EnergyTank parameter missing",
                                                     dtype,
                                                     num_envs,
                                                     true);
    if (!result.ok()) {
      return result;
    }
  }
  return internal::validate_scalar_parameter_metadata(find_parameter(parameters, kPowerNoise),
                                                      "power_noise parameter missing",
                                                      dtype,
                                                      num_envs,
                                                      false);
}

template <typename Scalar>
bool energy_parameters_are_valid(const ParameterSetView& parameters, const std::size_t env) {
  const Scalar upper = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kUpperEnergyLimit), env, std::numeric_limits<Scalar>::quiet_NaN());
  const Scalar lower = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kLowerEnergyLimit), env, std::numeric_limits<Scalar>::quiet_NaN());
  const Scalar initial = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kInitialEnergy), env, std::numeric_limits<Scalar>::quiet_NaN());
  const Scalar empty = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kEmptyThreshold), env, std::numeric_limits<Scalar>::quiet_NaN());
  const Scalar noise = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kPowerNoise), env, static_cast<Scalar>(0));
  return detail::validate_energy_tank_parameters(upper, lower, initial, empty) &&
         std::isfinite(noise) && noise >= static_cast<Scalar>(0);
}

template <typename Scalar>
Result apply_energy_parameters(const ParameterSetView& parameters,
                               const std::size_t num_envs,
                               MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  result = validate_energy_parameter_metadata(parameters, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!energy_parameters_are_valid<Scalar>(parameters, env)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result reset_energy(const ParameterSetView& parameters,
                    StatusView parameter_status,
                    StateView& state,
                    const TensorView& reset_mask,
                    MutableStatusView status) {
  StateEntryView* state_variable = find_state(state, kStateVariable);
  if (state_variable == nullptr) {
    return internal::invalid_view("EnergyTank state requires state_variable");
  }
  const std::size_t num_envs = static_cast<std::size_t>(state_variable->values.shape[0]);
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_energy_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_energy_parameters<Scalar>(
      parameters, num_envs, MutableStatusView{status.flags, status.size, status.device});
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    if ((status.flags[env] & status_bit(StatusFlag::kInvalidParameter)) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      continue;
    }
    const Scalar initial = internal::read_scalar_parameter<Scalar>(
        find_parameter(parameters, kInitialEnergy), env, static_cast<Scalar>(0));
    internal::write_scalar_value(
        state_variable->values, env, std::sqrt(initial) * std::sqrt(static_cast<Scalar>(2)));
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_energy(const TensorView& power_signal,
                   const TensorView& additional_power,
                   const TensorView& dt,
                   const ParameterSetView& parameters,
                   StatusView parameter_status,
                   StateView& state,
                   const MutableTensorView& scale,
                   const MutableTensorView& filtered_power,
                   MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = internal::validate_scalar_batch(
      power_signal, "power_signal tensor data is null", dtype_for<Scalar>(), 1);
  if (!result.ok()) {
    return result;
  }
  if (power_signal.rank == 0) {
    return internal::invalid_view("power_signal must have shape [num_envs]");
  }
  num_envs = static_cast<std::size_t>(power_signal.shape[0]);
  result = internal::validate_scalar_batch(
      additional_power, "additional_power tensor data is null", dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_scalar_batch(dt, "dt tensor data is null", dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_scalar_output(scale, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_scalar_output(filtered_power, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_energy_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_energy_parameters<Scalar>(
      parameters, num_envs, MutableStatusView{status.flags, status.size, status.device});
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* state_variable = find_state(state, kStateVariable);
  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar power_value = internal::read_scalar_value<Scalar>(power_signal, env);
    const Scalar additional_value = internal::read_scalar_value<Scalar>(additional_power, env);
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (additional_value < static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidInput);
    }
    if (!std::isfinite(power_value) || !std::isfinite(additional_value)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kUnsupportedMode) |
          status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kNonFiniteInput) |
          status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::write_scalar_value(scale, env, static_cast<Scalar>(0));
      internal::write_scalar_value(filtered_power, env, static_cast<Scalar>(0));
      continue;
    }

    const Scalar state_value = internal::read_scalar_value<Scalar>(state_variable->values, env);
    if (!std::isfinite(state_value) || state_value <= static_cast<Scalar>(0)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::write_scalar_value(scale, env, static_cast<Scalar>(0));
      internal::write_scalar_value(filtered_power, env, static_cast<Scalar>(0));
      continue;
    }

    const auto update = detail::energy_tank_step(
        state_value,
        power_value,
        additional_value,
        dt_value,
        internal::read_scalar_parameter<Scalar>(
            find_parameter(parameters, kUpperEnergyLimit), env, static_cast<Scalar>(0)),
        internal::read_scalar_parameter<Scalar>(
            find_parameter(parameters, kLowerEnergyLimit), env, static_cast<Scalar>(0)),
        internal::read_scalar_parameter<Scalar>(
            find_parameter(parameters, kEmptyThreshold), env, static_cast<Scalar>(0)),
        internal::read_scalar_parameter<Scalar>(
            find_parameter(parameters, kPowerNoise), env, static_cast<Scalar>(0)));
    if (!std::isfinite(update.state_variable) || update.state_variable <= Scalar(0) ||
        !std::isfinite(update.scale) || !std::isfinite(update.filtered_power)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::write_scalar_value(scale, env, Scalar(0));
      internal::write_scalar_value(filtered_power, env, Scalar(0));
      continue;
    }
    internal::write_scalar_value(scale, env, update.scale);
    internal::write_scalar_value(filtered_power, env, update.filtered_power);
    internal::write_scalar_value(state_variable->values, env, update.state_variable);
    if (update.saturated) {
      status.flags[env] |= status_bit(StatusFlag::kSaturation);
    }
  }
  return internal::ok();
}

}  // namespace

PrepareResult prepare_energy_tank(const TensorView& power_signal,
                                  const TensorView& additional_power,
                                  const TensorView& dt,
                                  const ParameterSetView& parameters,
                                  StateView& state,
                                  const MutableTensorView& scale,
                                  const MutableTensorView& filtered_power) {
  PrepareResult prepare_result;
  if (energy_lifecycle_uses_cuda(
          power_signal, additional_power, dt, parameters, state, scale, filtered_power)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    prepare_result.result = validate_cuda_prepare(
        power_signal, additional_power, dt, parameters, scale, filtered_power, &num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = 1;
    prepare_result.state.push_back(internal::vector_descriptor("state_variable",
                                                               power_signal.dtype,
                                                               num_envs,
                                                               "sqrt(joule)",
                                                               "none",
                                                               "persistent tank state variable"));
    prepare_result.state.back().device = Device::kCuda;
    prepare_result.outputs.push_back(internal::vector_descriptor(
        "scale", power_signal.dtype, num_envs, "unitless", "none", "energy tank passivity scale"));
    prepare_result.outputs.back().device = Device::kCuda;
    prepare_result.outputs.push_back(internal::vector_descriptor("filtered_power",
                                                                 power_signal.dtype,
                                                                 num_envs,
                                                                 "watt",
                                                                 "none",
                                                                 "power after noise thresholding"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  (void)state;
  if (power_signal.dtype != DType::kFloat32 && power_signal.dtype != DType::kFloat64) {
    prepare_result.result = internal::invalid_view("power_signal dtype must be float32 or float64");
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      power_signal, "power_signal tensor data is null", power_signal.dtype, 1);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (power_signal.rank == 0) {
    prepare_result.result = internal::invalid_view("power_signal must have shape [num_envs]");
    return prepare_result;
  }
  const std::size_t num_envs = static_cast<std::size_t>(power_signal.shape[0]);
  prepare_result.result = internal::validate_scalar_batch(
      additional_power, "additional_power tensor data is null", power_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_scalar_batch(dt, "dt tensor data is null", power_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_output(scale, power_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_scalar_output(filtered_power, power_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_energy_parameter_metadata(parameters, power_signal.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = 1;
  prepare_result.state.push_back(internal::vector_descriptor("state_variable",
                                                             power_signal.dtype,
                                                             num_envs,
                                                             "sqrt(joule)",
                                                             "none",
                                                             "persistent tank state variable"));
  prepare_result.outputs.push_back(internal::vector_descriptor(
      "scale", power_signal.dtype, num_envs, "unitless", "none", "energy tank passivity scale"));
  prepare_result.outputs.push_back(internal::vector_descriptor("filtered_power",
                                                               power_signal.dtype,
                                                               num_envs,
                                                               "watt",
                                                               "none",
                                                               "power after noise thresholding"));
  return prepare_result;
}

Result apply_energy_tank_parameters(const ParameterSetView& parameters,
                                    const std::size_t num_envs,
                                    MutableStatusView parameter_status) {
  if (energy_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_energy_tank_parameters_cuda(parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  DType dtype = DType::kFloat64;
  const ParameterView* parameter = find_parameter(parameters, kUpperEnergyLimit);
  if (parameter != nullptr) {
    dtype = parameter->values.dtype;
  }
  return dtype == DType::kFloat32
             ? apply_energy_parameters<float>(parameters, num_envs, parameter_status)
             : apply_energy_parameters<double>(parameters, num_envs, parameter_status);
}

Result reset_energy_tank(const ParameterSetView& parameters,
                         StatusView parameter_status,
                         StateView& state,
                         const TensorView& reset_mask,
                         MutableStatusView status) {
  if (energy_lifecycle_uses_cuda(parameters, parameter_status, state, reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_energy_tank_cuda(parameters, parameter_status, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* state_variable = find_state(state, kStateVariable);
  if (state_variable == nullptr) {
    return internal::invalid_view("EnergyTank state requires state_variable");
  }
  return state_variable->values.dtype == DType::kFloat32
             ? reset_energy<float>(parameters, parameter_status, state, reset_mask, status)
             : reset_energy<double>(parameters, parameter_status, state, reset_mask, status);
}

Result step_energy_tank(const TensorView& power_signal,
                        const TensorView& additional_power,
                        const TensorView& dt,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        StateView& state,
                        const MutableTensorView& scale,
                        const MutableTensorView& filtered_power,
                        MutableStatusView status) {
  if (energy_lifecycle_uses_cuda(power_signal,
                                 additional_power,
                                 dt,
                                 parameters,
                                 parameter_status,
                                 state,
                                 scale,
                                 filtered_power,
                                 status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_energy_tank_cuda(power_signal,
                                 additional_power,
                                 dt,
                                 parameters,
                                 parameter_status,
                                 state,
                                 scale,
                                 filtered_power,
                                 status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (power_signal.dtype == DType::kFloat32) {
    return step_energy<float>(power_signal,
                              additional_power,
                              dt,
                              parameters,
                              parameter_status,
                              state,
                              scale,
                              filtered_power,
                              status);
  }
  if (power_signal.dtype == DType::kFloat64) {
    return step_energy<double>(power_signal,
                               additional_power,
                               dt,
                               parameters,
                               parameter_status,
                               state,
                               scale,
                               filtered_power,
                               status);
  }
  return internal::invalid_view("power_signal dtype must be float32 or float64");
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
