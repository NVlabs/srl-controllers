/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file energy_tank_cuda.cu
 * @brief CUDA tensor kernels for EnergyTank.
 */

#include "tensor/energy_tank_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/math_core.hpp"

#include "tensor/cuda_component_tuning_profiles.hpp"
#include "tensor/cuda_stream.hpp"

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
constexpr std::uint64_t kUnsupportedModeFlag =
    static_cast<std::uint64_t>(StatusFlag::kUnsupportedMode);
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDtFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kSaturationFlag = static_cast<std::uint64_t>(StatusFlag::kSaturation);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kSafeTriggers = kUnsupportedModeFlag | kInvalidParameterFlag |
                                        kInvalidInputFlag | kNonFiniteInputFlag | kInvalidDtFlag;

struct CudaLaunchConfig {
  int blocks{0};
  int threads{0};
};

template <typename Scalar>
struct CudaScalarBatchView {
  const Scalar* data{nullptr};
  int rank{0};
  std::int64_t stride{0};
};

template <typename Scalar>
struct CudaMutableScalarBatchView {
  Scalar* data{nullptr};
  std::int64_t stride{0};
};

template <typename Scalar>
struct CudaScalarParameterView {
  const Scalar* data{nullptr};
  bool present{false};
  bool shared{false};
  int rank{0};
  std::int64_t stride{0};
};

template <typename Scalar>
struct CudaEnergyParameters {
  CudaScalarParameterView<Scalar> upper;
  CudaScalarParameterView<Scalar> lower;
  CudaScalarParameterView<Scalar> empty;
  CudaScalarParameterView<Scalar> initial;
  CudaScalarParameterView<Scalar> noise;
};

template <typename Scalar>
__device__ bool is_finite(const Scalar value) {
  return isfinite(value);
}

template <typename Scalar>
__device__ Scalar read_scalar_batch(const CudaScalarBatchView<Scalar> view, const std::size_t env) {
  if (view.rank == 0) {
    return view.data[0];
  }
  return view.data[env * static_cast<std::size_t>(view.stride)];
}

template <typename Scalar>
__device__ void write_scalar_batch(CudaMutableScalarBatchView<Scalar> view,
                                   const std::size_t env,
                                   const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_scalar_parameter(const CudaScalarParameterView<Scalar> parameter,
                                        const std::size_t env,
                                        const Scalar default_value) {
  if (!parameter.present || parameter.data == nullptr) {
    return default_value;
  }
  if (parameter.shared || parameter.rank == 0) {
    return parameter.data[0];
  }
  return parameter.data[env * static_cast<std::size_t>(parameter.stride)];
}

template <typename Scalar>
__device__ bool validate_energy_parameters(const Scalar upper,
                                           const Scalar lower,
                                           const Scalar initial,
                                           const Scalar empty,
                                           const Scalar noise) {
  return detail::validate_energy_tank_parameters(upper, lower, initial, empty) &&
         is_finite(noise) && noise >= static_cast<Scalar>(0);
}

template <typename Scalar>
__device__ bool energy_parameters_are_valid(const CudaEnergyParameters<Scalar> parameters,
                                            const std::size_t env) {
  return validate_energy_parameters(
      read_scalar_parameter(parameters.upper, env, static_cast<Scalar>(0)),
      read_scalar_parameter(parameters.lower, env, static_cast<Scalar>(0)),
      read_scalar_parameter(parameters.initial, env, static_cast<Scalar>(0)),
      read_scalar_parameter(parameters.empty, env, static_cast<Scalar>(0)),
      read_scalar_parameter(parameters.noise, env, static_cast<Scalar>(0)));
}

template <typename Scalar>
__device__ void zero_outputs(CudaMutableScalarBatchView<Scalar> scale,
                             CudaMutableScalarBatchView<Scalar> filtered_power,
                             const std::size_t env) {
  write_scalar_batch(scale, env, static_cast<Scalar>(0));
  write_scalar_batch(filtered_power, env, static_cast<Scalar>(0));
}

template <typename Scalar>
__global__ void apply_energy_parameters_kernel(CudaEnergyParameters<Scalar> parameters,
                                               std::size_t num_envs,
                                               std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] = energy_parameters_are_valid(parameters, env) ? 0U : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void reset_energy_kernel(CudaEnergyParameters<Scalar> parameters,
                                    const std::uint64_t* parameter_status,
                                    CudaMutableScalarBatchView<Scalar> state,
                                    const std::uint8_t* reset_mask,
                                    std::int64_t reset_mask_stride,
                                    std::size_t num_envs,
                                    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!energy_parameters_are_valid(parameters, env)) {
    flags |= kInvalidParameterFlag;
  }
  const bool reset =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_mask_stride)] != 0U;
  if (reset && (flags & kInvalidParameterFlag) != 0U) {
    flags |= kSafeModeFlag;
  } else if (reset) {
    const Scalar initial = read_scalar_parameter(parameters.initial, env, static_cast<Scalar>(0));
    write_scalar_batch(
        state, env, detail::square_root(initial) * detail::square_root(static_cast<Scalar>(2)));
    flags |= kResetAppliedFlag;
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_energy_kernel(CudaScalarBatchView<Scalar> power_signal,
                                   CudaScalarBatchView<Scalar> additional_power,
                                   CudaScalarBatchView<Scalar> dt,
                                   CudaEnergyParameters<Scalar> parameters,
                                   const std::uint64_t* parameter_status,
                                   CudaMutableScalarBatchView<Scalar> state,
                                   CudaMutableScalarBatchView<Scalar> scale,
                                   CudaMutableScalarBatchView<Scalar> filtered_power,
                                   std::size_t num_envs,
                                   std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!energy_parameters_are_valid(parameters, env)) {
    flags |= kInvalidParameterFlag;
  }

  const Scalar power_value = read_scalar_batch(power_signal, env);
  const Scalar additional_value = read_scalar_batch(additional_power, env);
  const Scalar dt_value = read_scalar_batch(dt, env);
  if (additional_value < Scalar(0)) flags |= kInvalidInputFlag;
  if (!is_finite(power_value) || !is_finite(additional_value)) {
    flags |= kNonFiniteInputFlag;
  }
  if (!is_finite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_outputs(scale, filtered_power, env);
    status[env] = flags;
    return;
  }

  const Scalar state_value = state.data[env * static_cast<std::size_t>(state.stride)];
  if (!is_finite(state_value) || state_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_outputs(scale, filtered_power, env);
    status[env] = flags;
    return;
  }

  const Scalar upper = read_scalar_parameter(parameters.upper, env, static_cast<Scalar>(0));
  const Scalar lower = read_scalar_parameter(parameters.lower, env, static_cast<Scalar>(0));
  const Scalar empty = read_scalar_parameter(parameters.empty, env, static_cast<Scalar>(0));
  const Scalar noise = read_scalar_parameter(parameters.noise, env, static_cast<Scalar>(0));
  const auto update = detail::energy_tank_step(
      state_value, power_value, additional_value, dt_value, upper, lower, empty, noise);
  if (!is_finite(update.state_variable) || update.state_variable <= Scalar(0) ||
      !is_finite(update.scale) || !is_finite(update.filtered_power)) {
    zero_outputs(scale, filtered_power, env);
    status[env] = flags | kInvalidInputFlag | kSafeModeFlag;
    return;
  }
  if (update.saturated) {
    flags |= kSaturationFlag;
  }

  write_scalar_batch(scale, env, update.scale);
  write_scalar_batch(filtered_power, env, update.filtered_power);
  write_scalar_batch(state, env, update.state_variable);
  status[env] = flags;
}

inline Result cuda_launch_result() {
  const cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return Result{ResultCode::kInvalidArgument, "CUDA kernel launch failed"};
  }
  return Result{ResultCode::kOk, "ok"};
}

template <CudaTensorLaunchFamily Family, typename Scalar>
inline CudaLaunchConfig launch_config(const std::size_t num_envs) {
  constexpr int kThreads = cuda_tensor_tuned_threads<Family, Scalar>();
  return CudaLaunchConfig{static_cast<int>((num_envs + kThreads - 1) / kThreads), kThreads};
}

inline Result validate_cuda_status(const MutableStatusView& status, const std::size_t num_envs) {
  if (status.flags == nullptr) {
    return Result{ResultCode::kInvalidView, "status flags are null"};
  }
  if (status.size < num_envs) {
    return Result{ResultCode::kInvalidView, "status size is smaller than num_envs"};
  }
  if (status.device != Device::kCuda) {
    return Result{ResultCode::kInvalidView, "status must be a CUDA status view"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_status(const StatusView& status,
                                   const std::size_t num_envs,
                                   const bool allow_empty) {
  if (status.flags == nullptr) {
    return allow_empty ? Result{ResultCode::kOk, "ok"}
                       : Result{ResultCode::kInvalidView, "status flags are null"};
  }
  if (status.size < num_envs) {
    return Result{ResultCode::kInvalidView, "status size is smaller than num_envs"};
  }
  if (status.device != Device::kCuda) {
    return Result{ResultCode::kInvalidView, "status must be a CUDA status view"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline bool is_float_dtype(const DType dtype) {
  return dtype == DType::kFloat32 || dtype == DType::kFloat64;
}

inline Result validate_cuda_scalar_batch(const TensorView& input,
                                         const char* name,
                                         const DType dtype,
                                         const std::size_t num_envs,
                                         const bool allow_scalar) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype) {
    return Result{ResultCode::kInvalidView, "scalar batch tensor must be CUDA and match dtype"};
  }
  if (allow_scalar && input.rank == 0) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "scalar batch tensor must be [num_envs]"};
}

inline Result validate_cuda_scalar_output(const MutableTensorView& output,
                                          const DType dtype,
                                          const std::size_t num_envs,
                                          const char* name) {
  if (output.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (output.device != Device::kCuda || output.dtype != dtype || output.rank != 1 ||
      output.shape[0] < static_cast<std::int64_t>(num_envs) || output.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "scalar output must be CUDA [num_envs] matching dtype"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_reset_mask(const TensorView& reset_mask, const std::size_t num_envs) {
  if (reset_mask.data == nullptr) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (reset_mask.device != Device::kCuda || reset_mask.dtype != DType::kBool ||
      reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs) ||
      reset_mask.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "reset mask must be CUDA bool [num_envs]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_scalar_parameter(const ParameterView* parameter,
                                             const char* missing_message,
                                             const DType dtype,
                                             const std::size_t num_envs,
                                             const bool required) {
  if (parameter == nullptr) {
    return required ? Result{ResultCode::kInvalidView, missing_message}
                    : Result{ResultCode::kOk, "ok"};
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "EnergyTank parameter data is null"};
  }
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "EnergyTank parameter must be a CUDA tensor matching component dtype"};
  }
  if (parameter->shared) {
    if (values.rank == 0 || (values.rank == 1 && values.shape[0] == 1 && values.strides[0] >= 1)) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView, "shared EnergyTank parameter must be scalar or [1]"};
  }
  if (values.rank == 1 && values.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      values.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "batched EnergyTank parameter must be [num_envs]"};
}

inline Result validate_cuda_energy_parameters(const ParameterSetView& parameters,
                                              const DType dtype,
                                              const std::size_t num_envs) {
  const char* required_names[] = {
      kUpperEnergyLimit, kLowerEnergyLimit, kEmptyThreshold, kInitialEnergy};
  for (const char* name : required_names) {
    Result result = validate_cuda_scalar_parameter(find_parameter(parameters, name),
                                                   "required EnergyTank parameter missing",
                                                   dtype,
                                                   num_envs,
                                                   true);
    if (!result.ok()) {
      return result;
    }
  }
  return validate_cuda_scalar_parameter(find_parameter(parameters, kPowerNoise),
                                        "power_noise parameter missing",
                                        dtype,
                                        num_envs,
                                        false);
}

inline StateEntryView* energy_state(StateView& state) { return find_state(state, kStateVariable); }

inline Result validate_cuda_energy_state(StateView& state,
                                         const DType dtype,
                                         const std::size_t num_envs) {
  StateEntryView* state_variable = energy_state(state);
  if (state_variable == nullptr) {
    return Result{ResultCode::kInvalidView, "EnergyTank state requires state_variable"};
  }
  const MutableTensorView& values = state_variable->values;
  if (values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "EnergyTank state_variable data is null"};
  }
  if (values.device != Device::kCuda || values.dtype != dtype || values.rank != 1 ||
      values.shape[0] < static_cast<std::int64_t>(num_envs) || values.strides[0] < 1) {
    return Result{ResultCode::kInvalidView,
                  "EnergyTank state_variable must be CUDA [num_envs] matching dtype"};
  }
  return Result{ResultCode::kOk, "ok"};
}

template <typename Scalar>
CudaScalarBatchView<Scalar> as_cuda_scalar_batch(const TensorView& view) {
  return CudaScalarBatchView<Scalar>{
      static_cast<const Scalar*>(view.data), view.rank, view.rank == 0 ? 0 : view.strides[0]};
}

template <typename Scalar>
CudaMutableScalarBatchView<Scalar> as_cuda_mutable_scalar_batch(const MutableTensorView& view) {
  return CudaMutableScalarBatchView<Scalar>{static_cast<Scalar*>(view.data), view.strides[0]};
}

template <typename Scalar>
CudaScalarParameterView<Scalar> as_cuda_scalar_parameter(const ParameterView* parameter) {
  if (parameter == nullptr) {
    return CudaScalarParameterView<Scalar>{};
  }
  const TensorView& values = parameter->values;
  return CudaScalarParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         true,
                                         parameter->shared || values.rank == 0,
                                         values.rank,
                                         values.rank == 0 ? 0 : values.strides[0]};
}

template <typename Scalar>
CudaEnergyParameters<Scalar> as_cuda_energy_parameters(const ParameterSetView& parameters) {
  return CudaEnergyParameters<Scalar>{
      as_cuda_scalar_parameter<Scalar>(find_parameter(parameters, kUpperEnergyLimit)),
      as_cuda_scalar_parameter<Scalar>(find_parameter(parameters, kLowerEnergyLimit)),
      as_cuda_scalar_parameter<Scalar>(find_parameter(parameters, kEmptyThreshold)),
      as_cuda_scalar_parameter<Scalar>(find_parameter(parameters, kInitialEnergy)),
      as_cuda_scalar_parameter<Scalar>(find_parameter(parameters, kPowerNoise))};
}

template <typename Scalar>
Result launch_apply_energy(const ParameterSetView& parameters,
                           const std::size_t num_envs,
                           MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kEnergyApply, Scalar>(num_envs);
  apply_energy_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_energy_parameters<Scalar>(parameters), num_envs, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_energy(const ParameterSetView& parameters,
                           StatusView parameter_status,
                           StateView& state,
                           const TensorView& reset_mask,
                           MutableStatusView status,
                           const std::size_t num_envs) {
  StateEntryView* state_variable = energy_state(state);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kEnergyReset, Scalar>(num_envs);
  reset_energy_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_energy_parameters<Scalar>(parameters),
      parameter_status.flags,
      as_cuda_mutable_scalar_batch<Scalar>(state_variable->values),
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_energy(const TensorView& power_signal,
                          const TensorView& additional_power,
                          const TensorView& dt,
                          const ParameterSetView& parameters,
                          StatusView parameter_status,
                          StateView& state,
                          const MutableTensorView& scale,
                          const MutableTensorView& filtered_power,
                          MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(power_signal.shape[0]);
  StateEntryView* state_variable = energy_state(state);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kEnergyStep, Scalar>(num_envs);
  step_energy_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_scalar_batch<Scalar>(power_signal),
      as_cuda_scalar_batch<Scalar>(additional_power),
      as_cuda_scalar_batch<Scalar>(dt),
      as_cuda_energy_parameters<Scalar>(parameters),
      parameter_status.flags,
      as_cuda_mutable_scalar_batch<Scalar>(state_variable->values),
      as_cuda_mutable_scalar_batch<Scalar>(scale),
      as_cuda_mutable_scalar_batch<Scalar>(filtered_power),
      num_envs,
      status.flags);
  return cuda_launch_result();
}

}  // namespace

Result apply_energy_tank_parameters_cuda(const ParameterSetView& parameters,
                                         const std::size_t num_envs,
                                         MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* upper = find_parameter(parameters, kUpperEnergyLimit);
  const DType dtype = upper == nullptr ? DType::kUnknown : upper->values.dtype;
  if (!is_float_dtype(dtype)) {
    return Result{ResultCode::kInvalidView,
                  "EnergyTank parameter dtype must be float32 or float64"};
  }
  result = validate_cuda_energy_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_energy<float>(parameters, num_envs, parameter_status)
             : launch_apply_energy<double>(parameters, num_envs, parameter_status);
}

Result reset_energy_tank_cuda(const ParameterSetView& parameters,
                              StatusView parameter_status,
                              StateView& state,
                              const TensorView& reset_mask,
                              MutableStatusView status) {
  StateEntryView* state_variable = energy_state(state);
  if (state_variable == nullptr) {
    return Result{ResultCode::kInvalidView, "EnergyTank state requires state_variable"};
  }
  const DType dtype = state_variable->values.dtype;
  if (!is_float_dtype(dtype) || state_variable->values.rank != 1 ||
      state_variable->values.shape[0] <= 0) {
    return Result{ResultCode::kInvalidView, "EnergyTank state_variable must be [num_envs]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(state_variable->values.shape[0]);
  Result result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_energy_state(state, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_energy_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_energy<float>(
                   parameters, parameter_status, state, reset_mask, status, num_envs)
             : launch_reset_energy<double>(
                   parameters, parameter_status, state, reset_mask, status, num_envs);
}

Result step_energy_tank_cuda(const TensorView& power_signal,
                             const TensorView& additional_power,
                             const TensorView& dt,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             StateView& state,
                             const MutableTensorView& scale,
                             const MutableTensorView& filtered_power,
                             MutableStatusView status) {
  if (power_signal.data == nullptr || power_signal.device != Device::kCuda ||
      !is_float_dtype(power_signal.dtype) || power_signal.rank != 1 || power_signal.shape[0] <= 0 ||
      power_signal.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "power_signal must be CUDA [num_envs]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(power_signal.shape[0]);
  Result result = validate_cuda_scalar_batch(
      additional_power, "additional_power tensor data is null", power_signal.dtype, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_cuda_scalar_batch(dt, "dt tensor data is null", power_signal.dtype, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_output(scale, power_signal.dtype, num_envs, "scale data is null");
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_output(
      filtered_power, power_signal.dtype, num_envs, "filtered_power data is null");
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_energy_state(state, power_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_energy_parameters(parameters, power_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return power_signal.dtype == DType::kFloat32 ? launch_step_energy<float>(power_signal,
                                                                           additional_power,
                                                                           dt,
                                                                           parameters,
                                                                           parameter_status,
                                                                           state,
                                                                           scale,
                                                                           filtered_power,
                                                                           status)
                                               : launch_step_energy<double>(power_signal,
                                                                            additional_power,
                                                                            dt,
                                                                            parameters,
                                                                            parameter_status,
                                                                            state,
                                                                            scale,
                                                                            filtered_power,
                                                                            status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
