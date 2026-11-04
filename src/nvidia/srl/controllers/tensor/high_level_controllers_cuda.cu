/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include "tensor/high_level_controllers_cuda.hpp"

#include <cuda_runtime.h>

#include <limits>

#include "tensor/cuda_component_tuning_profiles.hpp"
#include "tensor/cuda_stream.hpp"

namespace nvidia::srl::controllers::detail::tensor {
namespace {

constexpr int kDim = 6;
constexpr std::uint64_t kInvalidParameter =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInput = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFinite = static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDt = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kSaturation = static_cast<std::uint64_t>(StatusFlag::kSaturation);
constexpr std::uint64_t kReset = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeMode = static_cast<std::uint64_t>(StatusFlag::kSafeMode);

struct ParameterDescriptor {
  TensorView values{};
  bool shared{false};
};

struct UnifiedParameters {
  ParameterDescriptor stiffness, damping, mass, acceleration_limit, twist_limit, pose_limit;
  ParameterDescriptor kp, kd, ki, initial_integral, integral_lower, integral_upper, deadzone;
};

struct UnifiedState {
  MutableTensorView previous_error{}, differentiator_initialized{}, integrator_previous_error{};
  MutableTensorView integrated_error{}, integrator_initialized{}, pose{}, twist{};
  MutableTensorView pose_previous{}, twist_previous{}, pose_initialized{}, twist_initialized{};
};

Result ok() { return Result{ResultCode::kOk, "ok"}; }
Result invalid(const char* message) { return Result{ResultCode::kInvalidView, message}; }
Result launch_result() {
  return cudaPeekAtLastError() == cudaSuccess
             ? ok()
             : Result{ResultCode::kInvalidArgument, "CUDA kernel launch failed"};
}

const ParameterView* required(const ParameterSetView& parameters, const char* name) {
  return find_parameter(parameters, name);
}

ParameterDescriptor descriptor(const ParameterView* parameter) {
  return parameter == nullptr ? ParameterDescriptor{}
                              : ParameterDescriptor{parameter->values, parameter->shared};
}

Result validate_vector(const ParameterView* parameter,
                       DType dtype,
                       std::size_t envs,
                       std::size_t dim) {
  if (!parameter || !parameter->values.data || parameter->values.device != Device::kCuda ||
      parameter->values.dtype != dtype)
    return invalid("invalid CUDA vector parameter");
  const TensorView& view = parameter->values;
  if (parameter->shared)
    return view.rank == 1 && view.shape[0] == static_cast<std::int64_t>(dim)
               ? ok()
               : invalid("invalid shared vector shape");
  return view.rank == 2 && view.shape[0] == static_cast<std::int64_t>(envs) &&
                 view.shape[1] == static_cast<std::int64_t>(dim)
             ? ok()
             : invalid("invalid batched vector shape");
}

Result validate_matrix(const ParameterView* parameter,
                       DType dtype,
                       std::size_t envs,
                       std::size_t dim) {
  if (!parameter || !parameter->values.data || parameter->values.device != Device::kCuda ||
      parameter->values.dtype != dtype)
    return invalid("invalid CUDA matrix parameter");
  const TensorView& view = parameter->values;
  if (parameter->shared)
    return view.rank == 2 && view.shape[0] == static_cast<std::int64_t>(dim) &&
                   view.shape[1] == static_cast<std::int64_t>(dim)
               ? ok()
               : invalid("invalid shared matrix shape");
  return view.rank == 3 && view.shape[0] == static_cast<std::int64_t>(envs) &&
                 view.shape[1] == static_cast<std::int64_t>(dim) &&
                 view.shape[2] == static_cast<std::int64_t>(dim)
             ? ok()
             : invalid("invalid batched matrix shape");
}

Result validate_signal(const TensorView& view, DType dtype, std::size_t envs, std::size_t dim) {
  return view.data && view.device == Device::kCuda && view.dtype == dtype && view.rank == 2 &&
                 view.shape[0] == static_cast<std::int64_t>(envs) &&
                 view.shape[1] == static_cast<std::int64_t>(dim)
             ? ok()
             : invalid("invalid CUDA signal");
}

template <typename Scalar>
__device__ Scalar vector_value(ParameterDescriptor parameter, std::size_t env, std::size_t dim) {
  const auto* data = static_cast<const Scalar*>(parameter.values.data);
  return parameter.shared
             ? data[dim * parameter.values.strides[0]]
             : data[env * parameter.values.strides[0] + dim * parameter.values.strides[1]];
}

template <typename Scalar>
__device__ Scalar motion_limit(ParameterDescriptor parameter,
                               std::size_t env,
                               std::size_t dim,
                               Scalar fallback = std::numeric_limits<Scalar>::infinity()) {
  return parameter.values.data == nullptr ? fallback : vector_value<Scalar>(parameter, env, dim);
}

template <typename Scalar>
__device__ Scalar
matrix_value(ParameterDescriptor parameter, std::size_t env, std::size_t row, std::size_t col) {
  const auto* data = static_cast<const Scalar*>(parameter.values.data);
  return parameter.shared
             ? data[row * parameter.values.strides[0] + col * parameter.values.strides[1]]
             : data[env * parameter.values.strides[0] + row * parameter.values.strides[1] +
                    col * parameter.values.strides[2]];
}

template <typename Scalar>
__device__ Scalar signal_value(TensorView view, std::size_t env, std::size_t dim) {
  return static_cast<const Scalar*>(view.data)[env * view.strides[0] + dim * view.strides[1]];
}

template <typename Scalar>
__device__ Scalar state_value(MutableTensorView view, std::size_t env, std::size_t dim) {
  return static_cast<const Scalar*>(view.data)[env * view.strides[0] + dim * view.strides[1]];
}

template <typename Scalar>
__device__ void write_value(MutableTensorView view,
                            std::size_t env,
                            std::size_t dim,
                            Scalar value) {
  static_cast<Scalar*>(view.data)[env * view.strides[0] + dim * view.strides[1]] = value;
}

template <typename Scalar>
__global__ void apply_static_kernel(ComponentMode mode,
                                    ParameterDescriptor stiffness,
                                    ParameterDescriptor damping,
                                    std::size_t envs,
                                    std::size_t dim,
                                    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  std::uint64_t flags = 0;
  for (std::size_t row = 0; row < dim; ++row) {
    if (mode == ComponentMode::kDiagonal) {
      if (!isfinite(vector_value<Scalar>(stiffness, env, row)) ||
          !isfinite(vector_value<Scalar>(damping, env, row)))
        flags |= kInvalidParameter;
    } else {
      for (std::size_t col = 0; col < dim; ++col)
        if (!isfinite(matrix_value<Scalar>(stiffness, env, row, col)) ||
            !isfinite(matrix_value<Scalar>(damping, env, row, col)))
          flags |= kInvalidParameter;
    }
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void static_step_kernel(ComponentMode mode,
                                   TensorView position,
                                   TensorView velocity,
                                   ParameterDescriptor stiffness,
                                   ParameterDescriptor damping,
                                   const std::uint64_t* parameter_status,
                                   MutableTensorView command,
                                   std::uint64_t* status,
                                   std::size_t envs,
                                   std::size_t dim) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  std::uint64_t flags = parameter_status[env];
  for (std::size_t row = 0; row < dim; ++row) {
    Scalar output = 0;
    for (std::size_t col = 0; col < dim; ++col) {
      const Scalar p = signal_value<Scalar>(position, env, col);
      const Scalar v = signal_value<Scalar>(velocity, env, col);
      if (!isfinite(p) || !isfinite(v)) flags |= kNonFinite;
      if (mode == ComponentMode::kDiagonal && row == col)
        output += vector_value<Scalar>(stiffness, env, row) * p +
                  vector_value<Scalar>(damping, env, row) * v;
      else if (mode == ComponentMode::kDense)
        output += matrix_value<Scalar>(stiffness, env, row, col) * p +
                  matrix_value<Scalar>(damping, env, row, col) * v;
    }
    if (!isfinite(output)) flags |= kNonFinite;
    write_value(command, env, row, flags == 0 ? output : Scalar{0});
  }
  if (flags) flags |= kSafeMode;
  status[env] = flags;
}

__global__ void reset_stateless_kernel(std::size_t envs, TensorView mask, std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const bool selected =
      !mask.data || static_cast<const std::uint8_t*>(mask.data)[env * mask.strides[0]];
  status[env] = selected ? kReset : 0;
}

template <typename Scalar>
__global__ void apply_unified_kernel(UnifiedParameters parameters,
                                     std::size_t envs,
                                     std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  std::uint64_t flags = 0;
  for (int row = 0; row < kDim; ++row) {
    const ParameterDescriptor vectors[] = {parameters.acceleration_limit,
                                           parameters.twist_limit,
                                           parameters.pose_limit,
                                           parameters.kp,
                                           parameters.kd,
                                           parameters.ki,
                                           parameters.initial_integral,
                                           parameters.integral_lower,
                                           parameters.integral_upper,
                                           parameters.deadzone};
    for (ParameterDescriptor item : vectors)
      if (item.values.data != nullptr && !isfinite(vector_value<Scalar>(item, env, row)))
        flags |= kInvalidParameter;
    const Scalar low = motion_limit<Scalar>(
        parameters.integral_lower, env, row, -std::numeric_limits<Scalar>::infinity());
    const Scalar high = motion_limit<Scalar>(parameters.integral_upper, env, row);
    const Scalar initial = vector_value<Scalar>(parameters.initial_integral, env, row);
    if (low > initial || initial > high ||
        vector_value<Scalar>(parameters.deadzone, env, row) < 0 ||
        motion_limit<Scalar>(parameters.acceleration_limit, env, row) < 0 ||
        motion_limit<Scalar>(parameters.twist_limit, env, row) < 0 ||
        motion_limit<Scalar>(parameters.pose_limit, env, row) < 0)
      flags |= kInvalidParameter;
    for (int col = 0; col < kDim; ++col)
      if (!isfinite(matrix_value<Scalar>(parameters.stiffness, env, row, col)) ||
          !isfinite(matrix_value<Scalar>(parameters.damping, env, row, col)) ||
          !isfinite(matrix_value<Scalar>(parameters.mass, env, row, col)))
        flags |= kInvalidParameter;
  }
  status[env] = flags;
}

template <typename Scalar>
__device__ bool solve6(Scalar matrix[6][6], Scalar rhs[6], Scalar output[6]) {
  for (int pivot = 0; pivot < kDim; ++pivot) {
    int best = pivot;
    for (int row = pivot + 1; row < kDim; ++row)
      if (fabs(matrix[row][pivot]) > fabs(matrix[best][pivot])) best = row;
    if (!isfinite(matrix[best][pivot]) || fabs(matrix[best][pivot]) <= Scalar{1e-12}) return false;
    if (best != pivot) {
      for (int col = pivot; col < kDim; ++col) {
        const Scalar temporary = matrix[pivot][col];
        matrix[pivot][col] = matrix[best][col];
        matrix[best][col] = temporary;
      }
      const Scalar temporary = rhs[pivot];
      rhs[pivot] = rhs[best];
      rhs[best] = temporary;
    }
    for (int row = pivot + 1; row < kDim; ++row) {
      const Scalar factor = matrix[row][pivot] / matrix[pivot][pivot];
      for (int col = pivot; col < kDim; ++col) matrix[row][col] -= factor * matrix[pivot][col];
      rhs[row] -= factor * rhs[pivot];
    }
  }
  for (int row = kDim - 1; row >= 0; --row) {
    Scalar value = rhs[row];
    for (int col = row + 1; col < kDim; ++col) value -= matrix[row][col] * output[col];
    output[row] = value / matrix[row][row];
    if (!isfinite(output[row])) return false;
  }
  return true;
}

template <typename Scalar>
__global__ void unified_step_kernel(TensorView desired,
                                    TensorView external,
                                    TensorView feedforward,
                                    TensorView dt,
                                    UnifiedParameters parameters,
                                    const std::uint64_t* parameter_status,
                                    UnifiedState state,
                                    MutableTensorView acceleration,
                                    std::uint64_t* status,
                                    std::size_t envs) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  std::uint64_t flags = parameter_status[env];
  const Scalar step_dt = dt.rank == 0 ? static_cast<const Scalar*>(dt.data)[0]
                                      : static_cast<const Scalar*>(dt.data)[env * dt.strides[0]];
  if (!isfinite(step_dt) || step_dt <= 0) flags |= kInvalidDt;
  const Scalar safe_dt = flags == 0 ? step_dt : Scalar{1};
  const bool diff_initialized =
      static_cast<const std::uint8_t*>(state.differentiator_initialized.data)[env];
  const bool int_initialized =
      static_cast<const std::uint8_t*>(state.integrator_initialized.data)[env];
  Scalar error[6]{}, integral[6]{}, mass[6][6]{}, rhs[6]{}, candidate_acceleration[6]{};
  Scalar candidate_twist[6]{}, candidate_pose[6]{};
  for (int row = 0; row < kDim; ++row) {
    const Scalar desired_value = signal_value<Scalar>(desired, env, row);
    const Scalar external_value = signal_value<Scalar>(external, env, row);
    const Scalar ff = signal_value<Scalar>(feedforward, env, row);
    if (!isfinite(desired_value) || !isfinite(external_value) || !isfinite(ff)) flags |= kNonFinite;
    const Scalar raw = external_value - desired_value;
    const Scalar deadzone = vector_value<Scalar>(parameters.deadzone, env, row);
    error[row] = raw > deadzone ? raw - deadzone : raw < -deadzone ? raw + deadzone : Scalar{0};
    const Scalar derivative =
        diff_initialized
            ? (error[row] - state_value<Scalar>(state.previous_error, env, row)) / safe_dt
            : Scalar{0};
    const Scalar old_integral = int_initialized
                                    ? state_value<Scalar>(state.integrated_error, env, row)
                                    : vector_value<Scalar>(parameters.initial_integral, env, row);
    const Scalar old_error = int_initialized
                                 ? state_value<Scalar>(state.integrator_previous_error, env, row)
                                 : error[row];
    const Scalar low = motion_limit<Scalar>(
        parameters.integral_lower, env, row, -std::numeric_limits<Scalar>::infinity());
    const Scalar high = motion_limit<Scalar>(parameters.integral_upper, env, row);
    integral[row] =
        fmin(high, fmax(low, old_integral + Scalar{0.5} * safe_dt * (old_error + error[row])));
    rhs[row] = vector_value<Scalar>(parameters.kp, env, row) * error[row] +
               vector_value<Scalar>(parameters.kd, env, row) * derivative +
               vector_value<Scalar>(parameters.ki, env, row) * integral[row] + ff;
    for (int col = 0; col < kDim; ++col) {
      mass[row][col] = matrix_value<Scalar>(parameters.mass, env, row, col);
      rhs[row] -= matrix_value<Scalar>(parameters.stiffness, env, row, col) *
                      state_value<Scalar>(state.pose, env, col) +
                  matrix_value<Scalar>(parameters.damping, env, row, col) *
                      state_value<Scalar>(state.twist, env, col);
    }
  }
  if (flags == 0 && !solve6(mass, rhs, candidate_acceleration)) flags |= kInvalidInput;
  if (flags == 0) {
    for (int dim = 0; dim < kDim; ++dim) {
      const Scalar al = fabs(motion_limit<Scalar>(parameters.acceleration_limit, env, dim));
      const Scalar vl = fabs(motion_limit<Scalar>(parameters.twist_limit, env, dim));
      const Scalar pl = fabs(motion_limit<Scalar>(parameters.pose_limit, env, dim));
      const Scalar raw_acceleration = candidate_acceleration[dim];
      candidate_acceleration[dim] = fmin(al, fmax(-al, raw_acceleration));
      const Scalar old_twist = state_value<Scalar>(state.twist, env, dim);
      candidate_twist[dim] = fmin(vl, fmax(-vl, old_twist + candidate_acceleration[dim] * safe_dt));
      const Scalar old_pose = state_value<Scalar>(state.pose, env, dim);
      candidate_pose[dim] = fmin(
          pl, fmax(-pl, old_pose + Scalar{0.5} * safe_dt * (old_twist + candidate_twist[dim])));
      if (candidate_acceleration[dim] != raw_acceleration || fabs(candidate_twist[dim]) == vl ||
          fabs(candidate_pose[dim]) == pl)
        flags |= kSaturation;
    }
  }
  if ((flags & ~kSaturation) != 0) {
    flags |= kSafeMode;
    for (int dim = 0; dim < kDim; ++dim) write_value(acceleration, env, dim, Scalar{0});
  } else {
    for (int dim = 0; dim < kDim; ++dim) {
      write_value(acceleration, env, dim, candidate_acceleration[dim]);
      write_value(state.previous_error, env, dim, error[dim]);
      write_value(state.integrator_previous_error, env, dim, error[dim]);
      write_value(state.integrated_error, env, dim, integral[dim]);
      write_value(state.pose_previous, env, dim, state_value<Scalar>(state.pose, env, dim));
      write_value(state.twist_previous, env, dim, state_value<Scalar>(state.twist, env, dim));
      write_value(state.pose, env, dim, candidate_pose[dim]);
      write_value(state.twist, env, dim, candidate_twist[dim]);
    }
    static_cast<std::uint8_t*>(state.differentiator_initialized.data)[env] = 1;
    static_cast<std::uint8_t*>(state.integrator_initialized.data)[env] = 1;
    static_cast<std::uint8_t*>(state.pose_initialized.data)[env] = 1;
    static_cast<std::uint8_t*>(state.twist_initialized.data)[env] = 1;
  }
  status[env] = flags;
}

__global__ void reset_unified_kernel(
    std::size_t envs, TensorView mask, UnifiedState state, std::uint64_t* status, DType dtype) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const bool selected =
      !mask.data || static_cast<const std::uint8_t*>(mask.data)[env * mask.strides[0]];
  status[env] = selected ? kReset : 0;
  if (!selected) return;
  MutableTensorView floats[] = {state.previous_error,
                                state.integrator_previous_error,
                                state.integrated_error,
                                state.pose,
                                state.twist,
                                state.pose_previous,
                                state.twist_previous};
  for (MutableTensorView view : floats)
    for (int dim = 0; dim < kDim; ++dim)
      if (dtype == DType::kFloat32)
        write_value(view, env, dim, 0.0F);
      else
        write_value(view, env, dim, 0.0);
  MutableTensorView flags[] = {state.differentiator_initialized,
                               state.integrator_initialized,
                               state.pose_initialized,
                               state.twist_initialized};
  for (MutableTensorView view : flags) static_cast<std::uint8_t*>(view.data)[env] = 0;
}

UnifiedParameters parse_unified(const ParameterSetView& parameters) {
  return {descriptor(required(parameters, "stiffness_matrix")),
          descriptor(required(parameters, "damping_matrix")),
          descriptor(required(parameters, "virtual_mass_matrix")),
          descriptor(required(parameters, "acceleration_limit")),
          descriptor(required(parameters, "twist_limit")),
          descriptor(required(parameters, "pose_limit")),
          descriptor(required(parameters, "proportional_gain")),
          descriptor(required(parameters, "differentiator_gain")),
          descriptor(required(parameters, "integral_gain")),
          descriptor(required(parameters, "initial_integral")),
          descriptor(required(parameters, "integral_lower_limit")),
          descriptor(required(parameters, "integral_upper_limit")),
          descriptor(required(parameters, "error_deadzone"))};
}

UnifiedState parse_state(StateView& state) {
  return {find_state(state, "differentiator_previous_error")->values,
          find_state(state, "differentiator_initialized")->values,
          find_state(state, "integrator_previous_error")->values,
          find_state(state, "integrated_error")->values,
          find_state(state, "integrator_initialized")->values,
          find_state(state, "admittance_pose")->values,
          find_state(state, "admittance_twist")->values,
          find_state(state, "pose_previous_signal")->values,
          find_state(state, "twist_previous_signal")->values,
          find_state(state, "pose_integrator_initialized")->values,
          find_state(state, "twist_integrator_initialized")->values};
}

template <CudaTensorLaunchFamily Family, typename Scalar>
int threads() {
  return cuda_tensor_tuned_threads<Family, Scalar>();
}

}  // namespace

Result apply_static_impedance_parameters_cuda(ComponentMode mode,
                                              const ParameterSetView& parameters,
                                              std::size_t envs,
                                              std::size_t dim,
                                              MutableStatusView status) {
  const ParameterView* stiffness = required(parameters, "stiffness");
  const ParameterView* damping = required(parameters, "damping");
  if (!stiffness || !damping || !status.flags || status.device != Device::kCuda)
    return invalid("invalid static impedance storage");
  Result result = mode == ComponentMode::kDiagonal
                      ? validate_vector(stiffness, stiffness->values.dtype, envs, dim)
                      : validate_matrix(stiffness, stiffness->values.dtype, envs, dim);
  if (!result.ok()) return result;
  result = mode == ComponentMode::kDiagonal
               ? validate_vector(damping, stiffness->values.dtype, envs, dim)
               : validate_matrix(damping, stiffness->values.dtype, envs, dim);
  if (!result.ok()) return result;
  if (stiffness->values.dtype == DType::kFloat32) {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelApplyStaticImpedance, float>();
    apply_static_kernel<float><<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(
        mode, descriptor(stiffness), descriptor(damping), envs, dim, status.flags);
  } else {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelApplyStaticImpedance, double>();
    apply_static_kernel<double><<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(
        mode, descriptor(stiffness), descriptor(damping), envs, dim, status.flags);
  }
  return launch_result();
}

Result step_static_impedance_cuda(ComponentMode mode,
                                  const TensorView& position,
                                  const TensorView& velocity,
                                  const ParameterSetView& parameters,
                                  StatusView parameter_status,
                                  const MutableTensorView& command,
                                  MutableStatusView status) {
  const std::size_t envs = position.shape[0], dim = position.shape[1];
  Result result = validate_signal(position, position.dtype, envs, dim);
  if (!result.ok()) return result;
  result = validate_signal(velocity, position.dtype, envs, dim);
  if (!result.ok() || !command.data || !status.flags || !parameter_status.flags)
    return invalid("invalid static impedance step storage");
  const auto stiffness = descriptor(required(parameters, "stiffness"));
  const auto damping = descriptor(required(parameters, "damping"));
  if (position.dtype == DType::kFloat32) {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelStepStaticImpedance, float>();
    static_step_kernel<float>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(mode,
                                                             position,
                                                             velocity,
                                                             stiffness,
                                                             damping,
                                                             parameter_status.flags,
                                                             command,
                                                             status.flags,
                                                             envs,
                                                             dim);
  } else {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelStepStaticImpedance, double>();
    static_step_kernel<double>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(mode,
                                                             position,
                                                             velocity,
                                                             stiffness,
                                                             damping,
                                                             parameter_status.flags,
                                                             command,
                                                             status.flags,
                                                             envs,
                                                             dim);
  }
  return launch_result();
}

Result reset_high_level_stateless_cuda(std::size_t envs,
                                       const TensorView& mask,
                                       MutableStatusView status) {
  const int t = 128;
  reset_stateless_kernel<<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(
      envs, mask, status.flags);
  return launch_result();
}

Result apply_unified_force_admittance_parameters_cuda(const ParameterSetView& parameters,
                                                      std::size_t envs,
                                                      MutableStatusView status) {
  const ParameterView* first = required(parameters, "stiffness_matrix");
  if (!first || !status.flags || status.device != Device::kCuda)
    return invalid("invalid unified parameter storage");
  const char* matrices[] = {"stiffness_matrix", "damping_matrix", "virtual_mass_matrix"};
  for (const char* name : matrices) {
    Result r = validate_matrix(required(parameters, name), first->values.dtype, envs, kDim);
    if (!r.ok()) return r;
  }
  const char* limits[] = {"acceleration_limit",
                          "twist_limit",
                          "pose_limit",
                          "integral_lower_limit",
                          "integral_upper_limit"};
  for (const char* name : limits) {
    const ParameterView* parameter = find_parameter(parameters, name);
    if (parameter == nullptr) continue;
    Result r = validate_vector(parameter, first->values.dtype, envs, kDim);
    if (!r.ok()) return r;
  }
  const char* vectors[] = {"proportional_gain",
                           "differentiator_gain",
                           "integral_gain",
                           "initial_integral",
                           "error_deadzone"};
  for (const char* name : vectors) {
    Result r = validate_vector(required(parameters, name), first->values.dtype, envs, kDim);
    if (!r.ok()) return r;
  }
  const UnifiedParameters parsed = parse_unified(parameters);
  if (first->values.dtype == DType::kFloat32) {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelApplyUnifiedForceAdmittance, float>();
    apply_unified_kernel<float>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(parsed, envs, status.flags);
  } else {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelApplyUnifiedForceAdmittance, double>();
    apply_unified_kernel<double>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(parsed, envs, status.flags);
  }
  return launch_result();
}

Result reset_unified_force_admittance_cuda(std::size_t envs,
                                           StateView& state,
                                           const TensorView& mask,
                                           MutableStatusView status) {
  UnifiedState parsed = parse_state(state);
  const int t = 128;
  reset_unified_kernel<<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(
      envs, mask, parsed, status.flags, parsed.pose.dtype);
  return launch_result();
}

Result step_unified_force_admittance_cuda(const TensorView& desired,
                                          const TensorView& external,
                                          const TensorView& feedforward,
                                          const TensorView& dt,
                                          const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          StateView& state,
                                          const MutableTensorView& acceleration,
                                          MutableStatusView status) {
  const std::size_t envs = desired.shape[0];
  Result result = validate_signal(desired, desired.dtype, envs, kDim);
  if (!result.ok()) return result;
  result = validate_signal(external, desired.dtype, envs, kDim);
  if (!result.ok()) return result;
  result = validate_signal(feedforward, desired.dtype, envs, kDim);
  if (!result.ok()) return result;
  if (!dt.data || dt.device != Device::kCuda || dt.dtype != desired.dtype || !acceleration.data ||
      !status.flags || !parameter_status.flags)
    return invalid("invalid unified force/admittance step storage");
  const UnifiedParameters parsed_parameters = parse_unified(parameters);
  UnifiedState parsed_state = parse_state(state);
  if (desired.dtype == DType::kFloat32) {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelStepUnifiedForceAdmittance, float>();
    unified_step_kernel<float>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(desired,
                                                             external,
                                                             feedforward,
                                                             dt,
                                                             parsed_parameters,
                                                             parameter_status.flags,
                                                             parsed_state,
                                                             acceleration,
                                                             status.flags,
                                                             envs);
  } else {
    const int t = threads<CudaTensorLaunchFamily::kHighLevelStepUnifiedForceAdmittance, double>();
    unified_step_kernel<double>
        <<<(envs + t - 1) / t, t, 0, active_cuda_stream()>>>(desired,
                                                             external,
                                                             feedforward,
                                                             dt,
                                                             parsed_parameters,
                                                             parameter_status.flags,
                                                             parsed_state,
                                                             acceleration,
                                                             status.flags,
                                                             envs);
  }
  return launch_result();
}

}  // namespace nvidia::srl::controllers::detail::tensor
