/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include "tensor/high_level_controllers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/high_level_controllers_cuda.hpp"
#endif

namespace nvidia::srl::controllers::detail::tensor {
namespace {

constexpr std::size_t kDim = 6;
constexpr const char* kStiffness = "stiffness";
constexpr const char* kDamping = "damping";
constexpr const char* kStiffnessMatrix = "stiffness_matrix";
constexpr const char* kDampingMatrix = "damping_matrix";
constexpr const char* kMassMatrix = "virtual_mass_matrix";
constexpr const char* kAccelerationLimit = "acceleration_limit";
constexpr const char* kTwistLimit = "twist_limit";
constexpr const char* kPoseLimit = "pose_limit";
constexpr const char* kKp = "proportional_gain";
constexpr const char* kKd = "differentiator_gain";
constexpr const char* kKi = "integral_gain";
constexpr const char* kInitialIntegral = "initial_integral";
constexpr const char* kIntegralLower = "integral_lower_limit";
constexpr const char* kIntegralUpper = "integral_upper_limit";
constexpr const char* kDeadzone = "error_deadzone";

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
bool uses_cuda(const ParameterSetView& parameters) noexcept {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.device == Device::kCuda) return true;
  }
  return false;
}
#endif

template <typename Scalar>
Scalar vector_parameter(const ParameterView* parameter,
                        const std::size_t env,
                        const std::size_t dim) noexcept {
  const TensorView& view = parameter->values;
  const auto* data = static_cast<const Scalar*>(view.data);
  if (parameter->shared) return data[dim * static_cast<std::size_t>(view.strides[0])];
  return data[env * static_cast<std::size_t>(view.strides[0]) +
              dim * static_cast<std::size_t>(view.strides[1])];
}

template <typename Scalar>
Scalar motion_limit(const ParameterView* parameter,
                    const std::size_t env,
                    const std::size_t dim,
                    const Scalar fallback = std::numeric_limits<Scalar>::infinity()) noexcept {
  return parameter == nullptr ? fallback : vector_parameter<Scalar>(parameter, env, dim);
}

template <typename Scalar>
Scalar matrix_parameter(const ParameterView* parameter,
                        const std::size_t env,
                        const std::size_t row,
                        const std::size_t col) noexcept {
  const TensorView& view = parameter->values;
  const auto* data = static_cast<const Scalar*>(view.data);
  if (parameter->shared) {
    return data[row * static_cast<std::size_t>(view.strides[0]) +
                col * static_cast<std::size_t>(view.strides[1])];
  }
  return data[env * static_cast<std::size_t>(view.strides[0]) +
              row * static_cast<std::size_t>(view.strides[1]) +
              col * static_cast<std::size_t>(view.strides[2])];
}

template <typename Scalar>
bool finite_vector_parameter(const ParameterView* parameter,
                             const std::size_t env,
                             const std::size_t dim) noexcept {
  for (std::size_t i = 0; i < dim; ++i) {
    if (!std::isfinite(vector_parameter<Scalar>(parameter, env, i))) return false;
  }
  return true;
}

template <typename Scalar>
bool finite_matrix_parameter(const ParameterView* parameter,
                             const std::size_t env,
                             const std::size_t dim) noexcept {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!std::isfinite(matrix_parameter<Scalar>(parameter, env, row, col))) return false;
    }
  }
  return true;
}

Result validate_vector(const ParameterView* parameter,
                       const DType dtype,
                       const std::size_t num_envs,
                       const std::size_t dim) noexcept {
  return internal::validate_vector_parameter_metadata(
      parameter, "required vector parameter is missing", dtype, num_envs, dim, true);
}

Result validate_matrix(const ParameterView* parameter,
                       const DType dtype,
                       const std::size_t num_envs,
                       const std::size_t dim) noexcept {
  return internal::validate_matrix_parameter_metadata(
      parameter, "required matrix parameter is missing", dtype, num_envs, dim, dim, true);
}

Result validate_signal(const TensorView& signal,
                       const DType dtype,
                       const std::size_t num_envs,
                       const std::size_t dim) noexcept {
  std::size_t found_envs = 0;
  std::size_t found_dim = 0;
  Result result =
      internal::validate_batched_signal(signal, "signal is missing", &found_envs, &found_dim);
  if (!result.ok()) return result;
  return signal.dtype == dtype && found_envs == num_envs && found_dim == dim
             ? internal::ok()
             : internal::invalid_view("signal shape or dtype does not match controller");
}

template <typename Scalar>
Result apply_static_cpu(const ComponentMode mode,
                        const ParameterSetView& parameters,
                        const std::size_t num_envs,
                        const std::size_t dim,
                        const char* first_name,
                        const char* second_name,
                        const MutableStatusView status) noexcept {
  const ParameterView* first = find_parameter(parameters, first_name);
  const ParameterView* second = find_parameter(parameters, second_name);
  Result result = mode == ComponentMode::kDiagonal
                      ? validate_vector(first, dtype_for<Scalar>(), num_envs, dim)
                      : validate_matrix(first, dtype_for<Scalar>(), num_envs, dim);
  if (!result.ok()) return result;
  result = mode == ComponentMode::kDiagonal
               ? validate_vector(second, dtype_for<Scalar>(), num_envs, dim)
               : validate_matrix(second, dtype_for<Scalar>(), num_envs, dim);
  if (!result.ok()) return result;
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  internal::clear_status(status, num_envs);
  for (std::size_t env = 0; env < num_envs; ++env) {
    const bool finite = mode == ComponentMode::kDiagonal
                            ? finite_vector_parameter<Scalar>(first, env, dim) &&
                                  finite_vector_parameter<Scalar>(second, env, dim)
                            : finite_matrix_parameter<Scalar>(first, env, dim) &&
                                  finite_matrix_parameter<Scalar>(second, env, dim);
    if (!finite) status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_static_cpu(const ComponentMode mode,
                       const TensorView& position,
                       const TensorView& velocity,
                       const ParameterSetView& parameters,
                       const StatusView parameter_status,
                       const MutableTensorView& command,
                       const MutableStatusView status) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(position.shape[0]);
  const std::size_t dim = static_cast<std::size_t>(position.shape[1]);
  Result result = validate_signal(position, dtype_for<Scalar>(), num_envs, dim);
  if (!result.ok()) return result;
  result = validate_signal(velocity, dtype_for<Scalar>(), num_envs, dim);
  if (!result.ok()) return result;
  result = internal::validate_matching_output(command, dtype_for<Scalar>(), num_envs, dim);
  if (!result.ok()) return result;
  result = internal::validate_status_view(parameter_status, num_envs, false);
  if (!result.ok()) return result;
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  const ParameterView* stiffness = find_parameter(parameters, kStiffness);
  const ParameterView* damping = find_parameter(parameters, kDamping);
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags[env];
    for (std::size_t row = 0; row < dim; ++row) {
      Scalar value = 0;
      for (std::size_t col = 0; col < dim; ++col) {
        const Scalar p = internal::read_matrix_value<Scalar>(position, env, col);
        const Scalar v = internal::read_matrix_value<Scalar>(velocity, env, col);
        if (!std::isfinite(p) || !std::isfinite(v))
          flags |= status_bit(StatusFlag::kNonFiniteInput);
        if (mode == ComponentMode::kDiagonal && row == col) {
          value += vector_parameter<Scalar>(stiffness, env, row) * p +
                   vector_parameter<Scalar>(damping, env, row) * v;
        } else if (mode == ComponentMode::kDense) {
          value += matrix_parameter<Scalar>(stiffness, env, row, col) * p +
                   matrix_parameter<Scalar>(damping, env, row, col) * v;
        }
      }
      internal::write_matrix_value(
          command, env, row, flags == 0U && std::isfinite(value) ? value : Scalar{0});
      if (!std::isfinite(value)) flags |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if (flags != 0U) flags |= status_bit(StatusFlag::kSafeMode);
    status.flags[env] = flags;
  }
  return internal::ok();
}

template <typename Scalar>
bool solve6(Scalar matrix[6][6], Scalar rhs[6], Scalar output[6]) noexcept {
  for (std::size_t pivot = 0; pivot < kDim; ++pivot) {
    std::size_t best = pivot;
    for (std::size_t row = pivot + 1; row < kDim; ++row) {
      if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) best = row;
    }
    if (!std::isfinite(matrix[best][pivot]) || std::abs(matrix[best][pivot]) <= Scalar{1e-12})
      return false;
    if (best != pivot) {
      for (std::size_t col = pivot; col < kDim; ++col)
        std::swap(matrix[pivot][col], matrix[best][col]);
      std::swap(rhs[pivot], rhs[best]);
    }
    for (std::size_t row = pivot + 1; row < kDim; ++row) {
      const Scalar factor = matrix[row][pivot] / matrix[pivot][pivot];
      for (std::size_t col = pivot; col < kDim; ++col)
        matrix[row][col] -= factor * matrix[pivot][col];
      rhs[row] -= factor * rhs[pivot];
    }
  }
  for (std::size_t reverse = 0; reverse < kDim; ++reverse) {
    const std::size_t row = kDim - 1 - reverse;
    Scalar value = rhs[row];
    for (std::size_t col = row + 1; col < kDim; ++col) value -= matrix[row][col] * output[col];
    output[row] = value / matrix[row][row];
    if (!std::isfinite(output[row])) return false;
  }
  return true;
}

template <typename Scalar>
Result apply_unified_cpu(const ParameterSetView& parameters,
                         const std::size_t num_envs,
                         const MutableStatusView status) noexcept {
  const char* matrices[] = {kStiffnessMatrix, kDampingMatrix, kMassMatrix};
  for (const char* name : matrices) {
    Result result =
        validate_matrix(find_parameter(parameters, name), dtype_for<Scalar>(), num_envs, kDim);
    if (!result.ok()) return result;
  }
  const char* limits[] = {
      kAccelerationLimit, kTwistLimit, kPoseLimit, kIntegralLower, kIntegralUpper};
  for (const char* name : limits) {
    const ParameterView* parameter = find_parameter(parameters, name);
    if (parameter == nullptr) continue;
    Result result = validate_vector(parameter, dtype_for<Scalar>(), num_envs, kDim);
    if (!result.ok()) return result;
  }
  const char* vectors[] = {kKp, kKd, kKi, kInitialIntegral, kDeadzone};
  for (const char* name : vectors) {
    Result result =
        validate_vector(find_parameter(parameters, name), dtype_for<Scalar>(), num_envs, kDim);
    if (!result.ok()) return result;
  }
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  internal::clear_status(status, num_envs);
  for (std::size_t env = 0; env < num_envs; ++env) {
    bool valid = true;
    for (const char* name : matrices)
      valid = valid && finite_matrix_parameter<Scalar>(find_parameter(parameters, name), env, kDim);
    for (const char* name : vectors)
      valid = valid && finite_vector_parameter<Scalar>(find_parameter(parameters, name), env, kDim);
    for (const char* name : limits) {
      const ParameterView* parameter = find_parameter(parameters, name);
      if (parameter != nullptr)
        valid = valid && finite_vector_parameter<Scalar>(parameter, env, kDim);
    }
    for (std::size_t dim = 0; dim < kDim; ++dim) {
      const Scalar low = motion_limit<Scalar>(find_parameter(parameters, kIntegralLower),
                                              env,
                                              dim,
                                              -std::numeric_limits<Scalar>::infinity());
      const Scalar high =
          motion_limit<Scalar>(find_parameter(parameters, kIntegralUpper), env, dim);
      const Scalar initial =
          vector_parameter<Scalar>(find_parameter(parameters, kInitialIntegral), env, dim);
      const Scalar deadzone =
          vector_parameter<Scalar>(find_parameter(parameters, kDeadzone), env, dim);
      const Scalar acceleration_limit =
          motion_limit<Scalar>(find_parameter(parameters, kAccelerationLimit), env, dim);
      const Scalar twist_limit =
          motion_limit<Scalar>(find_parameter(parameters, kTwistLimit), env, dim);
      const Scalar pose_limit =
          motion_limit<Scalar>(find_parameter(parameters, kPoseLimit), env, dim);
      valid = valid && low <= initial && initial <= high && deadzone >= Scalar{0} &&
              acceleration_limit >= Scalar{0} && twist_limit >= Scalar{0} &&
              pose_limit >= Scalar{0};
    }
    if (!valid) status.flags[env] = status_bit(StatusFlag::kInvalidParameter);
  }
  return internal::ok();
}

bool validate_state_entry(const StateEntryView* entry,
                          const DType dtype,
                          const std::size_t num_envs,
                          const bool flag) noexcept {
  if (entry == nullptr) return false;
  const MutableTensorView& view = entry->values;
  if (flag)
    return view.data != nullptr && view.device == Device::kCpu && view.dtype == DType::kBool &&
           view.rank == 1 && view.shape[0] == static_cast<std::int64_t>(num_envs);
  return view.data != nullptr && view.device == Device::kCpu && view.dtype == dtype &&
         internal::is_row_major_matrix(view) &&
         view.shape[0] == static_cast<std::int64_t>(num_envs) && view.shape[1] == 6;
}

template <typename Scalar>
Result step_unified_cpu(const TensorView& desired,
                        const TensorView& external,
                        const TensorView& feedforward,
                        const TensorView& dt,
                        const ParameterSetView& parameters,
                        const StatusView parameter_status,
                        StateView& state,
                        const MutableTensorView& acceleration,
                        const MutableStatusView status) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(desired.shape[0]);
  for (const TensorView& signal : {desired, external, feedforward}) {
    Result result = validate_signal(signal, dtype_for<Scalar>(), num_envs, kDim);
    if (!result.ok()) return result;
  }
  Result result =
      internal::validate_scalar_batch(dt, "dt is missing", dtype_for<Scalar>(), num_envs);
  if (!result.ok()) return result;
  result = internal::validate_matching_output(acceleration, dtype_for<Scalar>(), num_envs, kDim);
  if (!result.ok()) return result;
  result = internal::validate_status_view(parameter_status, num_envs, false);
  if (!result.ok()) return result;
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  StateEntryView* previous_error = find_state(state, "differentiator_previous_error");
  StateEntryView* differentiator_initialized = find_state(state, "differentiator_initialized");
  StateEntryView* integrator_previous_error = find_state(state, "integrator_previous_error");
  StateEntryView* integrated_error = find_state(state, "integrated_error");
  StateEntryView* integrator_initialized = find_state(state, "integrator_initialized");
  StateEntryView* pose = find_state(state, "admittance_pose");
  StateEntryView* twist = find_state(state, "admittance_twist");
  StateEntryView* pose_previous = find_state(state, "pose_previous_signal");
  StateEntryView* twist_previous = find_state(state, "twist_previous_signal");
  StateEntryView* pose_initialized = find_state(state, "pose_integrator_initialized");
  StateEntryView* twist_initialized = find_state(state, "twist_integrator_initialized");
  StateEntryView* float_states[] = {previous_error,
                                    integrator_previous_error,
                                    integrated_error,
                                    pose,
                                    twist,
                                    pose_previous,
                                    twist_previous};
  for (StateEntryView* entry : float_states)
    if (!validate_state_entry(entry, dtype_for<Scalar>(), num_envs, false))
      return internal::invalid_view("force/admittance state must be contiguous [batch_size,6]");
  StateEntryView* bool_states[] = {
      differentiator_initialized, integrator_initialized, pose_initialized, twist_initialized};
  for (StateEntryView* entry : bool_states)
    if (!validate_state_entry(entry, dtype_for<Scalar>(), num_envs, true))
      return internal::invalid_view("force/admittance flags must be bool [batch_size]");

  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags[env];
    const Scalar step_dt = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(step_dt) || step_dt <= Scalar{0})
      flags |= status_bit(StatusFlag::kInvalidDt);
    const Scalar integration_dt = flags == 0U ? step_dt : Scalar{1};
    Scalar error[6]{}, integral[6]{}, drive[6]{}, candidate_twist[6]{}, candidate_pose[6]{};
    Scalar mass[6][6]{}, rhs[6]{}, candidate_acceleration[6]{};
    const auto* diff_init =
        static_cast<const std::uint8_t*>(differentiator_initialized->values.data);
    const auto* int_init = static_cast<const std::uint8_t*>(integrator_initialized->values.data);
    for (std::size_t dim = 0; dim < kDim; ++dim) {
      const Scalar desired_value = internal::read_matrix_value<Scalar>(desired, env, dim);
      const Scalar external_value = internal::read_matrix_value<Scalar>(external, env, dim);
      const Scalar ff = internal::read_matrix_value<Scalar>(feedforward, env, dim);
      if (!std::isfinite(desired_value) || !std::isfinite(external_value) || !std::isfinite(ff))
        flags |= status_bit(StatusFlag::kNonFiniteInput);
      const Scalar raw_error = external_value - desired_value;
      const Scalar deadzone =
          vector_parameter<Scalar>(find_parameter(parameters, kDeadzone), env, dim);
      error[dim] = raw_error > deadzone    ? raw_error - deadzone
                   : raw_error < -deadzone ? raw_error + deadzone
                                           : Scalar{0};
      const Scalar previous = internal::read_matrix_value<Scalar>(previous_error->values, env, dim);
      const Scalar derivative =
          diff_init[env] != 0U ? (error[dim] - previous) / integration_dt : Scalar{0};
      const Scalar old_integral =
          int_init[env] != 0U
              ? internal::read_matrix_value<Scalar>(integrated_error->values, env, dim)
              : vector_parameter<Scalar>(find_parameter(parameters, kInitialIntegral), env, dim);
      const Scalar previous_integrand =
          int_init[env] != 0U
              ? internal::read_matrix_value<Scalar>(integrator_previous_error->values, env, dim)
              : error[dim];
      const Scalar low = motion_limit<Scalar>(find_parameter(parameters, kIntegralLower),
                                              env,
                                              dim,
                                              -std::numeric_limits<Scalar>::infinity());
      const Scalar high =
          motion_limit<Scalar>(find_parameter(parameters, kIntegralUpper), env, dim);
      integral[dim] = std::clamp(
          old_integral + Scalar{0.5} * integration_dt * (previous_integrand + error[dim]),
          low,
          high);
      drive[dim] =
          vector_parameter<Scalar>(find_parameter(parameters, kKp), env, dim) * error[dim] +
          vector_parameter<Scalar>(find_parameter(parameters, kKd), env, dim) * derivative +
          vector_parameter<Scalar>(find_parameter(parameters, kKi), env, dim) * integral[dim] + ff;
      rhs[dim] = drive[dim];
      for (std::size_t col = 0; col < kDim; ++col) {
        mass[dim][col] =
            matrix_parameter<Scalar>(find_parameter(parameters, kMassMatrix), env, dim, col);
        rhs[dim] -=
            matrix_parameter<Scalar>(find_parameter(parameters, kStiffnessMatrix), env, dim, col) *
                internal::read_matrix_value<Scalar>(pose->values, env, col) +
            matrix_parameter<Scalar>(find_parameter(parameters, kDampingMatrix), env, dim, col) *
                internal::read_matrix_value<Scalar>(twist->values, env, col);
      }
    }
    if (flags == 0U && !solve6(mass, rhs, candidate_acceleration))
      flags |= status_bit(StatusFlag::kInvalidInput);
    for (std::size_t dim = 0; dim < kDim && (flags & ~status_bit(StatusFlag::kSaturation)) == 0U;
         ++dim) {
      const Scalar acceleration_limit =
          std::abs(motion_limit<Scalar>(find_parameter(parameters, kAccelerationLimit), env, dim));
      const Scalar twist_limit =
          std::abs(motion_limit<Scalar>(find_parameter(parameters, kTwistLimit), env, dim));
      const Scalar pose_limit =
          std::abs(motion_limit<Scalar>(find_parameter(parameters, kPoseLimit), env, dim));
      const Scalar unclamped_acceleration = candidate_acceleration[dim];
      candidate_acceleration[dim] =
          std::clamp(unclamped_acceleration, -acceleration_limit, acceleration_limit);
      const Scalar old_twist = internal::read_matrix_value<Scalar>(twist->values, env, dim);
      candidate_twist[dim] = std::clamp(
          old_twist + candidate_acceleration[dim] * integration_dt, -twist_limit, twist_limit);
      const Scalar old_pose = internal::read_matrix_value<Scalar>(pose->values, env, dim);
      candidate_pose[dim] =
          std::clamp(old_pose + Scalar{0.5} * integration_dt * (old_twist + candidate_twist[dim]),
                     -pose_limit,
                     pose_limit);
      if (candidate_acceleration[dim] != unclamped_acceleration ||
          std::abs(candidate_twist[dim]) == twist_limit ||
          std::abs(candidate_pose[dim]) == pose_limit)
        flags |= status_bit(StatusFlag::kSaturation);
    }
    if ((flags & ~(status_bit(StatusFlag::kSaturation))) != 0U) {
      flags |= status_bit(StatusFlag::kSafeMode);
      for (std::size_t dim = 0; dim < kDim; ++dim)
        internal::write_matrix_value(acceleration, env, dim, Scalar{0});
    } else {
      for (std::size_t dim = 0; dim < kDim; ++dim) {
        internal::write_matrix_value(acceleration, env, dim, candidate_acceleration[dim]);
        internal::write_matrix_value(previous_error->values, env, dim, error[dim]);
        internal::write_matrix_value(integrator_previous_error->values, env, dim, error[dim]);
        internal::write_matrix_value(integrated_error->values, env, dim, integral[dim]);
        internal::write_matrix_value(twist_previous->values,
                                     env,
                                     dim,
                                     internal::read_matrix_value<Scalar>(twist->values, env, dim));
        internal::write_matrix_value(pose_previous->values,
                                     env,
                                     dim,
                                     internal::read_matrix_value<Scalar>(pose->values, env, dim));
        internal::write_matrix_value(twist->values, env, dim, candidate_twist[dim]);
        internal::write_matrix_value(pose->values, env, dim, candidate_pose[dim]);
      }
      auto set_flag = [env](StateEntryView* entry) {
        static_cast<std::uint8_t*>(entry->values.data)[env] = 1U;
      };
      set_flag(differentiator_initialized);
      set_flag(integrator_initialized);
      set_flag(pose_initialized);
      set_flag(twist_initialized);
    }
    status.flags[env] = flags;
  }
  return internal::ok();
}

}  // namespace

Result apply_static_impedance_parameters(const ComponentMode mode,
                                         const ParameterSetView& parameters,
                                         const std::size_t num_envs,
                                         const std::size_t signal_dim,
                                         const MutableStatusView parameter_status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense)
    return internal::invalid_argument("static joint impedance mode must be diagonal or dense");
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (uses_cuda(parameters) || parameter_status.device == Device::kCuda)
    return apply_static_impedance_parameters_cuda(
        mode, parameters, num_envs, signal_dim, parameter_status);
#endif
  const ParameterView* first = find_parameter(parameters, kStiffness);
  if (first == nullptr) return internal::invalid_view("stiffness parameter is missing");
  return first->values.dtype == DType::kFloat32
             ? apply_static_cpu<float>(
                   mode, parameters, num_envs, signal_dim, kStiffness, kDamping, parameter_status)
             : apply_static_cpu<double>(
                   mode, parameters, num_envs, signal_dim, kStiffness, kDamping, parameter_status);
}

Result step_static_impedance(const ComponentMode mode,
                             const TensorView& position_error,
                             const TensorView& velocity_error,
                             const ParameterSetView& parameters,
                             const StatusView parameter_status,
                             const MutableTensorView& command,
                             const MutableStatusView status) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (position_error.device == Device::kCuda)
    return step_static_impedance_cuda(
        mode, position_error, velocity_error, parameters, parameter_status, command, status);
#endif
  return position_error.dtype == DType::kFloat32 ? step_static_cpu<float>(mode,
                                                                          position_error,
                                                                          velocity_error,
                                                                          parameters,
                                                                          parameter_status,
                                                                          command,
                                                                          status)
                                                 : step_static_cpu<double>(mode,
                                                                           position_error,
                                                                           velocity_error,
                                                                           parameters,
                                                                           parameter_status,
                                                                           command,
                                                                           status);
}

Result reset_high_level_stateless(const std::size_t num_envs,
                                  const TensorView& reset_mask,
                                  const MutableStatusView status) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (status.device == Device::kCuda)
    return reset_high_level_stateless_cuda(num_envs, reset_mask, status);
#endif
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) return result;
  for (std::size_t env = 0; env < num_envs; ++env)
    status.flags[env] =
        internal::reset_mask_value(reset_mask, env) ? status_bit(StatusFlag::kResetApplied) : 0U;
  return internal::ok();
}

Result apply_unified_force_admittance_parameters(const ParameterSetView& parameters,
                                                 const std::size_t num_envs,
                                                 const MutableStatusView parameter_status) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (uses_cuda(parameters) || parameter_status.device == Device::kCuda)
    return apply_unified_force_admittance_parameters_cuda(parameters, num_envs, parameter_status);
#endif
  const ParameterView* first = find_parameter(parameters, kStiffnessMatrix);
  if (first == nullptr) return internal::invalid_view("stiffness_matrix parameter is missing");
  return first->values.dtype == DType::kFloat32
             ? apply_unified_cpu<float>(parameters, num_envs, parameter_status)
             : apply_unified_cpu<double>(parameters, num_envs, parameter_status);
}

Result reset_unified_force_admittance(const std::size_t num_envs,
                                      StateView& state,
                                      const TensorView& reset_mask,
                                      const MutableStatusView status) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (status.device == Device::kCuda)
    return reset_unified_force_admittance_cuda(num_envs, state, reset_mask, status);
#endif
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) return result;
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) return result;
  for (std::size_t env = 0; env < num_envs; ++env) {
    const bool reset = internal::reset_mask_value(reset_mask, env);
    status.flags[env] = reset ? status_bit(StatusFlag::kResetApplied) : 0U;
    if (!reset) continue;
    for (std::size_t entry_index = 0; entry_index < state.entry_count; ++entry_index) {
      MutableTensorView view = state.entries[entry_index].values;
      if (view.dtype == DType::kBool) {
        static_cast<std::uint8_t*>(view.data)[env] = 0U;
      } else if (view.dtype == DType::kFloat32) {
        for (std::size_t dim = 0; dim < kDim; ++dim) {
          internal::write_matrix_value(view, env, dim, 0.0F);
        }
      } else {
        for (std::size_t dim = 0; dim < kDim; ++dim) {
          internal::write_matrix_value(view, env, dim, 0.0);
        }
      }
    }
  }
  return internal::ok();
}

Result step_unified_force_admittance(const TensorView& desired_wrench,
                                     const TensorView& external_wrench,
                                     const TensorView& feedforward_wrench,
                                     const TensorView& dt,
                                     const ParameterSetView& parameters,
                                     const StatusView parameter_status,
                                     StateView& state,
                                     const MutableTensorView& commanded_acceleration,
                                     const MutableStatusView status) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (desired_wrench.device == Device::kCuda)
    return step_unified_force_admittance_cuda(desired_wrench,
                                              external_wrench,
                                              feedforward_wrench,
                                              dt,
                                              parameters,
                                              parameter_status,
                                              state,
                                              commanded_acceleration,
                                              status);
#endif
  return desired_wrench.dtype == DType::kFloat32 ? step_unified_cpu<float>(desired_wrench,
                                                                           external_wrench,
                                                                           feedforward_wrench,
                                                                           dt,
                                                                           parameters,
                                                                           parameter_status,
                                                                           state,
                                                                           commanded_acceleration,
                                                                           status)
                                                 : step_unified_cpu<double>(desired_wrench,
                                                                            external_wrench,
                                                                            feedforward_wrench,
                                                                            dt,
                                                                            parameters,
                                                                            parameter_status,
                                                                            state,
                                                                            commanded_acceleration,
                                                                            status);
}

}  // namespace nvidia::srl::controllers::detail::tensor
