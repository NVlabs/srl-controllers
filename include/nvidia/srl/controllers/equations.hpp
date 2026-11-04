/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 * @brief Canonical allocation-free control equations for user-defined CPU and CUDA controllers.
 * Raw equations require valid dimensions, finite operands and their documented mathematical
 * domains. They do not own state, validate array allocations or apply a command safety policy.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#ifdef __CUDACC__
#define NVIDIA_SRL_CONTROLLERS_HOST_DEVICE __host__ __device__
#else
#define NVIDIA_SRL_CONTROLLERS_HOST_DEVICE
#endif
namespace nvidia {
namespace srl {
namespace controllers {
namespace equations {
template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar absolute_value(const Scalar value) noexcept {
#ifdef __CUDA_ARCH__
  return ::fabs(value);
#else
  using std::fabs;
  return fabs(value);
#endif
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar square_root(const Scalar value) noexcept {
#ifdef __CUDA_ARCH__
  return ::sqrt(value);
#else
  using std::sqrt;
  return sqrt(value);
#endif
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar cosine(const Scalar value) noexcept {
#ifdef __CUDA_ARCH__
  return ::cos(value);
#else
  using std::cos;
  return cos(value);
#endif
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar arc_tangent2(const Scalar y,
                                                              const Scalar x) noexcept {
#ifdef __CUDA_ARCH__
  return ::atan2(y, x);
#else
  using std::atan2;
  return atan2(y, x);
#endif
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool finite_value(const Scalar value) noexcept {
#ifdef __CUDA_ARCH__
  return ::isfinite(value);
#else
  using std::isfinite;
  return isfinite(value);
#endif
}

constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename Scalar>
struct LimitResult {
  Scalar value{};
  bool saturated{false};
};

template <typename Scalar>
struct EnergyTankStepResult {
  Scalar scale{};
  Scalar filtered_power{};
  Scalar state_variable{};
  bool saturated{false};
};

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar proportional_value(
    const Scalar gain, const Scalar input_signal) noexcept {
  return gain * input_signal;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar differentiator_value(
    const Scalar input_signal, const Scalar previous_signal, const Scalar t_delta) noexcept {
  return (input_signal - previous_signal) / t_delta;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline LimitResult<Scalar> clamp_value(
    const Scalar value, const Scalar lower_limit, const Scalar upper_limit) noexcept {
  LimitResult<Scalar> result;
  result.value = value;
  if (value > upper_limit) {
    result.value = upper_limit;
    result.saturated = true;
  } else if (value < lower_limit) {
    result.value = lower_limit;
    result.saturated = true;
  }
  return result;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline LimitResult<Scalar> integrator_trapezoid_value(
    const Scalar integrated_signal,
    const Scalar input_signal,
    const Scalar previous_signal,
    const Scalar t_delta,
    const Scalar lower_limit,
    const Scalar upper_limit) noexcept {
  return clamp_value(
      integrated_signal + static_cast<Scalar>(0.5) * (input_signal + previous_signal) * t_delta,
      lower_limit,
      upper_limit);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar joint_diagonal_feedback_value(
    const Scalar diagonal_gain, const Scalar joint_error) noexcept {
  return diagonal_gain * joint_error;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar shaping_factor_value(
    const Scalar error, const Scalar maximum_distance, const Scalar shaping_distance) noexcept {
  const Scalar abs_error = absolute_value(error);
  if (abs_error >= maximum_distance) {
    return static_cast<Scalar>(0);
  }
  if (abs_error >= shaping_distance) {
    return static_cast<Scalar>(1) -
           ((abs_error - shaping_distance) / (maximum_distance - shaping_distance));
  }
  return static_cast<Scalar>(1);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar cartesian_force_command_value(
    const Scalar proportional_term,
    const Scalar differentiator_term,
    const Scalar integral_term,
    const Scalar desired_wrench,
    const Scalar feedforward_wrench,
    const Scalar shaping_factor) noexcept {
  return (proportional_term + differentiator_term + integral_term + desired_wrench +
          feedforward_wrench) *
         shaping_factor;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool validate_energy_tank_parameters(
    const Scalar upper_energy_limit,
    const Scalar lower_energy_limit,
    const Scalar initial_energy,
    const Scalar empty_threshold) noexcept {
  return finite_value(upper_energy_limit) && finite_value(lower_energy_limit) &&
         finite_value(initial_energy) && finite_value(empty_threshold) &&
         empty_threshold > static_cast<Scalar>(0) && lower_energy_limit > static_cast<Scalar>(0) &&
         upper_energy_limit > static_cast<Scalar>(0) && upper_energy_limit > lower_energy_limit &&
         upper_energy_limit > lower_energy_limit + empty_threshold &&
         initial_energy > lower_energy_limit && initial_energy <= upper_energy_limit;
}

template <typename Scalar>
/** Scale depends only on energy; positive scaled power recharges the tank.
 * Additional power must be nonnegative and recovered from physical dissipation.
 * Negative raw power is debited even inside the diagnostic deadband. The returned
 * scale bounds work using start-of-step energy; recharge is available next step.
 * Invalid domains/arithmetic return state_variable == -1 for checked adapters.
 */
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline EnergyTankStepResult<Scalar> energy_tank_step(
    const Scalar state_variable,
    const Scalar power_signal,
    const Scalar additional_power,
    const Scalar t_delta,
    const Scalar upper_energy_limit,
    const Scalar lower_energy_limit,
    const Scalar empty_threshold,
    const Scalar power_noise) noexcept {
  const EnergyTankStepResult<Scalar> invalid{Scalar(0), Scalar(0), Scalar(-1), false};
  if (!finite_value(state_variable) || state_variable <= Scalar(0) || !finite_value(power_signal) ||
      !finite_value(additional_power) || additional_power < Scalar(0) || !finite_value(t_delta) ||
      t_delta <= Scalar(0) || !finite_value(power_noise) || power_noise < Scalar(0) ||
      !validate_energy_tank_parameters(
          upper_energy_limit, lower_energy_limit, upper_energy_limit, empty_threshold)) {
    return invalid;
  }
  Scalar tank_energy = Scalar(0.5) * state_variable * state_variable;
  const Scalar tolerance = Scalar(8) * std::numeric_limits<Scalar>::epsilon();
  if (!finite_value(tank_energy) ||
      (tank_energy < lower_energy_limit &&
       lower_energy_limit - tank_energy > tolerance * lower_energy_limit) ||
      (tank_energy > upper_energy_limit &&
       tank_energy - upper_energy_limit > tolerance * upper_energy_limit)) {
    return invalid;
  }
  tank_energy = tank_energy < lower_energy_limit ? lower_energy_limit : tank_energy;
  tank_energy = tank_energy > upper_energy_limit ? upper_energy_limit : tank_energy;
  EnergyTankStepResult<Scalar> result;
  result.filtered_power = power_signal;
  if (absolute_value(result.filtered_power) < power_noise) {
    result.filtered_power = static_cast<Scalar>(0);
  }

  Scalar alpha = static_cast<Scalar>(0);
  if (tank_energy >= lower_energy_limit + empty_threshold) {
    alpha = static_cast<Scalar>(1);
  } else if (tank_energy >= lower_energy_limit &&
             tank_energy <= lower_energy_limit + empty_threshold) {
    alpha = static_cast<Scalar>(0.5) *
            (static_cast<Scalar>(1) - cosine(static_cast<Scalar>(kPi) *
                                             (tank_energy - lower_energy_limit) / empty_threshold));
  }

  result.scale = alpha;

  const Scalar requested_work = power_signal < Scalar(0) ? -power_signal * t_delta : Scalar(0);
  const Scalar main_recharge = result.filtered_power > Scalar(0)
                                   ? result.scale * result.filtered_power * t_delta
                                   : Scalar(0);
  const Scalar extra_recharge = additional_power * t_delta;
  if (!finite_value(requested_work) || !finite_value(main_recharge) ||
      !finite_value(extra_recharge)) {
    return invalid;
  }
  const Scalar available = tank_energy - lower_energy_limit;
  if (requested_work > Scalar(0) && result.scale * requested_work > available) {
    result.scale = available / requested_work;
    result.saturated = true;
  }
  Scalar remaining = tank_energy - result.scale * requested_work;
  remaining = remaining < lower_energy_limit ? lower_energy_limit : remaining;
  // Accept recharge separately to avoid overflowing their sum before saturation.
  const Scalar main_credit = main_recharge < upper_energy_limit - remaining
                                 ? main_recharge
                                 : upper_energy_limit - remaining;
  remaining += main_credit;
  const Scalar extra_credit = extra_recharge < upper_energy_limit - remaining
                                  ? extra_recharge
                                  : upper_energy_limit - remaining;
  remaining += extra_credit;
  result.saturated =
      result.saturated || main_credit < main_recharge || extra_credit < extra_recharge;
  result.state_variable = square_root(remaining) * square_root(Scalar(2));
  if (!finite_value(result.state_variable) || result.state_variable <= Scalar(0)) return invalid;
  return result;
}

template <typename Scalar>
struct ConstVectorView {
  const Scalar* data{nullptr};
  std::size_t size{0};
  std::ptrdiff_t stride{1};

  NVIDIA_SRL_CONTROLLERS_HOST_DEVICE Scalar operator[](const std::size_t index) const noexcept {
    return data[static_cast<std::ptrdiff_t>(index) * stride];
  }
};

template <typename Scalar>
struct MutableVectorView {
  Scalar* data{nullptr};
  std::size_t size{0};
  std::ptrdiff_t stride{1};

  NVIDIA_SRL_CONTROLLERS_HOST_DEVICE Scalar& operator[](const std::size_t index) const noexcept {
    return data[static_cast<std::ptrdiff_t>(index) * stride];
  }
};

template <typename Scalar>
struct ConstMatrixView {
  const Scalar* data{nullptr};
  std::size_t rows{0};
  std::size_t cols{0};
  std::ptrdiff_t row_stride{0};
  std::ptrdiff_t col_stride{0};

  NVIDIA_SRL_CONTROLLERS_HOST_DEVICE Scalar operator()(const std::size_t row,
                                                       const std::size_t col) const noexcept {
    return data[static_cast<std::ptrdiff_t>(row) * row_stride +
                static_cast<std::ptrdiff_t>(col) * col_stride];
  }
};

template <typename Scalar>
struct MutableMatrixView {
  Scalar* data{nullptr};
  std::size_t rows{0};
  std::size_t cols{0};
  std::ptrdiff_t row_stride{0};
  std::ptrdiff_t col_stride{0};

  NVIDIA_SRL_CONTROLLERS_HOST_DEVICE Scalar& operator()(const std::size_t row,
                                                        const std::size_t col) const noexcept {
    return data[static_cast<std::ptrdiff_t>(row) * row_stride +
                static_cast<std::ptrdiff_t>(col) * col_stride];
  }
};

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE constexpr Scalar rotation_tolerance() noexcept {
  return sizeof(Scalar) == sizeof(float) ? static_cast<Scalar>(1.0e-3)
                                         : static_cast<Scalar>(1.0e-8);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar vector_norm(
    const ConstVectorView<Scalar>& vector) noexcept {
  Scalar squared_norm = static_cast<Scalar>(0);
  for (std::size_t index = 0; index < vector.size; ++index) {
    squared_norm += vector[index] * vector[index];
  }
  return square_root(squared_norm);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool rotation_is_valid(
    const ConstMatrixView<Scalar>& pose) noexcept {
  const Scalar tolerance = rotation_tolerance<Scalar>();
  for (std::size_t col = 0; col < 4; ++col) {
    const Scalar expected = col == 3 ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    if (absolute_value(pose(3, col) - expected) > tolerance) {
      return false;
    }
  }
  for (std::size_t first = 0; first < 3; ++first) {
    for (std::size_t second = first; second < 3; ++second) {
      Scalar dot = static_cast<Scalar>(0);
      for (std::size_t row = 0; row < 3; ++row) {
        dot += pose(row, first) * pose(row, second);
      }
      const Scalar expected = first == second ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
      if (absolute_value(dot - expected) > tolerance) {
        return false;
      }
    }
  }
  const Scalar determinant = pose(0, 0) * (pose(1, 1) * pose(2, 2) - pose(1, 2) * pose(2, 1)) -
                             pose(0, 1) * (pose(1, 0) * pose(2, 2) - pose(1, 2) * pose(2, 0)) +
                             pose(0, 2) * (pose(1, 0) * pose(2, 1) - pose(1, 1) * pose(2, 0));
  return absolute_value(determinant - static_cast<Scalar>(1)) <= static_cast<Scalar>(4) * tolerance;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool so3_log_error(
    const ConstMatrixView<Scalar>& current_pose,
    const ConstMatrixView<Scalar>& goal_pose,
    const MutableVectorView<Scalar>& rotation_error) noexcept {
  if (!rotation_is_valid(current_pose) || !rotation_is_valid(goal_pose)) {
    return false;
  }
  Scalar relative[3][3]{};
  Scalar trace = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        relative[row][col] += goal_pose(row, inner) * current_pose(col, inner);
      }
    }
    trace += relative[row][row];
  }
  const Scalar raw_cosine = (trace - static_cast<Scalar>(1)) / static_cast<Scalar>(2);
  const Scalar cosine_value =
      raw_cosine < static_cast<Scalar>(-1)
          ? static_cast<Scalar>(-1)
          : (raw_cosine > static_cast<Scalar>(1) ? static_cast<Scalar>(1) : raw_cosine);
  Scalar skew[3]{static_cast<Scalar>(0.5) * (relative[2][1] - relative[1][2]),
                 static_cast<Scalar>(0.5) * (relative[0][2] - relative[2][0]),
                 static_cast<Scalar>(0.5) * (relative[1][0] - relative[0][1])};
  const Scalar sine = vector_norm(ConstVectorView<Scalar>{skew, 3, 1});
  const Scalar theta = arc_tangent2(sine, cosine_value);
  const Scalar tolerance = rotation_tolerance<Scalar>();
  if (sine > tolerance) {
    const Scalar scale = theta / sine;
    for (std::size_t dim = 0; dim < 3; ++dim) {
      rotation_error[dim] = scale * skew[dim];
    }
    return true;
  }
  if (cosine_value >= static_cast<Scalar>(0)) {
    for (std::size_t dim = 0; dim < 3; ++dim) {
      rotation_error[dim] = skew[dim];
    }
    return true;
  }

  Scalar axis[3]{
      square_root((relative[0][0] + static_cast<Scalar>(1)) > static_cast<Scalar>(0)
                      ? (relative[0][0] + static_cast<Scalar>(1)) / static_cast<Scalar>(2)
                      : static_cast<Scalar>(0)),
      square_root((relative[1][1] + static_cast<Scalar>(1)) > static_cast<Scalar>(0)
                      ? (relative[1][1] + static_cast<Scalar>(1)) / static_cast<Scalar>(2)
                      : static_cast<Scalar>(0)),
      square_root((relative[2][2] + static_cast<Scalar>(1)) > static_cast<Scalar>(0)
                      ? (relative[2][2] + static_cast<Scalar>(1)) / static_cast<Scalar>(2)
                      : static_cast<Scalar>(0))};
  std::size_t largest = axis[1] > axis[0] ? 1 : 0;
  if (axis[2] > axis[largest]) {
    largest = 2;
  }
  if (axis[largest] <= tolerance) {
    return false;
  }
  if (largest == 0) {
    axis[1] = (relative[0][1] + relative[1][0]) / (static_cast<Scalar>(4) * axis[0]);
    axis[2] = (relative[0][2] + relative[2][0]) / (static_cast<Scalar>(4) * axis[0]);
  } else if (largest == 1) {
    axis[0] = (relative[0][1] + relative[1][0]) / (static_cast<Scalar>(4) * axis[1]);
    axis[2] = (relative[1][2] + relative[2][1]) / (static_cast<Scalar>(4) * axis[1]);
  } else {
    axis[0] = (relative[0][2] + relative[2][0]) / (static_cast<Scalar>(4) * axis[2]);
    axis[1] = (relative[1][2] + relative[2][1]) / (static_cast<Scalar>(4) * axis[2]);
  }
  const Scalar axis_norm = vector_norm(ConstVectorView<Scalar>{axis, 3, 1});
  if (!finite_value(axis_norm) || axis_norm <= tolerance) {
    return false;
  }
  for (std::size_t dim = 0; dim < 3; ++dim) {
    rotation_error[dim] = theta * axis[dim] / axis_norm;
  }
  return true;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar dense_feedback_row(
    const ConstMatrixView<Scalar>& gain,
    const ConstVectorView<Scalar>& error,
    const std::size_t row) noexcept {
  Scalar output = static_cast<Scalar>(0);
  for (std::size_t col = 0; col < gain.cols; ++col) {
    output += gain(row, col) * error[col];
  }
  return output;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar cartesian_stiffness_value(
    const ConstMatrixView<Scalar>& stiffness,
    const ConstVectorView<Scalar>& pose_error,
    const std::size_t row) noexcept {
  return dense_feedback_row(stiffness, pose_error, row);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar cartesian_damping_value(
    const ConstMatrixView<Scalar>& damping,
    const ConstVectorView<Scalar>& velocity_error,
    const std::size_t row) noexcept {
  return dense_feedback_row(damping, velocity_error, row);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar joint_stiffness_value(
    const ConstMatrixView<Scalar>& stiffness,
    const ConstVectorView<Scalar>& position_error,
    const std::size_t row) noexcept {
  return dense_feedback_row(stiffness, position_error, row);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar joint_damping_value(
    const ConstMatrixView<Scalar>& damping,
    const ConstVectorView<Scalar>& velocity_error,
    const std::size_t row) noexcept {
  return dense_feedback_row(damping, velocity_error, row);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar variable_stiffness_value(
    const Scalar previous,
    const Scalar requested,
    const Scalar lower,
    const Scalar upper,
    const Scalar maximum_rate,
    const Scalar dt) noexcept {
  const Scalar maximum_delta = maximum_rate * dt;
  const Scalar rate_limited =
      clamp_value(requested, previous - maximum_delta, previous + maximum_delta).value;
  return clamp_value(rate_limited, lower, upper).value;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar deadband_value(const Scalar value,
                                                                const Scalar threshold) noexcept {
  return absolute_value(value) < threshold ? static_cast<Scalar>(0) : value;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar force_error_value(const Scalar desired,
                                                                   const Scalar measured) noexcept {
  return desired - measured;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar matrix_derivative_value(const Scalar current,
                                                                         const Scalar previous,
                                                                         const Scalar dt) noexcept {
  return differentiator_value(current, previous, dt);
}

template <typename Scalar>
struct FlowFunnelGeometry {
  Scalar length{};
  Scalar projection_distance{};
  Scalar radius{};
  bool valid{false};
};

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline FlowFunnelGeometry<Scalar> flow_funnel_geometry(
    const ConstVectorView<Scalar>& start,
    const ConstVectorView<Scalar>& end,
    const ConstVectorView<Scalar>& position,
    const MutableVectorView<Scalar>& unit_direction,
    const MutableVectorView<Scalar>& unit_direction_to_projection) noexcept {
  FlowFunnelGeometry<Scalar> geometry;
  for (std::size_t dim = 0; dim < 3; ++dim) {
    unit_direction[dim] = end[dim] - start[dim];
  }
  geometry.length = vector_norm(
      ConstVectorView<Scalar>{unit_direction.data, unit_direction.size, unit_direction.stride});
  if (!finite_value(geometry.length) || geometry.length <= static_cast<Scalar>(0)) {
    return geometry;
  }
  for (std::size_t dim = 0; dim < 3; ++dim) {
    unit_direction[dim] /= geometry.length;
    geometry.projection_distance += (position[dim] - start[dim]) * unit_direction[dim];
  }
  const Scalar clipped =
      clamp_value(geometry.projection_distance, static_cast<Scalar>(0), geometry.length).value;
  Scalar to_projection[3]{};
  for (std::size_t dim = 0; dim < 3; ++dim) {
    to_projection[dim] = start[dim] + clipped * unit_direction[dim] - position[dim];
  }
  geometry.radius = vector_norm(ConstVectorView<Scalar>{to_projection, 3, 1});
  if (!finite_value(geometry.radius)) {
    return geometry;
  }
  for (std::size_t dim = 0; dim < 3; ++dim) {
    unit_direction_to_projection[dim] = geometry.radius > static_cast<Scalar>(0)
                                            ? to_projection[dim] / geometry.radius
                                            : static_cast<Scalar>(0);
  }
  geometry.valid = true;
  return geometry;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar flow_funnel_weight(
    const Scalar radius, const Scalar funnel_radius, const Scalar maximum_distance) noexcept {
  if (radius <= funnel_radius) {
    return static_cast<Scalar>(1);
  }
  if (radius <= maximum_distance) {
    return static_cast<Scalar>(1) -
           static_cast<Scalar>(2) * (radius - funnel_radius) / (maximum_distance - funnel_radius);
  }
  return static_cast<Scalar>(-1);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar flow_funnel_direction_value(
    const Scalar radius,
    const Scalar funnel_radius,
    const Scalar maximum_distance,
    const Scalar weight,
    const Scalar funnel_direction,
    const Scalar projection_direction) noexcept {
  if (radius <= funnel_radius) {
    return funnel_direction;
  }
  if (radius < maximum_distance) {
    return weight * funnel_direction +
           (static_cast<Scalar>(1) - absolute_value(weight)) * projection_direction;
  }
  return -funnel_direction;
}

/** @brief Unilateral inward wall force in N; signed distance is positive inside, in m.
 * Normal velocity is positive inward, in m/s. Gains use N/m and Ns/m.
 * Damping activation distance must be positive, in m; gains must be nonnegative.
 */
template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar virtual_wall_force_value(
    const Scalar signed_distance,
    const Scalar normal_velocity,
    const Scalar stiffness,
    const Scalar damping,
    const Scalar damping_activation_distance) noexcept {
  if (signed_distance >= static_cast<Scalar>(0)) return static_cast<Scalar>(0);
  const Scalar penetration = -signed_distance;
  const Scalar fraction =
      clamp_value(
          penetration / damping_activation_distance, static_cast<Scalar>(0), static_cast<Scalar>(1))
          .value;
  const Scalar activation =
      fraction * fraction * (static_cast<Scalar>(3) - static_cast<Scalar>(2) * fraction);
  const Scalar force = stiffness * penetration - damping * activation * normal_velocity;
  return force > static_cast<Scalar>(0) ? force : static_cast<Scalar>(0);
}

/** @brief Outward spherical repulsion magnitude in N, using positive surface clearance in m.
 * Influence distance must be positive, in m; gain must be nonnegative, in N m^3.
 * Callers must reject contact/penetration before evaluating this equation.
 */
template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar spherical_apf_repulsion_value(
    const Scalar surface_clearance, const Scalar influence_distance, const Scalar gain) noexcept {
  if (surface_clearance >= influence_distance) return static_cast<Scalar>(0);
  return gain *
         (static_cast<Scalar>(1) / surface_clearance -
          static_cast<Scalar>(1) / influence_distance) /
         (surface_clearance * surface_clearance);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool validate_inverse_apf_parameters(
    const Scalar alpha,
    const Scalar gamma,
    const Scalar minimum_damping,
    const Scalar maximum_damping,
    const Scalar deadzone) noexcept {
  return finite_value(alpha) && finite_value(gamma) && finite_value(minimum_damping) &&
         finite_value(maximum_damping) && finite_value(deadzone) &&
         alpha >= static_cast<Scalar>(0) && gamma >= static_cast<Scalar>(0) &&
         minimum_damping >= static_cast<Scalar>(0) && maximum_damping >= minimum_damping &&
         deadzone >= static_cast<Scalar>(0);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar inverse_apf_damping(
    const Scalar error_norm_squared,
    const Scalar minimum_damping,
    const Scalar maximum_damping,
    const Scalar gamma) noexcept {
  return minimum_damping + (maximum_damping - minimum_damping) /
                               (static_cast<Scalar>(1) + gamma * error_norm_squared);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar inverse_apf_wrench_value(
    const Scalar error,
    const Scalar velocity,
    const Scalar error_norm,
    const Scalar error_norm_squared,
    const Scalar alpha,
    const Scalar deadzone,
    const Scalar damping) noexcept {
  Scalar potential = static_cast<Scalar>(0);
  if (error_norm > deadzone) {
    potential = -alpha /
                (static_cast<Scalar>(1) + alpha * (error_norm_squared - deadzone * deadzone)) *
                error;
  }
  return potential - damping * velocity;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar damping_factorization_value(
    const Scalar mass_times_damping_times_stiffness,
    const Scalar stiffness_times_damping_times_mass) noexcept {
  return mass_times_damping_times_stiffness + stiffness_times_damping_times_mass;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar damping_double_diagonalization_value(
    const Scalar mass_times_damping_times_stiffness_times_mass_transpose) noexcept {
  return static_cast<Scalar>(2) * mass_times_damping_times_stiffness_times_mass_transpose;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool command_observer_accepts(
    const Scalar command, const Scalar absolute_limit) noexcept {
  return finite_value(command) && absolute_value(command) <= absolute_limit;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar inertial_coupling_value(
    const Scalar mass_derivative_times_velocity,
    const Scalar mass_times_jacobian_derivative_velocity) noexcept {
  return static_cast<Scalar>(0.5) * mass_derivative_times_velocity -
         mass_times_jacobian_derivative_velocity;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar cartesian_impedance_value(
    const Scalar stiffness_wrench,
    const Scalar damping_wrench,
    const Scalar inertial_wrench,
    const Scalar acceleration_wrench) noexcept {
  return stiffness_wrench + damping_wrench + inertial_wrench + acceleration_wrench;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar joint_impedance_value(
    const Scalar stiffness_torque,
    const Scalar damping_torque,
    const Scalar acceleration_torque) noexcept {
  return stiffness_torque + damping_torque + acceleration_torque;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar nullspace_feedback_value(
    const Scalar stiffness_torque, const Scalar damping_torque) noexcept {
  return stiffness_torque + damping_torque;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar nullspace_projector_value(
    const std::size_t row,
    const std::size_t col,
    const Scalar dynamically_consistent_inverse_term) noexcept {
  return (row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0)) -
         dynamically_consistent_inverse_term;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar nullspace_projected_torque_value(
    const Scalar desired_torque, const Scalar task_space_correction) noexcept {
  return desired_torque - task_space_correction;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar unified_force_impedance_value(
    const Scalar impedance_wrench, const Scalar force_wrench) noexcept {
  return impedance_wrench + force_wrench;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar admittance_force_balance_value(
    const Scalar external_wrench,
    const Scalar feedforward_wrench,
    const Scalar stiffness_wrench,
    const Scalar damping_wrench) noexcept {
  return external_wrench + feedforward_wrench - stiffness_wrench - damping_wrench;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar operational_space_acceleration_value(
    const Scalar desired_acceleration,
    const Scalar position_feedback,
    const Scalar velocity_feedback) noexcept {
  return desired_acceleration + position_feedback + velocity_feedback;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool operational_space_mass_parameters_valid(
    const Scalar regularization, const Scalar minimum_pivot) noexcept {
  return finite_value(regularization) && regularization >= static_cast<Scalar>(0) &&
         finite_value(minimum_pivot) && minimum_pivot > static_cast<Scalar>(0);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar symmetric_average(
    const Scalar forward, const Scalar transpose) noexcept {
  return static_cast<Scalar>(0.5) * (forward + transpose);
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline bool symmetric_pair_is_valid(
    const Scalar forward, const Scalar transpose, const Scalar relative_tolerance) noexcept {
  const Scalar scale = static_cast<Scalar>(1) + (absolute_value(forward) > absolute_value(transpose)
                                                     ? absolute_value(forward)
                                                     : absolute_value(transpose));
  return finite_value(forward) && finite_value(transpose) &&
         absolute_value(forward - transpose) <= relative_tolerance * scale;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar operational_space_inverse_mass_value(
    const Scalar forward,
    const Scalar transpose,
    const Scalar regularization,
    const bool diagonal) noexcept {
  return symmetric_average(forward, transpose) +
         (diagonal ? regularization : static_cast<Scalar>(0));
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar torque_filter_value(const Scalar requested,
                                                                     const Scalar previous,
                                                                     const Scalar lower_limit,
                                                                     const Scalar upper_limit,
                                                                     const Scalar maximum_rate,
                                                                     const Scalar dt,
                                                                     const bool initialized,
                                                                     bool* saturated) noexcept {
  const auto magnitude = clamp_value(requested, lower_limit, upper_limit);
  Scalar filtered = magnitude.value;
  bool limited = magnitude.saturated;
  if (initialized) {
    const Scalar maximum_delta = maximum_rate * dt;
    const auto rate = clamp_value(filtered, previous - maximum_delta, previous + maximum_delta);
    filtered = rate.value;
    limited = limited || rate.saturated;
  }
  if (saturated != nullptr) {
    *saturated = limited;
  }
  return filtered;
}

template <typename Scalar>
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline Scalar safe_output_value(
    const bool safe_mode,
    const bool hold_last_valid,
    const bool has_last_valid,
    const Scalar current,
    const Scalar last_valid) noexcept {
  if (!safe_mode) {
    return current;
  }
  return hold_last_valid && has_last_valid ? last_valid : static_cast<Scalar>(0);
}
}  // namespace equations
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
