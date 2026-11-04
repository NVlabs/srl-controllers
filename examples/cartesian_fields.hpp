// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "nvidia/srl/controllers/equations.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace examples {

enum class CartesianFieldKind { kVirtualWalls, kPotentialField };
using FieldVector3 = std::array<double, 3>;

/** @brief Fixed example geometry in base axes, relative to the measured starting TCP.
 * Distances are m; stiffness is N/m, damping Ns/m, and repulsion gain N m^3.
 * These virtual surfaces constrain a point, not the robot links or tool volume.
 */
struct CartesianFieldParameters {
  FieldVector3 box_half_extent_m{0.1, 0.1, 0.1};
  FieldVector3 goal_offset_m{0.0, 0.12, 0.0};
  FieldVector3 obstacle_offset_m{-0.025, 0.06, 0.0};
  double wall_stiffness_n_per_m{1000.0};
  double wall_damping_ns_per_m{40.0};
  double damping_activation_distance_m{0.005};
  double goal_stiffness_n_per_m{50.0};
  double field_damping_ns_per_m{20.0};
  double obstacle_radius_m{0.02};
  double influence_distance_m{0.035};
  double repulsion_gain_n_m3{0.00005};
  double ramp_duration_seconds{1.0};
};

struct CartesianFieldSample {
  FieldVector3 force_n{};
  double clearance_m{std::numeric_limits<double>::infinity()};
  double goal_error_m{0.0};
  bool repulsion_active{false};
};

inline double field_norm(const FieldVector3& vector) noexcept {
  return std::hypot(vector[0], vector[1], vector[2]);
}

inline bool field_vector_is_finite(const FieldVector3& vector) noexcept {
  return std::all_of(
      vector.begin(), vector.end(), [](double scalar) { return std::isfinite(scalar); });
}

inline bool validate_cartesian_field_parameters(const CartesianFieldParameters& parameters) {
  if (!field_vector_is_finite(parameters.box_half_extent_m) ||
      !field_vector_is_finite(parameters.goal_offset_m) ||
      !field_vector_is_finite(parameters.obstacle_offset_m))
    return false;
  for (const double half_extent : parameters.box_half_extent_m)
    if (half_extent <= 0.0) return false;
  const std::array<double, 6> gains{parameters.wall_stiffness_n_per_m,
                                    parameters.wall_damping_ns_per_m,
                                    parameters.goal_stiffness_n_per_m,
                                    parameters.field_damping_ns_per_m,
                                    parameters.obstacle_radius_m,
                                    parameters.repulsion_gain_n_m3};
  for (const double gain : gains)
    if (!std::isfinite(gain) || gain < 0.0) return false;
  const std::array<double, 3> positive{parameters.damping_activation_distance_m,
                                       parameters.influence_distance_m,
                                       parameters.ramp_duration_seconds};
  for (const double scalar : positive)
    if (!std::isfinite(scalar) || scalar <= 0.0) return false;
  return true;
}

/** @brief Evaluate an allocation-free base-frame translation field, with zero output on failure.
 * Reference/current positions are m, velocity m/s, elapsed time s. Repulsion requires
 * positive surface clearance; no minimum-clearance cutoff or Cartesian force cap is applied.
 * Validate parameters once before starting the control loop.
 */
inline bool evaluate_cartesian_field(const CartesianFieldKind kind,
                                     const CartesianFieldParameters& parameters,
                                     const FieldVector3& reference_position_m,
                                     const FieldVector3& current_position_m,
                                     const FieldVector3& velocity_m_per_s,
                                     const double elapsed_seconds,
                                     CartesianFieldSample* sample) noexcept {
  if (sample == nullptr) return false;
  *sample = CartesianFieldSample{};
  if (!field_vector_is_finite(reference_position_m) ||
      !field_vector_is_finite(current_position_m) || !field_vector_is_finite(velocity_m_per_s) ||
      !std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0)
    return false;
  FieldVector3 displacement{}, goal_error{}, obstacle_to_tcp{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    displacement[axis] = current_position_m[axis] - reference_position_m[axis];
    goal_error[axis] = parameters.goal_offset_m[axis] - displacement[axis];
    obstacle_to_tcp[axis] = displacement[axis] - parameters.obstacle_offset_m[axis];
  }
  sample->goal_error_m = field_norm(goal_error);
  FieldVector3 force{};
  if (kind == CartesianFieldKind::kVirtualWalls) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double lower_force = equations::virtual_wall_force_value(
          parameters.box_half_extent_m[axis] + displacement[axis],
          velocity_m_per_s[axis],
          parameters.wall_stiffness_n_per_m,
          parameters.wall_damping_ns_per_m,
          parameters.damping_activation_distance_m);
      const double upper_force = equations::virtual_wall_force_value(
          parameters.box_half_extent_m[axis] - displacement[axis],
          -velocity_m_per_s[axis],
          parameters.wall_stiffness_n_per_m,
          parameters.wall_damping_ns_per_m,
          parameters.damping_activation_distance_m);
      force[axis] = lower_force - upper_force;
    }
  } else {
    const double center_distance = field_norm(obstacle_to_tcp);
    sample->clearance_m = center_distance - parameters.obstacle_radius_m;
    if (!std::isfinite(sample->clearance_m) || sample->clearance_m <= 0.0) return false;
    const double repulsion = equations::spherical_apf_repulsion_value(
        sample->clearance_m, parameters.influence_distance_m, parameters.repulsion_gain_n_m3);
    sample->repulsion_active = repulsion > 0.0;
    const double fraction =
        std::clamp(elapsed_seconds / parameters.ramp_duration_seconds, 0.0, 1.0);
    const double ramp =
        fraction * fraction * fraction * (10.0 + fraction * (-15.0 + 6.0 * fraction));
    for (std::size_t axis = 0; axis < 3; ++axis)
      force[axis] = ramp * (parameters.goal_stiffness_n_per_m * goal_error[axis] +
                            repulsion * obstacle_to_tcp[axis] / center_distance) -
                    parameters.field_damping_ns_per_m * velocity_m_per_s[axis];
  }
  const double magnitude = field_norm(force);
  if (!std::isfinite(magnitude) || !std::isfinite(sample->goal_error_m)) return false;
  sample->force_n = force;
  return true;
}

}  // namespace examples
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
