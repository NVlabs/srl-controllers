/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file example_specs.hpp
 * @brief Typed registry shared by physical-robot and Isaac Lab examples.
 */

#pragma once

#include <array>
#include <cstddef>

#include "nvidia/srl/controllers/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {

enum class ExampleRobotFamily { kFranka, kRizon, kFlexivDual, kUniversalRobots = 4 };

enum class ExampleControllerKind {
  kCartesianAdmittanceTorque,
  kCartesianUnifiedForceImpedanceTorque,
  kCartesianImpedanceTorque,
  kJointImpedance,
  kOperationalSpace,
  kVariableCartesianImpedanceTorque,
  kBimanualCartesianImpedanceTorque,
  kCartesianAdmittancePose,
  kStaticJointImpedance,
  kStaticCartesianImpedanceTorque,
  kCartesianUnifiedForceAdmittancePose,
  kDualCartesianImpedanceTorque,
  kCartesianForceImpedanceWithTankTorque,
  kCartesianVirtualWallsTorque,
  kCartesianPotentialFieldTorque,
};

/**
 * @brief Static identity and execution contract for one hardware/simulation example pair.
 *
 * The registry is compile-time data: no file parsing or allocation occurs in a control loop.
 * Joint quantities use radians, radians/second, and Nm. Cartesian vectors use
 * `[translation_m; rotation_vector_rad]`; twists use `[linear_mps; angular_radps]`; wrenches use
 * `[force_N; torque_Nm]`. `nominal_period_seconds` is passed only as the simulator physics period;
 * controllers always receive the measured step duration.
 */
struct ControllerExampleSpec {
  const char* hardware_target;
  const char* isaac_launcher;
  ExampleRobotFamily robot_family;
  ExampleControllerKind controller_kind;
  const char* default_robot_model;
  const char* supported_robot_models;
  std::size_t joint_dim;
  std::size_t task_dim;
  CartesianFrame cartesian_frame;
  double nominal_period_seconds;
  double default_duration_seconds;
  int controlled_force_axis;  ///< -1 when the example does not close a force loop.
};

NVIDIA_SRL_CONTROLLERS_API const std::array<ControllerExampleSpec, 20>&
controller_example_specs() noexcept;

/** @brief Non-owning view of every canonical example, including newer companions. */
struct ControllerExampleSpecView {
  const ControllerExampleSpec* data;
  std::size_t count;

  const ControllerExampleSpec* begin() const noexcept { return data; }
  const ControllerExampleSpec* end() const noexcept { return data + count; }
};

/** @brief Complete registry; the fixed-size accessor remains for existing consumers. */
NVIDIA_SRL_CONTROLLERS_API ControllerExampleSpecView all_controller_example_specs() noexcept;
NVIDIA_SRL_CONTROLLERS_API const ControllerExampleSpec* find_controller_example_spec(
    const char* hardware_target) noexcept;

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
