// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#pragma once

#include <franka/exception.h>
#include <franka/model.h>
#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "./franka_utils.hpp"
#include "examples/cartesian_fields.hpp"
#include "examples/cartesian_math.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace franka_example {
namespace cartesian_field {
namespace controllers = nvidia::srl::controllers;
namespace field = controllers::examples;
using Vector6 = Eigen::Vector<double, 6>;
using Vector7 = Eigen::Vector<double, 7>;
using Matrix7 = Eigen::Matrix<double, 7, 7>;
using Jacobian = Eigen::Matrix<double, 6, 7>;

// RT hot-path storage: all views and buffers are owned for the complete run.
template <std::size_t Size>
struct FixedFieldOutput {
  std::array<double, Size> command{}, last_valid{};
  std::array<std::uint64_t, 1> parameter_status{}, status{}, hold_count{};
  std::array<std::uint8_t, 1> valid{};
  controllers::ControllerOutputStorage storage() {
    return {controllers::write_view(command, controllers::Device::kCpu, 1, Size),
            controllers::write_status(parameter_status, controllers::Device::kCpu),
            controllers::write_status(status, controllers::Device::kCpu),
            controllers::make_safety_state(
                controllers::write_view(last_valid, controllers::Device::kCpu, 1, Size),
                controllers::write_view(valid, controllers::Device::kCpu, 1),
                controllers::write_view(hold_count, controllers::Device::kCpu, 1))};
  }
};

/** @brief Wall-only wrench or prepared APF orientation and posture control.
 * Translation forces are composed independently from the explicit field equations.
 * Prepare/reset only at startup; step uses fixed-size CPU float64 arrays.
 */
class FieldWorkspace {
 public:
  FieldWorkspace() = default;
  FieldWorkspace(const FieldWorkspace&) = delete;
  FieldWorkspace& operator=(const FieldWorkspace&) = delete;
  FieldWorkspace(FieldWorkspace&&) = delete;
  FieldWorkspace& operator=(FieldWorkspace&&) = delete;
  controllers::Result prepare(
      const field::CartesianFieldKind kind = field::CartesianFieldKind::kPotentialField) {
    kind_ = kind;
    if (kind_ == field::CartesianFieldKind::kVirtualWalls) return {};
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               7,
                                               {},
                                               controllers::CartesianFrame::kRobotBase};
    for (std::size_t axis = 3; axis < 6; ++axis) {
      arrays_.stiffness[axis * 6 + axis] = 10.0;
      arrays_.damping[axis * 6 + axis] = 3.0;
    }
    const std::array<double, 7> stiffness{120, 80, 60, 40, 30, 20, 10};
    const std::array<double, 7> damping{12, 8, 6, 4, 3, 2, 1};
    for (std::size_t joint = 0; joint < 7; ++joint) {
      arrays_.nullspace_stiffness[joint * 7 + joint] = stiffness[joint];
      arrays_.nullspace_damping[joint * 7 + joint] = damping[joint];
    }
    const controllers::StaticCartesianImpedanceController::Parameters orientation_parameters{
        controllers::shared(
            controllers::read_view(arrays_.stiffness, controllers::Device::kCpu, 6, 6)),
        controllers::shared(
            controllers::read_view(arrays_.damping, controllers::Device::kCpu, 6, 6))};
    const controllers::NullspaceController::Parameters nullspace_parameters{
        controllers::shared(
            controllers::read_view(arrays_.nullspace_stiffness, controllers::Device::kCpu, 7, 7)),
        controllers::shared(
            controllers::read_view(arrays_.nullspace_damping, controllers::Device::kCpu, 7, 7))};
    orientation_inputs_ = {
        controllers::read_view(arrays_.pose_error, controllers::Device::kCpu, 1, 6),
        controllers::read_view(arrays_.velocity_error, controllers::Device::kCpu, 1, 6)};
    nullspace_inputs_ = {
        controllers::read_view(arrays_.joint_error, controllers::Device::kCpu, 1, 7),
        controllers::read_view(arrays_.joint_velocity, controllers::Device::kCpu, 1, 7),
        controllers::read_view(arrays_.mass, controllers::Device::kCpu, 1, 7, 7),
        controllers::read_view(arrays_.jacobian, controllers::Device::kCpu, 1, 6, 7)};
    controllers::Result result =
        orientation_.prepare(config, orientation_parameters, orientation_output_.storage());
    if (!result.ok()) return result;
    result = nullspace_.prepare(config, nullspace_parameters, nullspace_output_.storage());
    if (!result.ok()) return result;
    result = orientation_.reset();
    if (!result.ok()) return result;
    return nullspace_.reset();
  }

  bool step(const Vector6& pose_error,
            const Vector6& twist,
            const Vector7& joint_error,
            const Vector7& joint_velocity,
            const Matrix7& mass,
            const Jacobian& jacobian,
            const field::CartesianFieldSample& sample) {
    wrench_.fill(0.0);
    if (kind_ == field::CartesianFieldKind::kVirtualWalls) {
      for (std::size_t axis = 0; axis < 3; ++axis) wrench_[axis] = sample.force_n[axis];
      return all_finite(wrench_);
    }
    for (std::size_t axis = 0; axis < 6; ++axis) {
      arrays_.pose_error[axis] = pose_error[axis];
      arrays_.velocity_error[axis] = -twist[axis];
      for (std::size_t joint = 0; joint < 7; ++joint)
        arrays_.jacobian[axis * 7 + joint] = jacobian(axis, joint);
    }
    for (std::size_t joint = 0; joint < 7; ++joint) {
      arrays_.joint_error[joint] = joint_error[joint];
      arrays_.joint_velocity[joint] = joint_velocity[joint];
      for (std::size_t column = 0; column < 7; ++column)
        arrays_.mass[joint * 7 + column] = mass(joint, column);
    }
    const controllers::Result orientation_result = orientation_.step(orientation_inputs_);
    if (!valid_step(orientation_result, orientation_output_.status[0])) return false;
    const controllers::Result nullspace_result = nullspace_.step(nullspace_inputs_);
    if (!valid_step(nullspace_result, nullspace_output_.status[0])) return false;
    const field::FieldVector3 moment{orientation_output_.command[3],
                                     orientation_output_.command[4],
                                     orientation_output_.command[5]};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      wrench_[axis] = sample.force_n[axis];
      wrench_[axis + 3] = moment[axis];
    }
    return all_finite(wrench_) && all_finite(nullspace_output_.command);
  }

  const std::array<double, 6>& wrench() const { return wrench_; }
  const std::array<double, 7>& nullspace_torque() const { return nullspace_output_.command; }

  Vector7 joint_torques(const Jacobian& jacobian, const Vector7& coriolis) const {
    Vector7 torque = jacobian.transpose() * Eigen::Map<const Vector6>(wrench_.data());
    if (kind_ == field::CartesianFieldKind::kPotentialField)
      torque += Eigen::Map<const Vector7>(nullspace_output_.command.data()) + coriolis;
    return torque;
  }

 private:
  static bool valid_step(const controllers::Result result, const std::uint64_t status) {
    return result.ok() && !controllers::has_status_flag(status, controllers::StatusFlag::kSafeMode);
  }
  struct Arrays {
    std::array<double, 36> stiffness{}, damping{};
    std::array<double, 49> nullspace_stiffness{}, nullspace_damping{}, mass{};
    std::array<double, 42> jacobian{};
    std::array<double, 6> pose_error{}, velocity_error{};
    std::array<double, 7> joint_error{}, joint_velocity{};
  } arrays_;
  FixedFieldOutput<6> orientation_output_;
  FixedFieldOutput<7> nullspace_output_;
  controllers::StaticCartesianImpedanceController orientation_;
  controllers::NullspaceController nullspace_;
  controllers::StaticCartesianImpedanceController::Inputs orientation_inputs_{};
  controllers::NullspaceController::Inputs nullspace_inputs_{};
  std::array<double, 6> wrench_{};
  field::CartesianFieldKind kind_{field::CartesianFieldKind::kPotentialField};
};

inline field::FieldVector3 extract_position(const Eigen::Matrix4d& pose) {
  return {pose(0, 3), pose(1, 3), pose(2, 3)};
}

struct FieldRunMeasurements {
  double minimum_clearance_m{std::numeric_limits<double>::infinity()};
  double final_goal_error_m{0.0};
  double settled_seconds{0.0};
  std::size_t repulsion_samples{0};
};

// Non-RT startup/reporting; libfranka owns scheduling and callback cadence.
inline int run_cartesian_field_example(const int argc,
                                       char** argv,
                                       const field::CartesianFieldKind kind) {
  ProgramArguments arguments;
  if (!parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  print_realtime_mode(arguments.non_rt_requested);
  const field::CartesianFieldParameters parameters;
  if (!field::validate_cartesian_field_parameters(parameters)) return EXIT_FAILURE;
  FieldWorkspace workspace;
  const controllers::Result prepared = workspace.prepare(kind);
  if (!prepared.ok()) {
    std::cerr << "Failed to prepare Cartesian field: " << prepared.message << '\n';
    return EXIT_FAILURE;
  }
  const bool is_apf = kind == field::CartesianFieldKind::kPotentialField;
  const auto initial_joint_configuration = kExtendedInitialJointConfiguration;
  const char* description =
      is_apf ? "Goal offset [0,0.12,0] m; virtual sphere offset [-0.025,0.06,0] m, radius 0.02 m; "
               "influence 0.035 m. Attraction 50 N/m; damping 20 Ns/m; "
               "repulsion gain 0.00005 N m^3; conservative forces ramp for 1 s.\n"
             : "Zero requested torque inside six compliant walls; libfranka compensates gravity. "
               "Box half extents [0.1,0.1,0.1] m. Wall stiffness 1000 N/m, normal damping 40 Ns/m "
               "activated over 0.005 m penetration. No Cartesian force cap or orientation/posture "
               "hold; joint torque magnitude/rate limits still apply.\n";
  std::cout
      << description
      << "Frame: O is the Franka base; geometry uses O axes relative to measured starting EE.\n"
      << (is_apf ? "Orientation hold: 10 Nm/rad and 3 Nm s/rad. No clearance cutoff or Cartesian "
                   "force/moment caps; joint torque magnitude/rate limits still apply.\n"
                 : "Command: wall force mapped through J transpose; no Coriolis feedforward.\n")
      << "Virtual geometry protects the configured EE point only; walls permit penetration.\n";
  if (!franka_example::confirm_execution(
          is_apf ? "Franka goal attraction and obstacle repulsion" : "Franka virtual walls",
          is_apf ? "Command: base-frame field forces with measured orientation and nullspace hold. "
                   "Geometry is captured after initial positioning; no collision planner is used."
                 : "Command: gravity compensation inside the box and uncapped base-frame wall "
                   "forces outside. Geometry is captured after initial positioning; no collision "
                   "planner is used.",
          initial_joint_configuration,
          arguments.run_duration_seconds))
    return EXIT_FAILURE;
  RunSummary summary;
  FieldRunMeasurements measurements;
  try {
    franka::Robot robot(arguments.robot_host, realtime_config(arguments.non_rt_requested));
    set_collision_behavior(robot);
    franka::Model model = robot.loadModel();
    franka_example::move_to_initial_configuration(robot, initial_joint_configuration);
    if (!franka_example::confirm_controller_start()) return EXIT_FAILURE;
    const franka::RobotState initial_state = robot.readOnce();
    if (!state_is_finite(initial_state)) return EXIT_FAILURE;
    const Eigen::Matrix4d reference_pose =
        Eigen::Map<const Eigen::Matrix4d>(initial_state.O_T_EE.data());
    const Vector7 reference_q = Eigen::Map<const Vector7>(initial_state.q.data());
    const field::FieldVector3 reference_position = extract_position(reference_pose);
    robot.control([&](const franka::RobotState& state,
                      const franka::Duration period) -> franka::Torques {
      const auto callback_start = std::chrono::steady_clock::now();
      ++summary.samples;
      const double dt_seconds = period.toSec();
      const auto stop = [&](const char* reason) {
        summary.stop_reason = reason;
        record_callback_time(callback_start, summary);
        return finish_with_previous_torques(state);
      };
      if (!state_is_finite(state) || !std::isfinite(dt_seconds) || dt_seconds < 0.0 ||
          (summary.samples > 1 && dt_seconds == 0.0))
        return stop("invalid state or control period");
      if (dt_seconds == 0.0) return franka::Torques(state.tau_J_d);
      summary.elapsed_seconds += dt_seconds;
      Matrix7 mass = Matrix7::Zero();
      const Jacobian jacobian =
          Eigen::Map<const Jacobian>(model.zeroJacobian(franka::Frame::kEndEffector, state).data());
      Vector7 coriolis = Vector7::Zero();
      if (is_apf) {
        mass = Eigen::Map<const Matrix7>(model.mass(state).data());
        coriolis = Eigen::Map<const Vector7>(model.coriolis(state).data());
      }
      const Eigen::Matrix4d pose = Eigen::Map<const Eigen::Matrix4d>(state.O_T_EE.data());
      const Vector7 q = Eigen::Map<const Vector7>(state.q.data());
      const Vector7 dq = Eigen::Map<const Vector7>(state.dq.data());
      if (!all_finite(mass) || !all_finite(jacobian) || !all_finite(coriolis))
        return stop("non-finite model output");
      const Vector6 twist = jacobian * dq;
      field::CartesianFieldSample sample;
      const bool field_valid = field::evaluate_cartesian_field(kind,
                                                               parameters,
                                                               reference_position,
                                                               extract_position(pose),
                                                               {twist[0], twist[1], twist[2]},
                                                               summary.elapsed_seconds,
                                                               &sample);
      measurements.minimum_clearance_m =
          std::min(measurements.minimum_clearance_m, sample.clearance_m);
      if (!field_valid) return stop("invalid field input or non-finite field output");
      measurements.final_goal_error_m = sample.goal_error_m;
      measurements.repulsion_samples += sample.repulsion_active ? 1 : 0;
      measurements.settled_seconds = sample.goal_error_m <= 0.005 && twist.head<3>().norm() <= 0.01
                                         ? measurements.settled_seconds + dt_seconds
                                         : 0.0;
      const Vector6 pose_error = controllers::examples::cartesian_pose_error(reference_pose, pose);
      if (!workspace.step(pose_error, twist, reference_q - q, dq, mass, jacobian, sample))
        return stop("controller rejected input or entered safe mode");
      const Vector7 desired_torques = workspace.joint_torques(jacobian, coriolis);
      if (!valid_joint_torques(desired_torques)) return stop("non-finite controller output");
      record_callback_time(callback_start, summary);
      if (summary.elapsed_seconds >= arguments.run_duration_seconds) {
        summary.completed = true;
        summary.stop_reason = "configured duration reached";
        return finish_with_filtered_torques(desired_torques, state);
      }
      return filtered_torques(desired_torques, state);
    });
  } catch (const std::exception& error) {
    summary.stop_reason = "control exception";
    std::cerr << "Franka Cartesian field failed: " << error.what() << '\n';
  }
  print_summary(summary);
  if (is_apf)
    std::cout << "Measured goal error [m]: " << measurements.final_goal_error_m
              << "; minimum clearance [m]: " << measurements.minimum_clearance_m
              << "; goal attained: " << (measurements.settled_seconds >= 0.5)
              << "; repulsion samples: " << measurements.repulsion_samples << '\n';
  return summary.completed ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cartesian_field
}  // namespace franka_example
