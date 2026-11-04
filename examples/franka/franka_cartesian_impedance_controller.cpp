/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_cartesian_impedance_controller.cpp
 * @brief Holds the measured end-effector pose with the controller API.
 */

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <eigen3/Eigen/Dense>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "./franka_utils.hpp"
#include "examples/cartesian_math.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace {

namespace controllers = nvidia::srl::controllers;

constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;

class CartesianHoldWorkspace {
 public:
  CartesianHoldWorkspace() {
    const std::array<double, kCartesianDim> cartesian_diagonal{2000, 2000, 2000, 100, 100, 100};
    const std::array<double, kJointCount> nullspace_stiffness_diagonal{120, 80, 60, 40, 30, 20, 10};
    const std::array<double, kJointCount> nullspace_damping_diagonal{12, 8, 6, 4, 3, 2, 1};
    for (std::size_t index = 0; index < kCartesianDim; ++index) {
      impedance_stiffness_[index * kCartesianDim + index] = cartesian_diagonal[index];
      impedance_damping_[index] = 0.7;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      nullspace_stiffness_[index * kJointCount + index] = nullspace_stiffness_diagonal[index];
      nullspace_damping_[index * kJointCount + index] = nullspace_damping_diagonal[index];
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               kJointCount,
                                               {},
                                               controllers::CartesianFrame::kRobotBase};

    const controllers::OperationalSpaceMass::Parameters mass_parameters{
        controllers::shared(controllers::read_view(regularization_, controllers::Device::kCpu, 1)),
        {}};
    const controllers::OperationalSpaceMass::Storage mass_storage{
        controllers::write_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::write_status(mass_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(mass_status_, controllers::Device::kCpu),
        {controllers::write_view(
             joint_factor_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
         controllers::write_view(
             solve_rhs_, controllers::Device::kCpu, 1, kJointCount, kCartesianDim),
         controllers::write_view(
             cartesian_inverse_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim)}};

    const controllers::CartesianImpedanceController::Parameters impedance_parameters{
        controllers::shared(controllers::read_view(
            impedance_stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(impedance_damping_, controllers::Device::kCpu, kCartesianDim))};
    const controllers::CartesianImpedanceController::Storage impedance_storage{
        controllers::write_view(commanded_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_status(impedance_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(impedance_status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(impedance_safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(impedance_hold_count_, controllers::Device::kCpu, 1)),
        {controllers::write_view(
             previous_cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
         controllers::write_view(
             previous_jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
         controllers::write_view(impedance_initialized_, controllers::Device::kCpu, 1)}};

    const controllers::NullspaceController::Parameters nullspace_parameters{
        controllers::shared(controllers::read_view(
            nullspace_stiffness_, controllers::Device::kCpu, kJointCount, kJointCount)),
        controllers::shared(controllers::read_view(
            nullspace_damping_, controllers::Device::kCpu, kJointCount, kJointCount))};
    const controllers::NullspaceController::Storage nullspace_storage{
        controllers::write_view(nullspace_torque_, controllers::Device::kCpu, 1, kJointCount),
        controllers::write_status(nullspace_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(nullspace_status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_nullspace_torque_, controllers::Device::kCpu, 1, kJointCount),
            controllers::write_view(nullspace_safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(nullspace_hold_count_, controllers::Device::kCpu, 1))};

    mass_inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    impedance_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    nullspace_inputs_ = {
        controllers::read_view(joint_position_error_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};

    controllers::Result result = mass_controller_.prepare(config, mass_parameters, mass_storage);
    if (!result.ok()) {
      return result;
    }
    result = impedance_controller_.prepare(config, impedance_parameters, impedance_storage);
    if (!result.ok()) {
      return result;
    }
    result = nullspace_controller_.prepare(config, nullspace_parameters, nullspace_storage);
    if (!result.ok()) {
      return result;
    }
    result = mass_controller_.reset();
    if (!result.ok()) {
      return result;
    }
    result = impedance_controller_.reset();
    if (!result.ok()) {
      return result;
    }
    return nullspace_controller_.reset();
  }

  bool step(const Eigen::Vector<double, kCartesianDim>& pose_error,
            const Eigen::Vector<double, kCartesianDim>& velocity_error,
            const Eigen::Vector<double, kJointCount>& joint_position_error,
            const Eigen::Vector<double, kJointCount>& joint_velocity,
            const Eigen::Matrix<double, kJointCount, kJointCount>& mass,
            const Eigen::Matrix<double, kCartesianDim, kJointCount>& jacobian,
            const double dt_seconds) {
    for (std::size_t row = 0; row < kJointCount; ++row) {
      joint_position_error_[row] = joint_position_error[static_cast<Eigen::Index>(row)];
      joint_velocity_[row] = joint_velocity[static_cast<Eigen::Index>(row)];
      for (std::size_t column = 0; column < kJointCount; ++column) {
        joint_mass_[row * kJointCount + column] =
            mass(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      pose_error_[row] = pose_error[static_cast<Eigen::Index>(row)];
      velocity_error_[row] = velocity_error[static_cast<Eigen::Index>(row)];
      for (std::size_t column = 0; column < kJointCount; ++column) {
        jacobian_[row * kJointCount + column] =
            jacobian(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
    dt_[0] = dt_seconds;
    mass_status_[0] = 0;
    impedance_status_[0] = 0;
    nullspace_status_[0] = 0;
    const controllers::Result mass_result = mass_controller_.step(mass_inputs_);
    const controllers::Result impedance_result = impedance_controller_.step(impedance_inputs_);
    const controllers::Result nullspace_result = nullspace_controller_.step(nullspace_inputs_);
    return mass_result.ok() && impedance_result.ok() && nullspace_result.ok() &&
           !controllers::has_status_flag(mass_status_[0], controllers::StatusFlag::kSafeMode) &&
           !controllers::has_status_flag(impedance_status_[0],
                                         controllers::StatusFlag::kSafeMode) &&
           !controllers::has_status_flag(nullspace_status_[0], controllers::StatusFlag::kSafeMode);
  }

  const std::array<double, kCartesianDim>& wrench() const { return commanded_wrench_; }
  const std::array<double, kJointCount>& nullspace_torque() const { return nullspace_torque_; }

 private:
  std::array<double, kCartesianDim * kCartesianDim> impedance_stiffness_{};
  std::array<double, kCartesianDim> impedance_damping_{};
  std::array<double, kJointCount * kJointCount> nullspace_stiffness_{};
  std::array<double, kJointCount * kJointCount> nullspace_damping_{};
  std::array<double, 1> regularization_{1e-6};
  std::array<double, kJointCount * kJointCount> joint_mass_{};
  std::array<double, kCartesianDim * kJointCount> jacobian_{};
  std::array<double, kCartesianDim * kCartesianDim> cartesian_mass_{};
  std::array<double, kJointCount * kJointCount> joint_factor_{};
  std::array<double, kJointCount * kCartesianDim> solve_rhs_{};
  std::array<double, kCartesianDim * kCartesianDim> cartesian_inverse_{};
  std::array<double, kCartesianDim> pose_error_{};
  std::array<double, kCartesianDim> velocity_error_{};
  std::array<double, kCartesianDim> desired_acceleration_{};
  std::array<double, kJointCount> joint_position_error_{};
  std::array<double, kJointCount> joint_velocity_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> commanded_wrench_{};
  std::array<double, kJointCount> nullspace_torque_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_cartesian_mass_{};
  std::array<double, kCartesianDim * kJointCount> previous_jacobian_{};
  std::array<std::uint8_t, 1> impedance_initialized_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> impedance_safety_valid_{};
  std::array<std::uint64_t, 1> impedance_hold_count_{};
  std::array<double, kJointCount> last_valid_nullspace_torque_{};
  std::array<std::uint8_t, 1> nullspace_safety_valid_{};
  std::array<std::uint64_t, 1> nullspace_hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> impedance_parameter_status_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  std::array<std::uint64_t, 1> nullspace_parameter_status_{};
  std::array<std::uint64_t, 1> nullspace_status_{};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  controllers::CartesianImpedanceController::Inputs impedance_inputs_{};
  controllers::NullspaceController::Inputs nullspace_inputs_{};
  controllers::OperationalSpaceMass mass_controller_;
  controllers::CartesianImpedanceController impedance_controller_;
  controllers::NullspaceController nullspace_controller_;
};

}  // namespace

int main(int argc, char** argv) {
  franka_example::ProgramArguments arguments;
  if (!franka_example::parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  franka_example::print_realtime_mode(arguments.non_rt_requested);

  CartesianHoldWorkspace controller;
  const controllers::Result prepare_result = controller.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare the Cartesian impedance controllers: " << prepare_result.message
              << '\n';
    return EXIT_FAILURE;
  }

  constexpr auto kInitialJointPosition = franka_example::kInitialJointConfiguration;
  if (!franka_example::confirm_execution(
          "Franka Cartesian impedance hold",
          "Command: hold the measured start transform O_T_EE (translation in m, rotation matrix), "
          "with the measured start joint angles as the nullspace target.\n"
          "Frame: O is the Franka base frame; EE is the configured end-effector frame.",
          kInitialJointPosition, arguments.run_duration_seconds)) {
    return EXIT_FAILURE;
  }

  franka_example::RunSummary summary;
  try {
    franka::Robot robot(arguments.robot_host,
                        franka_example::realtime_config(arguments.non_rt_requested));
    franka_example::set_collision_behavior(robot);
    franka::Model model = robot.loadModel();
    franka_example::move_to_initial_configuration(robot, kInitialJointPosition);
    if (!franka_example::confirm_controller_start()) {
      return EXIT_FAILURE;
    }
    const franka::RobotState initial_state = robot.readOnce();
    if (!franka_example::state_is_finite(initial_state)) {
      std::cerr << "Initial robot state contains a non-finite value.\n";
      return EXIT_FAILURE;
    }
    const Eigen::Matrix4d desired_pose =
        Eigen::Map<const Eigen::Matrix<double, 4, 4>>(initial_state.O_T_EE.data());
    const Eigen::Vector<double, 7> desired_q =
        Eigen::Map<const Eigen::Vector<double, 7>>(initial_state.q.data());

    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::Torques {
      const auto callback_start = std::chrono::steady_clock::now();
      ++summary.samples;
      const double dt_seconds = period.toSec();
      if (!franka_example::state_is_finite(state) || !std::isfinite(dt_seconds) ||
          dt_seconds < 0.0 || (summary.samples > 1 && dt_seconds == 0.0)) {
        summary.stop_reason = "invalid state or control period";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
      if (dt_seconds == 0.0) {
        franka_example::record_callback_time(callback_start, summary);
        return franka::Torques(state.tau_J_d);
      }
      summary.elapsed_seconds += dt_seconds;

      const Eigen::Matrix<double, 7, 7> mass =
          Eigen::Map<const Eigen::Matrix<double, 7, 7>>(model.mass(state).data());
      const Eigen::Matrix<double, 6, 7> jacobian = Eigen::Map<const Eigen::Matrix<double, 6, 7>>(
          model.zeroJacobian(franka::Frame::kEndEffector, state).data());
      const Eigen::Vector<double, 7> coriolis =
          Eigen::Map<const Eigen::Vector<double, 7>>(model.coriolis(state).data());
      const Eigen::Matrix4d pose =
          Eigen::Map<const Eigen::Matrix<double, 4, 4>>(state.O_T_EE.data());
      const Eigen::Vector<double, 7> q = Eigen::Map<const Eigen::Vector<double, 7>>(state.q.data());
      const Eigen::Vector<double, 7> dq =
          Eigen::Map<const Eigen::Vector<double, 7>>(state.dq.data());

      if (!franka_example::all_finite(mass) || !franka_example::all_finite(jacobian) ||
          !franka_example::all_finite(coriolis)) {
        summary.stop_reason = "non-finite model output";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      const Eigen::Vector<double, 6> pose_error =
          controllers::examples::cartesian_pose_error(desired_pose, pose);
      if (!controller.step(
              pose_error, -jacobian * dq, desired_q - q, dq, mass, jacobian, dt_seconds)) {
        summary.stop_reason = "controller entered safe mode or rejected an input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
      const Eigen::Map<const Eigen::Vector<double, 6>> wrench(controller.wrench().data());
      const Eigen::Map<const Eigen::Vector<double, 7>> nullspace_torque(
          controller.nullspace_torque().data());
      Eigen::Vector<double, 7> desired_torques = jacobian.transpose() * wrench;
      desired_torques += nullspace_torque + coriolis;

      if (!franka_example::all_finite(wrench) ||
          !franka_example::valid_joint_torques(desired_torques)) {
        summary.stop_reason = "non-finite controller output";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      franka_example::record_callback_time(callback_start, summary);
      if (summary.elapsed_seconds >= arguments.run_duration_seconds) {
        summary.completed = true;
        summary.stop_reason = "configured duration reached";
        return franka_example::finish_with_filtered_torques(desired_torques, state);
      }
      return franka_example::filtered_torques(desired_torques, state);
    });
  } catch (const franka::Exception& error) {
    summary.stop_reason = "libfranka exception";
    std::cerr << "Franka control failed: " << error.what() << '\n';
  } catch (const std::exception& error) {
    summary.stop_reason = "controller exception";
    std::cerr << "Controller failed: " << error.what() << '\n';
  }

  franka_example::print_summary(summary);
  return summary.completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
