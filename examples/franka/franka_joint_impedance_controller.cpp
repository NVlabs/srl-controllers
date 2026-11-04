/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_joint_impedance_controller.cpp
 * @brief Holds the measured joint position with the batch-one controller API.
 */

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <eigen3/Eigen/Dense>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "./franka_utils.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

int main(int argc, char** argv) {
  franka_example::ProgramArguments arguments;
  if (!franka_example::parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  franka_example::print_realtime_mode(arguments.non_rt_requested);

  namespace controllers = nvidia::srl::controllers;
  constexpr std::size_t kJointCount = 7;
  std::array<double, kJointCount * kJointCount> stiffness{};
  std::array<double, kJointCount> damping_factors{};
  std::array<double, kJointCount> position_error{};
  std::array<double, kJointCount> velocity_error{};
  std::array<double, kJointCount> desired_acceleration{};
  std::array<double, kJointCount * kJointCount> mass_row_major{};
  std::array<double, kJointCount> controller_torque{};
  std::array<double, kJointCount> last_valid_torque{};
  std::array<std::uint8_t, 1> safety_valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    constexpr std::array<double, kJointCount> kStiffness{1500, 1200, 800, 500, 300, 200, 50};
    stiffness[joint * kJointCount + joint] = kStiffness[joint];
    damping_factors[joint] = 0.7;
  }

  controllers::JointImpedanceController controller;
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::per_batch(controllers::read_view(
          stiffness, controllers::Device::kCpu, 1, kJointCount, kJointCount)),
      controllers::shared(controllers::read_view(damping_factors, controllers::Device::kCpu, 7))};
  const controllers::JointImpedanceController::Storage controller_storage{
      controllers::write_view(controller_torque, controllers::Device::kCpu, 1, kJointCount),
      controllers::write_status(parameter_status, controllers::Device::kCpu),
      controllers::write_status(status, controllers::Device::kCpu),
      controllers::make_safety_state(
          controllers::write_view(last_valid_torque, controllers::Device::kCpu, 1, kJointCount),
          controllers::write_view(safety_valid, controllers::Device::kCpu, 1),
          controllers::write_view(hold_count, controllers::Device::kCpu, 1))};
  const controllers::JointImpedanceController::Inputs inputs{
      controllers::read_view(position_error, controllers::Device::kCpu, 1, kJointCount),
      controllers::read_view(velocity_error, controllers::Device::kCpu, 1, kJointCount),
      controllers::read_view(desired_acceleration, controllers::Device::kCpu, 1, kJointCount),
      controllers::read_view(
          mass_row_major, controllers::Device::kCpu, 1, kJointCount, kJointCount)};
  const controllers::Result prepare_result =
      controller.prepare({controllers::Device::kCpu, controllers::DType::kFloat64, 1, 7, {}},
                         parameters,
                         controller_storage);
  if (!prepare_result.ok()) {
    std::cerr << "Joint impedance preparation failed: " << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
  const controllers::Result reset_result = controller.reset();
  if (!reset_result.ok()) {
    std::cerr << "Joint impedance reset failed: " << reset_result.message << '\n';
    return EXIT_FAILURE;
  }

  constexpr auto kInitialJointPosition = franka_example::kInitialJointConfiguration;
  if (!franka_example::confirm_execution(
          "Franka joint impedance hold",
          "Command: hold the measured start angles (rad) for all seven joints.\n"
          "Feedback: measured q and dq; model terms: mass matrix and Coriolis torque.",
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

      const Eigen::Vector<double, 7> q = Eigen::Map<const Eigen::Vector<double, 7>>(state.q.data());
      const Eigen::Vector<double, 7> dq =
          Eigen::Map<const Eigen::Vector<double, 7>>(state.dq.data());
      const Eigen::Matrix<double, 7, 7> mass =
          Eigen::Map<const Eigen::Matrix<double, 7, 7>>(model.mass(state).data());
      const Eigen::Vector<double, 7> coriolis =
          Eigen::Map<const Eigen::Vector<double, 7>>(model.coriolis(state).data());

      if (!franka_example::all_finite(mass) || !franka_example::all_finite(coriolis)) {
        summary.stop_reason = "non-finite model output";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        position_error[joint] =
            desired_q[static_cast<Eigen::Index>(joint)] - q[static_cast<Eigen::Index>(joint)];
        velocity_error[joint] = -dq[static_cast<Eigen::Index>(joint)];
        for (std::size_t column = 0; column < kJointCount; ++column) {
          mass_row_major[joint * kJointCount + column] =
              mass(static_cast<Eigen::Index>(joint), static_cast<Eigen::Index>(column));
        }
      }
      const controllers::Result controller_result = controller.step(inputs);
      Eigen::Vector<double, 7> desired_torques;
      for (std::size_t joint = 0; joint < kJointCount; ++joint) {
        desired_torques[static_cast<Eigen::Index>(joint)] =
            controller_torque[joint] + coriolis[static_cast<Eigen::Index>(joint)];
      }
      if (!controller_result.ok() ||
          controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode)) {
        summary.stop_reason = "controller entered safe mode";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
      if (!franka_example::valid_joint_torques(desired_torques)) {
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
