// NOLINTNEXTLINE(whitespace/line_length)
// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

/**
 * @file franka_cartesian_force_impedance_controller_with_tank.cpp
 * @brief Base-frame force/impedance control with a force energy tank.
 */

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "./franka_force_tank.hpp"
#include "./franka_utils.hpp"
#include "examples/cartesian_math.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace {

namespace controllers = nvidia::srl::controllers;

using franka_example::force_tank::ForceTankWorkspace;
using franka_example::force_tank::kForceRampSeconds;
using franka_example::force_tank::kTargetForceNewtons;

}  // namespace

int main(int argc, char** argv) {
  franka_example::ProgramArguments arguments;
  if (!franka_example::parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  franka_example::print_realtime_mode(arguments.non_rt_requested);

  ForceTankWorkspace controller;
  const controllers::Result prepare_result = controller.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare the base-frame force/impedance with energy tank controller: "
              << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }

  constexpr auto kInitialJointPosition = franka_example::kExtendedInitialJointConfiguration;
  if (!franka_example::confirm_execution(
          "Franka base-frame force/impedance with energy tank",
          "Setup: place the tool in unloaded contact with a compliant surface whose normal is "
          "the Franka base-frame -z axis. Verify the configured stiffness frame and payload "
          "first.\n"
          "Command: hold measured O_T_EE and ramp O-frame z force from 0 to -10 N over 2 s. "
          "Force shaping is disabled. The force contribution uses a 1 J energy tank; this is not a "
          "force or travel limit.",
          kInitialJointPosition,
          arguments.run_duration_seconds)) {
    return EXIT_FAILURE;
  }

  franka_example::RunSummary summary;
  double minimum_energy_joules = controller.energy_joules();
  double minimum_scale = 1.0;
  std::size_t limiting_samples = 0;
  double final_measured_force_newtons = 0.0;
  double maximum_absolute_force_newtons = 0.0;
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
      const Eigen::Vector<double, 7> dq =
          Eigen::Map<const Eigen::Vector<double, 7>>(state.dq.data());
      const Eigen::Vector<double, 6> external_wrench =
          Eigen::Map<const Eigen::Vector<double, 6>>(state.O_F_ext_hat_K.data());
      const Eigen::Vector<double, 6> pose_error =
          controllers::examples::cartesian_pose_error(desired_pose, pose);

      final_measured_force_newtons = external_wrench(2);
      maximum_absolute_force_newtons =
          std::max(maximum_absolute_force_newtons, std::abs(final_measured_force_newtons));

      if (!franka_example::all_finite(mass) || !franka_example::all_finite(jacobian) ||
          !franka_example::all_finite(coriolis) || !franka_example::all_finite(external_wrench) ||
          !pose_error.allFinite()) {
        summary.stop_reason = "invalid input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      Eigen::Vector<double, 6> desired_wrench = Eigen::Vector<double, 6>::Zero();
      desired_wrench(2) =
          std::min(1.0, summary.elapsed_seconds / kForceRampSeconds) * kTargetForceNewtons;
      if (!controller.step(
              pose_error, dq, mass, jacobian, external_wrench, desired_wrench, dt_seconds)) {
        summary.stop_reason = "controller entered safe mode or rejected an input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
      minimum_energy_joules = std::min(minimum_energy_joules, controller.energy_joules());
      minimum_scale = std::min(minimum_scale, controller.scale());
      limiting_samples += controller.scale() < 1.0;
      const Eigen::Map<const Eigen::Vector<double, 6>> wrench(controller.wrench().data());
      Eigen::Vector<double, 7> desired_torques = jacobian.transpose() * wrench + coriolis;

      if (!wrench.allFinite() || !franka_example::valid_joint_torques(desired_torques)) {
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
  std::cout << "Force tank: final_J=" << controller.energy_joules()
            << ", min_J=" << minimum_energy_joules << ", min_scale=" << minimum_scale
            << ", limiting_samples=" << limiting_samples << '\n';
  std::cout << std::fixed << std::setprecision(3)
            << "Observed base-frame z force: final_N=" << final_measured_force_newtons
            << ", max_abs_N=" << maximum_absolute_force_newtons
            << ". These measurements do not establish physical task success.\n";
  return summary.completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
