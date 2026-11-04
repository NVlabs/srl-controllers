/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_cartesian_admittance_controller.cpp
 * @brief Runs filtered Cartesian admittance through FER pose or FR3 torque control.
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

#include "./franka_admittance.hpp"
#include "./franka_utils.hpp"
#include "examples/cartesian_math.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace controllers = nvidia::srl::controllers;

int main(int argc, char** argv) {
  franka_example::ProgramArguments arguments;
  if (!franka_example::parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  franka_example::print_realtime_mode(arguments.non_rt_requested);

  franka_example::AdmittanceWorkspace controller;
  const controllers::Result prepare_result = controller.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare the Cartesian admittance controller: " << prepare_result.message
              << '\n';
    return EXIT_FAILURE;
  }
#if !defined(NVIDIA_SRL_FRANKA_FER)
  franka_example::CartesianImpedanceWorkspace impedance;
  const controllers::Result impedance_prepare_result = impedance.prepare();
  if (!impedance_prepare_result.ok()) {
    std::cerr << "Failed to prepare the Cartesian impedance controller: "
              << impedance_prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
#endif
  constexpr auto kInitialJointPosition = franka_example::kExtendedInitialJointConfiguration;
#if defined(NVIDIA_SRL_FRANKA_FER)
  constexpr const char* kCommandDescription =
      "Command: 7 Hz low-pass filtered base-frame wrench drives admittance pose offsets through "
      "the Franka Cartesian position interface.\n"
      "The first pose command matches the last commanded O_T_EE_c.\n"
      "Frame: O is the Franka base frame; wrench is O_F_ext_hat_K in N and Nm.";
#else
  constexpr const char* kCommandDescription =
      "Command: 7 Hz low-pass filtered base-frame wrench drives admittance pose, twist, "
      "and acceleration targets.\n"
      "A full Cartesian impedance controller converts those targets to joint torques.\n"
      "Frame: O is the Franka base frame; wrench is O_F_ext_hat_K in N and Nm.";
#endif
  if (!franka_example::confirm_execution(
          "Franka Cartesian admittance",
          kCommandDescription,
          kInitialJointPosition,
          arguments.run_duration_seconds)) {
    return EXIT_FAILURE;
  }

  franka_example::RunSummary summary;
  try {
    franka::Robot robot(arguments.robot_host,
                        franka_example::realtime_config(arguments.non_rt_requested));
    franka_example::set_collision_behavior(robot);
#if !defined(NVIDIA_SRL_FRANKA_FER)
    franka::Model model = robot.loadModel();
#endif
    franka_example::move_to_initial_configuration(robot, kInitialJointPosition);
    if (!franka_example::confirm_controller_start()) {
      return EXIT_FAILURE;
    }
    const franka::RobotState initial_state = robot.readOnce();
    if (!franka_example::state_is_finite(initial_state)) {
      std::cerr << "Initial robot state contains a non-finite value.\n";
      return EXIT_FAILURE;
    }
#if defined(NVIDIA_SRL_FRANKA_FER)
    if (!franka_example::all_finite(initial_state.O_T_EE_c)) {
      std::cerr << "Initial commanded Cartesian pose contains a non-finite value.\n";
      return EXIT_FAILURE;
    }
    std::array<double, 16> last_commanded_pose = initial_state.O_T_EE_c;
    Eigen::Matrix4d reference_pose = Eigen::Matrix4d::Identity();
    bool reference_initialized = false;
    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose {
          const auto callback_start = std::chrono::steady_clock::now();
          ++summary.samples;
          const double dt_seconds = period.toSec();
          auto finish_with_previous = [&]() -> franka::CartesianPose {
            franka_example::record_callback_time(callback_start, summary);
            const auto& previous_pose = franka_example::all_finite(state.O_T_EE_c)
                                            ? state.O_T_EE_c
                                            : last_commanded_pose;
            return franka::MotionFinished(franka::CartesianPose(previous_pose));
          };

          if (!franka_example::state_is_finite(state) ||
              !franka_example::all_finite(state.O_T_EE_c) || !std::isfinite(dt_seconds) ||
              dt_seconds < 0.0 || (summary.samples > 1 && dt_seconds == 0.0)) {
            summary.stop_reason = "invalid state or control period";
            return finish_with_previous();
          }
          if (!reference_initialized) {
            reference_pose =
                Eigen::Map<const Eigen::Matrix<double, 4, 4>>(state.O_T_EE_c.data());
            last_commanded_pose = state.O_T_EE_c;
            reference_initialized = true;
            franka_example::record_callback_time(callback_start, summary);
            return franka::CartesianPose(state.O_T_EE_c);
          }
          summary.elapsed_seconds += dt_seconds;
          const Eigen::Vector<double, 6> external_wrench =
              Eigen::Map<const Eigen::Vector<double, 6>>(state.O_F_ext_hat_K.data());
          if (!controller.step_admittance(external_wrench, dt_seconds) ||
              !franka_example::all_finite(controller.acceleration()) ||
              !franka_example::all_finite(controller.twist()) ||
              !franka_example::all_finite(controller.pose_offset())) {
            summary.stop_reason = "invalid admittance input or output";
            return finish_with_previous();
          }

          const Eigen::Map<const Eigen::Vector<double, 6>> offset(controller.pose_offset().data());
          const Eigen::Matrix4d desired_pose =
              controllers::examples::apply_cartesian_pose_offset(reference_pose, offset);
          if (!desired_pose.allFinite()) {
            summary.stop_reason = "invalid desired pose";
            return finish_with_previous();
          }
          Eigen::Map<Eigen::Matrix<double, 4, 4>>(last_commanded_pose.data()) = desired_pose;
          const franka::CartesianPose command(last_commanded_pose);
          franka_example::record_callback_time(callback_start, summary);
          if (summary.elapsed_seconds >= arguments.run_duration_seconds) {
            summary.completed = true;
            summary.stop_reason = "configured duration reached";
            return franka::MotionFinished(command);
          }
          return command;
        },
        franka::ControllerMode::kJointImpedance,
        false);
#else
    const Eigen::Matrix4d initial_pose =
        Eigen::Map<const Eigen::Matrix<double, 4, 4>>(initial_state.O_T_EE.data());
    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques {
          const auto callback_start = std::chrono::steady_clock::now();
          ++summary.samples;
          const double dt_seconds = period.toSec();
          auto finish_with_previous = [&]() -> franka::Torques {
            franka_example::record_callback_time(callback_start, summary);
            return franka_example::finish_with_previous_torques(state);
          };

          if (!franka_example::state_is_finite(state) || !std::isfinite(dt_seconds) ||
              dt_seconds < 0.0 || (summary.samples > 1 && dt_seconds == 0.0)) {
            summary.stop_reason = "invalid state or control period";
            return finish_with_previous();
          }
          if (dt_seconds == 0.0) {
            franka_example::record_callback_time(callback_start, summary);
            return franka::Torques(state.tau_J_d);
          }
          summary.elapsed_seconds += dt_seconds;

          const Eigen::Matrix<double, 7, 7> mass =
              Eigen::Map<const Eigen::Matrix<double, 7, 7>>(model.mass(state).data());
          const Eigen::Matrix<double, 6, 7> jacobian =
              Eigen::Map<const Eigen::Matrix<double, 6, 7>>(
                  model.zeroJacobian(franka::Frame::kEndEffector, state).data());
          const Eigen::Vector<double, 7> coriolis =
              Eigen::Map<const Eigen::Vector<double, 7>>(model.coriolis(state).data());
          const Eigen::Matrix4d pose =
              Eigen::Map<const Eigen::Matrix<double, 4, 4>>(state.O_T_EE.data());
          const Eigen::Vector<double, 7> dq =
              Eigen::Map<const Eigen::Vector<double, 7>>(state.dq.data());
          const Eigen::Vector<double, 6> external_wrench =
              Eigen::Map<const Eigen::Vector<double, 6>>(state.O_F_ext_hat_K.data());
          if (!franka_example::all_finite(mass) || !franka_example::all_finite(jacobian) ||
              !franka_example::all_finite(coriolis)) {
            summary.stop_reason = "non-finite model output";
            return finish_with_previous();
          }

          if (!controller.step_admittance(external_wrench, dt_seconds)) {
            summary.stop_reason = "invalid wrench input or admittance safe mode";
            return finish_with_previous();
          }

          if (!franka_example::all_finite(controller.acceleration()) ||
              !franka_example::all_finite(controller.twist()) ||
              !franka_example::all_finite(controller.pose_offset())) {
            summary.stop_reason = "invalid admittance target";
            return finish_with_previous();
          }

          const Eigen::Map<const Eigen::Vector<double, 6>> offset(controller.pose_offset().data());
          const Eigen::Map<const Eigen::Vector<double, 6>> desired_twist(controller.twist().data());
          const Eigen::Map<const Eigen::Vector<double, 6>> desired_acceleration(
              controller.acceleration().data());
          const Eigen::Matrix4d desired_pose =
              controllers::examples::apply_cartesian_pose_offset(initial_pose, offset);
          if (!desired_pose.allFinite()) {
            summary.stop_reason = "invalid desired pose";
            return finish_with_previous();
          }
          const Eigen::Vector<double, 6> pose_error =
              controllers::examples::cartesian_pose_error(desired_pose, pose);
          const Eigen::Vector<double, 6> velocity_error = desired_twist - jacobian * dq;
          if (!impedance.step(pose_error, velocity_error, desired_acceleration,
                              dq, mass, jacobian, dt_seconds)) {
            summary.stop_reason = "Cartesian impedance safe mode or invalid input";
            return finish_with_previous();
          }
          const Eigen::Map<const Eigen::Vector<double, 6>> wrench(impedance.wrench().data());
          const Eigen::Vector<double, 7> desired_torques = jacobian.transpose() * wrench + coriolis;
          if (!franka_example::all_finite(wrench) ||
              !franka_example::valid_joint_torques(desired_torques)) {
            summary.stop_reason = "non-finite torque output";
            return finish_with_previous();
          }
          const franka::Torques command = franka_example::filtered_torques(desired_torques, state);
          franka_example::record_callback_time(callback_start, summary);

          if (summary.elapsed_seconds >= arguments.run_duration_seconds) {
            summary.completed = true;
            summary.stop_reason = "configured duration reached";
            return franka::MotionFinished(command);
          }
          return command;
        });
#endif
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
