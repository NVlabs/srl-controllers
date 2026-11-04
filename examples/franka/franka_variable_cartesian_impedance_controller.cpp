/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_variable_cartesian_impedance_controller.cpp
 * @brief Holds pose with rate-limited variable Cartesian impedance.
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

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;

class VariableImpedanceWorkspace {
 public:
  VariableImpedanceWorkspace() {
    const std::array<double, kCartesianDim> initial{2000, 2000, 2000, 100, 100, 100};
    maximum_stiffness_ = {5000, 5000, 5000, 250, 250, 250};
    maximum_stiffness_rate_ = maximum_stiffness_;
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      initial_stiffness_[axis * kCartesianDim + axis] = initial[axis];
      requested_stiffness_[axis * kCartesianDim + axis] = initial[axis];
      damping_factors_[axis] = 0.7;
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

    const controllers::VariableCartesianImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            initial_stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(maximum_stiffness_, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(minimum_stiffness_, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(controllers::read_view(
            maximum_stiffness_rate_, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, kCartesianDim))};
    const controllers::VariableCartesianImpedanceController::Storage storage{
        {controllers::write_view(commanded_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_status(parameter_status_, controllers::Device::kCpu),
         controllers::write_status(status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(
                 last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
             controllers::write_view(safety_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(hold_count_, controllers::Device::kCpu, 1))},
        {controllers::write_view(
             current_stiffness_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
         controllers::write_view(
             previous_stiffness_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
         controllers::write_view(stiffness_initialized_, controllers::Device::kCpu, 1),
         {controllers::write_view(
              previous_cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
          controllers::write_view(
              previous_jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
          controllers::write_view(inertial_initialized_, controllers::Device::kCpu, 1)}},
        {controllers::write_view(stiffness_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(damping_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(inertial_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(stiffness_status_, controllers::Device::kCpu, 1),
         controllers::write_view(damping_status_, controllers::Device::kCpu, 1),
         controllers::write_view(inertial_status_, controllers::Device::kCpu, 1)}};

    mass_inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    controller_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(
            requested_stiffness_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};

    controllers::Result result = mass_controller_.prepare(config, mass_parameters, mass_storage);
    if (!result.ok()) {
      return result;
    }
    result = controller_.prepare(config, parameters, storage);
    if (!result.ok()) {
      return result;
    }
    result = mass_controller_.reset();
    if (!result.ok()) {
      return result;
    }
    return controller_.reset();
  }

  bool step(const Eigen::Vector<double, kCartesianDim>& pose_error,
            const Eigen::Vector<double, kCartesianDim>& velocity_error,
            const Eigen::Vector<double, kJointCount>& joint_velocity,
            const Eigen::Matrix<double, kJointCount, kJointCount>& mass,
            const Eigen::Matrix<double, kCartesianDim, kJointCount>& jacobian,
            const double requested_z_stiffness,
            const double dt_seconds) {
    requested_stiffness_[2 * kCartesianDim + 2] = requested_z_stiffness;
    for (std::size_t row = 0; row < kJointCount; ++row) {
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
    status_[0] = 0;
    const controllers::Result mass_result = mass_controller_.step(mass_inputs_);
    const controllers::Result result = controller_.step(controller_inputs_);
    return mass_result.ok() && result.ok() &&
           !controllers::has_status_flag(mass_status_[0], controllers::StatusFlag::kSafeMode) &&
           !controllers::has_status_flag(status_[0], controllers::StatusFlag::kSafeMode);
  }

  const std::array<double, kCartesianDim>& wrench() const { return commanded_wrench_; }

 private:
  std::array<double, kCartesianDim * kCartesianDim> initial_stiffness_{};
  std::array<double, kCartesianDim> maximum_stiffness_{};
  std::array<double, kCartesianDim> minimum_stiffness_{};
  std::array<double, kCartesianDim> maximum_stiffness_rate_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kCartesianDim * kCartesianDim> requested_stiffness_{};
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
  std::array<double, kJointCount> joint_velocity_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> commanded_wrench_{};
  std::array<double, kCartesianDim * kCartesianDim> current_stiffness_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_stiffness_{};
  std::array<std::uint8_t, 1> stiffness_initialized_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_cartesian_mass_{};
  std::array<double, kCartesianDim * kJointCount> previous_jacobian_{};
  std::array<std::uint8_t, 1> inertial_initialized_{};
  std::array<double, kCartesianDim> stiffness_wrench_{};
  std::array<double, kCartesianDim> damping_wrench_{};
  std::array<double, kCartesianDim> inertial_wrench_{};
  std::array<std::uint64_t, 1> stiffness_status_{};
  std::array<std::uint64_t, 1> damping_status_{};
  std::array<std::uint64_t, 1> inertial_status_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> safety_valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  controllers::VariableCartesianImpedanceController::Inputs controller_inputs_{};
  controllers::OperationalSpaceMass mass_controller_;
  controllers::VariableCartesianImpedanceController controller_;
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

  VariableImpedanceWorkspace controller;
  const controllers::Result prepare_result = controller.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare the variable Cartesian impedance controller: "
              << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }

  constexpr auto kInitialJointPosition = franka_example::kExtendedInitialJointConfiguration;
  if (!franka_example::confirm_execution(
          "Franka variable Cartesian impedance hold",
          "Command: hold the measured start transform O_T_EE while base-frame z stiffness varies "
          "sinusoidally from approximately 0 to 2000 N/m at 0.1 Hz.\n"
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
      const double z_stiffness =
          1000.0 * (std::sin(2.0 * kPi * 0.1 * summary.elapsed_seconds) + 1.001);

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

      if (!franka_example::all_finite(mass) || !franka_example::all_finite(jacobian) ||
          !franka_example::all_finite(coriolis)) {
        summary.stop_reason = "non-finite model input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      const Eigen::Vector<double, 6> pose_error =
          controllers::examples::cartesian_pose_error(desired_pose, pose);
      if (!controller.step(
              pose_error, -jacobian * dq, dq, mass, jacobian, z_stiffness, dt_seconds)) {
        summary.stop_reason = "controller entered safe mode or rejected an input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
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
  return summary.completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
