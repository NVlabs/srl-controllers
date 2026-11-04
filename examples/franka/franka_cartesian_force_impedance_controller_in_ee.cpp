/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_cartesian_force_impedance_controller_in_ee.cpp
 * @brief Force/impedance control in the Franka end-effector frame.
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

#include "./franka_utils.hpp"
#include "examples/cartesian_math.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace {

namespace controllers = nvidia::srl::controllers;

constexpr double kTargetForceNewtons = 10.0;
constexpr double kForceRampSeconds = 2.0;
constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;

class ForceImpedanceWorkspace {
 public:
  ForceImpedanceWorkspace() {
    const std::array<double, kCartesianDim> stiffness{5000, 5000, 20, 250, 250, 250};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = stiffness[axis];
      damping_factors_[axis] = 0.7;
      maximum_distance_[axis] = 0.1;
      shaping_distance_[axis] = 0.06;
    }
    proportional_gain_[2] = 1.0;
    differentiator_gain_[2] = 0.01;
  }

  controllers::Result prepare() {
    controllers::CompositeControllerConfig config;
    config.device = controllers::Device::kCpu;
    config.dtype = controllers::DType::kFloat64;
    config.batch_size = 1;
    config.joint_dim = kJointCount;
    config.cartesian_frame = controllers::CartesianFrame::kEndEffector;
    config.use_shaping_function = true;

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

    const controllers::CartesianUnifiedForceImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, kCartesianDim)),
        {controllers::shared(
             controllers::read_view(proportional_gain_, controllers::Device::kCpu, kCartesianDim)),
         controllers::shared(controllers::read_view(
             differentiator_gain_, controllers::Device::kCpu, kCartesianDim)),
         controllers::shared(
             controllers::read_view(integral_gain_, controllers::Device::kCpu, kCartesianDim)),
         controllers::shared(
             controllers::read_view(initial_output_, controllers::Device::kCpu, kCartesianDim)),
         controllers::shared(
             controllers::read_view(maximum_distance_, controllers::Device::kCpu, kCartesianDim)),
         controllers::shared(
             controllers::read_view(shaping_distance_, controllers::Device::kCpu, kCartesianDim))}};
    const controllers::CartesianUnifiedForceImpedanceController::Storage storage{
        {controllers::write_view(commanded_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_status(parameter_status_, controllers::Device::kCpu),
         controllers::write_status(status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(
                 last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
             controllers::write_view(safety_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(hold_count_, controllers::Device::kCpu, 1))},
        {{controllers::write_view(
              previous_cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
          controllers::write_view(
              previous_jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
          controllers::write_view(inertial_initialized_, controllers::Device::kCpu, 1)},
         {controllers::write_view(
              differentiator_previous_error_, controllers::Device::kCpu, 1, kCartesianDim),
          controllers::write_view(differentiator_initialized_, controllers::Device::kCpu, 1),
          controllers::write_view(
              integrator_previous_error_, controllers::Device::kCpu, 1, kCartesianDim),
          controllers::write_view(integrated_error_, controllers::Device::kCpu, 1, kCartesianDim),
          controllers::write_view(integrator_initialized_, controllers::Device::kCpu, 1)}},
        {controllers::write_view(impedance_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(force_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(impedance_status_, controllers::Device::kCpu, 1),
         controllers::write_view(force_status_, controllers::Device::kCpu, 1)}};

    mass_inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    controller_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(external_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(feedforward_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
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
            const Eigen::Vector<double, kCartesianDim>& external_wrench,
            const Eigen::Vector<double, kCartesianDim>& desired_wrench,
            const double dt_seconds) {
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
      external_wrench_[row] = external_wrench[static_cast<Eigen::Index>(row)];
      desired_wrench_[row] = desired_wrench[static_cast<Eigen::Index>(row)];
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
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kCartesianDim> proportional_gain_{};
  std::array<double, kCartesianDim> differentiator_gain_{};
  std::array<double, kCartesianDim> integral_gain_{};
  std::array<double, kCartesianDim> initial_output_{};
  std::array<double, kCartesianDim> maximum_distance_{};
  std::array<double, kCartesianDim> shaping_distance_{};
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
  std::array<double, kCartesianDim> external_wrench_{};
  std::array<double, kCartesianDim> desired_wrench_{};
  std::array<double, kCartesianDim> feedforward_wrench_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> commanded_wrench_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_cartesian_mass_{};
  std::array<double, kCartesianDim * kJointCount> previous_jacobian_{};
  std::array<std::uint8_t, 1> inertial_initialized_{};
  std::array<double, kCartesianDim> differentiator_previous_error_{};
  std::array<std::uint8_t, 1> differentiator_initialized_{};
  std::array<double, kCartesianDim> integrator_previous_error_{};
  std::array<double, kCartesianDim> integrated_error_{};
  std::array<std::uint8_t, 1> integrator_initialized_{};
  std::array<double, kCartesianDim> impedance_wrench_{};
  std::array<double, kCartesianDim> force_wrench_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  std::array<std::uint64_t, 1> force_status_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> safety_valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  controllers::CartesianUnifiedForceImpedanceController::Inputs controller_inputs_{};
  controllers::OperationalSpaceMass mass_controller_;
  controllers::CartesianUnifiedForceImpedanceController controller_;
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

  ForceImpedanceWorkspace controller;
  const controllers::Result prepare_result = controller.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare the end-effector-frame force/impedance controller: "
              << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }

  constexpr auto kInitialJointPosition = [] {
    auto joints = franka_example::kExtendedInitialJointConfiguration;
    joints[4] += franka_example::kPi / 4.0;
    return joints;
  }();
  if (!franka_example::confirm_execution(
          "Franka end-effector-frame force/impedance",
          "Setup: place the tool in unloaded contact with a compliant surface whose normal is "
          "the end-effector -z axis. This assumes the configured stiffness and EE frames are "
          "aligned.\n"
          "Command: hold measured O_T_EE and ramp EE-frame z force from 0 to +10 N over 2 s. "
          "Force contribution tapers over 0.06 to 0.10 m/rad of pose error.",
          kInitialJointPosition, arguments.run_duration_seconds)) {
    return EXIT_FAILURE;
  }

  franka_example::RunSummary summary;
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
    const Eigen::Matrix4d initial_pose =
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
      const Eigen::Matrix<double, 6, 7> body_jacobian =
          Eigen::Map<const Eigen::Matrix<double, 6, 7>>(
              model.bodyJacobian(franka::Frame::kEndEffector, state).data());
      const Eigen::Vector<double, 7> coriolis =
          Eigen::Map<const Eigen::Vector<double, 7>>(model.coriolis(state).data());
      const Eigen::Matrix4d base_to_ee =
          Eigen::Map<const Eigen::Matrix<double, 4, 4>>(state.O_T_EE.data());
      const Eigen::Vector<double, 7> dq =
          Eigen::Map<const Eigen::Vector<double, 7>>(state.dq.data());
      const Eigen::Vector<double, 6> base_wrench =
          Eigen::Map<const Eigen::Vector<double, 6>>(state.O_F_ext_hat_K.data());

      const Eigen::Matrix3d ee_rotation_base = base_to_ee.topLeftCorner<3, 3>().transpose();
      Eigen::Vector<double, 6> ee_wrench;
      ee_wrench.head<3>() = ee_rotation_base * base_wrench.head<3>();
      ee_wrench.tail<3>() = ee_rotation_base * base_wrench.tail<3>();
      const Eigen::Matrix4d ee_to_desired = base_to_ee.inverse() * initial_pose;
      const Eigen::Vector<double, 6> pose_error =
          controllers::examples::cartesian_pose_error(ee_to_desired, Eigen::Matrix4d::Identity());

      final_measured_force_newtons = ee_wrench(2);
      maximum_absolute_force_newtons =
          std::max(maximum_absolute_force_newtons, std::abs(final_measured_force_newtons));

      if (!franka_example::all_finite(mass) || !franka_example::all_finite(body_jacobian) ||
          !franka_example::all_finite(coriolis) || !ee_wrench.allFinite() ||
          !ee_to_desired.allFinite() || !pose_error.allFinite()) {
        summary.stop_reason = "invalid input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }

      Eigen::Vector<double, 6> desired_wrench = Eigen::Vector<double, 6>::Zero();
      desired_wrench(2) =
          std::min(1.0, summary.elapsed_seconds / kForceRampSeconds) * kTargetForceNewtons;
      if (!controller.step(pose_error,
                           -body_jacobian * dq,
                           dq,
                           mass,
                           body_jacobian,
                           ee_wrench,
                           desired_wrench,
                           dt_seconds)) {
        summary.stop_reason = "controller entered safe mode or rejected an input";
        franka_example::record_callback_time(callback_start, summary);
        return franka_example::finish_with_previous_torques(state);
      }
      const Eigen::Map<const Eigen::Vector<double, 6>> wrench(controller.wrench().data());
      Eigen::Vector<double, 7> desired_torques = body_jacobian.transpose() * wrench + coriolis;

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
  std::cout << std::fixed << std::setprecision(3)
            << "Observed end-effector z force: final_N=" << final_measured_force_newtons
            << ", max_abs_N=" << maximum_absolute_force_newtons
            << ". These measurements do not establish physical task success.\n";
  return summary.completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
