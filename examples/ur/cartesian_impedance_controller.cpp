/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_impedance_controller.cpp
 * @brief Holds the measured UR TCP pose using Jacobian-transpose direct torque control.
 */

#include <ur_client_library/example_robot_wrapper.h>
#include <ur_client_library/types.h>
#include <ur_client_library/ur/ur_driver.h>

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include "./impedance_common.hpp"
#include "./ur_kinematics.hpp"
#include "examples/cartesian_math.hpp"
#include "examples/ur/initial_motion.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace {

namespace controllers = nvidia::srl::controllers;
using ur_impedance_example::Jacobian;
using ur_impedance_example::kCartesianDim;
using ur_impedance_example::kJointCount;

class CartesianImpedanceWorkspace {
 public:
  CartesianImpedanceWorkspace() {
    constexpr std::array<double, kCartesianDim> kStiffness{
        1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0};
    constexpr std::array<double, kCartesianDim> kDamping{178.8854382000,
                                                         178.8854382000,
                                                         178.8854382000,
                                                         18.9736659612,
                                                         18.9736659612,
                                                         18.9736659612};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = kStiffness[axis];
      damping_[axis * kCartesianDim + axis] = kDamping[axis];
    }
    inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim)};
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               kJointCount,
                                               {},
                                               controllers::CartesianFrame::kRobotBase};
    const controllers::StaticCartesianImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            damping_, controllers::Device::kCpu, kCartesianDim, kCartesianDim))};
    const controllers::StaticCartesianImpedanceController::Storage storage{
        controllers::write_view(wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_status(parameter_status_, controllers::Device::kCpu),
        controllers::write_status(status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(valid_, controllers::Device::kCpu, 1),
            controllers::write_view(hold_count_, controllers::Device::kCpu, 1))};
    controllers::Result result = controller_.prepare(config, parameters, storage);
    return result.ok() ? controller_.reset() : result;
  }

  bool step(const Eigen::Matrix4d& desired_pose,
            const Eigen::Matrix4d& measured_pose,
            const Jacobian& jacobian,
            const urcl::vector6d_t& measured_velocity,
            std::array<double, kJointCount>* torque) noexcept {
    if (torque == nullptr || !desired_pose.allFinite() || !measured_pose.allFinite() ||
        !jacobian.allFinite()) {
      return false;
    }
    const Eigen::Vector<double, kCartesianDim> pose_error =
        controllers::examples::cartesian_pose_error(desired_pose, measured_pose);
    Eigen::Vector<double, kJointCount> joint_velocity;
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      joint_velocity[static_cast<Eigen::Index>(joint)] = measured_velocity[joint];
    }
    const Eigen::Vector<double, kCartesianDim> velocity_error = -jacobian * joint_velocity;
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      pose_error_[axis] = pose_error[static_cast<Eigen::Index>(axis)];
      velocity_error_[axis] = velocity_error[static_cast<Eigen::Index>(axis)];
    }
    status_[0] = 0U;
    const controllers::Result result = controller_.step(inputs_);
    const Eigen::Map<const Eigen::Vector<double, kCartesianDim>> wrench(wrench_.data());
    const Eigen::Vector<double, kJointCount> joint_torque = jacobian.transpose() * wrench;
    if (!result.ok() ||
        controllers::has_status_flag(status_[0], controllers::StatusFlag::kSafeMode) ||
        !pose_error.allFinite() || !velocity_error.allFinite() || !wrench.allFinite() ||
        !joint_torque.allFinite()) {
      return false;
    }
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      (*torque)[joint] = joint_torque[static_cast<Eigen::Index>(joint)];
    }
    return true;
  }

 private:
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim * kCartesianDim> damping_{};
  std::array<double, kCartesianDim> pose_error_{};
  std::array<double, kCartesianDim> velocity_error_{};
  std::array<double, kCartesianDim> wrench_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::StaticCartesianImpedanceController::Inputs inputs_{};
  controllers::StaticCartesianImpedanceController controller_;
};

bool same_tcp_offset(const urcl::vector6d_t& expected, const urcl::vector6d_t& observed) noexcept {
  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    if (std::abs(expected[axis] - observed[axis]) > 1.0e-12) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  ur_impedance_example::ProgramArguments arguments;
  if (!ur_impedance_example::parse_arguments(argc, argv, true, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_IP --robot-model=ur5e|ur10e [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }

  const ur_impedance_example::ResourcePaths paths =
      ur_impedance_example::resource_paths(arguments.robot_model);
  CartesianImpedanceWorkspace impedance;
  ur_impedance_example::TorqueFilterWorkspace torque_filter(
      ur_impedance_example::torque_limits_for_model(arguments.robot_model));
  const controllers::Result impedance_prepare = impedance.prepare();
  const controllers::Result filter_prepare = torque_filter.prepare();
  if (!impedance_prepare.ok() || !filter_prepare.ok()) {
    std::cerr << "Controller preparation failed: "
              << (!impedance_prepare.ok() ? impedance_prepare.message : filter_prepare.message)
              << '\n';
    return EXIT_FAILURE;
  }

  if (!ur_impedance_example::confirm_connection(
          "Universal Robots Cartesian impedance hold",
          "Command: hold the first measured TCP pose in the robot base frame. A nominal URDF "
          "and the configured TCP offset supply only the base-aligned Jacobian; measured TCP "
          "pose supplies the hold error. Translation is m, rotation is axis-angle rad, wrench "
          "is [N; Nm], and joint command is Nm.",
          arguments)) {
    return EXIT_FAILURE;
  }

  urcl::setLogLevel(urcl::LogLevel::INFO);
  std::size_t samples = 0;
  double elapsed_seconds = 0.0;
  double maximum_loop_seconds = 0.0;
  const char* stop_reason = "not started";
  bool completed = false;

  try {
    ur_impedance_example::UrKinematics kinematics(paths.model);
    urcl::ExampleRobotWrapper robot(
        arguments.robot_ip, paths.torque_output_recipe, paths.input_recipe, true, "", paths.script);
    if (!robot.isHealthy()) {
      std::cerr << "Universal Robots client initialization failed.\n";
      return EXIT_FAILURE;
    }
    urcl::UrDriver& driver = *robot.getUrDriver();
    const urcl::VersionInformation version = driver.getVersion();
    if (!ur_impedance_example::supports_direct_torque(version)) {
      std::cerr << "Direct torque requires PolyScope 5.23.0 / PolyScope X 10.10.0 or newer; "
                << "connected version is " << version.toString() << ".\n";
      return EXIT_FAILURE;
    }
    if (!nvidia::srl::controllers::examples::ur::move_to_initial_joint_position(
            robot.getUrDriver())) {
      std::cerr << "Failed to move to the initial joint position.\n";
      return EXIT_FAILURE;
    }
    if (!nvidia::srl::controllers::examples::ur::confirm_controller_start()) {
      return EXIT_FAILURE;
    }
    ur_impedance_example::announce_torque_start(arguments, version);

    ur_impedance_example::ControlStopGuard control_guard(driver);
    urcl::rtde_interface::DataPackage data_package(driver.getRTDEOutputRecipe());
    urcl::vector6d_t initial_joint_position{};
    urcl::vector6d_t initial_joint_velocity{};
    urcl::vector6d_t initial_tcp_pose_vector{};
    urcl::vector6d_t configured_tcp_offset{};
    double previous_timestamp_seconds = 0.0;
    if (!driver.getDataPackage(data_package) ||
        !data_package.getData("actual_q", initial_joint_position) ||
        !data_package.getData("actual_qd", initial_joint_velocity) ||
        !data_package.getData("actual_TCP_pose", initial_tcp_pose_vector) ||
        !data_package.getData("tcp_offset", configured_tcp_offset) ||
        !data_package.getData("timestamp", previous_timestamp_seconds) ||
        !ur_impedance_example::finite_vector(initial_joint_position) ||
        !ur_impedance_example::finite_vector(initial_joint_velocity) ||
        !ur_impedance_example::finite_vector(initial_tcp_pose_vector) ||
        !ur_impedance_example::finite_vector(configured_tcp_offset) ||
        !std::isfinite(previous_timestamp_seconds) ||
        !kinematics.configure_tcp(configured_tcp_offset)) {
      stop_reason = "invalid initial RTDE state or TCP configuration";
    } else {
      const Eigen::Matrix4d desired_pose =
          ur_impedance_example::pose_vector_to_transform(initial_tcp_pose_vector);
      Eigen::Matrix4d nominal_pose = Eigen::Matrix4d::Identity();
      Jacobian jacobian = Jacobian::Zero();
      const urcl::vector6d_t zero_torque{};
      if (!desired_pose.allFinite() ||
          !kinematics.compute(initial_joint_position, &nominal_pose, &jacobian) ||
          !driver.writeJointCommand(zero_torque,
                                    urcl::comm::ControlMode::MODE_TORQUE,
                                    urcl::RobotReceiveTimeout::millisec(20))) {
        stop_reason = "initial kinematics or zero-torque command failed";
      } else {
        ++samples;
        std::array<double, kJointCount> unfiltered_torque{};
        urcl::vector6d_t filtered_torque{};
        while (!completed) {
          const auto loop_start = std::chrono::steady_clock::now();
          if (!driver.getDataPackage(data_package)) {
            stop_reason = "RTDE package unavailable";
            break;
          }
          ++samples;

          urcl::vector6d_t measured_position{};
          urcl::vector6d_t measured_velocity{};
          urcl::vector6d_t measured_tcp_pose_vector{};
          urcl::vector6d_t observed_tcp_offset{};
          double timestamp_seconds = 0.0;
          if (!data_package.getData("actual_q", measured_position) ||
              !data_package.getData("actual_qd", measured_velocity) ||
              !data_package.getData("actual_TCP_pose", measured_tcp_pose_vector) ||
              !data_package.getData("tcp_offset", observed_tcp_offset) ||
              !data_package.getData("timestamp", timestamp_seconds)) {
            stop_reason = "RTDE package missing a required field";
            break;
          }
          if (!ur_impedance_example::finite_vector(measured_position) ||
              !ur_impedance_example::finite_vector(measured_velocity) ||
              !ur_impedance_example::finite_vector(measured_tcp_pose_vector) ||
              !ur_impedance_example::finite_vector(observed_tcp_offset) ||
              !std::isfinite(timestamp_seconds) ||
              !same_tcp_offset(configured_tcp_offset, observed_tcp_offset)) {
            stop_reason = "non-finite RTDE input or changed TCP offset";
            break;
          }

          const double dt_seconds = timestamp_seconds - previous_timestamp_seconds;
          previous_timestamp_seconds = timestamp_seconds;
          if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0 ||
              dt_seconds > ur_impedance_example::kMaximumPeriodSeconds) {
            stop_reason = "invalid RTDE timestamp delta";
            break;
          }
          elapsed_seconds += dt_seconds;

          const Eigen::Matrix4d measured_pose =
              ur_impedance_example::pose_vector_to_transform(measured_tcp_pose_vector);
          if (!kinematics.compute(measured_position, &nominal_pose, &jacobian) ||
              !impedance.step(
                  desired_pose, measured_pose, jacobian, measured_velocity, &unfiltered_torque) ||
              !torque_filter.step(unfiltered_torque, dt_seconds, &filtered_torque)) {
            stop_reason = "kinematics or controller entered safe mode";
            break;
          }
          if (!driver.writeJointCommand(filtered_torque,
                                        urcl::comm::ControlMode::MODE_TORQUE,
                                        urcl::RobotReceiveTimeout::millisec(20))) {
            stop_reason = "torque command rejected";
            break;
          }

          maximum_loop_seconds = std::max(
              maximum_loop_seconds,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count());
          if (elapsed_seconds >= arguments.run_duration_seconds) {
            completed = true;
            stop_reason = "configured duration reached";
          }
        }
      }
    }

    if (!ur_impedance_example::best_effort_zero_torque(driver)) {
      completed = false;
      stop_reason = "zero-torque shutdown command failed";
    }
    if (!control_guard.stop()) {
      completed = false;
      stop_reason = "stopControl failed";
    }
  } catch (const std::exception& error) {
    stop_reason = "client, model, or controller exception";
    std::cerr << "Universal Robots example failed: " << error.what() << '\n';
  }

  std::cout << std::fixed << std::setprecision(6) << "Run summary: samples=" << samples
            << ", elapsed_s=" << elapsed_seconds << ", max_loop_s=" << maximum_loop_seconds
            << ", stop_reason=" << stop_reason << '\n';
  return completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
