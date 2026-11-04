/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_impedance_controller.cpp
 * @brief Holds the measured UR joint position using direct torque impedance control.
 */

#include <ur_client_library/example_robot_wrapper.h>
#include <ur_client_library/types.h>
#include <ur_client_library/ur/ur_driver.h>

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
#include "examples/ur/initial_motion.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace {

namespace controllers = nvidia::srl::controllers;
using ur_impedance_example::kJointCount;

class JointImpedanceWorkspace {
 public:
  JointImpedanceWorkspace() {
    constexpr std::array<double, kJointCount> kStiffness{1000.0, 800.0, 600.0, 200.0, 100.0, 50.0};
    constexpr std::array<double, kJointCount> kDamping{
        56.5685424948, 101.1928851256, 37.9473319220, 17.8885438200, 8.9442719100, 4.8989794856};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      stiffness_[joint * kJointCount + joint] = kStiffness[joint];
      damping_[joint * kJointCount + joint] = kDamping[joint];
    }
    inputs_ = {controllers::read_view(position_error_, controllers::Device::kCpu, 1, kJointCount),
               controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kJointCount)};
  }

  controllers::Result prepare() {
    controllers::JointBlockConfig config;
    config.device = controllers::Device::kCpu;
    config.dtype = controllers::DType::kFloat64;
    config.batch_size = 1;
    config.joint_dim = kJointCount;
    config.mode = controllers::ComponentMode::kDense;
    const controllers::StaticJointImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            stiffness_, controllers::Device::kCpu, kJointCount, kJointCount)),
        controllers::shared(
            controllers::read_view(damping_, controllers::Device::kCpu, kJointCount, kJointCount))};
    const controllers::StaticJointImpedanceController::Storage storage{
        controllers::write_view(command_, controllers::Device::kCpu, 1, kJointCount),
        controllers::write_status(parameter_status_, controllers::Device::kCpu),
        controllers::write_status(status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(last_valid_command_, controllers::Device::kCpu, 1, kJointCount),
            controllers::write_view(valid_, controllers::Device::kCpu, 1),
            controllers::write_view(hold_count_, controllers::Device::kCpu, 1))};
    controllers::Result result = controller_.prepare(config, parameters, storage);
    return result.ok() ? controller_.reset() : result;
  }

  bool step(const urcl::vector6d_t& desired_position,
            const urcl::vector6d_t& measured_position,
            const urcl::vector6d_t& measured_velocity,
            std::array<double, kJointCount>* torque) noexcept {
    if (torque == nullptr) {
      return false;
    }
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      position_error_[joint] = desired_position[joint] - measured_position[joint];
      velocity_error_[joint] = -measured_velocity[joint];
    }
    status_[0] = 0U;
    const controllers::Result result = controller_.step(inputs_);
    if (!result.ok() ||
        controllers::has_status_flag(status_[0], controllers::StatusFlag::kSafeMode) ||
        !std::all_of(command_.begin(), command_.end(), [](const double value) {
          return std::isfinite(value);
        })) {
      return false;
    }
    *torque = command_;
    return true;
  }

 private:
  std::array<double, kJointCount * kJointCount> stiffness_{};
  std::array<double, kJointCount * kJointCount> damping_{};
  std::array<double, kJointCount> position_error_{};
  std::array<double, kJointCount> velocity_error_{};
  std::array<double, kJointCount> command_{};
  std::array<double, kJointCount> last_valid_command_{};
  std::array<std::uint8_t, 1> valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::StaticJointImpedanceController::Inputs inputs_{};
  controllers::StaticJointImpedanceController controller_;
};

}  // namespace

int main(int argc, char* argv[]) {
  ur_impedance_example::ProgramArguments arguments;
  if (!ur_impedance_example::parse_arguments(argc, argv, true, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_IP --robot-model=ur5e|ur10e [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }

  JointImpedanceWorkspace impedance;
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
          "Universal Robots joint impedance hold",
          "Command: hold the first measured six-joint position using static stiffness, manual "
          "damping, and gravity-compensated direct torque. Joint position is rad, velocity is "
          "rad/s, and command is Nm.",
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
    const ur_impedance_example::ResourcePaths paths = ur_impedance_example::resource_paths();
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
    urcl::vector6d_t desired_position{};
    bool initialized = false;
    double previous_timestamp_seconds = 0.0;
    std::array<double, kJointCount> unfiltered_torque{};
    urcl::vector6d_t filtered_torque{};
    urcl::rtde_interface::DataPackage data_package(driver.getRTDEOutputRecipe());

    while (!completed) {
      const auto loop_start = std::chrono::steady_clock::now();
      if (!driver.getDataPackage(data_package)) {
        stop_reason = "RTDE package unavailable";
        break;
      }
      ++samples;

      urcl::vector6d_t measured_position{};
      urcl::vector6d_t measured_velocity{};
      double timestamp_seconds = 0.0;
      if (!data_package.getData("actual_q", measured_position) ||
          !data_package.getData("actual_qd", measured_velocity) ||
          !data_package.getData("timestamp", timestamp_seconds)) {
        stop_reason = "RTDE package missing a required field";
        break;
      }
      if (!ur_impedance_example::finite_vector(measured_position) ||
          !ur_impedance_example::finite_vector(measured_velocity) ||
          !std::isfinite(timestamp_seconds)) {
        stop_reason = "non-finite RTDE input";
        break;
      }

      if (!initialized) {
        desired_position = measured_position;
        previous_timestamp_seconds = timestamp_seconds;
        initialized = true;
        if (!driver.writeJointCommand(filtered_torque,
                                      urcl::comm::ControlMode::MODE_TORQUE,
                                      urcl::RobotReceiveTimeout::millisec(20))) {
          stop_reason = "initial zero-torque command rejected";
          break;
        }
        continue;
      }

      const double dt_seconds = timestamp_seconds - previous_timestamp_seconds;
      previous_timestamp_seconds = timestamp_seconds;
      if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0 ||
          dt_seconds > ur_impedance_example::kMaximumPeriodSeconds) {
        stop_reason = "invalid RTDE timestamp delta";
        break;
      }
      elapsed_seconds += dt_seconds;

      if (!impedance.step(
              desired_position, measured_position, measured_velocity, &unfiltered_torque) ||
          !torque_filter.step(unfiltered_torque, dt_seconds, &filtered_torque)) {
        stop_reason = "controller entered safe mode or rejected an input";
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

    if (!ur_impedance_example::best_effort_zero_torque(driver)) {
      completed = false;
      stop_reason = "zero-torque shutdown command failed";
    }
    if (!control_guard.stop()) {
      completed = false;
      stop_reason = "stopControl failed";
    }
  } catch (const std::exception& error) {
    stop_reason = "client or controller exception";
    std::cerr << "Universal Robots example failed: " << error.what() << '\n';
  }

  std::cout << std::fixed << std::setprecision(6) << "Run summary: samples=" << samples
            << ", elapsed_s=" << elapsed_seconds << ", max_loop_s=" << maximum_loop_seconds
            << ", stop_reason=" << stop_reason << '\n';
  return completed ? EXIT_SUCCESS : EXIT_FAILURE;
}
