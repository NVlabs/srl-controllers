/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file impedance_common.hpp
 * @brief Shared startup, controller, and torque-filter support for UR impedance examples.
 */

#pragma once

#include <ur_client_library/example_robot_wrapper.h>
#include <ur_client_library/types.h>
#include <ur_client_library/ur/ur_driver.h>
#include <ur_client_library/ur/version_information.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <string>

#include "nvidia/srl/controllers/controllers.hpp"

#ifndef UR_EXAMPLE_RESOURCE_RELATIVE_DIR
#error "UR_EXAMPLE_RESOURCE_RELATIVE_DIR must identify the installed UR example resources"
#endif

namespace ur_impedance_example {

namespace controllers = nvidia::srl::controllers;

constexpr std::size_t kJointCount = 6;
constexpr std::size_t kCartesianDim = 6;
constexpr double kDefaultRunDurationSeconds = 10.0;
constexpr double kMaximumPeriodSeconds = 0.1;
constexpr double kTorqueRateLimitNmPerSecond = 500.0;
inline constexpr std::array<double, kJointCount> kUr5eTorqueLimitsNm{
    150.0, 150.0, 150.0, 28.0, 28.0, 28.0};
inline constexpr std::array<double, kJointCount> kUr10eTorqueLimitsNm{
    330.0, 330.0, 150.0, 54.0, 54.0, 54.0};

inline const std::array<double, kJointCount>& torque_limits_for_model(
    const std::string& robot_model) noexcept {
  return robot_model == "ur10e" ? kUr10eTorqueLimitsNm : kUr5eTorqueLimitsNm;
}

struct ProgramArguments {
  std::string robot_ip;
  std::string robot_model;
  double run_duration_seconds{kDefaultRunDurationSeconds};
  bool non_rt_requested{false};
};

inline bool parse_arguments(const int argc,
                            char** argv,
                            const bool require_robot_model,
                            ProgramArguments* arguments) {
  if (arguments == nullptr || argv == nullptr || argc < 2 || argv[1] == nullptr ||
      argv[1][0] == '\0' || argv[1][0] == '-') {
    return false;
  }
  arguments->robot_ip = argv[1];
  arguments->robot_model.clear();
  arguments->run_duration_seconds = kDefaultRunDurationSeconds;
  arguments->non_rt_requested = false;
  bool has_duration = false;
  bool has_robot_model = false;
  for (int index = 2; index < argc; ++index) {
    if (argv[index] == nullptr) {
      return false;
    }
    const std::string argument = argv[index];
    constexpr const char* kDurationPrefix = "--duration-seconds=";
    constexpr const char* kRobotModelPrefix = "--robot-model=";
    if (argument == "--non-rt") {
      if (arguments->non_rt_requested) {
        return false;
      }
      arguments->non_rt_requested = true;
    } else if (!has_duration && argument.rfind(kDurationPrefix, 0) == 0) {
      const std::string value = argument.substr(std::char_traits<char>::length(kDurationPrefix));
      try {
        std::size_t parsed_size = 0;
        arguments->run_duration_seconds = std::stod(value, &parsed_size);
        if (parsed_size != value.size() || !std::isfinite(arguments->run_duration_seconds) ||
            arguments->run_duration_seconds <= 0.0) {
          return false;
        }
      } catch (const std::exception&) {
        return false;
      }
      has_duration = true;
    } else if (require_robot_model && !has_robot_model &&
               argument.rfind(kRobotModelPrefix, 0) == 0) {
      arguments->robot_model = argument.substr(std::char_traits<char>::length(kRobotModelPrefix));
      if (arguments->robot_model != "ur5e" && arguments->robot_model != "ur10e") {
        return false;
      }
      has_robot_model = true;
    } else {
      return false;
    }
  }
  return !require_robot_model || has_robot_model;
}

inline bool confirm_connection(const char* title,
                               const char* command_description,
                               const ProgramArguments& arguments) {
  std::cout << title << "\n\n"
            << command_description << '\n'
            << "The configured robot safety limits remain authoritative. No motion planning or "
               "additional collision checking is performed. Keep an emergency stop within reach.\n"
            << "After connecting, the robot moves to [0, -90, -90, -90, 90, 0] deg before "
               "the controller captures its hold target.\n"
            << "Direct torque commands require PolyScope 5.23.0 / PolyScope X 10.10.0 or newer.\n"
            << "Torque clamp for " << arguments.robot_model << ": +/-[";
  const std::array<double, kJointCount>& torque_limits =
      torque_limits_for_model(arguments.robot_model);
  for (std::size_t joint = 0; joint < torque_limits.size(); ++joint) {
    std::cout << torque_limits[joint] << (joint + 1 == torque_limits.size() ? "]" : ", ");
  }
  std::cout << " Nm; torque-rate cap: " << kTorqueRateLimitNmPerSecond
            << " Nm/s per joint. Published joint maxima do not supersede configured robot safety "
               "limits.\n"
            << "Run duration: " << arguments.run_duration_seconds << " s.\n"
            << "non_rt_requested=" << (arguments.non_rt_requested ? "true" : "false") << '\n';
  if (!arguments.robot_model.empty()) {
    std::cout << "URDF model: " << arguments.robot_model << ".\n";
  }
  if (arguments.non_rt_requested) {
    std::cout << "Warning: ur_client_library has no switch to disable its warning-only FIFO "
                 "scheduling attempt. The client may still acquire real-time scheduling.\n";
  }
  std::cout << "The controller uses preallocated storage, but the vendor client and Pinocchio "
               "calls are not hard-real-time qualification.\n\n"
            << "Press Enter to connect to the robot and execute the initial joint motion: "
            << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before connecting to the robot.\n";
    return false;
  }
  return true;
}

inline void announce_torque_start(const ProgramArguments& arguments,
                                  const urcl::VersionInformation& version) {
  std::cout << "Connected to " << arguments.robot_ip << " with PolyScope " << version.toString()
            << ".\n";
  if (!arguments.robot_model.empty()) {
    std::cout << "Selected nominal kinematic model: " << arguments.robot_model << ".\n";
  }
  std::cout << "The next valid RTDE sample becomes the hold target; starting direct torque "
               "control.\n";
}

inline bool supports_direct_torque(const urcl::VersionInformation& version) {
  const urcl::VersionInformation minimum_g5 = urcl::VersionInformation::fromString("5.23.0");
  const urcl::VersionInformation minimum_x = urcl::VersionInformation::fromString("10.10.0");
  return (version.major == 5 && version >= minimum_g5) ||
         (version.major >= 10 && version >= minimum_x);
}

inline bool finite_vector(const urcl::vector6d_t& values) {
  return std::all_of(
      values.begin(), values.end(), [](const double value) { return std::isfinite(value); });
}

struct ResourcePaths {
  std::string input_recipe;
  std::string torque_output_recipe;
  std::string script;
  std::string model;
};

inline ResourcePaths resource_paths(const std::string& robot_model = {}) {
  const std::filesystem::path executable_directory =
      std::filesystem::canonical("/proc/self/exe").parent_path();
  const std::filesystem::path resource_directory =
      executable_directory / UR_EXAMPLE_RESOURCE_RELATIVE_DIR;
  ResourcePaths paths;
  paths.input_recipe = (resource_directory / "rtde_input_recipe.txt").string();
  paths.torque_output_recipe = (resource_directory / "rtde_torque_output_recipe.txt").string();
  paths.script = (resource_directory / "external_control.urscript").string();
  if (!robot_model.empty()) {
    paths.model = (resource_directory / "models" / (robot_model + ".urdf")).string();
  }
  return paths;
}

class ControlStopGuard {
 public:
  explicit ControlStopGuard(urcl::UrDriver& driver) : driver_(driver) {
    driver_.startRTDECommunication();
  }

  ControlStopGuard(const ControlStopGuard&) = delete;
  ControlStopGuard& operator=(const ControlStopGuard&) = delete;

  ~ControlStopGuard() noexcept {
    if (active_) {
      try {
        driver_.stopControl();
      } catch (...) {
        // A destructor on an exception path must not terminate the process.
      }
    }
  }

  bool stop() {
    if (!active_) {
      return true;
    }
    const bool stopped = driver_.stopControl();
    active_ = !stopped;
    return stopped;
  }

 private:
  urcl::UrDriver& driver_;
  bool active_{true};
};

class TorqueFilterWorkspace {
 public:
  explicit TorqueFilterWorkspace(const std::array<double, kJointCount>& torque_limits) {
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      lower_limit_[joint] = -torque_limits[joint];
      upper_limit_[joint] = torque_limits[joint];
    }
    maximum_rate_.fill(kTorqueRateLimitNmPerSecond);
    inputs_ = {controllers::read_view(command_, controllers::Device::kCpu, 1, kJointCount),
               controllers::read_view(dt_, controllers::Device::kCpu, 1)};
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{
        controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointCount, {}};
    const controllers::TorqueFilter::Parameters parameters{
        controllers::shared(
            controllers::read_view(lower_limit_, controllers::Device::kCpu, kJointCount)),
        controllers::shared(
            controllers::read_view(upper_limit_, controllers::Device::kCpu, kJointCount)),
        controllers::shared(
            controllers::read_view(maximum_rate_, controllers::Device::kCpu, kJointCount))};
    const controllers::TorqueFilter::Storage storage{
        {controllers::write_view(filtered_command_, controllers::Device::kCpu, 1, kJointCount),
         controllers::write_status(parameter_status_, controllers::Device::kCpu),
         controllers::write_status(status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(
                 last_valid_command_, controllers::Device::kCpu, 1, kJointCount),
             controllers::write_view(valid_, controllers::Device::kCpu, 1),
             controllers::write_view(hold_count_, controllers::Device::kCpu, 1))},
        {controllers::write_view(previous_command_, controllers::Device::kCpu, 1, kJointCount),
         controllers::write_view(initialized_, controllers::Device::kCpu, 1)}};
    controllers::Result result = filter_.prepare(config, parameters, storage);
    return result.ok() ? filter_.reset() : result;
  }

  bool step(const std::array<double, kJointCount>& command,
            const double dt_seconds,
            urcl::vector6d_t* filtered_command) noexcept {
    if (filtered_command == nullptr || !std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
      return false;
    }
    command_ = command;
    dt_[0] = dt_seconds;
    status_[0] = 0U;
    const controllers::Result result = filter_.step(inputs_);
    if (!result.ok() ||
        controllers::has_status_flag(status_[0], controllers::StatusFlag::kSafeMode) ||
        !std::all_of(filtered_command_.begin(), filtered_command_.end(), [](const double value) {
          return std::isfinite(value);
        })) {
      return false;
    }
    std::copy(filtered_command_.begin(), filtered_command_.end(), filtered_command->begin());
    return true;
  }

 private:
  std::array<double, kJointCount> lower_limit_{};
  std::array<double, kJointCount> upper_limit_{};
  std::array<double, kJointCount> maximum_rate_{};
  std::array<double, kJointCount> command_{};
  std::array<double, kJointCount> filtered_command_{};
  std::array<double, kJointCount> previous_command_{};
  std::array<double, kJointCount> last_valid_command_{};
  std::array<double, 1> dt_{};
  std::array<std::uint8_t, 1> initialized_{};
  std::array<std::uint8_t, 1> valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::TorqueFilter::Inputs inputs_{};
  controllers::TorqueFilter filter_;
};

inline bool best_effort_zero_torque(urcl::UrDriver& driver) {
  const urcl::vector6d_t zero_torque{};
  return driver.writeJointCommand(
      zero_torque, urcl::comm::ControlMode::MODE_TORQUE, urcl::RobotReceiveTimeout::millisec(20));
}

}  // namespace ur_impedance_example
