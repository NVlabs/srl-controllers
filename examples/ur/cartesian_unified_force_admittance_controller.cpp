/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_unified_force_admittance_controller.cpp
 * @brief Runs unified Cartesian force/admittance control on a Universal Robots arm.
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
#include <filesystem>  // NOLINT(build/c++17)
#include <iomanip>
#include <iostream>
#include <string>

#include "examples/cartesian_math.hpp"
#include "examples/ur/initial_motion.hpp"
#include "examples/ur/unified_force_admittance_example.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

#ifndef UR_EXAMPLE_RESOURCE_RELATIVE_DIR
#error "UR_EXAMPLE_RESOURCE_RELATIVE_DIR must identify the installed UR example resources"
#endif

namespace {

namespace controllers = nvidia::srl::controllers;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultRunDurationSeconds = 10.0;
constexpr double kMaximumPeriodSeconds = 0.1;
constexpr double kVirtualMassScale = 1.3;
constexpr double kDampingScale = 2.0;
constexpr double kForceAxisDampingScale = 1.5;
constexpr double kEndEffectorForceAxisDampingScale = 3.0;
constexpr double kTargetForceMagnitude = 10.0;
constexpr double kMaximumShapingDistance = 0.1;
constexpr double kFullForceDistance = 0.06;
constexpr std::size_t kCartesianDim = 6;
constexpr std::size_t kForceAxis = 2;

class LowPassFilter6 {
 public:
  explicit LowPassFilter6(double cutoff_hertz) : cutoff_hertz_(cutoff_hertz) {}

  Eigen::Vector<double, 6> update(const Eigen::Vector<double, 6>& input, double dt_seconds) {
    const double time_constant = 1.0 / (2.0 * kPi * cutoff_hertz_);
    const double alpha = dt_seconds / (time_constant + dt_seconds);
    if (!initialized_) {
      output_ = input;
      initialized_ = true;
    } else {
      output_ += alpha * (input - output_);
    }
    return output_;
  }

 private:
  double cutoff_hertz_;
  Eigen::Vector<double, 6> output_{Eigen::Vector<double, 6>::Zero()};
  bool initialized_{false};
};

class ControlStopGuard {
 public:
  explicit ControlStopGuard(urcl::UrDriver& driver) : driver_(driver) {
    driver_.startRTDECommunication();
  }

  ControlStopGuard(const ControlStopGuard&) = delete;
  ControlStopGuard& operator=(const ControlStopGuard&) = delete;

  ~ControlStopGuard() {
    if (active_) {
      driver_.stopControl();
    }
  }

  bool stop() {
    if (!active_) {
      return true;
    }
    active_ = false;
    return driver_.stopControl();
  }

 private:
  urcl::UrDriver& driver_;
  bool active_{true};
};

struct ProgramArguments {
  std::string robot_ip;
  double run_duration_seconds{kDefaultRunDurationSeconds};
  bool non_rt_requested{false};
};

bool parse_arguments(const int argc, char** argv, ProgramArguments* arguments) {
  if (arguments == nullptr || argv == nullptr || argc < 2 || argc > 4 || argv[1] == nullptr ||
      argv[1][0] == '\0' || argv[1][0] == '-') {
    return false;
  }
  arguments->robot_ip = argv[1];
  arguments->run_duration_seconds = kDefaultRunDurationSeconds;
  arguments->non_rt_requested = false;
  bool has_duration = false;
  for (int index = 2; index < argc; ++index) {
    if (argv[index] == nullptr) {
      return false;
    }
    const std::string argument = argv[index];
    constexpr const char* kDurationPrefix = "--duration-seconds=";
    if (argument == "--non-rt") {
      if (arguments->non_rt_requested) {
        return false;
      }
      arguments->non_rt_requested = true;
      continue;
    }
    if (has_duration || argument.rfind(kDurationPrefix, 0) != 0) {
      return false;
    }
    const std::string value = argument.substr(std::char_traits<char>::length(kDurationPrefix));
    try {
      std::size_t parsed_size = 0;
      arguments->run_duration_seconds = std::stod(value, &parsed_size);
      has_duration = true;
      if (parsed_size != value.size() || !std::isfinite(arguments->run_duration_seconds) ||
          arguments->run_duration_seconds <= 0.0) {
        return false;
      }
    } catch (const std::exception&) {
      return false;
    }
  }
  return true;
}

bool confirm_execution(const double run_duration_seconds,
                       const bool non_rt_requested,
                       const bool uses_end_effector_force_target) {
  std::cout << "Universal Robots unified Cartesian force/admittance";
  if (uses_end_effector_force_target) {
    std::cout << " in the end-effector frame\n\n"
              << "Command: the measured TCP wrench is rotated from robot-base axes into "
                 "end-effector axes. Desired robot-applied end-effector-z force ramps from 0 to "
                 "+10 N over 2 s.\n";
  } else {
    std::cout << " in the robot base frame\n\n"
              << "Command: base-frame TCP force error drives a virtual 6-DoF system from the "
                 "measured start pose without a software pose envelope. Desired base-z force "
                 "applied by the robot ramps from 0 to -10 N over 2 s.\n";
  }
  std::cout
      << "Units: TCP translation in m, rotation as an axis-angle vector in rad, wrench in N and "
         "Nm. RTDE timestamp is the control clock.\n"
      << "No contact surface is required: force shaping bounds the free-space equilibrium. The "
         "operator must verify that the printed force direction is intended for the workcell.\n"
      << "The configured robot safety limits remain authoritative. Keep an emergency stop within "
         "reach. The controller hot path uses caller-owned storage, but this vendor "
         "client loop is not qualified as hard real time.\n";
  std::cout << "After connecting, the robot moves to [0, -90, -90, -90, "
            << (uses_end_effector_force_target ? "135" : "90")
            << ", 0] deg before the controller captures its equilibrium pose.\n"
            << "Run duration: " << run_duration_seconds << " s.\n"
            << "non_rt_requested=" << (non_rt_requested ? "true" : "false") << '\n';
  if (non_rt_requested) {
    std::cout << "Warning: ur_client_library has no switch to disable its warning-only FIFO "
                 "scheduling attempt. --non-rt is accepted here without changing that upstream "
                 "behavior, and the client may still acquire real-time scheduling.\n";
  }
  std::cout << '\n' << "Press Enter to connect to the robot and execute the initial joint motion: "
            << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before connecting to the robot.\n";
    return false;
  }
  return true;
}

Eigen::Matrix4d pose_vector_to_transform(const Eigen::Vector<double, 6>& pose) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topRightCorner<3, 1>() = pose.head<3>();
  const Eigen::Vector3d rotation_vector = pose.tail<3>();
  const double angle = rotation_vector.norm();
  if (angle > 1e-12) {
    transform.topLeftCorner<3, 3>() =
        Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
  }
  return transform;
}

urcl::vector6d_t transform_to_pose_vector(const Eigen::Matrix4d& transform) {
  urcl::vector6d_t pose{};
  const Eigen::AngleAxisd rotation(transform.topLeftCorner<3, 3>());
  Eigen::Vector3d rotation_vector = Eigen::Vector3d::Zero();
  if (rotation.angle() > 1e-12) {
    rotation_vector = rotation.angle() * rotation.axis();
  }
  for (std::size_t index = 0; index < 3; ++index) {
    pose[index] = transform(static_cast<Eigen::Index>(index), 3);
    pose[index + 3] = rotation_vector(static_cast<Eigen::Index>(index));
  }
  return pose;
}

bool finite_vector(const urcl::vector6d_t& values) {
  return std::all_of(
      values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

void update_desired_external_wrench(const double elapsed_seconds,
                                    const bool uses_end_effector_force_target,
                                    std::array<double, kCartesianDim>* desired_external_wrench) {
  desired_external_wrench->fill(0.0);
  const double ramp = std::min(elapsed_seconds / 2.0, 1.0);
  const double commanded_robot_force =
      uses_end_effector_force_target ? kTargetForceMagnitude : -kTargetForceMagnitude;
  (*desired_external_wrench)[kForceAxis] = -commanded_robot_force * ramp;
}

Eigen::Vector<double, kCartesianDim> rotate_cartesian_vector(
    const Eigen::Matrix3d& rotation_new_from_old,
    const Eigen::Vector<double, kCartesianDim>& vector) {
  Eigen::Vector<double, kCartesianDim> rotated;
  rotated.head<3>() = rotation_new_from_old * vector.head<3>();
  rotated.tail<3>() = rotation_new_from_old * vector.tail<3>();
  return rotated;
}

void update_cartesian_matrix(const Eigen::Matrix3d& rotation_base_from_controller,
                             const std::array<double, kCartesianDim>& diagonal,
                             std::array<double, kCartesianDim * kCartesianDim>* matrix) {
  matrix->fill(0.0);
  for (std::size_t block = 0; block < 2; ++block) {
    const std::size_t offset = block * 3;
    Eigen::Vector3d block_diagonal;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      block_diagonal(static_cast<Eigen::Index>(axis)) = diagonal[offset + axis];
    }
    const Eigen::Matrix3d rotated_block = rotation_base_from_controller *
                                          block_diagonal.asDiagonal() *
                                          rotation_base_from_controller.transpose();
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t column = 0; column < 3; ++column) {
        (*matrix)[(offset + row) * kCartesianDim + offset + column] =
            rotated_block(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
  }
}

}  // namespace

int nvidia::srl::controllers::examples::ur::run_unified_force_admittance_example(
    const int argc, char* argv[], const ForceTargetFrame force_target_frame) {
  const bool uses_end_effector_force_target = force_target_frame == ForceTargetFrame::kEndEffector;
  ProgramArguments arguments;
  if (!parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0] << " ROBOT_IP [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }

  std::array<double, kCartesianDim * kCartesianDim> stiffness{};
  std::array<double, kCartesianDim * kCartesianDim> damping{};
  std::array<double, kCartesianDim * kCartesianDim> virtual_mass{};
  const std::array<double, kCartesianDim> stiffness_diagonal{1000, 1000, 10, 30, 30, 30};
  const std::array<double, kCartesianDim> mass_diagonal{15, 15, 1.5, 0.5, 0.5, 0.5};
  std::array<double, kCartesianDim> damping_diagonal{};
  std::array<double, kCartesianDim> virtual_mass_diagonal{};
  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    virtual_mass_diagonal[axis] = kVirtualMassScale * mass_diagonal[axis];
    const double force_axis_damping_scale =
        uses_end_effector_force_target ? kEndEffectorForceAxisDampingScale : kForceAxisDampingScale;
    const double damping_scale = axis == kForceAxis ? force_axis_damping_scale : kDampingScale;
    const double damping_stiffness =
        axis == kForceAxis && uses_end_effector_force_target
            ? stiffness_diagonal[axis] +
                  kTargetForceMagnitude / (kMaximumShapingDistance - kFullForceDistance)
            : stiffness_diagonal[axis];
    damping_diagonal[axis] =
        damping_scale * 2.0 * std::sqrt(damping_stiffness * virtual_mass_diagonal[axis]);
  }
  update_cartesian_matrix(Eigen::Matrix3d::Identity(), stiffness_diagonal, &stiffness);
  update_cartesian_matrix(Eigen::Matrix3d::Identity(), damping_diagonal, &damping);
  update_cartesian_matrix(Eigen::Matrix3d::Identity(), virtual_mass_diagonal, &virtual_mass);
  const std::array<double, kCartesianDim> proportional_gain{1, 1, 1, 1, 1, 1};
  const std::array<double, kCartesianDim> differentiator_gain{};
  const std::array<double, kCartesianDim> integral_gain{};
  const std::array<double, kCartesianDim> initial_integral{};
  const std::array<double, kCartesianDim> error_deadzone{0.5, 0.5, 0.5, 0.05, 0.05, 0.05};
  const std::array<double, kCartesianDim> maximum_distance{kMaximumShapingDistance,
                                                           kMaximumShapingDistance,
                                                           kMaximumShapingDistance,
                                                           kMaximumShapingDistance,
                                                           kMaximumShapingDistance,
                                                           kMaximumShapingDistance};
  const std::array<double, kCartesianDim> shaping_distance{kFullForceDistance,
                                                           kFullForceDistance,
                                                           kFullForceDistance,
                                                           kFullForceDistance,
                                                           kFullForceDistance,
                                                           kFullForceDistance};
  std::array<double, kCartesianDim> measured_wrench{};
  std::array<double, kCartesianDim> target_wrench{};
  std::array<double, kCartesianDim> external_wrench{};
  std::array<double, kCartesianDim> desired_wrench{};
  std::array<double, kCartesianDim> feedforward_wrench{};
  std::array<double, kCartesianDim> shaping_error{};
  std::array<double, kCartesianDim> shaping_factor{};
  std::array<double, 1> dt{};
  std::array<double, kCartesianDim> commanded_acceleration{};
  std::array<double, kCartesianDim> pose_offset{};
  std::array<double, kCartesianDim> twist{};
  std::array<double, kCartesianDim> pose_previous_signal{};
  std::array<double, kCartesianDim> twist_previous_signal{};
  std::array<std::uint8_t, 1> pose_integrator_initialized{};
  std::array<std::uint8_t, 1> twist_integrator_initialized{};
  std::array<double, kCartesianDim> differentiator_previous_error{};
  std::array<std::uint8_t, 1> differentiator_initialized{};
  std::array<double, kCartesianDim> integrator_previous_error{};
  std::array<double, kCartesianDim> integrated_error{};
  std::array<std::uint8_t, 1> force_integrator_initialized{};
  std::array<double, kCartesianDim> last_valid_acceleration{};
  std::array<std::uint8_t, 1> safety_valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  std::array<double, kCartesianDim> last_valid_shaping_factor{};
  std::array<std::uint8_t, 1> shaping_safety_valid{};
  std::array<std::uint64_t, 1> shaping_hold_count{};
  std::array<std::uint64_t, 1> shaping_parameter_status{};
  std::array<std::uint64_t, 1> shaping_status{};

  controllers::ShapingFunction shaping_function;
  const controllers::ShapingFunction::Parameters shaping_parameters{
      controllers::shared(
          controllers::read_view(maximum_distance, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(
          controllers::read_view(shaping_distance, controllers::Device::kCpu, kCartesianDim))};
  const controllers::ShapingFunction::Storage shaping_storage{
      {controllers::write_view(shaping_factor, controllers::Device::kCpu, 1, kCartesianDim),
       controllers::write_status(shaping_parameter_status, controllers::Device::kCpu),
       controllers::write_status(shaping_status, controllers::Device::kCpu),
       controllers::make_safety_state(
           controllers::write_view(
               last_valid_shaping_factor, controllers::Device::kCpu, 1, kCartesianDim),
           controllers::write_view(shaping_safety_valid, controllers::Device::kCpu, 1),
           controllers::write_view(shaping_hold_count, controllers::Device::kCpu, 1))},
      {},
      {}};
  const controllers::ShapingFunction::Inputs shaping_inputs{
      controllers::read_view(shaping_error, controllers::Device::kCpu, 1, kCartesianDim)};
  controllers::CartesianUnifiedForceAdmittanceController controller;
  const controllers::CartesianUnifiedForceAdmittanceController::Parameters parameters{
      controllers::shared(controllers::read_view(
          stiffness, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
      controllers::shared(
          controllers::read_view(damping, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
      controllers::shared(controllers::read_view(
          virtual_mass, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
      {},
      {},
      {},
      {controllers::shared(
           controllers::read_view(proportional_gain, controllers::Device::kCpu, kCartesianDim)),
       controllers::shared(
           controllers::read_view(differentiator_gain, controllers::Device::kCpu, kCartesianDim)),
       controllers::shared(
           controllers::read_view(integral_gain, controllers::Device::kCpu, kCartesianDim)),
       controllers::shared(
           controllers::read_view(initial_integral, controllers::Device::kCpu, kCartesianDim)),
       {},
       {},
       controllers::shared(
           controllers::read_view(error_deadzone, controllers::Device::kCpu, kCartesianDim))}};
  const controllers::CartesianUnifiedForceAdmittanceController::Storage controller_storage{
      {controllers::write_view(commanded_acceleration, controllers::Device::kCpu, 1, kCartesianDim),
       controllers::write_status(parameter_status, controllers::Device::kCpu),
       controllers::write_status(status, controllers::Device::kCpu),
       controllers::make_safety_state(
           controllers::write_view(
               last_valid_acceleration, controllers::Device::kCpu, 1, kCartesianDim),
           controllers::write_view(safety_valid, controllers::Device::kCpu, 1),
           controllers::write_view(hold_count, controllers::Device::kCpu, 1))},
      {{controllers::write_view(
            differentiator_previous_error, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(differentiator_initialized, controllers::Device::kCpu, 1),
        controllers::write_view(
            integrator_previous_error, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(integrated_error, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(force_integrator_initialized, controllers::Device::kCpu, 1)},
       {controllers::write_view(pose_offset, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(twist, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(pose_previous_signal, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(twist_previous_signal, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(pose_integrator_initialized, controllers::Device::kCpu, 1),
        controllers::write_view(twist_integrator_initialized, controllers::Device::kCpu, 1)}}};
  const controllers::CartesianUnifiedForceAdmittanceController::Inputs inputs{
      controllers::read_view(desired_wrench, controllers::Device::kCpu, 1, kCartesianDim),
      controllers::read_view(external_wrench, controllers::Device::kCpu, 1, kCartesianDim),
      controllers::read_view(feedforward_wrench, controllers::Device::kCpu, 1, kCartesianDim),
      controllers::read_view(dt, controllers::Device::kCpu, 1)};
  const controllers::ControllerConfig controller_config{controllers::Device::kCpu,
                                                        controllers::DType::kFloat64,
                                                        1,
                                                        0,
                                                        {},
                                                        controllers::CartesianFrame::kRobotBase};
  const controllers::SignalBlockConfig shaping_config{
      controllers::Device::kCpu, controllers::DType::kFloat64, 1, kCartesianDim, {}};
  const controllers::Result shaping_prepare_result =
      shaping_function.prepare(shaping_config, shaping_parameters, shaping_storage);
  if (!shaping_prepare_result.ok()) {
    std::cerr << "Shaping function preparation failed: " << shaping_prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
  const controllers::Result prepare_result =
      controller.prepare(controller_config, parameters, controller_storage);
  if (!prepare_result.ok()) {
    std::cerr << "Controller preparation failed: " << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
  const controllers::Result shaping_reset_result = shaping_function.reset();
  const controllers::Result reset_result = controller.reset();
  if (shaping_parameter_status[0] != 0 || parameter_status[0] != 0 || !shaping_reset_result.ok() ||
      !reset_result.ok()) {
    if (!shaping_reset_result.ok()) {
      std::cerr << "Shaping function reset failed: " << shaping_reset_result.message << '\n';
    }
    if (!reset_result.ok()) {
      std::cerr << "Controller reset failed: " << reset_result.message << '\n';
    }
    std::cerr << "Failed to prepare the unified force/admittance controller.\n";
    return EXIT_FAILURE;
  }
  if (!confirm_execution(arguments.run_duration_seconds,
                         arguments.non_rt_requested,
                         uses_end_effector_force_target)) {
    return EXIT_FAILURE;
  }

  urcl::setLogLevel(urcl::LogLevel::INFO);
  std::size_t samples = 0;
  double elapsed_seconds = 0.0;
  double maximum_loop_seconds = 0.0;
  const char* stop_reason = "not started";
  bool completed = false;

  try {
    const std::filesystem::path executable_directory =
        std::filesystem::canonical("/proc/self/exe").parent_path();
    const std::filesystem::path resource_directory =
        executable_directory / UR_EXAMPLE_RESOURCE_RELATIVE_DIR;
    const std::string output_recipe_file = (resource_directory / "rtde_output_recipe.txt").string();
    const std::string input_recipe_file = (resource_directory / "rtde_input_recipe.txt").string();
    const std::string script_file = (resource_directory / "external_control.urscript").string();
    urcl::ExampleRobotWrapper robot(
        arguments.robot_ip, output_recipe_file, input_recipe_file, true, "", script_file);
    if (!robot.isHealthy()) {
      std::cerr << "Universal Robots client initialization failed.\n";
      return EXIT_FAILURE;
    }
    const urcl::vector6d_t& initial_joint_position =
        uses_end_effector_force_target
            ? nvidia::srl::controllers::examples::ur::kEndEffectorForceInitialJointPosition
            : nvidia::srl::controllers::examples::ur::kInitialJointPosition;
    if (!nvidia::srl::controllers::examples::ur::move_to_initial_joint_position(
            robot.getUrDriver(), initial_joint_position)) {
      std::cerr << "Failed to move to the initial joint position.\n";
      return EXIT_FAILURE;
    }
    if (!nvidia::srl::controllers::examples::ur::confirm_controller_start()) {
      return EXIT_FAILURE;
    }

    urcl::UrDriver& driver = *robot.getUrDriver();
    ControlStopGuard control_guard(driver);
    LowPassFilter6 force_filter(5.0);

    bool initialized = false;
    double previous_timestamp_seconds = 0.0;
    Eigen::Matrix4d initial_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d desired_pose = Eigen::Matrix4d::Identity();
    urcl::rtde_interface::DataPackage data_package(driver.getRTDEOutputRecipe());

    while (!completed) {
      const auto loop_start = std::chrono::steady_clock::now();
      if (!driver.getDataPackage(data_package)) {
        stop_reason = "RTDE package unavailable";
        break;
      }
      ++samples;

      urcl::vector6d_t raw_tcp_pose{};
      urcl::vector6d_t raw_tcp_force{};
      double timestamp_seconds = 0.0;
      if (!data_package.getData("actual_TCP_pose", raw_tcp_pose) ||
          !data_package.getData("actual_TCP_force", raw_tcp_force) ||
          !data_package.getData("timestamp", timestamp_seconds)) {
        stop_reason = "RTDE package missing a required field";
        break;
      }
      if (!finite_vector(raw_tcp_pose) || !finite_vector(raw_tcp_force) ||
          !std::isfinite(timestamp_seconds)) {
        stop_reason = "non-finite RTDE input";
        break;
      }

      Eigen::Vector<double, 6> tcp_pose;
      Eigen::Vector<double, 6> tcp_force;
      for (std::size_t index = 0; index < 6; ++index) {
        tcp_pose(static_cast<Eigen::Index>(index)) = raw_tcp_pose[index];
        tcp_force(static_cast<Eigen::Index>(index)) = raw_tcp_force[index];
      }
      const Eigen::Matrix4d measured_pose = pose_vector_to_transform(tcp_pose);
      if (!measured_pose.allFinite()) {
        stop_reason = "invalid TCP transform";
        break;
      }

      if (!initialized) {
        initialized = true;
        previous_timestamp_seconds = timestamp_seconds;
        initial_pose = measured_pose;
        desired_pose = measured_pose;
        if (!driver.writeJointCommand(transform_to_pose_vector(desired_pose),
                                      urcl::comm::ControlMode::MODE_POSE,
                                      urcl::RobotReceiveTimeout::millisec(100))) {
          stop_reason = "initial pose command rejected";
          break;
        }
        continue;
      }

      const double dt_seconds = timestamp_seconds - previous_timestamp_seconds;
      previous_timestamp_seconds = timestamp_seconds;
      if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0 || dt_seconds > kMaximumPeriodSeconds) {
        stop_reason = "invalid RTDE timestamp delta";
        break;
      }
      elapsed_seconds += dt_seconds;

      const Eigen::Vector<double, 6> filtered_force = force_filter.update(tcp_force, dt_seconds);
      Eigen::Matrix3d rotation_base_from_controller = Eigen::Matrix3d::Identity();
      Eigen::Vector<double, 6> controller_wrench = filtered_force;
      Eigen::Vector<double, 6> controller_pose_offset =
          Eigen::Map<const Eigen::Vector<double, 6>>(pose_offset.data());
      if (uses_end_effector_force_target) {
        rotation_base_from_controller = measured_pose.topLeftCorner<3, 3>();
        const Eigen::Matrix3d rotation_controller_from_base =
            rotation_base_from_controller.transpose();
        controller_wrench = rotate_cartesian_vector(rotation_controller_from_base, filtered_force);
        controller_pose_offset =
            rotate_cartesian_vector(rotation_controller_from_base, controller_pose_offset);
        update_cartesian_matrix(rotation_base_from_controller, stiffness_diagonal, &stiffness);
        update_cartesian_matrix(rotation_base_from_controller, damping_diagonal, &damping);
        update_cartesian_matrix(
            rotation_base_from_controller, virtual_mass_diagonal, &virtual_mass);
      }
      update_desired_external_wrench(
          elapsed_seconds, uses_end_effector_force_target, &target_wrench);
      for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
        measured_wrench[axis] = controller_wrench[static_cast<Eigen::Index>(axis)];
        shaping_error[axis] = -controller_pose_offset[static_cast<Eigen::Index>(axis)];
      }
      shaping_status[0] = 0;
      const controllers::Result shaping_result = shaping_function.step(shaping_inputs);
      for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
        external_wrench[axis] = shaping_factor[axis] * measured_wrench[axis];
        desired_wrench[axis] = shaping_factor[axis] * target_wrench[axis];
      }
      if (uses_end_effector_force_target) {
        const Eigen::Vector<double, 6> external_wrench_base = rotate_cartesian_vector(
            rotation_base_from_controller,
            Eigen::Map<const Eigen::Vector<double, 6>>(external_wrench.data()));
        const Eigen::Vector<double, 6> desired_wrench_base = rotate_cartesian_vector(
            rotation_base_from_controller,
            Eigen::Map<const Eigen::Vector<double, 6>>(desired_wrench.data()));
        std::copy(external_wrench_base.data(),
                  external_wrench_base.data() + kCartesianDim,
                  external_wrench.begin());
        std::copy(desired_wrench_base.data(),
                  desired_wrench_base.data() + kCartesianDim,
                  desired_wrench.begin());
      }
      dt[0] = dt_seconds;
      status[0] = 0;
      const controllers::Result controller_result = controller.step(inputs);
      const Eigen::Map<const Eigen::Vector<double, 6>> acceleration(commanded_acceleration.data());
      const Eigen::Map<const Eigen::Vector<double, 6>> desired_twist(twist.data());
      const Eigen::Map<const Eigen::Vector<double, 6>> desired_pose_offset(pose_offset.data());
      desired_pose =
          controllers::examples::apply_cartesian_pose_offset(initial_pose, desired_pose_offset);
      if (!shaping_result.ok() ||
          controllers::has_status_flag(shaping_status[0], controllers::StatusFlag::kSafeMode) ||
          !controller_result.ok() ||
          controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode) ||
          !filtered_force.allFinite() || !acceleration.allFinite() || !desired_twist.allFinite() ||
          !desired_pose_offset.allFinite() || !desired_pose.allFinite()) {
        stop_reason = "invalid force/admittance target";
        break;
      }

      if (!driver.writeJointCommand(transform_to_pose_vector(desired_pose),
                                    urcl::comm::ControlMode::MODE_POSE,
                                    urcl::RobotReceiveTimeout::millisec(100))) {
        stop_reason = "pose command rejected";
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

#ifndef UR_UNIFIED_FORCE_ADMITTANCE_NO_MAIN
int main(int argc, char* argv[]) {
  return nvidia::srl::controllers::examples::ur::run_unified_force_admittance_example(
      argc, argv, nvidia::srl::controllers::examples::ur::ForceTargetFrame::kRobotBase);
}
#endif
