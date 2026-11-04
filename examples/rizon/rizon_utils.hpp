/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file rizon_utils.hpp
 * @brief Explicit Flexiv Rizon lifecycle, frame, torque, and reporting helpers.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <flexiv/rdk/data.hpp>
#include <flexiv/rdk/mode.hpp>
#include <flexiv/rdk/model.hpp>
#include <flexiv/rdk/robot.hpp>
#include <flexiv/rdk/scheduler.hpp>

namespace rizon_example {

namespace rdk = flexiv::rdk;

constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;
constexpr double kNominalPeriodSeconds = 0.001;
constexpr double kDefaultRunDurationSeconds = 10.0;
constexpr double kEnableTimeoutSeconds = 30.0;
constexpr double kInitialMotionTimeoutSeconds = 60.0;
constexpr double kInitialMotionMaximumVelocity = 0.5;
constexpr double kInitialMotionMaximumAcceleration = 1.0;
constexpr double kInitialPositionTolerance = 0.01;
constexpr double kInitialVelocityTolerance = 0.02;
constexpr std::size_t kMaximumConsecutiveStaleStateSamples = 2;
constexpr double kTorqueMagnitudeLimitFactor = 1.0;
constexpr double kTorqueSlewRateNmPerSecond = 1000.0;
constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
constexpr std::array<double, kJointCount> kInitialJointPosition{
    0.0, -15.0 * kDegreesToRadians, 0.0, 75.0 * kDegreesToRadians, 0.0, 0.0, 0.0};
inline volatile std::sig_atomic_t g_signal_requested = 0;

using JointVector = Eigen::Matrix<double, kJointCount, 1>;
using JointMatrix = Eigen::Matrix<double, kJointCount, kJointCount>;
using CartesianVector = Eigen::Matrix<double, kCartesianDim, 1>;
using Jacobian = Eigen::Matrix<double, kCartesianDim, kJointCount>;

struct CycleData {
  JointVector joint_position{};
  JointVector joint_velocity{};
  JointVector measured_torque{};
  JointVector coriolis_torque{};
  JointMatrix mass{};
  Jacobian world_jacobian{};
  Jacobian tcp_jacobian{};
  Eigen::Matrix4d world_tcp_pose{Eigen::Matrix4d::Identity()};
  CartesianVector external_wrench_world{};
  CartesianVector external_wrench_tcp{};
  std::int64_t state_timestamp_nanoseconds{0};
};

struct CartesianMotionCommand {
  std::array<double, rdk::kPoseSize> pose{};
};

struct ExampleDescription {
  const char* title;
  const char* configuration;
  double torque_magnitude_limit_factor{kTorqueMagnitudeLimitFactor};
  double friction_compensation_percent{100.0};
  std::array<double, kJointCount> initial_joint_position{kInitialJointPosition};
};

struct ProgramArguments {
  std::string robot_serial_number;
  double run_duration_seconds{kDefaultRunDurationSeconds};
  bool non_rt_requested{false};
};

inline bool parse_arguments(const int argc, char** argv, ProgramArguments* arguments) {
  if (arguments == nullptr || argv == nullptr || argc < 2 || argc > 4 || argv[1] == nullptr ||
      argv[1][0] == '\0' || argv[1][0] == '-') {
    return false;
  }
  arguments->robot_serial_number = argv[1];
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
    if (argument.rfind(kDurationPrefix, 0) != 0 || has_duration) {
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
    } catch (...) {
      return false;
    }
  }
  return true;
}

enum class StartupAction { kReady, kEnable, kWait, kReject };

inline StartupAction startup_action(const rdk::OperationalStatus status,
                                    const bool fault,
                                    const bool reduced,
                                    const bool recovery,
                                    const bool estop_released) {
  if (fault || reduced || recovery || !estop_released) {
    return StartupAction::kReject;
  }
  switch (status) {
    case rdk::OperationalStatus::READY:
      return StartupAction::kReady;
    case rdk::OperationalStatus::NOT_ENABLED:
      return StartupAction::kEnable;
    case rdk::OperationalStatus::BOOTING:
    case rdk::OperationalStatus::RELEASING_BRAKE:
      return StartupAction::kWait;
    default:
      return StartupAction::kReject;
  }
}

inline std::string format_joint_position_degrees(
    const std::array<double, kJointCount>& joint_position) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    if (joint > 0) {
      stream << ", ";
    }
    stream << joint_position[joint] / kDegreesToRadians;
  }
  stream << ']';
  return stream.str();
}

inline bool confirm_execution(const ExampleDescription& description,
                              const double run_duration_seconds,
                              const bool non_rt_requested,
                              const bool uses_joint_torque_interface = true) {
  std::cout << description.title << "\n\n"
            << description.configuration << "\n"
            << "Before starting the controller, the robot will move in NRT_JOINT_POSITION mode "
               "to "
            << format_joint_position_degrees(description.initial_joint_position) << " degrees.\n"
            << "After that motion, a second confirmation is required. The controller then starts "
               "from measured state, retains configured robot safety settings, and runs for "
            << run_duration_seconds
            << " s. It never clears faults, runs recovery, homes, or "
               "executes a plan.\n";
  if (uses_joint_torque_interface) {
    std::cout << "Torque magnitude limit: " << description.torque_magnitude_limit_factor * 100.0
              << "% of RobotInfo::tau_max; friction compensation: "
              << description.friction_compensation_percent << "%.\n";
  } else {
    std::cout << "Interface: RT_CARTESIAN_MOTION_FORCE in pure Cartesian motion mode.\n";
  }
  std::cout << "Keep an emergency stop within reach. Flexiv RDK state/model calls allocate and "
               "can throw; this functional 1 kHz example is not hard-real-time qualified.\n"
            << "non_rt_requested=" << (non_rt_requested ? "true" : "false") << '\n';
  if (non_rt_requested) {
    std::cout << "Warning: --non-rt uses the RDK scheduler's non-real-time priority and leaves "
                 "the task unbound from any CPU. Runtime timeliness and safety checks remain "
                 "enabled.\n";
  }
  std::cout << '\n'
            << "Press Enter to connect and execute the initial joint motion: " << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before the initial motion.\n";
    return false;
  }
  return true;
}

inline bool confirm_controller_start(
    const std::array<double, kJointCount>& initial_joint_position = kInitialJointPosition) {
  std::cout << "Initial position reached: " << format_joint_position_degrees(initial_joint_position)
            << " degrees.\n"
            << "Press Enter to start the example controller: " << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before starting the controller.\n";
    return false;
  }
  return true;
}

template <typename Container>
inline bool all_finite(const Container& values) {
  return std::all_of(
      values.begin(), values.end(), [](const double value) { return std::isfinite(value); });
}

template <typename Derived>
inline bool all_finite(const Eigen::MatrixBase<Derived>& values) {
  return values.allFinite();
}

inline bool valid_robot_info(const rdk::RobotInfo& info) {
  if (info.DoF_e != 0 || info.DoF_m != kJointCount || info.DoF != kJointCount ||
      info.q_min.size() != kJointCount || info.q_max.size() != kJointCount ||
      info.dq_max.size() != kJointCount || info.tau_max.size() != kJointCount ||
      !all_finite(info.q_min) || !all_finite(info.q_max) || !all_finite(info.dq_max) ||
      !all_finite(info.tau_max)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    if (!(info.q_min[joint] < info.q_max[joint]) || info.dq_max[joint] <= 0.0 ||
        info.tau_max[joint] <= 0.0) {
      return false;
    }
  }
  return true;
}

inline bool valid_robot_state(const rdk::RobotStates& state) {
  return state.q.size() == kJointCount && state.dq.size() == kJointCount &&
         state.tau.size() == kJointCount && all_finite(state.q) && all_finite(state.dq) &&
         all_finite(state.tau) && all_finite(state.tcp_pose) && all_finite(state.flange_pose) &&
         all_finite(state.ext_wrench_in_world) && all_finite(state.ext_wrench_in_tcp);
}

inline bool is_initial_position_valid(
    const rdk::RobotInfo& info,
    const std::array<double, kJointCount>& initial_joint_position = kInitialJointPosition) {
  if (!valid_robot_info(info)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    if (initial_joint_position[joint] < info.q_min[joint] ||
        initial_joint_position[joint] > info.q_max[joint]) {
      return false;
    }
  }
  return true;
}

inline bool has_reached_initial_position(
    const rdk::RobotStates& state,
    const std::array<double, kJointCount>& initial_joint_position = kInitialJointPosition) {
  if (state.q.size() != kJointCount || state.dq.size() != kJointCount || !all_finite(state.q) ||
      !all_finite(state.dq)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    if (std::abs(state.q[joint] - initial_joint_position[joint]) > kInitialPositionTolerance ||
        std::abs(state.dq[joint]) > kInitialVelocityTolerance) {
      return false;
    }
  }
  return true;
}

inline bool move_to_initial_position(rdk::Robot& robot,
                                     const rdk::RobotInfo& info,
                                     const std::array<double, kJointCount>& initial_joint_position,
                                     std::string& failure) {
  if (!is_initial_position_valid(info, initial_joint_position)) {
    failure = "configured initial position is outside the robot's software joint limits";
    return false;
  }
  const std::vector<double> target(initial_joint_position.begin(), initial_joint_position.end());
  const std::vector<double> target_velocity(kJointCount, 0.0);
  std::vector<double> maximum_velocity(kJointCount, kInitialMotionMaximumVelocity);
  const std::vector<double> maximum_acceleration(kJointCount, kInitialMotionMaximumAcceleration);
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    maximum_velocity[joint] = std::min(maximum_velocity[joint], info.dq_max[joint]);
  }
  robot.SwitchMode(rdk::Mode::NRT_JOINT_POSITION);
  robot.SendJointPosition(target, target_velocity, maximum_velocity, maximum_acceleration);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(kInitialMotionTimeoutSeconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (g_signal_requested != 0) {
      failure = "operator signal received during the initial motion";
      return false;
    }
    if (robot.fault() || robot.operational_status() != rdk::OperationalStatus::READY) {
      failure = "robot left READY state during the initial motion";
      return false;
    }
    if (has_reached_initial_position(robot.states(), initial_joint_position)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  failure = "initial motion did not converge within 60 seconds";
  return false;
}

inline bool convert_timestamp_to_nanoseconds(const std::pair<int, int>& timestamp,
                                             std::int64_t& nanoseconds) {
  constexpr std::int64_t kNanosecondsPerSecond = 1000000000;
  if (timestamp.first < 0 || timestamp.second < 0 || timestamp.second >= kNanosecondsPerSecond) {
    return false;
  }
  nanoseconds = static_cast<std::int64_t>(timestamp.first) * kNanosecondsPerSecond +
                static_cast<std::int64_t>(timestamp.second);
  return true;
}

inline bool pose_from_rdk(const std::array<double, rdk::kPoseSize>& pose,
                          Eigen::Matrix4d& transform) {
  if (!all_finite(pose)) {
    return false;
  }
  Eigen::Quaterniond quaternion(pose[3], pose[4], pose[5], pose[6]);
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    return false;
  }
  quaternion.normalize();
  transform.setIdentity();
  transform.topLeftCorner<3, 3>() = quaternion.toRotationMatrix();
  transform.topRightCorner<3, 1>() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
  return transform.allFinite();
}

inline bool pose_to_rdk(const Eigen::Matrix4d& transform,
                        std::array<double, rdk::kPoseSize>& pose) {
  if (!transform.allFinite()) {
    return false;
  }
  Eigen::Quaterniond quaternion(transform.topLeftCorner<3, 3>());
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    return false;
  }
  quaternion.normalize();
  pose = {transform(0, 3),
          transform(1, 3),
          transform(2, 3),
          quaternion.w(),
          quaternion.x(),
          quaternion.y(),
          quaternion.z()};
  return all_finite(pose);
}

inline bool fill_cartesian_cycle_data(const rdk::RobotStates& state, CycleData& data) {
  if (!valid_robot_state(state) ||
      !convert_timestamp_to_nanoseconds(state.timestamp, data.state_timestamp_nanoseconds) ||
      !pose_from_rdk(state.tcp_pose, data.world_tcp_pose)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    data.joint_position[static_cast<Eigen::Index>(joint)] = state.q[joint];
    data.joint_velocity[static_cast<Eigen::Index>(joint)] = state.dq[joint];
    data.measured_torque[static_cast<Eigen::Index>(joint)] = state.tau[joint];
  }
  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    data.external_wrench_world[static_cast<Eigen::Index>(axis)] = state.ext_wrench_in_world[axis];
    data.external_wrench_tcp[static_cast<Eigen::Index>(axis)] = state.ext_wrench_in_tcp[axis];
  }
  return data.joint_position.allFinite() && data.joint_velocity.allFinite() &&
         data.measured_torque.allFinite() && data.world_tcp_pose.allFinite() &&
         data.external_wrench_world.allFinite() && data.external_wrench_tcp.allFinite();
}

inline bool derive_tcp_jacobians(const Jacobian& flange_world_jacobian,
                                 const Eigen::Matrix4d& world_flange_pose,
                                 const Eigen::Matrix4d& world_tcp_pose,
                                 Jacobian& world_tcp_jacobian,
                                 Jacobian& tcp_jacobian) {
  if (!flange_world_jacobian.allFinite() || !world_flange_pose.allFinite() ||
      !world_tcp_pose.allFinite()) {
    return false;
  }
  const Eigen::Vector3d offset_world =
      world_tcp_pose.topRightCorner<3, 1>() - world_flange_pose.topRightCorner<3, 1>();
  Eigen::Matrix3d skew;
  skew << 0.0, -offset_world.z(), offset_world.y(), offset_world.z(), 0.0, -offset_world.x(),
      -offset_world.y(), offset_world.x(), 0.0;
  world_tcp_jacobian.bottomRows<3>() = flange_world_jacobian.bottomRows<3>();
  world_tcp_jacobian.topRows<3>() =
      flange_world_jacobian.topRows<3>() - skew * flange_world_jacobian.bottomRows<3>();
  const Eigen::Matrix3d world_rotation_tcp = world_tcp_pose.topLeftCorner<3, 3>();
  tcp_jacobian.topRows<3>() = world_rotation_tcp.transpose() * world_tcp_jacobian.topRows<3>();
  tcp_jacobian.bottomRows<3>() =
      world_rotation_tcp.transpose() * world_tcp_jacobian.bottomRows<3>();
  return world_tcp_jacobian.allFinite() && tcp_jacobian.allFinite();
}

inline bool fill_cycle_data(const rdk::RobotStates& state, rdk::Model& model, CycleData& data) {
  if (!valid_robot_state(state) ||
      !convert_timestamp_to_nanoseconds(state.timestamp, data.state_timestamp_nanoseconds)) {
    return false;
  }
  model.Update(state.q, state.dq);
  const Eigen::MatrixXd dynamic_mass = model.M();
  const Eigen::MatrixXd dynamic_flange_jacobian = model.J("flange");
  const Eigen::VectorXd dynamic_coriolis = model.c();
  if (dynamic_mass.rows() != static_cast<Eigen::Index>(kJointCount) ||
      dynamic_mass.cols() != static_cast<Eigen::Index>(kJointCount) ||
      dynamic_flange_jacobian.rows() != static_cast<Eigen::Index>(kCartesianDim) ||
      dynamic_flange_jacobian.cols() != static_cast<Eigen::Index>(kJointCount) ||
      dynamic_coriolis.size() != static_cast<Eigen::Index>(kJointCount) ||
      !dynamic_mass.allFinite() || !dynamic_flange_jacobian.allFinite() ||
      !dynamic_coriolis.allFinite()) {
    return false;
  }
  Eigen::Matrix4d world_flange_pose;
  if (!pose_from_rdk(state.flange_pose, world_flange_pose) ||
      !pose_from_rdk(state.tcp_pose, data.world_tcp_pose)) {
    return false;
  }
  data.mass = dynamic_mass;
  data.world_jacobian = dynamic_flange_jacobian;
  if (!derive_tcp_jacobians(dynamic_flange_jacobian,
                            world_flange_pose,
                            data.world_tcp_pose,
                            data.world_jacobian,
                            data.tcp_jacobian)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    data.joint_position[static_cast<Eigen::Index>(joint)] = state.q[joint];
    data.joint_velocity[static_cast<Eigen::Index>(joint)] = state.dq[joint];
    data.measured_torque[static_cast<Eigen::Index>(joint)] = state.tau[joint];
    data.coriolis_torque[static_cast<Eigen::Index>(joint)] =
        dynamic_coriolis[static_cast<Eigen::Index>(joint)];
  }
  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    data.external_wrench_world[static_cast<Eigen::Index>(axis)] = state.ext_wrench_in_world[axis];
    data.external_wrench_tcp[static_cast<Eigen::Index>(axis)] = state.ext_wrench_in_tcp[axis];
  }
  return data.joint_position.allFinite() && data.joint_velocity.allFinite() &&
         data.measured_torque.allFinite() && data.coriolis_torque.allFinite() &&
         data.mass.allFinite() && data.external_wrench_world.allFinite() &&
         data.external_wrench_tcp.allFinite();
}

inline bool filter_torques(const JointVector& desired_torques,
                           const std::vector<double>& torque_limits,
                           const double dt_seconds,
                           const double magnitude_limit_factor,
                           JointVector& previous_torques,
                           std::vector<double>& streamed_torques,
                           std::array<bool, kJointCount>& magnitude_limited,
                           std::array<bool, kJointCount>& slew_limited) {
  if (torque_limits.size() != kJointCount || streamed_torques.size() != kJointCount ||
      !desired_torques.allFinite() || !previous_torques.allFinite() || !std::isfinite(dt_seconds) ||
      dt_seconds <= 0.0 || !std::isfinite(magnitude_limit_factor) ||
      magnitude_limit_factor <= 0.0 || magnitude_limit_factor > 1.0) {
    return false;
  }
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    const double robot_limit = torque_limits[joint];
    const Eigen::Index index = static_cast<Eigen::Index>(joint);
    if (!std::isfinite(robot_limit) || robot_limit <= 0.0) {
      return false;
    }
    const double command_limit = magnitude_limit_factor * robot_limit;
    const double maximum_delta = kTorqueSlewRateNmPerSecond * dt_seconds;
    const double magnitude_limited_torque =
        std::clamp(desired_torques[index], -command_limit, command_limit);
    const double delta = std::clamp(
        magnitude_limited_torque - previous_torques[index], -maximum_delta, maximum_delta);
    magnitude_limited[joint] = magnitude_limited_torque != desired_torques[index];
    slew_limited[joint] = delta != magnitude_limited_torque - previous_torques[index];
    previous_torques[index] += delta;
    streamed_torques[joint] = previous_torques[index];
  }
  return true;
}

inline bool wait_until_ready(rdk::Robot& robot, std::string& failure) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(kEnableTimeoutSeconds);
  bool enable_requested = false;
  while (std::chrono::steady_clock::now() < deadline) {
    const rdk::OperationalStatus status = robot.operational_status();
    const StartupAction action = startup_action(
        status, robot.fault(), robot.reduced(), robot.recovery(), robot.estop_released());
    if (action == StartupAction::kReady) {
      return true;
    }
    if (action == StartupAction::kReject) {
      failure = "robot status rejected: " + rdk::kOpStatusNames[static_cast<std::size_t>(status)];
      return false;
    }
    if (action == StartupAction::kEnable) {
      if (!enable_requested) {
        robot.Enable();
        enable_requested = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  failure = "robot did not become READY within 30 seconds";
  return false;
}

class RobotStopGuard {
 public:
  explicit RobotStopGuard(rdk::Robot& robot) : robot_(robot) {}
  RobotStopGuard(const RobotStopGuard&) = delete;
  RobotStopGuard& operator=(const RobotStopGuard&) = delete;

  ~RobotStopGuard() {
    if (armed_) {
      try {
        robot_.Stop();
      } catch (...) {
      }
    }
  }

  void arm() { armed_ = true; }
  bool armed() const { return armed_; }

  void stop() {
    if (armed_) {
      robot_.Stop();
      armed_ = false;
    }
  }

 private:
  rdk::Robot& robot_;
  bool armed_{false};
};

class SchedulerStopGuard {
 public:
  explicit SchedulerStopGuard(rdk::Scheduler& scheduler) : scheduler_(scheduler) {}
  SchedulerStopGuard(const SchedulerStopGuard&) = delete;
  SchedulerStopGuard& operator=(const SchedulerStopGuard&) = delete;

  ~SchedulerStopGuard() {
    if (started_) {
      try {
        scheduler_.Stop();
      } catch (...) {
      }
    }
  }

  void mark_started() { started_ = true; }
  void stop() {
    if (started_) {
      scheduler_.Stop();
      started_ = false;
    }
  }

 private:
  rdk::Scheduler& scheduler_;
  bool started_{false};
};

enum class StopReason {
  kNone,
  kDurationReached,
  kSignal,
  kRobotNotReady,
  kStaleState,
  kInvalidState,
  kControllerRejected,
  kInvalidTorque,
  kTimelinessLimit,
  kRdkException,
};

inline const char* get_stop_reason_message(const StopReason reason) {
  switch (reason) {
    case StopReason::kNone:
      return "no stop reason was recorded";
    case StopReason::kDurationReached:
      return "requested duration reached";
    case StopReason::kSignal:
      return "interrupt signal received";
    case StopReason::kRobotNotReady:
      return "robot is not ready or entered a fault state";
    case StopReason::kStaleState:
      return "robot state data stopped updating";
    case StopReason::kInvalidState:
      return "robot or model state is invalid";
    case StopReason::kControllerRejected:
      return "controller rejected its input or output";
    case StopReason::kInvalidTorque:
      return "computed joint torque is invalid";
    case StopReason::kTimelinessLimit:
      return "Flexiv RDK timeliness failure limit was reached";
    case StopReason::kRdkException:
      return "Flexiv RDK operation threw an exception";
  }
  return "unknown stop reason";
}

struct RunState {
  std::atomic<bool> stop_requested{false};
  std::atomic<StopReason> stop_reason{StopReason::kNone};
  std::size_t samples{0};
  double elapsed_seconds{0.0};
  std::chrono::steady_clock::time_point previous_start{};

  void request_stop(const StopReason requested_reason) {
    StopReason expected = StopReason::kNone;
    stop_reason.compare_exchange_strong(expected, requested_reason);
    stop_requested.store(true);
  }
};

inline void signal_handler(int) { g_signal_requested = 1; }

template <typename Workspace>
int run_example(const int argc,
                char** argv,
                const ExampleDescription& description,
                Workspace& workspace) {
  ProgramArguments arguments;
  if (!parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_SERIAL_NUMBER [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }

  const auto prepare_result = workspace.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare controller: " << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
  if (!confirm_execution(description, arguments.run_duration_seconds, arguments.non_rt_requested)) {
    return EXIT_FAILURE;
  }

  RunState run_state;
  g_signal_requested = 0;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    rdk::Robot robot(arguments.robot_serial_number);
    RobotStopGuard robot_guard(robot);
    const rdk::RobotInfo info = robot.info();
    if (!valid_robot_info(info)) {
      std::cerr << "Rizon examples require exactly seven manipulator joints, no external axes, "
                   "and finite positive joint limits.\n";
      return EXIT_FAILURE;
    }
    std::string startup_failure;
    if (!wait_until_ready(robot, startup_failure)) {
      std::cerr << "Robot startup rejected: " << startup_failure << '\n';
      return EXIT_FAILURE;
    }
    robot_guard.arm();
    if (!move_to_initial_position(
            robot, info, description.initial_joint_position, startup_failure)) {
      std::cerr << "Initial motion rejected: " << startup_failure << '\n';
      return EXIT_FAILURE;
    }
    robot_guard.stop();
    if (!confirm_controller_start(description.initial_joint_position)) {
      return EXIT_FAILURE;
    }

    rdk::Model model(robot);
    CycleData initial_data;
    if (!fill_cycle_data(robot.states(), model, initial_data) ||
        !workspace.set_initial_state(initial_data)) {
      std::cerr << "Initial robot/model state is invalid or the controller rejected it.\n";
      return EXIT_FAILURE;
    }

    JointVector previous_torques = JointVector::Zero();
    std::vector<double> streamed_torques(kJointCount, 0.0);
    std::array<bool, kJointCount> magnitude_limited{};
    std::array<bool, kJointCount> slew_limited{};
    CycleData cycle_data;
    std::int64_t previous_state_timestamp_nanoseconds = initial_data.state_timestamp_nanoseconds;
    std::size_t consecutive_stale_state_samples = 0;
    rdk::Scheduler scheduler;
    SchedulerStopGuard scheduler_guard(scheduler);

    scheduler.AddTask(
        [&]() {
          const auto callback_start = std::chrono::steady_clock::now();
          double dt_seconds = kNominalPeriodSeconds;
          if (run_state.samples > 0) {
            dt_seconds =
                std::chrono::duration<double>(callback_start - run_state.previous_start).count();
          }
          run_state.previous_start = callback_start;
          ++run_state.samples;
          if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
            run_state.request_stop(StopReason::kInvalidState);
            return;
          }
          run_state.elapsed_seconds += dt_seconds;

          try {
            if (robot.operational_status() != rdk::OperationalStatus::READY || robot.fault()) {
              run_state.request_stop(StopReason::kRobotNotReady);
              return;
            }
            if (robot.reached_timeliness_failure_limit()) {
              run_state.request_stop(StopReason::kTimelinessLimit);
              return;
            }
            if (!fill_cycle_data(robot.states(), model, cycle_data)) {
              run_state.request_stop(StopReason::kInvalidState);
              return;
            }
            if (cycle_data.state_timestamp_nanoseconds < previous_state_timestamp_nanoseconds) {
              run_state.request_stop(StopReason::kStaleState);
              return;
            }
            if (cycle_data.state_timestamp_nanoseconds == previous_state_timestamp_nanoseconds) {
              ++consecutive_stale_state_samples;
              if (consecutive_stale_state_samples > kMaximumConsecutiveStaleStateSamples) {
                run_state.request_stop(StopReason::kStaleState);
              }
              return;
            }
            consecutive_stale_state_samples = 0;
            previous_state_timestamp_nanoseconds = cycle_data.state_timestamp_nanoseconds;
            JointVector desired_torques;
            if (!workspace.step(
                    cycle_data, run_state.elapsed_seconds, dt_seconds, desired_torques)) {
              run_state.request_stop(StopReason::kControllerRejected);
              return;
            }
            desired_torques += cycle_data.coriolis_torque;
            if (!filter_torques(desired_torques,
                                info.tau_max,
                                dt_seconds,
                                description.torque_magnitude_limit_factor,
                                previous_torques,
                                streamed_torques,
                                magnitude_limited,
                                slew_limited)) {
              run_state.request_stop(StopReason::kInvalidTorque);
              return;
            }
            robot.StreamJointTorque(
                streamed_torques, true, true, description.friction_compensation_percent);
          } catch (...) {
            run_state.request_stop(StopReason::kRdkException);
            return;
          }

          if (run_state.elapsed_seconds >= arguments.run_duration_seconds) {
            run_state.request_stop(StopReason::kDurationReached);
          }
        },
        "controllers-rizon-torque",
        1,
        arguments.non_rt_requested ? 0 : scheduler.max_priority(),
        -1);

    robot.SetTimelinessFailureLimit(2.0);
    robot.SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
    robot_guard.arm();
    scheduler.Start();
    scheduler_guard.mark_started();
    while (!run_state.stop_requested.load()) {
      if (g_signal_requested != 0) {
        run_state.request_stop(StopReason::kSignal);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler_guard.stop();
    robot_guard.stop();
    const StopReason stop_reason = run_state.stop_reason.load();
    const bool completed = stop_reason == StopReason::kDurationReached;
    if (!completed && stop_reason != StopReason::kSignal) {
      std::cerr << "Controller stopped after " << run_state.elapsed_seconds
                << " s: " << get_stop_reason_message(stop_reason) << ".\n";
    }
    workspace.print_observation();
    return completed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "Rizon control failed: " << error.what() << '\n';
    if (run_state.stop_reason.load() == StopReason::kNone) {
      run_state.request_stop(StopReason::kRdkException);
    }
    return EXIT_FAILURE;
  }
}

template <typename Workspace>
int run_cartesian_motion_example(const int argc,
                                 char** argv,
                                 const ExampleDescription& description,
                                 Workspace& workspace) {
  ProgramArguments arguments;
  if (!parse_arguments(argc, argv, &arguments)) {
    std::cerr << "Usage: " << argv[0]
              << " ROBOT_SERIAL_NUMBER [--duration-seconds=SECONDS] [--non-rt]\n";
    return EXIT_FAILURE;
  }
  const auto prepare_result = workspace.prepare();
  if (!prepare_result.ok()) {
    std::cerr << "Failed to prepare controller: " << prepare_result.message << '\n';
    return EXIT_FAILURE;
  }
  if (!confirm_execution(
          description, arguments.run_duration_seconds, arguments.non_rt_requested, false)) {
    return EXIT_FAILURE;
  }
  RunState run_state;
  g_signal_requested = 0;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    rdk::Robot robot(arguments.robot_serial_number);
    RobotStopGuard robot_guard(robot);
    const rdk::RobotInfo info = robot.info();
    if (!valid_robot_info(info)) {
      std::cerr << "Rizon examples require exactly seven manipulator joints, no external axes, "
                   "and finite positive joint limits.\n";
      return EXIT_FAILURE;
    }
    std::string startup_failure;
    if (!wait_until_ready(robot, startup_failure)) {
      std::cerr << "Robot startup rejected: " << startup_failure << '\n';
      return EXIT_FAILURE;
    }
    robot_guard.arm();
    if (!move_to_initial_position(
            robot, info, description.initial_joint_position, startup_failure)) {
      std::cerr << "Initial motion rejected: " << startup_failure << '\n';
      return EXIT_FAILURE;
    }
    robot_guard.stop();
    if (!confirm_controller_start(description.initial_joint_position)) {
      return EXIT_FAILURE;
    }
    CycleData initial_data;
    if (!fill_cartesian_cycle_data(robot.states(), initial_data) ||
        !workspace.set_initial_state(initial_data)) {
      std::cerr << "Initial robot state is invalid or the controller rejected it.\n";
      return EXIT_FAILURE;
    }
    CycleData cycle_data;
    CartesianMotionCommand command;
    std::int64_t previous_state_timestamp_nanoseconds = initial_data.state_timestamp_nanoseconds;
    std::size_t consecutive_stale_state_samples = 0;
    rdk::Scheduler scheduler;
    SchedulerStopGuard scheduler_guard(scheduler);
    scheduler.AddTask(
        [&]() {
          const auto callback_start = std::chrono::steady_clock::now();
          double dt_seconds = kNominalPeriodSeconds;
          if (run_state.samples > 0) {
            dt_seconds =
                std::chrono::duration<double>(callback_start - run_state.previous_start).count();
          }
          run_state.previous_start = callback_start;
          ++run_state.samples;
          if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
            run_state.request_stop(StopReason::kInvalidState);
            return;
          }
          run_state.elapsed_seconds += dt_seconds;
          try {
            if (robot.operational_status() != rdk::OperationalStatus::READY || robot.fault()) {
              run_state.request_stop(StopReason::kRobotNotReady);
              return;
            }
            if (robot.reached_timeliness_failure_limit()) {
              run_state.request_stop(StopReason::kTimelinessLimit);
              return;
            }
            if (!fill_cartesian_cycle_data(robot.states(), cycle_data)) {
              run_state.request_stop(StopReason::kInvalidState);
              return;
            }
            if (cycle_data.state_timestamp_nanoseconds < previous_state_timestamp_nanoseconds) {
              run_state.request_stop(StopReason::kStaleState);
              return;
            }
            if (cycle_data.state_timestamp_nanoseconds == previous_state_timestamp_nanoseconds) {
              ++consecutive_stale_state_samples;
              if (consecutive_stale_state_samples > kMaximumConsecutiveStaleStateSamples) {
                run_state.request_stop(StopReason::kStaleState);
              }
              return;
            }
            consecutive_stale_state_samples = 0;
            previous_state_timestamp_nanoseconds = cycle_data.state_timestamp_nanoseconds;
            if (!workspace.step(cycle_data, run_state.elapsed_seconds, dt_seconds, command)) {
              run_state.request_stop(StopReason::kControllerRejected);
              return;
            }
            robot.StreamCartesianMotionForce(command.pose);
          } catch (...) {
            run_state.request_stop(StopReason::kRdkException);
            return;
          }
          if (run_state.elapsed_seconds >= arguments.run_duration_seconds) {
            run_state.request_stop(StopReason::kDurationReached);
          }
        },
        "controllers-rizon-cartesian-motion",
        1,
        arguments.non_rt_requested ? 0 : scheduler.max_priority(),
        -1);
    constexpr std::array<double, kCartesianDim> kCartesianStiffness{
        2000, 2000, 2000, 200, 200, 200};
    constexpr std::array<double, kCartesianDim> kCartesianDampingRatio{
        0.7, 0.7, 0.7, 0.7, 0.7, 0.7};
    constexpr std::array<bool, kCartesianDim> kForceControlAxes{
        false, false, false, false, false, false};
    robot.SetTimelinessFailureLimit(2.0);
    robot.SwitchMode(rdk::Mode::RT_CARTESIAN_MOTION_FORCE);
    robot_guard.arm();
    robot.SetForceControlAxis(kForceControlAxes);
    robot.SetCartesianImpedance(kCartesianStiffness, kCartesianDampingRatio);
    scheduler.Start();
    scheduler_guard.mark_started();
    while (!run_state.stop_requested.load()) {
      if (g_signal_requested != 0) {
        run_state.request_stop(StopReason::kSignal);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler_guard.stop();
    robot_guard.stop();
    const StopReason stop_reason = run_state.stop_reason.load();
    const bool completed = stop_reason == StopReason::kDurationReached;
    if (!completed && stop_reason != StopReason::kSignal) {
      std::cerr << "Controller stopped after " << run_state.elapsed_seconds
                << " s: " << get_stop_reason_message(stop_reason) << ".\n";
    }
    workspace.print_observation();
    return completed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "Rizon control failed: " << error.what() << '\n';
    if (run_state.stop_reason.load() == StopReason::kNone) {
      run_state.request_stop(StopReason::kRdkException);
    }
    return EXIT_FAILURE;
  }
}

}  // namespace rizon_example
