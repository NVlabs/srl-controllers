/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2017 Franka Emika GmbH
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file franka_utils.hpp
 * @brief Small, explicit safety and reporting helpers for the Franka examples.
 */

#pragma once

#include <franka/control_types.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <eigen3/Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

namespace franka_example {

constexpr double kDefaultRunDurationSeconds = 10.0;
constexpr double kInitialMotionSpeedFactor = 0.5;
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<double, 7> kInitialJointConfiguration{
    0.0, 0.0, 0.0, -3.0 * kPi / 4.0, 0.0, 3.0 * kPi / 4.0, kPi / 4.0};
constexpr std::array<double, 7> kExtendedInitialJointConfiguration{
    0.0, 0.0, 0.0, -2.3 * kPi / 4.0, 0.0, 2.3 * kPi / 4.0, kPi / 4.0};
constexpr std::array<double, 7> kMaximumJointTorqueNm{87.0, 87.0, 87.0, 87.0,
                                                      12.0, 12.0, 12.0};
constexpr std::array<double, 7> kCollisionTorqueThresholdNm{60.0, 60.0, 60.0, 60.0,
                                                            10.0, 10.0, 10.0};
constexpr std::array<double, 6> kCollisionWrenchThreshold{40.0, 40.0, 40.0, 10.0, 10.0, 10.0};

struct RunSummary {
  std::size_t samples{0};
  double elapsed_seconds{0.0};
  double maximum_callback_seconds{0.0};
  const char* stop_reason{"not started"};
  bool completed{false};
};

struct ProgramArguments {
  std::string robot_host;
  double run_duration_seconds{kDefaultRunDurationSeconds};
  bool non_rt_requested{false};
};

inline bool parse_arguments(const int argc, char** argv, ProgramArguments* arguments) {
  if (arguments == nullptr || argv == nullptr || argc < 2 || argc > 4 || argv[1] == nullptr ||
      argv[1][0] == '\0' || argv[1][0] == '-') {
    return false;
  }
  arguments->robot_host = argv[1];
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

inline franka::RealtimeConfig realtime_config(const bool non_rt_requested) noexcept {
  return non_rt_requested ? franka::RealtimeConfig::kIgnore : franka::RealtimeConfig::kEnforce;
}

inline void print_realtime_mode(const bool non_rt_requested) {
  std::cout << "non_rt_requested=" << (non_rt_requested ? "true" : "false") << '\n';
  if (non_rt_requested) {
    std::cout << "Warning: --non-rt asks libfranka to ignore failures to configure real-time "
                 "scheduling and the absence of a PREEMPT_RT kernel. libfranka may still acquire "
                 "real-time scheduling when the host permits it.\n";
  }
}

class InitialMotionGenerator {
 public:
  InitialMotionGenerator(const double speed_factor,
                         const std::array<double, 7>& target_joint_position)
      : target_joint_position_(target_joint_position.data()) {
    maximum_velocity_ *= speed_factor;
    maximum_start_acceleration_ *= speed_factor;
    maximum_goal_acceleration_ *= speed_factor;
  }

  franka::JointPositions operator()(const franka::RobotState& state,
                                    const franka::Duration period) {
    if (!initialized_) {
      start_joint_position_ = Vector7d(state.q.data());
      joint_displacement_ = target_joint_position_ - start_joint_position_;
      calculate_synchronized_values();
      initialized_ = true;
    }
    elapsed_seconds_ += period.toSec();

    Vector7d desired_displacement;
    const bool motion_finished = calculate_desired_values(elapsed_seconds_, desired_displacement);
    std::array<double, 7> desired_joint_position{};
    Eigen::Map<Vector7d>(desired_joint_position.data()) =
        start_joint_position_ + desired_displacement;
    franka::JointPositions command(desired_joint_position);
    command.motion_finished = motion_finished;
    return command;
  }

 private:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector7i = Eigen::Matrix<int, 7, 1>;

  bool calculate_desired_values(const double time, Vector7d& desired_displacement) const {
    const Vector7i displacement_sign = joint_displacement_.cwiseSign().cast<int>();
    const Vector7d constant_velocity_duration =
        synchronized_deceleration_start_time_ - synchronized_acceleration_end_time_;
    const Vector7d deceleration_duration =
        synchronized_finish_time_ - synchronized_deceleration_start_time_;
    std::array<bool, 7> joint_motion_finished{};

    for (std::size_t joint = 0; joint < joint_motion_finished.size(); ++joint) {
      if (std::abs(joint_displacement_[joint]) < kDeltaQMotionFinished) {
        desired_displacement[joint] = 0.0;
        joint_motion_finished[joint] = true;
      } else if (time < synchronized_acceleration_end_time_[joint]) {
        desired_displacement[joint] = -synchronized_velocity_[joint] * displacement_sign[joint] *
                                      (0.5 * time - synchronized_acceleration_end_time_[joint]) *
                                      std::pow(time, 3.0) /
                                      std::pow(synchronized_acceleration_end_time_[joint], 3.0);
      } else if (time < synchronized_deceleration_start_time_[joint]) {
        desired_displacement[joint] = acceleration_displacement_[joint] +
                                      (time - synchronized_acceleration_end_time_[joint]) *
                                          synchronized_velocity_[joint] * displacement_sign[joint];
      } else if (time < synchronized_finish_time_[joint]) {
        const double time_after_acceleration = time - synchronized_acceleration_end_time_[joint];
        desired_displacement[joint] =
            joint_displacement_[joint] +
            0.5 *
                ((time_after_acceleration - 2.0 * deceleration_duration[joint] -
                  constant_velocity_duration[joint]) *
                     std::pow(time_after_acceleration - constant_velocity_duration[joint], 3.0) /
                     std::pow(deceleration_duration[joint], 3.0) +
                 2.0 * time_after_acceleration - deceleration_duration[joint] -
                 2.0 * constant_velocity_duration[joint]) *
                synchronized_velocity_[joint] * displacement_sign[joint];
      } else {
        desired_displacement[joint] = joint_displacement_[joint];
        joint_motion_finished[joint] = true;
      }
    }
    return std::all_of(joint_motion_finished.cbegin(),
                       joint_motion_finished.cend(),
                       [](const bool finished) { return finished; });
  }

  void calculate_synchronized_values() {
    Vector7d reachable_velocity = maximum_velocity_;
    Vector7d finish_time = Vector7d::Zero();
    Vector7d deceleration_duration = Vector7d::Zero();
    Vector7d acceleration_duration = Vector7d::Zero();
    const Vector7i displacement_sign = joint_displacement_.cwiseSign().cast<int>();

    for (Eigen::Index joint = 0; joint < joint_displacement_.size(); ++joint) {
      if (std::abs(joint_displacement_[joint]) <= kDeltaQMotionFinished) {
        continue;
      }
      const double displacement_at_maximum_velocity =
          0.75 * std::pow(maximum_velocity_[joint], 2.0) / maximum_start_acceleration_[joint] +
          0.75 * std::pow(maximum_velocity_[joint], 2.0) / maximum_goal_acceleration_[joint];
      if (std::abs(joint_displacement_[joint]) < displacement_at_maximum_velocity) {
        reachable_velocity[joint] =
            std::sqrt((4.0 / 3.0) * joint_displacement_[joint] * displacement_sign[joint] *
                      maximum_start_acceleration_[joint] * maximum_goal_acceleration_[joint] /
                      (maximum_start_acceleration_[joint] + maximum_goal_acceleration_[joint]));
      }
      acceleration_duration[joint] =
          1.5 * reachable_velocity[joint] / maximum_start_acceleration_[joint];
      deceleration_duration[joint] =
          1.5 * reachable_velocity[joint] / maximum_goal_acceleration_[joint];
      finish_time[joint] = 0.5 * acceleration_duration[joint] + 0.5 * deceleration_duration[joint] +
                           std::abs(joint_displacement_[joint]) / reachable_velocity[joint];
    }

    const double synchronized_finish_time = finish_time.maxCoeff();
    for (Eigen::Index joint = 0; joint < joint_displacement_.size(); ++joint) {
      if (std::abs(joint_displacement_[joint]) <= kDeltaQMotionFinished) {
        continue;
      }
      const double quadratic_a =
          0.75 * (maximum_goal_acceleration_[joint] + maximum_start_acceleration_[joint]);
      const double quadratic_b = -synchronized_finish_time * maximum_goal_acceleration_[joint] *
                                 maximum_start_acceleration_[joint];
      const double quadratic_c = std::abs(joint_displacement_[joint]) *
                                 maximum_goal_acceleration_[joint] *
                                 maximum_start_acceleration_[joint];
      const double discriminant =
          std::max(0.0, quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c);
      synchronized_velocity_[joint] =
          (-quadratic_b - std::sqrt(discriminant)) / (2.0 * quadratic_a);
      synchronized_acceleration_end_time_[joint] =
          1.5 * synchronized_velocity_[joint] / maximum_start_acceleration_[joint];
      const double synchronized_deceleration_duration =
          1.5 * synchronized_velocity_[joint] / maximum_goal_acceleration_[joint];
      synchronized_finish_time_[joint] =
          0.5 * synchronized_acceleration_end_time_[joint] +
          0.5 * synchronized_deceleration_duration +
          std::abs(joint_displacement_[joint] / synchronized_velocity_[joint]);
      synchronized_deceleration_start_time_[joint] =
          synchronized_finish_time_[joint] - synchronized_deceleration_duration;
      acceleration_displacement_[joint] = 0.5 * synchronized_acceleration_end_time_[joint] *
                                          synchronized_velocity_[joint] * displacement_sign[joint];
    }
  }

  static constexpr double kDeltaQMotionFinished = 1e-6;
  const Vector7d target_joint_position_;
  Vector7d start_joint_position_{Vector7d::Zero()};
  Vector7d joint_displacement_{Vector7d::Zero()};
  Vector7d synchronized_velocity_{Vector7d::Zero()};
  Vector7d synchronized_acceleration_end_time_{Vector7d::Zero()};
  Vector7d synchronized_deceleration_start_time_{Vector7d::Zero()};
  Vector7d synchronized_finish_time_{Vector7d::Zero()};
  Vector7d acceleration_displacement_{Vector7d::Zero()};
  double elapsed_seconds_{0.0};
  bool initialized_{false};
  Vector7d maximum_velocity_{(Vector7d() << 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5).finished()};
  Vector7d maximum_start_acceleration_{Vector7d::Constant(5.0)};
  Vector7d maximum_goal_acceleration_{Vector7d::Constant(5.0)};
};

inline bool confirm_execution(const char* example_name,
                              const char* configuration,
                              const std::array<double, 7>& initial_joint_position,
                              const double run_duration_seconds,
                              const char* reference_state = "measured") {
  std::cout << example_name << "\n\n"
            << configuration << "\n"
            << "Initial joint target q [rad]: [";
  for (std::size_t joint = 0; joint < initial_joint_position.size(); ++joint) {
    std::cout << (joint == 0 ? "" : ", ") << initial_joint_position[joint];
  }
  std::cout << "] at speed factor " << kInitialMotionSpeedFactor << ".\n"
            << "After confirmation, the robot moves to that configuration. A second Enter "
               "confirmation is required before it captures the "
            << reference_state << " state as the controller target, and runs the controller for "
            << run_duration_seconds << " s.\n"
            << "Joint collision torque thresholds [Nm]: [";
  for (std::size_t joint = 0; joint < kCollisionTorqueThresholdNm.size(); ++joint) {
    std::cout << (joint == 0 ? "" : ", ") << kCollisionTorqueThresholdNm[joint];
  }
  std::cout << "].\n"
            << "Cartesian collision wrench thresholds [N, N, N, Nm, Nm, Nm]: [";
  for (std::size_t axis = 0; axis < kCollisionWrenchThreshold.size(); ++axis) {
    std::cout << (axis == 0 ? "" : ", ") << kCollisionWrenchThreshold[axis];
  }
  std::cout << "].\n"
            << "Lower and upper thresholds are equal, including during acceleration.\n"
            << "Keep an emergency stop within reach. Vendor and geometry boundary code may "
               "allocate or throw; this example is not hard-real-time qualified.\n\n"
            << "Press Enter to connect and execute the initial joint motion "
               "(type anything to cancel): " << std::flush;

  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before connecting to the robot.\n";
    return false;
  }
  std::cout << "Confirmed. Connecting to the robot...\n" << std::flush;
  return true;
}

inline bool confirm_controller_start() {
  std::cout << "Press Enter to start the example controller (type anything to cancel): "
            << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before starting the controller.\n";
    return false;
  }
  return true;
}

inline void set_collision_behavior(franka::Robot& robot) {
  robot.setCollisionBehavior(kCollisionTorqueThresholdNm, kCollisionTorqueThresholdNm,
                             kCollisionWrenchThreshold, kCollisionWrenchThreshold);
}

inline void move_to_initial_configuration(franka::Robot& robot,
                                          const std::array<double, 7>& initial_joint_position) {
  std::cout << "Moving to the initial joint configuration...\n" << std::flush;
  InitialMotionGenerator motion_generator(kInitialMotionSpeedFactor, initial_joint_position);
  robot.control(motion_generator);
  std::cout << "Initial joint configuration reached.\n" << std::flush;
}

template <std::size_t N>
inline bool all_finite(const std::array<double, N>& values) {
  return std::all_of(
      values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

template <typename Derived>
inline bool all_finite(const Eigen::MatrixBase<Derived>& values) {
  return values.allFinite();
}

template <typename Derived>
inline bool valid_joint_torques(const Eigen::MatrixBase<Derived>& values) {
  return values.size() == 7 && all_finite(values);
}

inline bool state_is_finite(const franka::RobotState& state) {
  return all_finite(state.q) && all_finite(state.dq) && all_finite(state.tau_J_d) &&
         all_finite(state.O_T_EE) && all_finite(state.O_F_ext_hat_K);
}

template <typename Derived>
inline franka::Torques filtered_torques(const Eigen::MatrixBase<Derived>& desired_torques,
                                        const franka::RobotState& state) {
  std::array<double, 7> desired{};
  for (std::size_t joint = 0; joint < desired.size(); ++joint) {
    desired[joint] = std::clamp(
        desired_torques.derived().coeff(static_cast<Eigen::Index>(joint)),
        -kMaximumJointTorqueNm[joint],
        kMaximumJointTorqueNm[joint]);
  }
  return franka::Torques(franka::limitRate(franka::kMaxTorqueRate, desired, state.tau_J_d));
}

template <typename Derived>
inline franka::Torques finish_with_filtered_torques(
    const Eigen::MatrixBase<Derived>& desired_torques, const franka::RobotState& state) {
  return franka::MotionFinished(filtered_torques(desired_torques, state));
}

inline franka::Torques finish_with_previous_torques(const franka::RobotState& state) {
  std::array<double, 7> previous = state.tau_J_d;
  if (!all_finite(previous)) {
    previous.fill(0.0);
  }
  return franka::MotionFinished(franka::Torques(previous));
}

inline void record_callback_time(const std::chrono::steady_clock::time_point& callback_start,
                                 RunSummary& summary) {
  const double callback_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - callback_start).count();
  summary.maximum_callback_seconds = std::max(summary.maximum_callback_seconds, callback_seconds);
}

inline void print_summary(const RunSummary& summary) {
  std::cout << std::fixed << std::setprecision(6) << "Run summary: samples=" << summary.samples
            << ", elapsed_s=" << summary.elapsed_seconds
            << ", max_callback_s=" << summary.maximum_callback_seconds
            << ", stop_reason=" << summary.stop_reason << '\n';
}

}  // namespace franka_example
