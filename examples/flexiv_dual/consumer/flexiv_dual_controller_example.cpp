/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file flexiv_dual_controller_example.cpp
 * @brief Flexiv RDK 1.9.1 independent and cooperative two-Rizon examples.
 */

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <flexiv/rdk/data.hpp>
#include <flexiv/rdk/mode.hpp>
#include <flexiv/rdk/model.hpp>
#include <flexiv/rdk/robot.hpp>
#include <flexiv/rdk/scheduler.hpp>

#include "nvidia/srl/controllers/controllers.hpp"

namespace {
namespace controllers = nvidia::srl::controllers;
namespace rdk = flexiv::rdk;

constexpr std::size_t kArmJointDim = 7;
constexpr std::size_t kJointDim = 14;
constexpr std::size_t kSpatialDim = 6;
constexpr std::size_t kTaskDim = 12;
constexpr double kNominalPeriodSeconds = 0.001;
constexpr double kDefaultRunDurationSeconds = 10.0;
constexpr double kEnableTimeoutSeconds = 30.0;
constexpr double kInitialMotionTimeoutSeconds = 60.0;
constexpr double kInitialMotionMaximumVelocity = 0.5;
constexpr double kInitialMotionMaximumAcceleration = 1.0;
constexpr double kInitialPositionTolerance = 0.01;
constexpr double kInitialVelocityTolerance = 0.02;
constexpr double kTorqueMagnitudeLimitFactor = 1.0;
constexpr double kTorqueSlewRateNmPerSecond = 1000.0;
constexpr bool kEnableGravityCompensation = true;
constexpr bool kEnableSoftLimits = true;
constexpr double kFrictionCompensationPercent = 100.0;
constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
constexpr double kCircleDiameterMeters = 0.2;
constexpr double kCircleRadiusMeters = 0.5 * kCircleDiameterMeters;
constexpr double kCircleFrequencyHertz = 0.2;
constexpr double kCircleSpeedRampSeconds = 2.0;
constexpr std::array<double, kArmJointDim> kInitialJointPosition{
    0.0, -15.0 * kDegreesToRadians, 0.0, 75.0 * kDegreesToRadians, 0.0, 0.0, 0.0};
volatile std::sig_atomic_t g_signal_requested = 0;

using JointVector = Eigen::Matrix<double, kJointDim, 1>;
using JointMatrix = Eigen::Matrix<double, kJointDim, kJointDim>;
using ArmJointVector = Eigen::Matrix<double, kArmJointDim, 1>;
using ArmJointMatrix = Eigen::Matrix<double, kArmJointDim, kArmJointDim>;
using CartesianVector = Eigen::Matrix<double, kSpatialDim, 1>;
using ArmJacobian = Eigen::Matrix<double, kSpatialDim, kArmJointDim>;
using SpatialMatrix = Eigen::Matrix<double, kSpatialDim, kSpatialDim>;
using TaskVector = Eigen::Matrix<double, kTaskDim, 1>;
using TaskMatrix = Eigen::Matrix<double, kTaskDim, kTaskDim>;
using RowMajorTaskMatrix = Eigen::Matrix<double, kTaskDim, kTaskDim, Eigen::RowMajor>;
using TaskJacobian = Eigen::Matrix<double, kTaskDim, kJointDim>;
using RowMajorTaskJacobian = Eigen::Matrix<double, kTaskDim, kJointDim, Eigen::RowMajor>;

#ifndef FLEXIV_DUAL_EXAMPLE_PROFILE
#define FLEXIV_DUAL_EXAMPLE_PROFILE 0
#endif

constexpr int kExampleProfile = FLEXIV_DUAL_EXAMPLE_PROFILE;
static_assert(kExampleProfile >= 0 && kExampleProfile <= 1, "invalid dual-Rizon profile");

struct ProgramConfig {
  std::string arm1_robot_serial;
  std::string arm2_robot_serial;
  std::string arm1_flange_link;
  std::string arm2_flange_link;
  controllers::BimanualFrameConfig frames;
  double run_duration_seconds{kDefaultRunDurationSeconds};
  bool non_rt_requested{false};
};

struct CycleData {
  JointVector position{};
  JointVector motor_position{};
  JointVector velocity{};
  JointVector motor_velocity{};
  JointVector measured_torque{};
  JointVector desired_torque{};
  JointVector gravity{};
  JointVector coriolis{};
  JointMatrix mass{};
  Eigen::Matrix4d arm1_tcp_pose{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d arm2_tcp_pose{Eigen::Matrix4d::Identity()};
  CartesianVector arm1_tcp_velocity{};
  CartesianVector arm2_tcp_velocity{};
  ArmJacobian arm1_jacobian{};
  ArmJacobian arm2_jacobian{};
};

struct DebugSnapshot {
  std::atomic<std::uint64_t> sequence{0};
  std::atomic<std::uint64_t> sample{0};
  std::atomic<double> elapsed_seconds{0.0};
  std::atomic<double> dt_seconds{0.0};
  std::array<std::atomic<double>, kTaskDim> task_error{};
  std::array<std::atomic<double>, kTaskDim> task_velocity_error{};
  std::array<std::atomic<double>, kTaskDim> wrench{};
  std::array<std::atomic<double>, kTaskDim> task_mass_diagonal{};
  std::array<std::atomic<double>, kJointDim> task_torque{};
  std::array<std::atomic<double>, kJointDim> joint_position_error{};
  std::array<std::atomic<double>, kJointDim> joint_velocity{};
  std::array<std::atomic<double>, kJointDim> nullspace_torque{};
  std::array<std::atomic<double>, kJointDim> coriolis{};
  std::array<std::atomic<double>, kJointDim> gravity{};
  std::array<std::atomic<double>, kJointDim> measured_torque{};
  std::array<std::atomic<double>, kJointDim> desired_torque{};
  std::array<std::atomic<double>, kJointDim> motor_link_deflection{};
  std::array<std::atomic<double>, kJointDim> unfiltered_torque{};
  std::array<std::atomic<double>, kJointDim> filtered_torque{};
  std::array<std::atomic<double>, kTaskDim> jacobian_link_velocity{};
  std::array<std::atomic<double>, kTaskDim> jacobian_motor_velocity{};
  std::array<std::atomic<double>, kTaskDim> measured_tcp_velocity{};
  std::array<std::atomic<double>, kTaskDim> finite_difference_tcp_velocity{};
};

struct DebugValues {
  std::uint64_t sample{0};
  double elapsed_seconds{0.0};
  double dt_seconds{0.0};
  std::array<double, kTaskDim> task_error{};
  std::array<double, kTaskDim> task_velocity_error{};
  std::array<double, kTaskDim> wrench{};
  std::array<double, kTaskDim> task_mass_diagonal{};
  std::array<double, kJointDim> task_torque{};
  std::array<double, kJointDim> joint_position_error{};
  std::array<double, kJointDim> joint_velocity{};
  std::array<double, kJointDim> nullspace_torque{};
  std::array<double, kJointDim> coriolis{};
  std::array<double, kJointDim> gravity{};
  std::array<double, kJointDim> measured_torque{};
  std::array<double, kJointDim> desired_torque{};
  std::array<double, kJointDim> motor_link_deflection{};
  std::array<double, kJointDim> unfiltered_torque{};
  std::array<double, kJointDim> filtered_torque{};
  std::array<double, kTaskDim> jacobian_link_velocity{};
  std::array<double, kTaskDim> jacobian_motor_velocity{};
  std::array<double, kTaskDim> measured_tcp_velocity{};
  std::array<double, kTaskDim> finite_difference_tcp_velocity{};
};

template <std::size_t Size>
void store_debug_array(const std::array<double, Size>& source,
                       std::array<std::atomic<double>, Size>* destination) {
  for (std::size_t index = 0; index < Size; ++index) {
    (*destination)[index].store(source[index], std::memory_order_relaxed);
  }
}

template <std::size_t Size>
void load_debug_array(const std::array<std::atomic<double>, Size>& source,
                      std::array<double, Size>* destination) {
  for (std::size_t index = 0; index < Size; ++index) {
    (*destination)[index] = source[index].load(std::memory_order_relaxed);
  }
}

bool read_debug_snapshot(const DebugSnapshot& source, DebugValues* destination) {
  const std::uint64_t before = source.sequence.load(std::memory_order_acquire);
  if (destination == nullptr || before == 0U || before % 2U != 0U) {
    return false;
  }
  destination->sample = source.sample.load(std::memory_order_relaxed);
  destination->elapsed_seconds = source.elapsed_seconds.load(std::memory_order_relaxed);
  destination->dt_seconds = source.dt_seconds.load(std::memory_order_relaxed);
  load_debug_array(source.task_error, &destination->task_error);
  load_debug_array(source.task_velocity_error, &destination->task_velocity_error);
  load_debug_array(source.wrench, &destination->wrench);
  load_debug_array(source.task_mass_diagonal, &destination->task_mass_diagonal);
  load_debug_array(source.task_torque, &destination->task_torque);
  load_debug_array(source.joint_position_error, &destination->joint_position_error);
  load_debug_array(source.joint_velocity, &destination->joint_velocity);
  load_debug_array(source.nullspace_torque, &destination->nullspace_torque);
  load_debug_array(source.coriolis, &destination->coriolis);
  load_debug_array(source.gravity, &destination->gravity);
  load_debug_array(source.measured_torque, &destination->measured_torque);
  load_debug_array(source.desired_torque, &destination->desired_torque);
  load_debug_array(source.motor_link_deflection, &destination->motor_link_deflection);
  load_debug_array(source.unfiltered_torque, &destination->unfiltered_torque);
  load_debug_array(source.filtered_torque, &destination->filtered_torque);
  load_debug_array(source.jacobian_link_velocity, &destination->jacobian_link_velocity);
  load_debug_array(source.jacobian_motor_velocity, &destination->jacobian_motor_velocity);
  load_debug_array(source.measured_tcp_velocity, &destination->measured_tcp_velocity);
  load_debug_array(source.finite_difference_tcp_velocity,
                   &destination->finite_difference_tcp_velocity);
  const std::uint64_t after = source.sequence.load(std::memory_order_acquire);
  return before == after && after % 2U == 0U;
}

template <std::size_t Size>
void print_debug_array(const char* name, const std::array<double, Size>& values) {
  std::cout << name << "=[";
  for (std::size_t index = 0; index < Size; ++index) {
    std::cout << (index == 0 ? "" : ",") << values[index];
  }
  std::cout << "]\n";
}

void print_debug_values(const DebugValues& values) {
  std::cout << "\nBIMANUAL_DEBUG sample=" << values.sample
            << " elapsed_s=" << values.elapsed_seconds << " dt_s=" << values.dt_seconds << '\n';
  print_debug_array("task_error_abs_rel", values.task_error);
  print_debug_array("task_velocity_error_abs_rel", values.task_velocity_error);
  print_debug_array("task_wrench_abs_rel", values.wrench);
  print_debug_array("task_mass_diagonal_abs_rel", values.task_mass_diagonal);
  print_debug_array("task_torque_arm1_arm2", values.task_torque);
  print_debug_array("joint_position_error_arm1_arm2", values.joint_position_error);
  print_debug_array("joint_velocity_arm1_arm2", values.joint_velocity);
  print_debug_array("nullspace_torque_arm1_arm2", values.nullspace_torque);
  print_debug_array("coriolis_arm1_arm2", values.coriolis);
  print_debug_array("model_gravity_arm1_arm2", values.gravity);
  print_debug_array("rdk_measured_torque_arm1_arm2", values.measured_torque);
  print_debug_array("rdk_desired_torque_arm1_arm2", values.desired_torque);
  print_debug_array("motor_link_deflection_arm1_arm2", values.motor_link_deflection);
  print_debug_array("unfiltered_torque_arm1_arm2", values.unfiltered_torque);
  print_debug_array("filtered_torque_arm1_arm2", values.filtered_torque);
  print_debug_array("jacobian_times_dq_arm1_arm2", values.jacobian_link_velocity);
  print_debug_array("jacobian_times_dtheta_arm1_arm2", values.jacobian_motor_velocity);
  print_debug_array("rdk_tcp_velocity_arm1_arm2", values.measured_tcp_velocity);
  print_debug_array("finite_difference_tcp_velocity_arm1_arm2",
                    values.finite_difference_tcp_velocity);
  std::cout << std::flush;
}

template <typename Container>
bool has_only_finite_values(const Container& values) {
  return std::all_of(
      values.begin(), values.end(), [](const double value) { return std::isfinite(value); });
}

bool parse_transform(const std::string& text, std::array<double, 16>* transform) {
  if (transform == nullptr) {
    return false;
  }
  std::stringstream stream(text);
  std::string token;
  std::size_t index = 0;
  try {
    while (std::getline(stream, token, ',')) {
      if (index >= transform->size() || token.empty()) {
        return false;
      }
      std::size_t parsed_size = 0;
      const double value = std::stod(token, &parsed_size);
      if (parsed_size != token.size() || !std::isfinite(value)) {
        return false;
      }
      (*transform)[index++] = value;
    }
  } catch (...) {
    return false;
  }
  return index == transform->size();
}

bool parse_positive_double(const std::string& text, double* value) {
  if (value == nullptr || text.empty()) {
    return false;
  }
  try {
    std::size_t parsed_size = 0;
    const double parsed_value = std::stod(text, &parsed_size);
    if (parsed_size != text.size() || !std::isfinite(parsed_value) || parsed_value <= 0.0) {
      return false;
    }
    *value = parsed_value;
  } catch (...) {
    return false;
  }
  return true;
}

void print_usage(const char* executable) {
  std::cerr << "Usage: " << executable
            << " ARM1_ROBOT_SERIAL ARM2_ROBOT_SERIAL ARM1_FLANGE_LINK ARM2_FLANGE_LINK"
               " --arm1-t-arm2=m00,m01,...,m33"
               " [--main-frame=arm1|arm2|explicit-arm1|explicit-arm2]"
               " [--reference-t-main=m00,m01,...,m33]"
               " [--duration-seconds=SECONDS] [--non-rt]\n";
}

bool parse_arguments(const int argc, char** argv, ProgramConfig* config) {
  if (config == nullptr || argc < 6) {
    return false;
  }
  config->arm1_robot_serial = argv[1];
  config->arm2_robot_serial = argv[2];
  config->arm1_flange_link = argv[3];
  config->arm2_flange_link = argv[4];
  bool has_base_transform = false;
  bool has_main_frame = false;
  bool has_main_transform = false;
  bool has_duration = false;
  for (int index = 5; index < argc; ++index) {
    const std::string argument = argv[index];
    constexpr const char* kBasePrefix = "--arm1-t-arm2=";
    constexpr const char* kMainPrefix = "--main-frame=";
    constexpr const char* kExplicitPrefix = "--reference-t-main=";
    constexpr const char* kDurationPrefix = "--duration-seconds=";
    if (argument == "--non-rt") {
      if (config->non_rt_requested) {
        return false;
      }
      config->non_rt_requested = true;
    } else if (argument.rfind(kBasePrefix, 0) == 0) {
      if (has_base_transform) {
        return false;
      }
      has_base_transform =
          parse_transform(argument.substr(std::char_traits<char>::length(kBasePrefix)),
                          &config->frames.arm1_T_arm2);
      if (!has_base_transform) {
        return false;
      }
    } else if (argument.rfind(kMainPrefix, 0) == 0) {
      if (has_main_frame) {
        return false;
      }
      const std::string value = argument.substr(std::char_traits<char>::length(kMainPrefix));
      if (value == "arm1") {
        config->frames.main_frame = controllers::BimanualMainFrame::kArm1Base;
      } else if (value == "arm2") {
        config->frames.main_frame = controllers::BimanualMainFrame::kArm2Base;
      } else if (value == "explicit-arm1") {
        config->frames.main_frame = controllers::BimanualMainFrame::kExplicit;
        config->frames.explicit_reference = controllers::BimanualBase::kArm1;
      } else if (value == "explicit-arm2") {
        config->frames.main_frame = controllers::BimanualMainFrame::kExplicit;
        config->frames.explicit_reference = controllers::BimanualBase::kArm2;
      } else {
        return false;
      }
      has_main_frame = true;
    } else if (argument.rfind(kExplicitPrefix, 0) == 0) {
      if (has_main_transform) {
        return false;
      }
      has_main_transform =
          parse_transform(argument.substr(std::char_traits<char>::length(kExplicitPrefix)),
                          &config->frames.reference_T_main);
      if (!has_main_transform) {
        return false;
      }
    } else if (argument.rfind(kDurationPrefix, 0) == 0) {
      if (has_duration) {
        return false;
      }
      has_duration =
          parse_positive_double(argument.substr(std::char_traits<char>::length(kDurationPrefix)),
                                &config->run_duration_seconds);
      if (!has_duration) {
        return false;
      }
    } else {
      return false;
    }
  }
  return config->arm1_robot_serial != config->arm2_robot_serial && has_base_transform &&
         (config->frames.main_frame != controllers::BimanualMainFrame::kExplicit ||
          has_main_transform);
}

Eigen::Matrix4d transform_from_array(const std::array<double, 16>& values) {
  Eigen::Matrix4d transform;
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      transform(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          values[row * 4 + col];
    }
  }
  return transform;
}

bool is_rigid_transform(const std::array<double, 16>& values) {
  const Eigen::Matrix4d transform = transform_from_array(values);
  const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
  return transform.allFinite() &&
         (transform.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() < 1.0e-9 &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1.0e-9 &&
         std::abs(rotation.determinant() - 1.0) < 1.0e-9;
}

bool pose_from_rdk(const std::array<double, rdk::kPoseSize>& pose, Eigen::Matrix4d* transform) {
  if (transform == nullptr || !has_only_finite_values(pose)) {
    return false;
  }
  Eigen::Quaterniond quaternion(pose[3], pose[4], pose[5], pose[6]);
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm < 1.0e-12) {
    return false;
  }
  quaternion.normalize();
  transform->setIdentity();
  transform->topLeftCorner<3, 3>() = quaternion.toRotationMatrix();
  transform->topRightCorner<3, 1>() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
  return transform->allFinite();
}

Eigen::MatrixXd tcp_jacobian(const Eigen::MatrixXd& flange_jacobian,
                             const Eigen::Matrix4d& world_flange,
                             const Eigen::Matrix4d& world_tcp) {
  Eigen::MatrixXd result = flange_jacobian;
  const Eigen::Vector3d offset =
      world_tcp.topRightCorner<3, 1>() - world_flange.topRightCorner<3, 1>();
  Eigen::Matrix3d skew;
  skew << 0.0, -offset.z(), offset.y(), offset.z(), 0.0, -offset.x(), -offset.y(), offset.x(), 0.0;
  result.topRows<3>() = flange_jacobian.topRows<3>() - skew * flange_jacobian.bottomRows<3>();
  result.bottomRows<3>() = flange_jacobian.bottomRows<3>();
  return result;
}

bool valid_state(const rdk::RobotStates& state) {
  return state.q.size() == kArmJointDim && state.theta.size() == kArmJointDim &&
         state.dq.size() == kArmJointDim && state.dtheta.size() == kArmJointDim &&
         state.tau.size() == kArmJointDim && state.tau_des.size() == kArmJointDim &&
         has_only_finite_values(state.q) && has_only_finite_values(state.theta) &&
         has_only_finite_values(state.dq) && has_only_finite_values(state.dtheta) &&
         has_only_finite_values(state.tau) && has_only_finite_values(state.tau_des) &&
         has_only_finite_values(state.flange_pose) && has_only_finite_values(state.tcp_pose) &&
         has_only_finite_values(state.tcp_vel);
}

bool fill_cycle_data(const rdk::RobotStates& arm1,
                     const rdk::RobotStates& arm2,
                     rdk::Model& arm1_model,
                     rdk::Model& arm2_model,
                     const ProgramConfig& config,
                     CycleData* data) {
  if (data == nullptr) {
    return false;
  }
  if (!valid_state(arm1) || !valid_state(arm2)) {
    return false;
  }
  arm1_model.Update(arm1.q, arm1.dtheta);
  arm2_model.Update(arm2.q, arm2.dtheta);
  const Eigen::MatrixXd arm1_mass = arm1_model.M();
  const Eigen::MatrixXd arm2_mass = arm2_model.M();
  const Eigen::VectorXd arm1_gravity = arm1_model.g();
  const Eigen::VectorXd arm2_gravity = arm2_model.g();
  const Eigen::VectorXd arm1_coriolis = arm1_model.c();
  const Eigen::VectorXd arm2_coriolis = arm2_model.c();
  const Eigen::MatrixXd arm1_flange_jacobian = arm1_model.J(config.arm1_flange_link);
  const Eigen::MatrixXd arm2_flange_jacobian = arm2_model.J(config.arm2_flange_link);
  if (arm1_mass.rows() != kArmJointDim || arm1_mass.cols() != kArmJointDim ||
      arm2_mass.rows() != kArmJointDim || arm2_mass.cols() != kArmJointDim ||
      arm1_gravity.size() != kArmJointDim || arm2_gravity.size() != kArmJointDim ||
      arm1_coriolis.size() != kArmJointDim || arm2_coriolis.size() != kArmJointDim ||
      arm1_flange_jacobian.rows() != kSpatialDim || arm1_flange_jacobian.cols() != kArmJointDim ||
      arm2_flange_jacobian.rows() != kSpatialDim || arm2_flange_jacobian.cols() != kArmJointDim ||
      !arm1_mass.allFinite() || !arm2_mass.allFinite() || !arm1_gravity.allFinite() ||
      !arm2_gravity.allFinite() || !arm1_coriolis.allFinite() || !arm2_coriolis.allFinite() ||
      !arm1_flange_jacobian.allFinite() || !arm2_flange_jacobian.allFinite()) {
    return false;
  }
  Eigen::Matrix4d world_arm1_flange;
  Eigen::Matrix4d world_arm2_flange;
  Eigen::Matrix4d world_arm1_tcp;
  Eigen::Matrix4d world_arm2_tcp;
  if (!pose_from_rdk(arm1.flange_pose, &world_arm1_flange) ||
      !pose_from_rdk(arm2.flange_pose, &world_arm2_flange) ||
      !pose_from_rdk(arm1.tcp_pose, &world_arm1_tcp) ||
      !pose_from_rdk(arm2.tcp_pose, &world_arm2_tcp)) {
    return false;
  }
  const ArmJacobian arm1_tcp_jacobian =
      tcp_jacobian(arm1_flange_jacobian, world_arm1_flange, world_arm1_tcp);
  const ArmJacobian arm2_tcp_jacobian =
      tcp_jacobian(arm2_flange_jacobian, world_arm2_flange, world_arm2_tcp);
  data->position << Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm1.q.data()),
      Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm2.q.data());
  data->motor_position << Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(
      arm1.theta.data()),
      Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm2.theta.data());
  data->velocity << Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm1.dq.data()),
      Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm2.dq.data());
  data->motor_velocity << Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(
      arm1.dtheta.data()),
      Eigen::Map<const Eigen::Matrix<double, kArmJointDim, 1>>(arm2.dtheta.data());
  data->measured_torque << Eigen::Map<const ArmJointVector>(arm1.tau.data()),
      Eigen::Map<const ArmJointVector>(arm2.tau.data());
  data->desired_torque << Eigen::Map<const ArmJointVector>(arm1.tau_des.data()),
      Eigen::Map<const ArmJointVector>(arm2.tau_des.data());
  data->arm1_tcp_velocity = Eigen::Map<const CartesianVector>(arm1.tcp_vel.data());
  data->arm2_tcp_velocity = Eigen::Map<const CartesianVector>(arm2.tcp_vel.data());
  data->mass.setZero();
  data->mass.topLeftCorner<kArmJointDim, kArmJointDim>() = arm1_mass;
  data->mass.bottomRightCorner<kArmJointDim, kArmJointDim>() = arm2_mass;
  data->gravity << arm1_gravity, arm2_gravity;
  data->coriolis << arm1_coriolis, arm2_coriolis;
  data->arm1_tcp_pose = world_arm1_tcp;
  data->arm2_tcp_pose = world_arm2_tcp;
  data->arm1_jacobian = arm1_tcp_jacobian;
  data->arm2_jacobian = arm2_tcp_jacobian;
  return data->position.allFinite() && data->motor_position.allFinite() &&
         data->velocity.allFinite() && data->motor_velocity.allFinite() &&
         data->measured_torque.allFinite() && data->desired_torque.allFinite() &&
         data->mass.allFinite() && data->gravity.allFinite() && data->coriolis.allFinite() &&
         data->arm1_tcp_pose.allFinite() && data->arm2_tcp_pose.allFinite() &&
         data->arm1_tcp_velocity.allFinite() && data->arm2_tcp_velocity.allFinite() &&
         data->arm1_jacobian.allFinite() && data->arm2_jacobian.allFinite();
}

template <typename Derived, std::size_t Size>
void copy_matrix(const Eigen::MatrixBase<Derived>& source, std::array<double, Size>* destination) {
  for (std::size_t row = 0; row < static_cast<std::size_t>(source.rows()); ++row) {
    for (std::size_t col = 0; col < static_cast<std::size_t>(source.cols()); ++col) {
      (*destination)[row * static_cast<std::size_t>(source.cols()) + col] =
          source(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }
  }
}

template <typename Derived, std::size_t Size>
void copy_vector(const Eigen::MatrixBase<Derived>& source, std::array<double, Size>* destination) {
  for (std::size_t index = 0; index < Size; ++index) {
    (*destination)[index] = source[static_cast<Eigen::Index>(index)];
  }
}

bool is_safe(const std::uint64_t status) {
  return !controllers::has_status_flag(status, controllers::StatusFlag::kSafeMode);
}

CartesianVector calculate_pose_error(const Eigen::Matrix4d& desired_pose,
                                     const Eigen::Matrix4d& current_pose) {
  CartesianVector error;
  error.head<3>() = desired_pose.topRightCorner<3, 1>() - current_pose.topRightCorner<3, 1>();
  const Eigen::AngleAxisd rotation_error(desired_pose.topLeftCorner<3, 3>() *
                                         current_pose.topLeftCorner<3, 3>().transpose());
  error.tail<3>() = rotation_error.angle() * rotation_error.axis();
  return error;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& rotation) {
  Eigen::AngleAxisd angle_axis(rotation);
  return angle_axis.angle() * angle_axis.axis();
}

CartesianVector calculate_finite_difference_velocity(const Eigen::Matrix4d& previous_pose,
                                                     const Eigen::Matrix4d& current_pose,
                                                     const double dt_seconds) {
  CartesianVector velocity;
  velocity.head<3>() =
      (current_pose.topRightCorner<3, 1>() - previous_pose.topRightCorner<3, 1>()) / dt_seconds;
  velocity.tail<3>() = so3_log(current_pose.topLeftCorner<3, 3>() *
                               previous_pose.topLeftCorner<3, 3>().transpose()) /
                       dt_seconds;
  return velocity;
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle < 1.0e-12) {
    return Eigen::Matrix3d::Identity() + skew(rotation_vector);
  }
  return Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  const Eigen::Matrix3d rotation_skew = skew(rotation_vector);
  if (angle < 1.0e-6) {
    return Eigen::Matrix3d::Identity() + 0.5 * rotation_skew +
           (1.0 / 6.0) * rotation_skew * rotation_skew;
  }
  const double angle_squared = angle * angle;
  return Eigen::Matrix3d::Identity() + (1.0 - std::cos(angle)) / angle_squared * rotation_skew +
         (angle - std::sin(angle)) / (angle_squared * angle) * rotation_skew * rotation_skew;
}

Eigen::Matrix3d so3_left_jacobian_inverse(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  const Eigen::Matrix3d rotation_skew = skew(rotation_vector);
  if (angle < 1.0e-6) {
    return Eigen::Matrix3d::Identity() - 0.5 * rotation_skew +
           (1.0 / 12.0) * rotation_skew * rotation_skew;
  }
  const double angle_squared = angle * angle;
  const double coefficient =
      1.0 / angle_squared - (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle));
  return Eigen::Matrix3d::Identity() - 0.5 * rotation_skew +
         coefficient * rotation_skew * rotation_skew;
}

CartesianVector se3_log(const Eigen::Matrix4d& transform) {
  CartesianVector tangent;
  const Eigen::Vector3d rotation_vector = so3_log(transform.topLeftCorner<3, 3>());
  tangent.head<3>() = so3_left_jacobian_inverse(rotation_vector) * transform.topRightCorner<3, 1>();
  tangent.tail<3>() = rotation_vector;
  return tangent;
}

Eigen::Matrix4d se3_exp(const CartesianVector& tangent) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<3, 3>() = so3_exp(tangent.tail<3>());
  transform.topRightCorner<3, 1>() = so3_left_jacobian(tangent.tail<3>()) * tangent.head<3>();
  return transform;
}

SpatialMatrix adjoint(const Eigen::Matrix4d& transform) {
  SpatialMatrix result = SpatialMatrix::Zero();
  const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
  result.topLeftCorner<3, 3>() = rotation;
  result.topRightCorner<3, 3>() = skew(transform.topRightCorner<3, 1>()) * rotation;
  result.bottomRightCorner<3, 3>() = rotation;
  return result;
}

SpatialMatrix se3_adjoint_matrix(const CartesianVector& tangent) {
  SpatialMatrix result = SpatialMatrix::Zero();
  result.topLeftCorner<3, 3>() = skew(tangent.tail<3>());
  result.topRightCorner<3, 3>() = skew(tangent.head<3>());
  result.bottomRightCorner<3, 3>() = skew(tangent.tail<3>());
  return result;
}

SpatialMatrix se3_right_jacobian(const CartesianVector& tangent) {
  const SpatialMatrix negative_adjoint = -se3_adjoint_matrix(tangent);
  SpatialMatrix result = SpatialMatrix::Identity();
  SpatialMatrix term = SpatialMatrix::Identity();
  for (int order = 1; order <= 18; ++order) {
    term = term * negative_adjoint / static_cast<double>(order + 1);
    result += term;
  }
  return result;
}

SpatialMatrix rotate_spatial(const Eigen::Matrix3d& rotation) {
  SpatialMatrix result = SpatialMatrix::Zero();
  result.topLeftCorner<3, 3>() = rotation;
  result.bottomRightCorner<3, 3>() = rotation;
  return result;
}

bool calculate_mass_aware_damping(const TaskMatrix& task_mass,
                                  const TaskMatrix& stiffness,
                                  const TaskVector& damping_factors,
                                  TaskMatrix* damping) {
  if (damping == nullptr || !task_mass.allFinite() || !stiffness.allFinite() ||
      !damping_factors.allFinite()) {
    return false;
  }
  const TaskMatrix symmetric_mass = 0.5 * (task_mass + task_mass.transpose());
  const Eigen::SelfAdjointEigenSolver<TaskMatrix> mass_solver(symmetric_mass);
  if (mass_solver.info() != Eigen::Success || mass_solver.eigenvalues().minCoeff() <= 0.0) {
    return false;
  }
  const TaskVector mass_roots = mass_solver.eigenvalues().cwiseSqrt();
  const TaskMatrix mass_root =
      mass_solver.eigenvectors() * mass_roots.asDiagonal() * mass_solver.eigenvectors().transpose();
  const TaskMatrix mass_inverse_root = mass_solver.eigenvectors() *
                                       mass_roots.cwiseInverse().asDiagonal() *
                                       mass_solver.eigenvectors().transpose();
  TaskMatrix transformed_stiffness =
      mass_inverse_root * (0.5 * (stiffness + stiffness.transpose())) * mass_inverse_root;
  transformed_stiffness = 0.5 * (transformed_stiffness + transformed_stiffness.transpose());
  const Eigen::SelfAdjointEigenSolver<TaskMatrix> stiffness_solver(transformed_stiffness);
  constexpr double kNegativeEigenvalueTolerance = -1.0e-9;
  if (stiffness_solver.info() != Eigen::Success ||
      stiffness_solver.eigenvalues().minCoeff() < kNegativeEigenvalueTolerance) {
    return false;
  }
  const TaskVector transformed_roots = stiffness_solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const TaskMatrix transformed_root = stiffness_solver.eigenvectors() *
                                      transformed_roots.asDiagonal() *
                                      stiffness_solver.eigenvectors().transpose();
  *damping =
      2.0 * mass_root * damping_factors.asDiagonal() * transformed_root * mass_root.transpose();
  return damping->allFinite();
}

Eigen::Matrix3d main_rotation(const controllers::BimanualFrameConfig& frames) {
  const Eigen::Matrix4d arm1_T_arm2 = transform_from_array(frames.arm1_T_arm2);
  const Eigen::Matrix4d reference_T_main = transform_from_array(frames.reference_T_main);
  Eigen::Matrix3d arm1_R_main = Eigen::Matrix3d::Identity();
  if (frames.main_frame == controllers::BimanualMainFrame::kArm2Base) {
    arm1_R_main = arm1_T_arm2.topLeftCorner<3, 3>();
  } else if (frames.main_frame == controllers::BimanualMainFrame::kExplicit) {
    if (frames.explicit_reference == controllers::BimanualBase::kArm1) {
      arm1_R_main = reference_T_main.topLeftCorner<3, 3>();
    } else {
      arm1_R_main = arm1_T_arm2.topLeftCorner<3, 3>() * reference_T_main.topLeftCorner<3, 3>();
    }
  }
  return arm1_R_main.transpose();
}

bool cooperative_task_kinematics(const CycleData& cycle,
                                 const controllers::BimanualFrameConfig& frames,
                                 const Eigen::Matrix4d& desired_absolute_pose,
                                 const Eigen::Matrix4d& desired_relative_pose,
                                 TaskVector* task_error,
                                 TaskJacobian* task_jacobian) {
  // Translation uses decoupled cooperative coordinates in the arm-1 base:
  // p_a = 0.5 (p_1 + p_2), p_r = p_2 - p_1. Orientation retains the symmetric SO(3) midpoint and
  // relative rotation. Keeping translation out of T_1^-1 T_2 avoids turning transverse relative
  // displacement into arm-1 torque through the TCP-separation lever arm.
  if (task_error == nullptr || task_jacobian == nullptr) {
    return false;
  }
  const Eigen::Matrix4d arm1_T_arm2 = transform_from_array(frames.arm1_T_arm2);
  const Eigen::Matrix4d arm1_T_tcp1 = cycle.arm1_tcp_pose;
  const Eigen::Matrix4d arm1_T_tcp2 = arm1_T_arm2 * cycle.arm2_tcp_pose;
  const Eigen::Matrix4d relative_pose = arm1_T_tcp1.inverse() * arm1_T_tcp2;
  const CartesianVector relative_log = se3_log(relative_pose);
  const Eigen::Matrix4d midpoint_offset = se3_exp(0.5 * relative_log);
  Eigen::Matrix4d absolute_pose = arm1_T_tcp1 * midpoint_offset;
  absolute_pose.topRightCorner<3, 1>() =
      0.5 * (arm1_T_tcp1.topRightCorner<3, 1>() + arm1_T_tcp2.topRightCorner<3, 1>());

  const SpatialMatrix main_X_arm1 = rotate_spatial(main_rotation(frames));
  const SpatialMatrix arm1_X_absolute = rotate_spatial(absolute_pose.topLeftCorner<3, 3>());
  const Eigen::Matrix4d desired_arm2_pose = arm1_T_tcp1 * desired_relative_pose;
  task_error->head<kSpatialDim>() =
      main_X_arm1 * calculate_pose_error(desired_absolute_pose, absolute_pose);
  CartesianVector relative_error = calculate_pose_error(desired_arm2_pose, arm1_T_tcp2);
  relative_error.head<3>() =
      desired_relative_pose.topRightCorner<3, 1>() -
      (arm1_T_tcp2.topRightCorner<3, 1>() - arm1_T_tcp1.topRightCorner<3, 1>());
  task_error->tail<kSpatialDim>() = main_X_arm1 * relative_error;

  const SpatialMatrix inverse_relative_adjoint = adjoint(relative_pose.inverse());
  const SpatialMatrix inverse_midpoint_adjoint = adjoint(midpoint_offset.inverse());
  const SpatialMatrix relative_log_rate = se3_right_jacobian(relative_log).inverse();
  const SpatialMatrix midpoint_rate =
      0.5 * se3_right_jacobian(0.5 * relative_log) * relative_log_rate;
  const SpatialMatrix arm1_X_arm2 = rotate_spatial(arm1_T_arm2.topLeftCorner<3, 3>());
  const ArmJacobian arm2_jacobian_arm1 = arm1_X_arm2 * cycle.arm2_jacobian;
  task_jacobian->setZero();
  for (std::size_t joint = 0; joint < kJointDim; ++joint) {
    CartesianVector tcp1_body = CartesianVector::Zero();
    CartesianVector tcp2_body = CartesianVector::Zero();
    CartesianVector absolute_geometric = CartesianVector::Zero();
    CartesianVector relative_geometric = CartesianVector::Zero();
    if (joint < kArmJointDim) {
      const CartesianVector tcp1_geometric =
          cycle.arm1_jacobian.col(static_cast<Eigen::Index>(joint));
      tcp1_body = rotate_spatial(arm1_T_tcp1.topLeftCorner<3, 3>().transpose()) * tcp1_geometric;
      absolute_geometric.head<3>() = 0.5 * tcp1_geometric.head<3>();
      relative_geometric.head<3>() = -tcp1_geometric.head<3>();
      relative_geometric.tail<3>() = -tcp1_geometric.tail<3>();
    } else {
      const CartesianVector tcp2_geometric =
          arm2_jacobian_arm1.col(static_cast<Eigen::Index>(joint - kArmJointDim));
      tcp2_body = rotate_spatial(arm1_T_tcp2.topLeftCorner<3, 3>().transpose()) * tcp2_geometric;
      absolute_geometric.head<3>() = 0.5 * tcp2_geometric.head<3>();
      relative_geometric = tcp2_geometric;
    }
    const CartesianVector relative_body = tcp2_body - inverse_relative_adjoint * tcp1_body;
    const CartesianVector absolute_body =
        inverse_midpoint_adjoint * tcp1_body + midpoint_rate * relative_body;
    absolute_geometric.tail<3>() = (arm1_X_absolute * absolute_body).tail<3>();
    task_jacobian->block<kSpatialDim, 1>(0, static_cast<Eigen::Index>(joint)) =
        main_X_arm1 * absolute_geometric;
    task_jacobian->block<kSpatialDim, 1>(kSpatialDim, static_cast<Eigen::Index>(joint)) =
        main_X_arm1 * relative_geometric;
  }
  return task_error->allFinite() && task_jacobian->allFinite();
}

void integrate_spatial_twist(const CartesianVector& twist,
                             const double dt_seconds,
                             Eigen::Matrix4d* pose) {
  pose->topRightCorner<3, 1>() += dt_seconds * twist.head<3>();
  pose->topLeftCorner<3, 3>() = so3_exp(dt_seconds * twist.tail<3>()) * pose->topLeftCorner<3, 3>();
}

bool validate_cooperative_task_kinematics(double* maximum_residual) {
  if (maximum_residual == nullptr) {
    return false;
  }
  constexpr double kFiniteDifferenceStep = 1.0e-7;
  constexpr double kMaximumResidual = 2.0e-5;
  controllers::BimanualFrameConfig frames;
  const Eigen::Matrix4d arm1_T_arm2 =
      (Eigen::Translation3d(0.31, -0.17, 0.08) *
       Eigen::AngleAxisd(0.23, Eigen::Vector3d(0.2, -0.4, 0.7).normalized()))
          .matrix();
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      frames.arm1_T_arm2[row * 4 + col] =
          arm1_T_arm2(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }
  }
  CycleData cycle;
  cycle.arm1_tcp_pose = (Eigen::Translation3d(0.42, -0.11, 0.63) *
                         Eigen::AngleAxisd(-0.31, Eigen::Vector3d(0.6, 0.2, -0.3).normalized()))
                            .matrix();
  cycle.arm2_tcp_pose = (Eigen::Translation3d(-0.27, 0.19, 0.58) *
                         Eigen::AngleAxisd(0.37, Eigen::Vector3d(-0.1, 0.8, 0.4).normalized()))
                            .matrix();
  for (std::size_t row = 0; row < kSpatialDim; ++row) {
    for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
      const double scale = row < 3 ? 0.03 : 0.05;
      cycle.arm1_jacobian(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(joint)) =
          scale * std::sin(static_cast<double>((row + 1U) * (joint + 2U)));
      cycle.arm2_jacobian(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(joint)) =
          scale * std::cos(static_cast<double>((row + 2U) * (joint + 1U)));
    }
  }
  const Eigen::Matrix4d arm1_T_tcp2 = arm1_T_arm2 * cycle.arm2_tcp_pose;
  Eigen::Matrix4d desired_relative_pose = cycle.arm1_tcp_pose.inverse() * arm1_T_tcp2;
  desired_relative_pose.topRightCorner<3, 1>() =
      arm1_T_tcp2.topRightCorner<3, 1>() - cycle.arm1_tcp_pose.topRightCorner<3, 1>();
  Eigen::Matrix4d desired_absolute_pose =
      cycle.arm1_tcp_pose * se3_exp(0.5 * se3_log(cycle.arm1_tcp_pose.inverse() * arm1_T_tcp2));
  desired_absolute_pose.topRightCorner<3, 1>() =
      0.5 * (cycle.arm1_tcp_pose.topRightCorner<3, 1>() + arm1_T_tcp2.topRightCorner<3, 1>());
  TaskVector nominal_error;
  TaskJacobian nominal_jacobian;
  if (!cooperative_task_kinematics(cycle,
                                   frames,
                                   desired_absolute_pose,
                                   desired_relative_pose,
                                   &nominal_error,
                                   &nominal_jacobian) ||
      nominal_error.cwiseAbs().maxCoeff() > kMaximumResidual) {
    return false;
  }
  *maximum_residual = 0.0;
  for (std::size_t joint = 0; joint < kJointDim; ++joint) {
    CycleData perturbed = cycle;
    if (joint < kArmJointDim) {
      integrate_spatial_twist(cycle.arm1_jacobian.col(static_cast<Eigen::Index>(joint)),
                              kFiniteDifferenceStep,
                              &perturbed.arm1_tcp_pose);
    } else {
      integrate_spatial_twist(
          cycle.arm2_jacobian.col(static_cast<Eigen::Index>(joint - kArmJointDim)),
          kFiniteDifferenceStep,
          &perturbed.arm2_tcp_pose);
    }
    TaskVector perturbed_error;
    TaskJacobian perturbed_jacobian;
    if (!cooperative_task_kinematics(perturbed,
                                     frames,
                                     desired_absolute_pose,
                                     desired_relative_pose,
                                     &perturbed_error,
                                     &perturbed_jacobian)) {
      return false;
    }
    const TaskVector residual = perturbed_error / kFiniteDifferenceStep +
                                nominal_jacobian.col(static_cast<Eigen::Index>(joint));
    *maximum_residual = std::max(*maximum_residual, residual.cwiseAbs().maxCoeff());
  }
  return *maximum_residual <= kMaximumResidual;
}

bool report_bimanual_kinematics_self_check() {
  double maximum_residual = 0.0;
  if (!validate_cooperative_task_kinematics(&maximum_residual)) {
    std::cerr << "Bimanual kinematics finite-difference self-check failed; maximum residual="
              << maximum_residual << '\n';
    return false;
  }
  std::cout << "Bimanual kinematics finite-difference self-check passed; maximum residual="
            << maximum_residual << ".\n";
  return true;
}

class ArmHoldController {
 public:
  ArmHoldController() {
    constexpr std::array<double, kSpatialDim> kStiffness{1000, 1000, 1000, 100, 100, 100};
    constexpr std::array<double, kArmJointDim> kNullspaceStiffness{120, 80, 60, 40, 30, 20, 10};
    constexpr std::array<double, kArmJointDim> kNullspaceDamping{12, 8, 6, 4, 3, 2, 1};
    for (std::size_t axis = 0; axis < kSpatialDim; ++axis) {
      stiffness_[axis * kSpatialDim + axis] = kStiffness[axis];
      damping_factors_[axis] = 0.7;
    }
    for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
      nullspace_stiffness_[joint * kArmJointDim + joint] = kNullspaceStiffness[joint];
      nullspace_damping_[joint * kArmJointDim + joint] = kNullspaceDamping[joint];
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               kArmJointDim,
                                               {},
                                               controllers::CartesianFrame::kRobotBase,
                                               kSpatialDim};
    controllers::Result result = mass_.prepare(
        config,
        {controllers::shared(controllers::read_view(regularization_, controllers::Device::kCpu, 1)),
         controllers::shared(controllers::read_view(minimum_pivot_, controllers::Device::kCpu, 1))},
        {controllers::write_view(
             cartesian_mass_, controllers::Device::kCpu, 1, kSpatialDim, kSpatialDim),
         controllers::write_status(mass_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(mass_status_, controllers::Device::kCpu),
         {controllers::write_view(
              mass_joint_factor_, controllers::Device::kCpu, 1, kArmJointDim, kArmJointDim),
          controllers::write_view(
              mass_solve_rhs_, controllers::Device::kCpu, 1, kArmJointDim, kSpatialDim),
          controllers::write_view(
              cartesian_inverse_, controllers::Device::kCpu, 1, kSpatialDim, kSpatialDim)}});
    if (!result.ok()) {
      return result;
    }
    result = impedance_.prepare(
        config,
        {controllers::shared(controllers::read_view(
             stiffness_, controllers::Device::kCpu, kSpatialDim, kSpatialDim)),
         controllers::shared(
             controllers::read_view(damping_factors_, controllers::Device::kCpu, kSpatialDim))},
        {controllers::write_view(wrench_, controllers::Device::kCpu, 1, kSpatialDim),
         controllers::write_status(impedance_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(impedance_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(last_valid_wrench_, controllers::Device::kCpu, 1, kSpatialDim),
             controllers::write_view(impedance_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(impedance_hold_count_, controllers::Device::kCpu, 1)),
         {controllers::write_view(
              previous_cartesian_mass_, controllers::Device::kCpu, 1, kSpatialDim, kSpatialDim),
          controllers::write_view(
              previous_jacobian_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim),
          controllers::write_view(impedance_initialized_, controllers::Device::kCpu, 1)}});
    if (!result.ok()) {
      return result;
    }
    result = nullspace_.prepare(
        config,
        {controllers::shared(controllers::read_view(
             nullspace_stiffness_, controllers::Device::kCpu, kArmJointDim, kArmJointDim)),
         controllers::shared(controllers::read_view(
             nullspace_damping_, controllers::Device::kCpu, kArmJointDim, kArmJointDim))},
        {controllers::write_view(nullspace_torque_, controllers::Device::kCpu, 1, kArmJointDim),
         controllers::write_status(nullspace_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(nullspace_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(
                 last_valid_nullspace_torque_, controllers::Device::kCpu, 1, kArmJointDim),
             controllers::write_view(nullspace_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(nullspace_hold_count_, controllers::Device::kCpu, 1))});
    configure_inputs();
    if (result.ok()) {
      result = mass_.reset();
    }
    if (result.ok()) {
      result = impedance_.reset();
    }
    if (result.ok()) {
      result = nullspace_.reset();
    }
    return result;
  }

  bool set_target(const Eigen::Matrix4d& pose, const ArmJointVector& joint_position) {
    desired_pose_ = pose;
    desired_joint_position_ = joint_position;
    return desired_pose_.allFinite() && desired_joint_position_.allFinite();
  }

  bool set_pose_target(const Eigen::Matrix4d& pose) {
    desired_pose_ = pose;
    return desired_pose_.allFinite();
  }

  bool step(const Eigen::Matrix4d& pose,
            const ArmJacobian& jacobian,
            const ArmJointMatrix& mass,
            const ArmJointVector& position,
            const ArmJointVector& velocity,
            const CartesianVector& desired_velocity,
            const CartesianVector& desired_acceleration,
            const double dt_seconds,
            ArmJointVector* torque) {
    if (torque == nullptr) {
      return false;
    }
    copy_vector(calculate_pose_error(desired_pose_, pose), &pose_error_);
    copy_vector(desired_velocity - jacobian * velocity, &velocity_error_);
    copy_vector(desired_acceleration, &desired_acceleration_);
    copy_vector(desired_joint_position_ - position, &joint_position_error_);
    copy_vector(velocity, &joint_velocity_);
    copy_matrix(mass, &joint_mass_);
    copy_matrix(jacobian, &jacobian_);
    dt_[0] = dt_seconds;
    mass_status_[0] = 0U;
    impedance_status_[0] = 0U;
    nullspace_status_[0] = 0U;
    if (!mass_.step(mass_inputs_).ok() || !is_safe(mass_status_[0]) ||
        !impedance_.step(impedance_inputs_).ok() || !is_safe(impedance_status_[0]) ||
        !nullspace_.step(nullspace_inputs_).ok() || !is_safe(nullspace_status_[0])) {
      return false;
    }
    const Eigen::Map<const CartesianVector> wrench(wrench_.data());
    const Eigen::Map<const ArmJointVector> nullspace_torque(nullspace_torque_.data());
    *torque = jacobian.transpose() * wrench + nullspace_torque;
    return torque->allFinite();
  }

 private:
  void configure_inputs() {
    mass_inputs_ = {
        controllers::read_view(
            joint_mass_, controllers::Device::kCpu, 1, kArmJointDim, kArmJointDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim)};
    impedance_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kSpatialDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kSpatialDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kSpatialDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kArmJointDim),
        controllers::read_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kSpatialDim, kSpatialDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    nullspace_inputs_ = {
        controllers::read_view(joint_position_error_, controllers::Device::kCpu, 1, kArmJointDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kArmJointDim),
        controllers::read_view(
            joint_mass_, controllers::Device::kCpu, 1, kArmJointDim, kArmJointDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim)};
  }

  Eigen::Matrix4d desired_pose_{Eigen::Matrix4d::Identity()};
  ArmJointVector desired_joint_position_{ArmJointVector::Zero()};
  std::array<double, 1> regularization_{1.0e-6};
  std::array<double, 1> minimum_pivot_{1.0e-12};
  std::array<double, kSpatialDim * kSpatialDim> stiffness_{};
  std::array<double, kSpatialDim> damping_factors_{};
  std::array<double, kArmJointDim * kArmJointDim> nullspace_stiffness_{};
  std::array<double, kArmJointDim * kArmJointDim> nullspace_damping_{};
  std::array<double, kSpatialDim> pose_error_{};
  std::array<double, kSpatialDim> velocity_error_{};
  std::array<double, kSpatialDim> desired_acceleration_{};
  std::array<double, kArmJointDim> joint_position_error_{};
  std::array<double, kArmJointDim> joint_velocity_{};
  std::array<double, kArmJointDim * kArmJointDim> joint_mass_{};
  std::array<double, kSpatialDim * kArmJointDim> jacobian_{};
  std::array<double, 1> dt_{};
  std::array<double, kSpatialDim * kSpatialDim> cartesian_mass_{};
  std::array<double, kArmJointDim * kArmJointDim> mass_joint_factor_{};
  std::array<double, kArmJointDim * kSpatialDim> mass_solve_rhs_{};
  std::array<double, kSpatialDim * kSpatialDim> cartesian_inverse_{};
  std::array<double, kSpatialDim> wrench_{};
  std::array<double, kArmJointDim> nullspace_torque_{};
  std::array<double, kSpatialDim * kSpatialDim> previous_cartesian_mass_{};
  std::array<double, kSpatialDim * kArmJointDim> previous_jacobian_{};
  std::array<std::uint8_t, 1> impedance_initialized_{};
  std::array<double, kSpatialDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> impedance_valid_{};
  std::array<std::uint64_t, 1> impedance_hold_count_{};
  std::array<double, kArmJointDim> last_valid_nullspace_torque_{};
  std::array<std::uint8_t, 1> nullspace_valid_{};
  std::array<std::uint64_t, 1> nullspace_hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> impedance_parameter_status_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  std::array<std::uint64_t, 1> nullspace_parameter_status_{};
  std::array<std::uint64_t, 1> nullspace_status_{};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  controllers::CartesianImpedanceController::Inputs impedance_inputs_{};
  controllers::NullspaceController::Inputs nullspace_inputs_{};
  controllers::OperationalSpaceMass mass_;
  controllers::CartesianImpedanceController impedance_;
  controllers::NullspaceController nullspace_;
};

class DualIndependentCartesianImpedanceController {
 public:
  controllers::Result prepare(const controllers::BimanualFrameConfig& frames,
                              const std::vector<double>& torque_limits) {
    if (torque_limits.size() != kJointDim || !has_only_finite_values(torque_limits)) {
      return {controllers::ResultCode::kInvalidArgument,
              "expected fourteen finite joint-torque limits"};
    }
    if (!is_rigid_transform(frames.arm1_T_arm2) ||
        (frames.main_frame == controllers::BimanualMainFrame::kExplicit &&
         !is_rigid_transform(frames.reference_T_main))) {
      return {controllers::ResultCode::kInvalidArgument,
              "bimanual frame calibration must contain finite rigid transforms"};
    }
    frames_ = frames;
    controllers::Result result = arm1_controller_.prepare();
    if (result.ok()) {
      result = arm2_controller_.prepare();
    }
    if (!result.ok()) {
      return result;
    }
    for (std::size_t joint = 0; joint < kJointDim; ++joint) {
      lower_limit_[joint] = -kTorqueMagnitudeLimitFactor * torque_limits[joint];
      upper_limit_[joint] = kTorqueMagnitudeLimitFactor * torque_limits[joint];
      maximum_rate_[joint] = kTorqueSlewRateNmPerSecond;
    }
    const controllers::ControllerConfig filter_config{
        controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointDim};
    result = filter_.prepare(
        filter_config,
        {controllers::shared(
             controllers::read_view(lower_limit_, controllers::Device::kCpu, kJointDim)),
         controllers::shared(
             controllers::read_view(upper_limit_, controllers::Device::kCpu, kJointDim)),
         controllers::shared(
             controllers::read_view(maximum_rate_, controllers::Device::kCpu, kJointDim))},
        {{controllers::write_view(filtered_torque_, controllers::Device::kCpu, 1, kJointDim),
          controllers::write_status(filter_parameter_status_, controllers::Device::kCpu),
          controllers::write_status(filter_status_, controllers::Device::kCpu),
          controllers::make_safety_state(
              controllers::write_view(last_valid_filter_, controllers::Device::kCpu, 1, kJointDim),
              controllers::write_view(filter_valid_, controllers::Device::kCpu, 1),
              controllers::write_view(filter_hold_count_, controllers::Device::kCpu, 1))},
         {controllers::write_view(previous_command_, controllers::Device::kCpu, 1, kJointDim),
          controllers::write_view(filter_initialized_, controllers::Device::kCpu, 1)}});
    if (!result.ok()) {
      return result;
    }
    configure_inputs();
    result = reset_all();
    if (!result.ok()) {
      return result;
    }
    dt_[0] = kNominalPeriodSeconds;
    result = filter_.step(filter_inputs_);
    return result.ok() && is_safe(filter_status_[0])
               ? controllers::Result{}
               : controllers::Result{controllers::ResultCode::kInvalidArgument,
                                     "failed to seed the torque filter"};
  }

  bool set_initial_state(const CycleData& initial) {
    arm1_T_arm2_ = transform_from_array(frames_.arm1_T_arm2);
    initial_setpoint_ = initial.arm1_tcp_pose;
    setpoint_T_arm1_tcp_.setIdentity();
    setpoint_T_arm2_tcp_ = initial_setpoint_.inverse() * arm1_T_arm2_ * initial.arm2_tcp_pose;
    return arm1_controller_.set_target(initial_setpoint_ * setpoint_T_arm1_tcp_,
                                       initial.position.head<kArmJointDim>()) &&
           arm2_controller_.set_target(
               arm1_T_arm2_.inverse() * initial_setpoint_ * setpoint_T_arm2_tcp_,
               initial.position.tail<kArmJointDim>());
  }

  bool step(const CycleData& cycle,
            const double elapsed_seconds,
            const double dt_seconds,
            std::array<double, kJointDim>* torque) {
    if (torque == nullptr || !std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0 ||
        !std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
      return false;
    }
    constexpr double kPi = 3.14159265358979323846;
    const double normalized_ramp = std::min(1.0, elapsed_seconds / kCircleSpeedRampSeconds);
    const double ramp_phase_integral =
        normalized_ramp * normalized_ramp * normalized_ramp -
        0.5 * normalized_ramp * normalized_ramp * normalized_ramp * normalized_ramp;
    const double angular_frequency = 2.0 * kPi * kCircleFrequencyHertz;
    const double phase =
        angular_frequency * (kCircleSpeedRampSeconds * ramp_phase_integral +
                             std::max(0.0, elapsed_seconds - kCircleSpeedRampSeconds));
    const double phase_velocity =
        normalized_ramp < 1.0
            ? angular_frequency * (3.0 * normalized_ramp * normalized_ramp -
                                   2.0 * normalized_ramp * normalized_ramp * normalized_ramp)
            : angular_frequency;
    const double phase_acceleration =
        normalized_ramp < 1.0
            ? angular_frequency / kCircleSpeedRampSeconds *
                  (6.0 * normalized_ramp - 6.0 * normalized_ramp * normalized_ramp)
            : 0.0;
    Eigen::Matrix4d setpoint = initial_setpoint_;
    setpoint(1, 3) += kCircleRadiusMeters * std::sin(phase);
    setpoint(2, 3) += kCircleRadiusMeters * (std::cos(phase) - 1.0);
    if (!arm1_controller_.set_pose_target(setpoint * setpoint_T_arm1_tcp_) ||
        !arm2_controller_.set_pose_target(arm1_T_arm2_.inverse() * setpoint *
                                          setpoint_T_arm2_tcp_)) {
      return false;
    }
    CartesianVector arm1_desired_velocity = CartesianVector::Zero();
    CartesianVector arm1_desired_acceleration = CartesianVector::Zero();
    arm1_desired_velocity[1] = kCircleRadiusMeters * std::cos(phase) * phase_velocity;
    arm1_desired_velocity[2] = -kCircleRadiusMeters * std::sin(phase) * phase_velocity;
    arm1_desired_acceleration[1] =
        kCircleRadiusMeters *
        (-std::sin(phase) * phase_velocity * phase_velocity + std::cos(phase) * phase_acceleration);
    arm1_desired_acceleration[2] =
        -kCircleRadiusMeters *
        (std::cos(phase) * phase_velocity * phase_velocity + std::sin(phase) * phase_acceleration);
    CartesianVector arm2_desired_velocity = CartesianVector::Zero();
    CartesianVector arm2_desired_acceleration = CartesianVector::Zero();
    arm2_desired_velocity.head<3>() =
        arm1_T_arm2_.topLeftCorner<3, 3>().transpose() * arm1_desired_velocity.head<3>();
    arm2_desired_acceleration.head<3>() =
        arm1_T_arm2_.topLeftCorner<3, 3>().transpose() * arm1_desired_acceleration.head<3>();
    const ArmJointVector arm1_position = cycle.position.head<kArmJointDim>();
    const ArmJointVector arm2_position = cycle.position.tail<kArmJointDim>();
    const ArmJointVector arm1_velocity = cycle.velocity.head<kArmJointDim>();
    const ArmJointVector arm2_velocity = cycle.velocity.tail<kArmJointDim>();
    const ArmJointMatrix arm1_mass = cycle.mass.topLeftCorner<kArmJointDim, kArmJointDim>();
    const ArmJointMatrix arm2_mass = cycle.mass.bottomRightCorner<kArmJointDim, kArmJointDim>();
    ArmJointVector arm1_torque;
    ArmJointVector arm2_torque;
    if (!arm1_controller_.step(cycle.arm1_tcp_pose,
                               cycle.arm1_jacobian,
                               arm1_mass,
                               arm1_position,
                               arm1_velocity,
                               arm1_desired_velocity,
                               arm1_desired_acceleration,
                               dt_seconds,
                               &arm1_torque) ||
        !arm2_controller_.step(cycle.arm2_tcp_pose,
                               cycle.arm2_jacobian,
                               arm2_mass,
                               arm2_position,
                               arm2_velocity,
                               arm2_desired_velocity,
                               arm2_desired_acceleration,
                               dt_seconds,
                               &arm2_torque)) {
      return false;
    }
    for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
      unfiltered_torque_[joint] = arm1_torque[static_cast<Eigen::Index>(joint)] +
                                  cycle.coriolis[static_cast<Eigen::Index>(joint)];
      unfiltered_torque_[kArmJointDim + joint] =
          arm2_torque[static_cast<Eigen::Index>(joint)] +
          cycle.coriolis[static_cast<Eigen::Index>(kArmJointDim + joint)];
    }
    dt_[0] = dt_seconds;
    filter_status_[0] = 0U;
    if (!filter_.step(filter_inputs_).ok() || !is_safe(filter_status_[0])) {
      return false;
    }
    *torque = filtered_torque_;
    return has_only_finite_values(*torque);
  }

  void publish_debug(
      const CycleData&, const std::uint64_t, const double, const double, DebugSnapshot*) const {}

 private:
  void configure_inputs() {
    filter_inputs_ = {
        controllers::read_view(unfiltered_torque_, controllers::Device::kCpu, 1, kJointDim),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
  }

  controllers::Result reset_all() { return filter_.reset(); }

  controllers::BimanualFrameConfig frames_{};
  Eigen::Matrix4d arm1_T_arm2_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d initial_setpoint_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d setpoint_T_arm1_tcp_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d setpoint_T_arm2_tcp_{Eigen::Matrix4d::Identity()};
  std::array<double, kJointDim> lower_limit_{};
  std::array<double, kJointDim> upper_limit_{};
  std::array<double, kJointDim> maximum_rate_{};
  std::array<double, 1> dt_{};
  std::array<double, kJointDim> unfiltered_torque_{};
  std::array<double, kJointDim> filtered_torque_{};
  std::array<double, kJointDim> previous_command_{};
  std::array<std::uint8_t, 1> filter_initialized_{};
  std::array<double, kJointDim> last_valid_filter_{};
  std::array<std::uint8_t, 1> filter_valid_{};
  std::array<std::uint64_t, 1> filter_hold_count_{};
  std::array<std::uint64_t, 1> filter_parameter_status_{};
  std::array<std::uint64_t, 1> filter_status_{};
  controllers::TorqueFilter::Inputs filter_inputs_{};
  ArmHoldController arm1_controller_;
  ArmHoldController arm2_controller_;
  controllers::TorqueFilter filter_;
};

class CooperativeBimanualController {
 public:
  CooperativeBimanualController() {
    constexpr std::array<double, kSpatialDim> kCircleAbsoluteStiffness{
        1500, 1500, 1500, 150, 150, 150};
    constexpr std::array<double, kSpatialDim> kCircleRelativeStiffness{500, 500, 500, 50, 50, 50};
    constexpr std::array<double, kSpatialDim> kHoldAbsoluteStiffness{300, 300, 300, 30, 30, 30};
    constexpr std::array<double, kSpatialDim> kHoldRelativeStiffness{750, 750, 750, 75, 75, 75};
    constexpr std::array<double, kArmJointDim> kNullspaceStiffness{40, 30, 20, 16, 10, 6, 2};
    constexpr std::array<double, kArmJointDim> kNullspaceDamping{5, 4, 3, 2, 1.5, 1, 0.5};
    const std::array<double, kSpatialDim>& absolute_stiffness =
        kExampleProfile == 0 ? kCircleAbsoluteStiffness : kHoldAbsoluteStiffness;
    const std::array<double, kSpatialDim>& relative_stiffness =
        kExampleProfile == 0 ? kCircleRelativeStiffness : kHoldRelativeStiffness;
    for (std::size_t axis = 0; axis < kSpatialDim; ++axis) {
      stiffness_[axis * kTaskDim + axis] = absolute_stiffness[axis];
      stiffness_[(kSpatialDim + axis) * kTaskDim + kSpatialDim + axis] = relative_stiffness[axis];
      damping_factors_[axis] = 0.7;
      damping_factors_[kSpatialDim + axis] = 0.7;
    }
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const std::size_t offset = arm * kArmJointDim;
      for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
        nullspace_stiffness_[(offset + joint) * kJointDim + offset + joint] =
            kNullspaceStiffness[joint];
        nullspace_damping_[(offset + joint) * kJointDim + offset + joint] =
            kNullspaceDamping[joint];
      }
    }
  }

  controllers::Result prepare(const controllers::BimanualFrameConfig& frames,
                              const std::vector<double>& torque_limits) {
    if (torque_limits.size() != kJointDim || !has_only_finite_values(torque_limits)) {
      return {controllers::ResultCode::kInvalidArgument,
              "expected fourteen finite joint-torque limits"};
    }
    if (!is_rigid_transform(frames.arm1_T_arm2) ||
        (frames.main_frame == controllers::BimanualMainFrame::kExplicit &&
         !is_rigid_transform(frames.reference_T_main))) {
      return {controllers::ResultCode::kInvalidArgument,
              "bimanual frame calibration must contain finite rigid transforms"};
    }
    frames_ = frames;
    controllers::BimanualTaskConfig config;
    config.device = controllers::Device::kCpu;
    config.dtype = controllers::DType::kFloat64;
    config.batch_size = 1;
    config.joint_dim = kJointDim;
    config.cartesian_frame = controllers::CartesianFrame::kMainFrameOrientation;
    config.task_dim = kTaskDim;
    config.frames = frames_;
    config.coordination_mode = controllers::BimanualCoordinationMode::kDecoupledCooperative;
    for (std::size_t joint = 0; joint < kJointDim; ++joint) {
      lower_limit_[joint] = -kTorqueMagnitudeLimitFactor * torque_limits[joint];
      upper_limit_[joint] = kTorqueMagnitudeLimitFactor * torque_limits[joint];
      maximum_rate_[joint] = kTorqueSlewRateNmPerSecond;
    }
    controllers::Result result = controller_.prepare(
        config,
        {{controllers::shared(
              controllers::read_view(regularization_, controllers::Device::kCpu, 1)),
          controllers::shared(
              controllers::read_view(minimum_pivot_, controllers::Device::kCpu, 1))},
         {controllers::shared(
              controllers::read_view(stiffness_, controllers::Device::kCpu, kTaskDim, kTaskDim)),
          controllers::shared(
              controllers::read_view(damping_factors_, controllers::Device::kCpu, kTaskDim))},
         {controllers::shared(controllers::read_view(
              nullspace_stiffness_, controllers::Device::kCpu, kJointDim, kJointDim)),
          controllers::shared(controllers::read_view(
              nullspace_damping_, controllers::Device::kCpu, kJointDim, kJointDim))},
         {controllers::shared(
              controllers::read_view(lower_limit_, controllers::Device::kCpu, kJointDim)),
          controllers::shared(
              controllers::read_view(upper_limit_, controllers::Device::kCpu, kJointDim)),
          controllers::shared(
              controllers::read_view(maximum_rate_, controllers::Device::kCpu, kJointDim))}},
        {{controllers::write_view(task_error_, controllers::Device::kCpu, 1, kTaskDim),
          controllers::write_view(
              task_jacobian_, controllers::Device::kCpu, 1, kTaskDim, kJointDim),
          controllers::write_status(kinematics_status_, controllers::Device::kCpu)},
         {controllers::write_view(task_mass_, controllers::Device::kCpu, 1, kTaskDim, kTaskDim),
          controllers::write_status(mass_parameter_status_, controllers::Device::kCpu),
          controllers::write_status(mass_status_, controllers::Device::kCpu),
          {controllers::write_view(
               joint_factor_, controllers::Device::kCpu, 1, kJointDim, kJointDim),
           controllers::write_view(solve_rhs_, controllers::Device::kCpu, 1, kJointDim, kTaskDim),
           controllers::write_view(
               task_inverse_, controllers::Device::kCpu, 1, kTaskDim, kTaskDim)}},
         {{controllers::write_view(wrench_, controllers::Device::kCpu, 1, kTaskDim),
           controllers::write_status(impedance_parameter_status_, controllers::Device::kCpu),
           controllers::write_status(impedance_status_, controllers::Device::kCpu),
           controllers::make_safety_state(
               controllers::write_view(last_valid_wrench_, controllers::Device::kCpu, 1, kTaskDim),
               controllers::write_view(impedance_valid_, controllers::Device::kCpu, 1),
               controllers::write_view(impedance_hold_count_, controllers::Device::kCpu, 1))},
          {controllers::write_view(
               previous_task_mass_, controllers::Device::kCpu, 1, kTaskDim, kTaskDim),
           controllers::write_view(
               previous_task_jacobian_, controllers::Device::kCpu, 1, kTaskDim, kJointDim),
           controllers::write_view(impedance_initialized_, controllers::Device::kCpu, 1)}},
         {controllers::write_view(nullspace_torque_, controllers::Device::kCpu, 1, kJointDim),
          controllers::write_status(nullspace_parameter_status_, controllers::Device::kCpu),
          controllers::write_status(nullspace_status_, controllers::Device::kCpu),
          controllers::make_safety_state(
              controllers::write_view(
                  last_valid_nullspace_torque_, controllers::Device::kCpu, 1, kJointDim),
              controllers::write_view(nullspace_valid_, controllers::Device::kCpu, 1),
              controllers::write_view(nullspace_hold_count_, controllers::Device::kCpu, 1))},
         {controllers::write_view(unfiltered_torque_, controllers::Device::kCpu, 1, kJointDim),
          {{controllers::write_view(filtered_torque_, controllers::Device::kCpu, 1, kJointDim),
            controllers::write_status(filter_parameter_status_, controllers::Device::kCpu),
            controllers::write_status(filter_status_, controllers::Device::kCpu),
            controllers::make_safety_state(
                controllers::write_view(last_valid_filter_,
                                        controllers::Device::kCpu,
                                        1,
                                        kJointDim),
                controllers::write_view(filter_valid_, controllers::Device::kCpu, 1),
                controllers::write_view(filter_hold_count_, controllers::Device::kCpu, 1))},
           {controllers::write_view(previous_command_, controllers::Device::kCpu, 1, kJointDim),
            controllers::write_view(filter_initialized_, controllers::Device::kCpu, 1)}}},
         controllers::write_view(task_velocity_error_, controllers::Device::kCpu, 1, kTaskDim)});
    configure_inputs();
    return result.ok() ? controller_.reset() : result;
  }

  bool set_initial_state(const CycleData& initial) {
    const Eigen::Matrix4d arm1_T_arm2 = transform_from_array(frames_.arm1_T_arm2);
    const Eigen::Matrix4d arm1_T_tcp2 = arm1_T_arm2 * initial.arm2_tcp_pose;
    const Eigen::Matrix4d relative_transform = initial.arm1_tcp_pose.inverse() * arm1_T_tcp2;
    desired_relative_pose_ = relative_transform;
    desired_relative_pose_.topRightCorner<3, 1>() =
        arm1_T_tcp2.topRightCorner<3, 1>() - initial.arm1_tcp_pose.topRightCorner<3, 1>();
    desired_absolute_pose_ = initial.arm1_tcp_pose * se3_exp(0.5 * se3_log(relative_transform));
    desired_absolute_pose_.topRightCorner<3, 1>() =
        0.5 * (initial.arm1_tcp_pose.topRightCorner<3, 1>() + arm1_T_tcp2.topRightCorner<3, 1>());
    initial_absolute_pose_ = desired_absolute_pose_;
    has_previous_tcp_pose_ = false;
    copy_vector(initial.position, &desired_joint_position_);
    return desired_absolute_pose_.allFinite() && desired_relative_pose_.allFinite();
  }

  bool step(const CycleData& cycle,
            const double elapsed_seconds,
            const double dt_seconds,
            std::array<double, kJointDim>* torque) {
    if (torque == nullptr || !std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0 ||
        !std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
      return false;
    }
    if (has_previous_tcp_pose_) {
      arm1_finite_difference_velocity_ = calculate_finite_difference_velocity(
          previous_arm1_tcp_pose_, cycle.arm1_tcp_pose, dt_seconds);
      arm2_finite_difference_velocity_ = calculate_finite_difference_velocity(
          previous_arm2_tcp_pose_, cycle.arm2_tcp_pose, dt_seconds);
    } else {
      arm1_finite_difference_velocity_.setZero();
      arm2_finite_difference_velocity_.setZero();
      has_previous_tcp_pose_ = true;
    }
    previous_arm1_tcp_pose_ = cycle.arm1_tcp_pose;
    previous_arm2_tcp_pose_ = cycle.arm2_tcp_pose;
    CartesianVector absolute_velocity_arm1 = CartesianVector::Zero();
    CartesianVector absolute_acceleration_arm1 = CartesianVector::Zero();
    desired_absolute_pose_ = initial_absolute_pose_;
    if (kExampleProfile == 0) {
      constexpr double kPi = 3.14159265358979323846;
      const double ramp = std::min(1.0, elapsed_seconds / kCircleSpeedRampSeconds);
      const double ramp_integral = ramp * ramp * ramp - 0.5 * ramp * ramp * ramp * ramp;
      const double frequency = 2.0 * kPi * kCircleFrequencyHertz;
      const double phase = frequency * (kCircleSpeedRampSeconds * ramp_integral +
                                        std::max(0.0, elapsed_seconds - kCircleSpeedRampSeconds));
      const double phase_velocity =
          ramp < 1.0 ? frequency * (3.0 * ramp * ramp - 2.0 * ramp * ramp * ramp) : frequency;
      const double phase_acceleration =
          ramp < 1.0 ? frequency / kCircleSpeedRampSeconds * (6.0 * ramp - 6.0 * ramp * ramp) : 0.0;
      desired_absolute_pose_(1, 3) += kCircleRadiusMeters * std::sin(phase);
      desired_absolute_pose_(2, 3) += kCircleRadiusMeters * (std::cos(phase) - 1.0);
      absolute_velocity_arm1[1] = kCircleRadiusMeters * std::cos(phase) * phase_velocity;
      absolute_velocity_arm1[2] = -kCircleRadiusMeters * std::sin(phase) * phase_velocity;
      absolute_acceleration_arm1[1] =
          kCircleRadiusMeters * (-std::sin(phase) * phase_velocity * phase_velocity +
                                 std::cos(phase) * phase_acceleration);
      absolute_acceleration_arm1[2] =
          -kCircleRadiusMeters * (std::cos(phase) * phase_velocity * phase_velocity +
                                  std::sin(phase) * phase_acceleration);
    }
    copy_matrix(desired_absolute_pose_, &desired_absolute_pose_array_);
    copy_matrix(cycle.arm1_tcp_pose, &arm1_tcp_pose_array_);
    copy_matrix(cycle.arm2_tcp_pose, &arm2_tcp_pose_array_);
    copy_matrix(desired_relative_pose_, &desired_relative_pose_array_);
    copy_matrix(cycle.arm1_jacobian, &arm1_jacobian_array_);
    copy_matrix(cycle.arm2_jacobian, &arm2_jacobian_array_);
    const SpatialMatrix main_X_arm1 = rotate_spatial(main_rotation(frames_));
    TaskVector desired_velocity = TaskVector::Zero();
    TaskVector desired_acceleration = TaskVector::Zero();
    desired_velocity.head<kSpatialDim>() = main_X_arm1 * absolute_velocity_arm1;
    desired_acceleration.head<kSpatialDim>() = main_X_arm1 * absolute_acceleration_arm1;
    copy_matrix(cycle.mass, &joint_mass_);
    copy_vector(cycle.motor_velocity, &joint_velocity_);
    copy_vector(Eigen::Map<const JointVector>(desired_joint_position_.data()) - cycle.position,
                &joint_position_error_);
    copy_vector(desired_velocity, &desired_task_velocity_);
    copy_vector(desired_acceleration, &desired_acceleration_);
    copy_vector(cycle.coriolis, &joint_bias_torque_);
    dt_[0] = dt_seconds;
    filter_status_[0] = 0U;
    if (!controller_.step(controller_inputs_).ok() || !is_safe(filter_status_[0])) {
      return false;
    }
    *torque = filtered_torque_;
    return has_only_finite_values(*torque);
  }

  void publish_debug(const CycleData& cycle,
                     const std::uint64_t sample,
                     const double elapsed_seconds,
                     const double dt_seconds,
                     DebugSnapshot* snapshot) const {
    if (snapshot == nullptr) {
      return;
    }
    snapshot->sequence.fetch_add(1U, std::memory_order_acq_rel);
    snapshot->sample.store(sample, std::memory_order_relaxed);
    snapshot->elapsed_seconds.store(elapsed_seconds, std::memory_order_relaxed);
    snapshot->dt_seconds.store(dt_seconds, std::memory_order_relaxed);
    store_debug_array(task_error_, &snapshot->task_error);
    store_debug_array(task_velocity_error_, &snapshot->task_velocity_error);
    store_debug_array(wrench_, &snapshot->wrench);
    std::array<double, kTaskDim> task_mass_diagonal{};
    for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
      task_mass_diagonal[axis] = task_mass_[axis * kTaskDim + axis];
    }
    store_debug_array(task_mass_diagonal, &snapshot->task_mass_diagonal);
    std::array<double, kJointDim> task_torque{};
    for (std::size_t joint = 0; joint < kJointDim; ++joint) {
      for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
        task_torque[joint] += task_jacobian_[axis * kJointDim + joint] * wrench_[axis];
      }
    }
    store_debug_array(task_torque, &snapshot->task_torque);
    store_debug_array(joint_position_error_, &snapshot->joint_position_error);
    store_debug_array(joint_velocity_, &snapshot->joint_velocity);
    store_debug_array(nullspace_torque_, &snapshot->nullspace_torque);
    std::array<double, kJointDim> coriolis{};
    copy_vector(cycle.coriolis, &coriolis);
    store_debug_array(coriolis, &snapshot->coriolis);
    std::array<double, kJointDim> gravity{};
    copy_vector(cycle.gravity, &gravity);
    store_debug_array(gravity, &snapshot->gravity);
    std::array<double, kJointDim> measured_torque{};
    copy_vector(cycle.measured_torque, &measured_torque);
    store_debug_array(measured_torque, &snapshot->measured_torque);
    std::array<double, kJointDim> desired_torque{};
    copy_vector(cycle.desired_torque, &desired_torque);
    store_debug_array(desired_torque, &snapshot->desired_torque);
    std::array<double, kJointDim> motor_link_deflection{};
    copy_vector(cycle.motor_position - cycle.position, &motor_link_deflection);
    store_debug_array(motor_link_deflection, &snapshot->motor_link_deflection);
    store_debug_array(unfiltered_torque_, &snapshot->unfiltered_torque);
    store_debug_array(filtered_torque_, &snapshot->filtered_torque);
    TaskVector jacobian_link_velocity;
    jacobian_link_velocity.head<kSpatialDim>() =
        cycle.arm1_jacobian * cycle.velocity.head<kArmJointDim>();
    jacobian_link_velocity.tail<kSpatialDim>() =
        cycle.arm2_jacobian * cycle.velocity.tail<kArmJointDim>();
    TaskVector jacobian_motor_velocity;
    jacobian_motor_velocity.head<kSpatialDim>() =
        cycle.arm1_jacobian * cycle.motor_velocity.head<kArmJointDim>();
    jacobian_motor_velocity.tail<kSpatialDim>() =
        cycle.arm2_jacobian * cycle.motor_velocity.tail<kArmJointDim>();
    TaskVector measured_tcp_velocity;
    measured_tcp_velocity << cycle.arm1_tcp_velocity, cycle.arm2_tcp_velocity;
    TaskVector finite_difference_tcp_velocity;
    finite_difference_tcp_velocity << arm1_finite_difference_velocity_,
        arm2_finite_difference_velocity_;
    std::array<double, kTaskDim> velocity_debug{};
    copy_vector(jacobian_link_velocity, &velocity_debug);
    store_debug_array(velocity_debug, &snapshot->jacobian_link_velocity);
    copy_vector(jacobian_motor_velocity, &velocity_debug);
    store_debug_array(velocity_debug, &snapshot->jacobian_motor_velocity);
    copy_vector(measured_tcp_velocity, &velocity_debug);
    store_debug_array(velocity_debug, &snapshot->measured_tcp_velocity);
    copy_vector(finite_difference_tcp_velocity, &velocity_debug);
    store_debug_array(velocity_debug, &snapshot->finite_difference_tcp_velocity);
    snapshot->sequence.fetch_add(1U, std::memory_order_release);
  }

 private:
  void configure_inputs() {
    controller_inputs_ = {
        {controllers::read_view(desired_absolute_pose_array_, controllers::Device::kCpu, 1, 4, 4),
         controllers::read_view(arm1_tcp_pose_array_, controllers::Device::kCpu, 1, 4, 4),
         controllers::read_view(arm2_tcp_pose_array_, controllers::Device::kCpu, 1, 4, 4),
         controllers::read_view(desired_relative_pose_array_, controllers::Device::kCpu, 1, 4, 4),
         controllers::read_view(
             arm1_jacobian_array_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim),
         controllers::read_view(
             arm2_jacobian_array_, controllers::Device::kCpu, 1, kSpatialDim, kArmJointDim)},
        controllers::read_view(desired_task_velocity_, controllers::Device::kCpu, 1, kTaskDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kTaskDim),
        controllers::read_view(joint_position_error_, controllers::Device::kCpu, 1, kJointDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointDim),
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointDim, kJointDim),
        controllers::read_view(joint_bias_torque_, controllers::Device::kCpu, 1, kJointDim),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
  }

  controllers::BimanualFrameConfig frames_{};
  Eigen::Matrix4d initial_absolute_pose_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d desired_absolute_pose_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d desired_relative_pose_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d previous_arm1_tcp_pose_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix4d previous_arm2_tcp_pose_{Eigen::Matrix4d::Identity()};
  CartesianVector arm1_finite_difference_velocity_{CartesianVector::Zero()};
  CartesianVector arm2_finite_difference_velocity_{CartesianVector::Zero()};
  bool has_previous_tcp_pose_{false};
  std::array<double, 1> regularization_{1.0e-6};
  std::array<double, 1> minimum_pivot_{1.0e-12};
  std::array<double, kTaskDim * kTaskDim> stiffness_{};
  std::array<double, kTaskDim> damping_factors_{};
  std::array<double, kJointDim * kJointDim> nullspace_stiffness_{};
  std::array<double, kJointDim * kJointDim> nullspace_damping_{};
  std::array<double, kJointDim> lower_limit_{};
  std::array<double, kJointDim> upper_limit_{};
  std::array<double, kJointDim> maximum_rate_{};
  std::array<double, kTaskDim> task_error_{};
  std::array<double, kTaskDim * kJointDim> task_jacobian_{};
  std::array<double, 16> desired_absolute_pose_array_{};
  std::array<double, 16> arm1_tcp_pose_array_{};
  std::array<double, 16> arm2_tcp_pose_array_{};
  std::array<double, 16> desired_relative_pose_array_{};
  std::array<double, kSpatialDim * kArmJointDim> arm1_jacobian_array_{};
  std::array<double, kSpatialDim * kArmJointDim> arm2_jacobian_array_{};
  std::array<double, kJointDim * kJointDim> joint_mass_{};
  std::array<double, kJointDim> joint_velocity_{};
  std::array<double, kJointDim> joint_bias_torque_{};
  std::array<double, kJointDim> desired_joint_position_{};
  std::array<double, kJointDim> joint_position_error_{};
  std::array<double, kTaskDim> desired_task_velocity_{};
  std::array<double, kTaskDim> task_velocity_error_{};
  std::array<double, kTaskDim> desired_acceleration_{};
  std::array<double, 1> dt_{};
  std::array<double, kTaskDim * kTaskDim> task_mass_{};
  std::array<double, kJointDim * kJointDim> joint_factor_{};
  std::array<double, kJointDim * kTaskDim> solve_rhs_{};
  std::array<double, kTaskDim * kTaskDim> task_inverse_{};
  std::array<double, kTaskDim> wrench_{};
  std::array<double, kJointDim> nullspace_torque_{};
  std::array<double, kTaskDim * kTaskDim> previous_task_mass_{};
  std::array<double, kTaskDim * kJointDim> previous_task_jacobian_{};
  std::array<std::uint8_t, 1> impedance_initialized_{};
  std::array<double, kTaskDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> impedance_valid_{};
  std::array<std::uint64_t, 1> impedance_hold_count_{};
  std::array<double, kJointDim> unfiltered_torque_{};
  std::array<double, kJointDim> filtered_torque_{};
  std::array<double, kJointDim> previous_command_{};
  std::array<std::uint8_t, 1> filter_initialized_{};
  std::array<double, kJointDim> last_valid_filter_{};
  std::array<std::uint8_t, 1> filter_valid_{};
  std::array<std::uint64_t, 1> filter_hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> kinematics_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> impedance_parameter_status_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  std::array<double, kJointDim> last_valid_nullspace_torque_{};
  std::array<std::uint8_t, 1> nullspace_valid_{};
  std::array<std::uint64_t, 1> nullspace_hold_count_{};
  std::array<std::uint64_t, 1> nullspace_parameter_status_{};
  std::array<std::uint64_t, 1> nullspace_status_{};
  std::array<std::uint64_t, 1> filter_parameter_status_{};
  std::array<std::uint64_t, 1> filter_status_{};
  controllers::BimanualCartesianImpedanceTorqueController::Inputs controller_inputs_{};
  controllers::BimanualCartesianImpedanceTorqueController controller_;
};

bool confirm_execution(const ProgramConfig& config) {
  const char* title =
      kExampleProfile == 0 ? "bimanual circular tracking" : "bimanual compliant hold";
  std::cout << "Flexiv RDK 1.9.1 two-Rizon " << title << "\n\n"
            << "Arm 1: " << config.arm1_robot_serial << "; arm 2: " << config.arm2_robot_serial
            << ".\n";
  std::cout << "One 12D impedance uses decoupled cooperative coordinates: the Cartesian/SO(3) "
               "midpoint plus shared-base TCP displacement and relative orientation. Task "
               "vectors use [linear; angular] in the configured main-frame orientation.\n"
            << (kExampleProfile == 0
                    ? "The midpoint follows a 0.2 m diameter, 0.2 Hz circle in the arm-1-base "
                      "y/z plane, starting at the top. Absolute stiffness is "
                      "[1500,1500,1500,150,150,150] and relative stiffness is "
                      "[500,500,500,50,50,50].\n"
                    : "The midpoint is held with stiffness [300,300,300,30,30,30], while the "
                      "relative pose uses [750,750,750,75,75,75].\n");
  std::cout << "Bimanual wrench uses K*error + mass-aware D*velocity_error only; numerical "
               "mass/Jacobian derivative terms are disabled. Independent per-arm projected "
               "nullspace impedance holds each arm at its captured torque-mode start posture "
               "with stiffness "
               "[40,30,20,16,10,6,2] and damping [5,4,3,2,1.5,1,0.5].\n";
  std::cout << "RDK torque-mode compensation: gravity="
            << (kEnableGravityCompensation ? "enabled" : "disabled")
            << ", soft limits=" << (kEnableSoftLimits ? "enabled" : "disabled")
            << ", friction=" << kFrictionCompensationPercent << "%.\n";
  std::cout << "For moving profiles the reference ramps smoothly over 2 s in the arm-1-base "
               "y-z plane.\n"
            << "Each RDK WORLD frame is its arm's base; ARM1_BASE_T_ARM2_BASE supplies the full "
               "fixed calibration used to derive arm 2's target.\n"
            << "Flange links: " << config.arm1_flange_link << ", " << config.arm2_flange_link
            << ". Commands run for " << config.run_duration_seconds
            << " s with both robots in RT_JOINT_TORQUE mode.\n"
            << "Before controller startup, both arms move in NRT_JOINT_POSITION mode to "
               "[0, -15, 0, 75, 0, 0, 0] degrees. A second confirmation is then required.\n"
            << "The program does not clear faults, home, run recovery, change collision "
               "thresholds, or claim collision-free motion.\n"
            << "Keep an emergency stop within reach. RDK state/model calls allocate and can "
               "throw, so this is not a hard-real-time qualification.\n"
            << "non_rt_requested=" << (config.non_rt_requested ? "true" : "false") << '\n';
  if (config.non_rt_requested) {
    std::cout << "Warning: --non-rt uses the RDK scheduler's non-real-time priority and leaves "
                 "the task unbound from any CPU. Runtime timeliness and safety checks remain "
                 "enabled.\n";
  }
  std::cout << '\n'
            << "Press Enter to connect and execute both initial joint motions: " << std::flush;
  std::string response;
  return std::getline(std::cin, response) && response.empty();
}

bool confirm_controller_start() {
  std::cout << "Both arms reached [0, -15, 0, 75, 0, 0, 0] degrees.\n"
            << "Press Enter to start the dual-arm controller: " << std::flush;
  std::string response;
  return std::getline(std::cin, response) && response.empty();
}

bool valid_robot_info(const rdk::RobotInfo& info) {
  if (info.DoF_e != 0 || info.DoF != kArmJointDim || info.q_min.size() != kArmJointDim ||
      info.q_max.size() != kArmJointDim || info.dq_max.size() != kArmJointDim ||
      info.tau_max.size() != kArmJointDim || !has_only_finite_values(info.q_min) ||
      !has_only_finite_values(info.q_max) || !has_only_finite_values(info.dq_max) ||
      !has_only_finite_values(info.tau_max)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
    if (!(info.q_min[joint] < info.q_max[joint]) || info.dq_max[joint] <= 0.0 ||
        info.tau_max[joint] <= 0.0 || kInitialJointPosition[joint] < info.q_min[joint] ||
        kInitialJointPosition[joint] > info.q_max[joint]) {
      return false;
    }
  }
  return true;
}

bool wait_until_ready(rdk::Robot& robot, std::string* failure) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(kEnableTimeoutSeconds);
  bool enable_requested = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (robot.fault() || robot.reduced() || robot.recovery() || !robot.estop_released()) {
      *failure = "fault, reduced/recovery state, or engaged emergency stop";
      return false;
    }
    const rdk::OperationalStatus status = robot.operational_status();
    if (status == rdk::OperationalStatus::READY) {
      return true;
    }
    if (status == rdk::OperationalStatus::NOT_ENABLED && !enable_requested) {
      robot.Enable();
      enable_requested = true;
    } else if (status != rdk::OperationalStatus::NOT_ENABLED &&
               status != rdk::OperationalStatus::BOOTING &&
               status != rdk::OperationalStatus::RELEASING_BRAKE) {
      *failure = "operational status is not safe for automatic enable";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  *failure = "robot did not become READY within 30 seconds";
  return false;
}

bool has_reached_initial_position(const rdk::RobotStates& state) {
  if (state.q.size() != kArmJointDim || state.dq.size() != kArmJointDim ||
      !has_only_finite_values(state.q) || !has_only_finite_values(state.dq)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
    if (std::abs(state.q[joint] - kInitialJointPosition[joint]) > kInitialPositionTolerance ||
        std::abs(state.dq[joint]) > kInitialVelocityTolerance) {
      return false;
    }
  }
  return true;
}

void send_initial_position(rdk::Robot& robot, const rdk::RobotInfo& info) {
  const std::vector<double> target(kInitialJointPosition.begin(), kInitialJointPosition.end());
  const std::vector<double> target_velocity(kArmJointDim, 0.0);
  std::vector<double> maximum_velocity(kArmJointDim, kInitialMotionMaximumVelocity);
  const std::vector<double> maximum_acceleration(kArmJointDim, kInitialMotionMaximumAcceleration);
  for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
    maximum_velocity[joint] = std::min(maximum_velocity[joint], info.dq_max[joint]);
  }
  robot.SwitchMode(rdk::Mode::NRT_JOINT_POSITION);
  robot.SendJointPosition(target, target_velocity, maximum_velocity, maximum_acceleration);
}

bool move_to_initial_position(rdk::Robot& arm1_robot,
                              rdk::Robot& arm2_robot,
                              const rdk::RobotInfo& arm1_info,
                              const rdk::RobotInfo& arm2_info,
                              std::string* failure) {
  send_initial_position(arm1_robot, arm1_info);
  send_initial_position(arm2_robot, arm2_info);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(kInitialMotionTimeoutSeconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (g_signal_requested != 0) {
      *failure = "operator signal received during the initial motions";
      return false;
    }
    if (arm1_robot.fault() || arm2_robot.fault() ||
        arm1_robot.operational_status() != rdk::OperationalStatus::READY ||
        arm2_robot.operational_status() != rdk::OperationalStatus::READY) {
      *failure = "a robot left READY state during the initial motion";
      return false;
    }
    if (has_reached_initial_position(arm1_robot.states()) &&
        has_reached_initial_position(arm2_robot.states())) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  *failure = "initial motions did not converge within 60 seconds";
  return false;
}

class RobotStopGuard {
 public:
  explicit RobotStopGuard(rdk::Robot& robot) : robot_(robot) {}
  RobotStopGuard(const RobotStopGuard&) = delete;
  RobotStopGuard& operator=(const RobotStopGuard&) = delete;
  ~RobotStopGuard() {
    if (is_armed_) {
      try {
        robot_.Stop();
      } catch (...) {
      }
    }
  }
  void arm() { is_armed_ = true; }
  void stop() {
    if (is_armed_) {
      robot_.Stop();
      is_armed_ = false;
    }
  }

 private:
  rdk::Robot& robot_;
  bool is_armed_{false};
};

class SchedulerStopGuard {
 public:
  explicit SchedulerStopGuard(rdk::Scheduler& scheduler) : scheduler_(scheduler) {}
  SchedulerStopGuard(const SchedulerStopGuard&) = delete;
  SchedulerStopGuard& operator=(const SchedulerStopGuard&) = delete;
  ~SchedulerStopGuard() {
    if (is_started_) {
      try {
        scheduler_.Stop();
      } catch (...) {
      }
    }
  }
  void mark_started() { is_started_ = true; }
  void stop() {
    if (is_started_) {
      scheduler_.Stop();
      is_started_ = false;
    }
  }

 private:
  rdk::Scheduler& scheduler_;
  bool is_started_{false};
};

enum class StopReason {
  kNone,
  kDurationReached,
  kSignal,
  kRobotNotReady,
  kInvalidState,
  kControllerRejected,
  kTimelinessLimit,
  kRdkException,
};

struct RunState {
  std::atomic<bool> stop_requested{false};
  std::atomic<StopReason> stop_reason{StopReason::kNone};
  std::size_t samples{0};
  double elapsed_seconds{0.0};
  std::chrono::steady_clock::time_point previous_start{};

  void request_stop(const StopReason reason) {
    StopReason expected = StopReason::kNone;
    stop_reason.compare_exchange_strong(expected, reason);
    stop_requested.store(true);
  }
};

void signal_handler(int) { g_signal_requested = 1; }

int run(const ProgramConfig& config) {
  RunState run_state;
  DebugSnapshot debug_snapshot;
  g_signal_requested = 0;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    rdk::Robot arm1_robot(config.arm1_robot_serial);
    rdk::Robot arm2_robot(config.arm2_robot_serial);
    RobotStopGuard arm1_stop_guard(arm1_robot);
    RobotStopGuard arm2_stop_guard(arm2_robot);
    const rdk::RobotInfo arm1_info = arm1_robot.info();
    const rdk::RobotInfo arm2_info = arm2_robot.info();
    if (!valid_robot_info(arm1_info) || !valid_robot_info(arm2_info)) {
      throw std::runtime_error("expected two 7-DoF Rizon arms with finite torque limits");
    }
    std::string startup_failure;
    if (!wait_until_ready(arm1_robot, &startup_failure)) {
      throw std::runtime_error("arm-1 startup rejected: " + startup_failure);
    }
    if (!wait_until_ready(arm2_robot, &startup_failure)) {
      throw std::runtime_error("arm-2 startup rejected: " + startup_failure);
    }
    arm1_stop_guard.arm();
    arm2_stop_guard.arm();
    if (!move_to_initial_position(arm1_robot, arm2_robot, arm1_info, arm2_info, &startup_failure)) {
      throw std::runtime_error("initial motion rejected: " + startup_failure);
    }
    arm1_stop_guard.stop();
    arm2_stop_guard.stop();
    if (!confirm_controller_start()) {
      std::cout << "Execution cancelled before starting the dual-arm controller.\n";
      return EXIT_FAILURE;
    }
    rdk::Model arm1_model(arm1_robot);
    rdk::Model arm2_model(arm2_robot);
    CycleData cycle;
    if (!fill_cycle_data(
            arm1_robot.states(), arm2_robot.states(), arm1_model, arm2_model, config, &cycle)) {
      std::ostringstream message;
      message << "initial model data invalid; arm-1 links:";
      for (const std::string& link : arm1_model.link_names()) {
        message << ' ' << link;
      }
      message << "; arm-2 links:";
      for (const std::string& link : arm2_model.link_names()) {
        message << ' ' << link;
      }
      throw std::runtime_error(message.str());
    }
    std::vector<double> torque_limits = arm1_info.tau_max;
    torque_limits.insert(torque_limits.end(), arm2_info.tau_max.begin(), arm2_info.tau_max.end());
    CooperativeBimanualController controller;
    const controllers::Result prepare_result = controller.prepare(config.frames, torque_limits);
    if (!prepare_result.ok() || !controller.set_initial_state(cycle)) {
      throw std::runtime_error(std::string("controller preparation failed: ") +
                               prepare_result.message);
    }

    std::array<double, kJointDim> torque{};
    std::vector<double> arm1_command(kArmJointDim, 0.0);
    std::vector<double> arm2_command(kArmJointDim, 0.0);
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
          run_state.elapsed_seconds += dt_seconds;
          try {
            if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
              run_state.request_stop(StopReason::kInvalidState);
              return;
            }
            if (arm1_robot.operational_status() != rdk::OperationalStatus::READY ||
                arm2_robot.operational_status() != rdk::OperationalStatus::READY ||
                arm1_robot.fault() || arm2_robot.fault()) {
              run_state.request_stop(StopReason::kRobotNotReady);
              return;
            }
            if (arm1_robot.reached_timeliness_failure_limit() ||
                arm2_robot.reached_timeliness_failure_limit()) {
              run_state.request_stop(StopReason::kTimelinessLimit);
              return;
            }
            if (!fill_cycle_data(arm1_robot.states(),
                                 arm2_robot.states(),
                                 arm1_model,
                                 arm2_model,
                                 config,
                                 &cycle)) {
              run_state.request_stop(StopReason::kInvalidState);
              return;
            }
            if (!controller.step(cycle, run_state.elapsed_seconds, dt_seconds, &torque)) {
              run_state.request_stop(StopReason::kControllerRejected);
              return;
            }
            constexpr std::size_t kDebugPublishPeriodSamples = 10;
            if (run_state.samples % kDebugPublishPeriodSamples == 0U) {
              controller.publish_debug(cycle,
                                       static_cast<std::uint64_t>(run_state.samples),
                                       run_state.elapsed_seconds,
                                       dt_seconds,
                                       &debug_snapshot);
            }
            for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
              arm1_command[joint] = torque[joint];
              arm2_command[joint] = torque[kArmJointDim + joint];
            }
            arm1_robot.StreamJointTorque(arm1_command,
                                         kEnableGravityCompensation,
                                         kEnableSoftLimits,
                                         kFrictionCompensationPercent);
            arm2_robot.StreamJointTorque(arm2_command,
                                         kEnableGravityCompensation,
                                         kEnableSoftLimits,
                                         kFrictionCompensationPercent);
          } catch (...) {
            run_state.request_stop(StopReason::kRdkException);
            return;
          }
          if (run_state.elapsed_seconds >= config.run_duration_seconds) {
            run_state.request_stop(StopReason::kDurationReached);
          }
        },
        "controllers-flexiv-dual",
        1,
        config.non_rt_requested ? 0 : scheduler.max_priority(),
        -1);

    arm1_robot.SetTimelinessFailureLimit(2.0);
    arm2_robot.SetTimelinessFailureLimit(2.0);
    arm1_robot.SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
    arm1_stop_guard.arm();
    arm2_robot.SwitchMode(rdk::Mode::RT_JOINT_TORQUE);
    arm2_stop_guard.arm();
    scheduler.Start();
    scheduler_guard.mark_started();
    std::uint64_t last_debug_sample = 0U;
    auto next_debug_print = std::chrono::steady_clock::now();
    while (!run_state.stop_requested.load()) {
      if (g_signal_requested != 0) {
        run_state.request_stop(StopReason::kSignal);
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= next_debug_print) {
        DebugValues values;
        if (read_debug_snapshot(debug_snapshot, &values) && values.sample != last_debug_sample) {
          print_debug_values(values);
          last_debug_sample = values.sample;
        }
        next_debug_print = now + std::chrono::milliseconds(250);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler_guard.stop();
    arm2_stop_guard.stop();
    arm1_stop_guard.stop();
    return run_state.stop_reason.load() == StopReason::kDurationReached ? EXIT_SUCCESS
                                                                        : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "Flexiv dual-arm controller failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-check") {
    return report_bimanual_kinematics_self_check() ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  ProgramConfig config;
  if (!parse_arguments(argc, argv, &config)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (!report_bimanual_kinematics_self_check()) {
    return EXIT_FAILURE;
  }
  if (!confirm_execution(config)) {
    std::cout << "Execution cancelled before connecting to the robot.\n";
    return EXIT_FAILURE;
  }
  return run(config);
}
