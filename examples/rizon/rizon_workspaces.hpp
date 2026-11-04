/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file rizon_workspaces.hpp
 * @brief Controller workspaces for the six Flexiv Rizon examples.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>

#include "../cartesian_math.hpp"
#include "./rizon_utils.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace rizon_example {

namespace controllers = nvidia::srl::controllers;

inline controllers::ControllerConfig cartesian_config(const controllers::CartesianFrame frame) {
  return {controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointCount, {}, frame};
}

template <typename Derived, std::size_t Size>
inline void copy_matrix(const Eigen::MatrixBase<Derived>& source,
                        std::array<double, Size>& destination) {
  for (std::size_t row = 0; row < static_cast<std::size_t>(source.rows()); ++row) {
    for (std::size_t column = 0; column < static_cast<std::size_t>(source.cols()); ++column) {
      destination[row * static_cast<std::size_t>(source.cols()) + column] =
          source(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
    }
  }
}

template <typename Derived, std::size_t Size>
inline void copy_vector(const Eigen::MatrixBase<Derived>& source,
                        std::array<double, Size>& destination) {
  for (std::size_t index = 0; index < Size; ++index) {
    destination[index] = source[static_cast<Eigen::Index>(index)];
  }
}

template <std::size_t Size>
inline Eigen::Matrix<double, static_cast<int>(Size), 1> map_vector(
    const std::array<double, Size>& source) {
  Eigen::Matrix<double, static_cast<int>(Size), 1> result;
  for (std::size_t index = 0; index < Size; ++index) {
    result[static_cast<Eigen::Index>(index)] = source[index];
  }
  return result;
}

inline bool safe_status(const std::uint64_t status) {
  return !controllers::has_status_flag(status, controllers::StatusFlag::kSafeMode);
}

class CartesianLowPassFilter {
 public:
  explicit CartesianLowPassFilter(const double cutoff_hertz) : cutoff_hertz_(cutoff_hertz) {}

  CartesianVector update(const CartesianVector& input, const double dt_seconds) {
    constexpr double kPi = 3.14159265358979323846;
    const double time_constant = 1.0 / (2.0 * kPi * cutoff_hertz_);
    const double alpha = dt_seconds / (time_constant + dt_seconds);
    if (!is_initialized_) {
      output_ = input;
      is_initialized_ = true;
    } else {
      output_ += alpha * (input - output_);
    }
    return output_;
  }

  void reset() {
    output_.setZero();
    is_initialized_ = false;
  }

 private:
  double cutoff_hertz_;
  CartesianVector output_{CartesianVector::Zero()};
  bool is_initialized_{false};
};

class CartesianMassWorkspace {
 public:
  controllers::Result prepare(const controllers::ControllerConfig& config) {
    const controllers::OperationalSpaceMass::Parameters parameters{
        controllers::shared(controllers::read_view(regularization_, controllers::Device::kCpu, 1)),
        {}};
    const controllers::OperationalSpaceMass::Storage storage{
        controllers::write_view(
            cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::write_status(parameter_status_, controllers::Device::kCpu),
        controllers::write_status(status_, controllers::Device::kCpu),
        {controllers::write_view(
             joint_factor_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
         controllers::write_view(
             solve_rhs_, controllers::Device::kCpu, 1, kJointCount, kCartesianDim),
         controllers::write_view(
             cartesian_inverse_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim)}};
    inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    controllers::Result result = controller_.prepare(config, parameters, storage);
    if (result.ok()) {
      result = controller_.reset();
    }
    return result;
  }

  bool step(const JointMatrix& mass, const Jacobian& jacobian) {
    copy_matrix(mass, joint_mass_);
    copy_matrix(jacobian, jacobian_);
    status_[0] = 0;
    const controllers::Result result = controller_.step(inputs_);
    return result.ok() && safe_status(status_[0]);
  }

  const std::array<double, kCartesianDim * kCartesianDim>& cartesian_mass() const {
    return cartesian_mass_;
  }

 private:
  std::array<double, 1> regularization_{1e-6};
  std::array<double, kJointCount * kJointCount> joint_mass_{};
  std::array<double, kCartesianDim * kJointCount> jacobian_{};
  std::array<double, kCartesianDim * kCartesianDim> cartesian_mass_{};
  std::array<double, kJointCount * kJointCount> joint_factor_{};
  std::array<double, kJointCount * kCartesianDim> solve_rhs_{};
  std::array<double, kCartesianDim * kCartesianDim> cartesian_inverse_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::OperationalSpaceMass::Inputs inputs_{};
  controllers::OperationalSpaceMass controller_;
};

class JointImpedanceExample {
 public:
  JointImpedanceExample() {
    constexpr std::array<double, kJointCount> kStiffness{1500, 1200, 800, 500, 300, 200, 50};
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      stiffness_[joint * kJointCount + joint] = kStiffness[joint];
      damping_factors_[joint] = 0.7;
    }
  }

  controllers::Result prepare() {
    const controllers::JointImpedanceController::Parameters parameters{
        controllers::per_batch(controllers::read_view(
            stiffness_, controllers::Device::kCpu, 1, kJointCount, kJointCount)),
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, kJointCount))};
    const controllers::JointImpedanceController::Storage storage{
        controllers::write_view(controller_torque_, controllers::Device::kCpu, 1, kJointCount),
        controllers::write_status(parameter_status_, controllers::Device::kCpu),
        controllers::write_status(status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(last_valid_torque_, controllers::Device::kCpu, 1, kJointCount),
            controllers::write_view(safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(hold_count_, controllers::Device::kCpu, 1))};
    inputs_ = {
        controllers::read_view(position_error_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount)};
    controllers::Result result = controller_.prepare(
        {controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointCount, {}},
        parameters,
        storage);
    if (result.ok()) {
      result = controller_.reset();
    }
    return result;
  }

  bool set_initial_state(const CycleData& initial) {
    desired_position_ = initial.joint_position;
    return desired_position_.allFinite();
  }

  bool step(const CycleData& data, double, double, JointVector& torque) {
    copy_vector(desired_position_ - data.joint_position, position_error_);
    copy_vector(-data.joint_velocity, velocity_error_);
    copy_matrix(data.mass, mass_);
    status_[0] = 0;
    const controllers::Result result = controller_.step(inputs_);
    torque = map_vector(controller_torque_);
    return result.ok() && safe_status(status_[0]) && torque.allFinite();
  }

  void print_observation() const {}

 private:
  JointVector desired_position_{};
  std::array<double, kJointCount * kJointCount> stiffness_{};
  std::array<double, kJointCount> damping_factors_{};
  std::array<double, kJointCount> position_error_{};
  std::array<double, kJointCount> velocity_error_{};
  std::array<double, kJointCount> desired_acceleration_{};
  std::array<double, kJointCount * kJointCount> mass_{};
  std::array<double, kJointCount> controller_torque_{};
  std::array<double, kJointCount> last_valid_torque_{};
  std::array<std::uint8_t, 1> safety_valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::JointImpedanceController::Inputs inputs_{};
  controllers::JointImpedanceController controller_;
};

class CartesianImpedanceExample {
 public:
  CartesianImpedanceExample() {
    constexpr std::array<double, kCartesianDim> kCartesianStiffness{
        2000, 2000, 2000, 200, 200, 200};
    constexpr std::array<double, kJointCount> kNullspaceStiffness{120, 80, 60, 40, 30, 20, 10};
    constexpr std::array<double, kJointCount> kNullspaceDamping{12, 8, 6, 4, 3, 2, 1};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = kCartesianStiffness[axis];
      damping_factors_[axis] = 0.7;
    }
    for (std::size_t joint = 0; joint < kJointCount; ++joint) {
      nullspace_stiffness_[joint * kJointCount + joint] = kNullspaceStiffness[joint];
      nullspace_damping_[joint * kJointCount + joint] = kNullspaceDamping[joint];
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config =
        cartesian_config(controllers::CartesianFrame::kRobotBase);
    controllers::Result result = mass_workspace_.prepare(config);
    if (!result.ok()) {
      return result;
    }
    const controllers::CartesianImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, kCartesianDim))};
    const controllers::CartesianImpedanceController::Storage storage{
        controllers::write_view(wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_status(parameter_status_, controllers::Device::kCpu),
        controllers::write_status(status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(hold_count_, controllers::Device::kCpu, 1)),
        {controllers::write_view(
             previous_cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
         controllers::write_view(
             previous_jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
         controllers::write_view(initialized_, controllers::Device::kCpu, 1)}};
    inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(mass_workspace_.cartesian_mass(),
                               controllers::Device::kCpu,
                               1,
                               kCartesianDim,
                               kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    const controllers::NullspaceController::Parameters nullspace_parameters{
        controllers::shared(controllers::read_view(
            nullspace_stiffness_, controllers::Device::kCpu, kJointCount, kJointCount)),
        controllers::shared(controllers::read_view(
            nullspace_damping_, controllers::Device::kCpu, kJointCount, kJointCount))};
    const controllers::NullspaceController::Storage nullspace_storage{
        controllers::write_view(nullspace_torque_, controllers::Device::kCpu, 1, kJointCount),
        controllers::write_status(nullspace_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(nullspace_status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_nullspace_torque_, controllers::Device::kCpu, 1, kJointCount),
            controllers::write_view(nullspace_safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(nullspace_hold_count_, controllers::Device::kCpu, 1))};
    nullspace_inputs_ = {
        controllers::read_view(joint_position_error_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    result = controller_.prepare(config, parameters, storage);
    if (result.ok()) {
      result = controller_.reset();
    }
    if (result.ok()) {
      result = nullspace_controller_.prepare(config, nullspace_parameters, nullspace_storage);
    }
    if (result.ok()) {
      result = nullspace_controller_.reset();
    }
    return result;
  }

  bool set_initial_state(const CycleData& initial) {
    desired_pose_ = initial.world_tcp_pose;
    desired_joint_position_ = initial.joint_position;
    return desired_pose_.allFinite() && desired_joint_position_.allFinite();
  }

  bool step(const CycleData& data, double, const double dt_seconds, JointVector& torque) {
    copy_vector(controllers::examples::cartesian_pose_error(desired_pose_, data.world_tcp_pose),
                pose_error_);
    copy_vector(-data.world_jacobian * data.joint_velocity, velocity_error_);
    copy_vector(desired_joint_position_ - data.joint_position, joint_position_error_);
    copy_vector(data.joint_velocity, joint_velocity_);
    copy_matrix(data.mass, joint_mass_);
    copy_matrix(data.world_jacobian, jacobian_);
    dt_[0] = dt_seconds;
    status_[0] = 0;
    nullspace_status_[0] = 0;
    if (!mass_workspace_.step(data.mass, data.world_jacobian)) {
      return false;
    }
    const controllers::Result result = controller_.step(inputs_);
    const controllers::Result nullspace_result = nullspace_controller_.step(nullspace_inputs_);
    const CartesianVector wrench = map_vector(wrench_);
    torque = data.world_jacobian.transpose() * wrench + map_vector(nullspace_torque_);
    return result.ok() && nullspace_result.ok() && safe_status(status_[0]) &&
           safe_status(nullspace_status_[0]) && wrench.allFinite() && torque.allFinite();
  }

  void print_observation() const {}

 private:
  Eigen::Matrix4d desired_pose_{Eigen::Matrix4d::Identity()};
  JointVector desired_joint_position_{};
  CartesianMassWorkspace mass_workspace_;
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kJointCount * kJointCount> nullspace_stiffness_{};
  std::array<double, kJointCount * kJointCount> nullspace_damping_{};
  std::array<double, kCartesianDim> pose_error_{};
  std::array<double, kCartesianDim> velocity_error_{};
  std::array<double, kCartesianDim> desired_acceleration_{};
  std::array<double, kJointCount> joint_position_error_{};
  std::array<double, kJointCount> joint_velocity_{};
  std::array<double, kJointCount * kJointCount> joint_mass_{};
  std::array<double, kCartesianDim * kJointCount> jacobian_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> wrench_{};
  std::array<double, kJointCount> nullspace_torque_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_cartesian_mass_{};
  std::array<double, kCartesianDim * kJointCount> previous_jacobian_{};
  std::array<std::uint8_t, 1> initialized_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> safety_valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<double, kJointCount> last_valid_nullspace_torque_{};
  std::array<std::uint8_t, 1> nullspace_safety_valid_{};
  std::array<std::uint64_t, 1> nullspace_hold_count_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  std::array<std::uint64_t, 1> nullspace_parameter_status_{};
  std::array<std::uint64_t, 1> nullspace_status_{};
  controllers::CartesianImpedanceController::Inputs inputs_{};
  controllers::NullspaceController::Inputs nullspace_inputs_{};
  controllers::CartesianImpedanceController controller_;
  controllers::NullspaceController nullspace_controller_;
};

class VariableCartesianImpedanceExample {
 public:
  VariableCartesianImpedanceExample() {
    constexpr std::array<double, kCartesianDim> kInitial{2000, 2000, 2000, 200, 200, 200};
    maximum_stiffness_ = {2000, 2000, 2000, 200, 200, 200};
    maximum_stiffness_rate_ = maximum_stiffness_;
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      initial_stiffness_[axis * kCartesianDim + axis] = kInitial[axis];
      requested_stiffness_[axis * kCartesianDim + axis] = kInitial[axis];
      damping_factors_[axis] = 0.7;
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config =
        cartesian_config(controllers::CartesianFrame::kRobotBase);
    controllers::Result result = mass_workspace_.prepare(config);
    if (!result.ok()) {
      return result;
    }
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
        {controllers::write_view(wrench_, controllers::Device::kCpu, 1, kCartesianDim),
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
    inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(
            requested_stiffness_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(mass_workspace_.cartesian_mass(),
                               controllers::Device::kCpu,
                               1,
                               kCartesianDim,
                               kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    result = controller_.prepare(config, parameters, storage);
    if (result.ok()) {
      result = controller_.reset();
    }
    return result;
  }

  bool set_initial_state(const CycleData& initial) {
    desired_pose_ = initial.world_tcp_pose;
    return desired_pose_.allFinite();
  }

  bool step(const CycleData& data,
            const double elapsed_seconds,
            const double dt_seconds,
            JointVector& torque) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kMinimumStiffness = 1.0;
    constexpr double kMaximumStiffness = 2000.0;
    constexpr double kFrequencyHertz = 0.1;
    constexpr double kStiffnessOffset = 0.5 * (kMaximumStiffness + kMinimumStiffness);
    constexpr double kStiffnessAmplitude = 0.5 * (kMaximumStiffness - kMinimumStiffness);
    requested_stiffness_[2 * kCartesianDim + 2] =
        kStiffnessOffset +
        kStiffnessAmplitude * std::sin(2.0 * kPi * kFrequencyHertz * elapsed_seconds);
    copy_vector(controllers::examples::cartesian_pose_error(desired_pose_, data.world_tcp_pose),
                pose_error_);
    copy_vector(-data.world_jacobian * data.joint_velocity, velocity_error_);
    copy_vector(data.joint_velocity, joint_velocity_);
    copy_matrix(data.world_jacobian, jacobian_);
    dt_[0] = dt_seconds;
    status_[0] = 0;
    if (!mass_workspace_.step(data.mass, data.world_jacobian)) {
      return false;
    }
    const controllers::Result result = controller_.step(inputs_);
    const CartesianVector wrench = map_vector(wrench_);
    torque = data.world_jacobian.transpose() * wrench;
    return result.ok() && safe_status(status_[0]) && wrench.allFinite() && torque.allFinite();
  }

  void print_observation() const {}

 private:
  Eigen::Matrix4d desired_pose_{Eigen::Matrix4d::Identity()};
  CartesianMassWorkspace mass_workspace_;
  std::array<double, kCartesianDim * kCartesianDim> initial_stiffness_{};
  std::array<double, kCartesianDim> maximum_stiffness_{};
  std::array<double, kCartesianDim> minimum_stiffness_{};
  std::array<double, kCartesianDim> maximum_stiffness_rate_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kCartesianDim * kCartesianDim> requested_stiffness_{};
  std::array<double, kCartesianDim> pose_error_{};
  std::array<double, kCartesianDim> velocity_error_{};
  std::array<double, kCartesianDim> desired_acceleration_{};
  std::array<double, kJointCount> joint_velocity_{};
  std::array<double, kCartesianDim * kJointCount> jacobian_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> wrench_{};
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
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::VariableCartesianImpedanceController::Inputs inputs_{};
  controllers::VariableCartesianImpedanceController controller_;
};

class CartesianAdmittanceExample {
 public:
  CartesianAdmittanceExample() : external_wrench_filter_(kExternalWrenchCutoffHertz) {
    constexpr std::array<double, kCartesianDim> kAdmittanceStiffness{
        1000, 1000, 1000, 100, 100, 100};
    constexpr std::array<double, kCartesianDim> kAdmittanceDamping{100, 100, 100, 20, 20, 20};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      admittance_stiffness_[axis * kCartesianDim + axis] = kAdmittanceStiffness[axis];
      admittance_damping_[axis * kCartesianDim + axis] = kAdmittanceDamping[axis];
      virtual_mass_[axis * kCartesianDim + axis] = 1.0;
      force_filter_threshold_[axis] = 0.5;
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config =
        cartesian_config(controllers::CartesianFrame::kRobotBase);
    const controllers::CartesianAdmittancePoseController::Parameters admittance_parameters{
        controllers::shared(controllers::read_view(
            admittance_stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            admittance_damping_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            virtual_mass_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            force_filter_threshold_, controllers::Device::kCpu, kCartesianDim)),
        {},
        {},
        {}};
    const controllers::CartesianAdmittancePoseController::Storage admittance_storage{
        {controllers::write_view(acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_status(admittance_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(admittance_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(
                 last_valid_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
             controllers::write_view(admittance_safety_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(admittance_hold_count_, controllers::Device::kCpu, 1))},
        {controllers::write_view(pose_offset_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(twist_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(
             pose_previous_signal_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(
             twist_previous_signal_, controllers::Device::kCpu, 1, kCartesianDim),
         controllers::write_view(pose_integrator_initialized_, controllers::Device::kCpu, 1),
         controllers::write_view(twist_integrator_initialized_, controllers::Device::kCpu, 1)}};
    admittance_inputs_ = {
        controllers::read_view(external_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(feedforward_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    controllers::Result result =
        admittance_controller_.prepare(config, admittance_parameters, admittance_storage);
    if (result.ok()) {
      result = admittance_controller_.reset();
    }
    return result;
  }

  bool set_initial_state(const CycleData& initial) {
    initial_pose_ = initial.world_tcp_pose;
    external_wrench_filter_.reset();
    return initial_pose_.allFinite();
  }

  bool step(const CycleData& data,
            double,
            const double dt_seconds,
            CartesianMotionCommand& command) {
    const CartesianVector filtered_external_wrench =
        external_wrench_filter_.update(data.external_wrench_world, dt_seconds);
    copy_vector(filtered_external_wrench, external_wrench_);
    dt_[0] = dt_seconds;
    admittance_status_[0] = 0;
    const controllers::Result admittance_result = admittance_controller_.step(admittance_inputs_);
    if (!admittance_result.ok() || !safe_status(admittance_status_[0])) {
      return false;
    }
    const CartesianVector acceleration = map_vector(acceleration_);
    const CartesianVector desired_twist = map_vector(twist_);
    const CartesianVector pose_offset = map_vector(pose_offset_);
    const Eigen::Matrix4d desired_pose =
        controllers::examples::apply_cartesian_pose_offset(initial_pose_, pose_offset);
    if (!acceleration.allFinite() || !desired_twist.allFinite() || !pose_offset.allFinite() ||
        !desired_pose.allFinite() || !pose_to_rdk(desired_pose, command.pose)) {
      return false;
    }
    return true;
  }

  void print_observation() const {}

 private:
  static constexpr double kExternalWrenchCutoffHertz = 7.0;
  Eigen::Matrix4d initial_pose_{Eigen::Matrix4d::Identity()};
  CartesianLowPassFilter external_wrench_filter_;
  std::array<double, kCartesianDim * kCartesianDim> admittance_stiffness_{};
  std::array<double, kCartesianDim * kCartesianDim> admittance_damping_{};
  std::array<double, kCartesianDim * kCartesianDim> virtual_mass_{};
  std::array<double, kCartesianDim> force_filter_threshold_{};
  std::array<double, kCartesianDim> external_wrench_{};
  std::array<double, kCartesianDim> feedforward_wrench_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> acceleration_{};
  std::array<double, kCartesianDim> pose_offset_{};
  std::array<double, kCartesianDim> twist_{};
  std::array<double, kCartesianDim> pose_previous_signal_{};
  std::array<double, kCartesianDim> twist_previous_signal_{};
  std::array<std::uint8_t, 1> pose_integrator_initialized_{};
  std::array<std::uint8_t, 1> twist_integrator_initialized_{};
  std::array<double, kCartesianDim> last_valid_acceleration_{};
  std::array<std::uint8_t, 1> admittance_safety_valid_{};
  std::array<std::uint64_t, 1> admittance_hold_count_{};
  std::array<std::uint64_t, 1> admittance_parameter_status_{};
  std::array<std::uint64_t, 1> admittance_status_{};
  controllers::CartesianAdmittancePoseController::Inputs admittance_inputs_{};
  controllers::CartesianAdmittancePoseController admittance_controller_;
};

class ForceImpedanceExample {
 public:
  ForceImpedanceExample(const controllers::CartesianFrame frame, const double target_force_newtons)
      : frame_(frame), target_force_newtons_(target_force_newtons) {
    constexpr std::array<double, kCartesianDim> kStiffness{2000, 2000, 50, 200, 200, 200};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = kStiffness[axis];
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
    config.cartesian_frame = frame_;
    config.use_shaping_function = true;
    controllers::Result result = mass_workspace_.prepare(config);
    if (!result.ok()) {
      return result;
    }
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
        {controllers::write_view(wrench_, controllers::Device::kCpu, 1, kCartesianDim),
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
    inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(mass_workspace_.cartesian_mass(),
                               controllers::Device::kCpu,
                               1,
                               kCartesianDim,
                               kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(external_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(feedforward_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    result = controller_.prepare(config, parameters, storage);
    if (result.ok()) {
      result = controller_.reset();
    }
    return result;
  }

  bool set_initial_state(const CycleData& initial) {
    desired_pose_ = initial.world_tcp_pose;
    return desired_pose_.allFinite();
  }

  bool step(const CycleData& data,
            const double elapsed_seconds,
            const double dt_seconds,
            JointVector& torque) {
    constexpr double kForceRampSeconds = 2.0;
    const bool tcp_frame = frame_ == controllers::CartesianFrame::kEndEffector;
    const Jacobian& jacobian = tcp_frame ? data.tcp_jacobian : data.world_jacobian;
    const CartesianVector& external_wrench =
        tcp_frame ? data.external_wrench_tcp : data.external_wrench_world;
    CartesianVector pose_error;
    if (tcp_frame) {
      const Eigen::Matrix4d tcp_to_desired = data.world_tcp_pose.inverse() * desired_pose_;
      pose_error =
          controllers::examples::cartesian_pose_error(tcp_to_desired, Eigen::Matrix4d::Identity());
    } else {
      pose_error = controllers::examples::cartesian_pose_error(desired_pose_, data.world_tcp_pose);
    }
    if (!pose_error.allFinite()) {
      return false;
    }
    CartesianVector desired_wrench = CartesianVector::Zero();
    desired_wrench(2) = std::min(1.0, elapsed_seconds / kForceRampSeconds) * target_force_newtons_;
    copy_vector(pose_error, pose_error_);
    copy_vector(-jacobian * data.joint_velocity, velocity_error_);
    copy_vector(data.joint_velocity, joint_velocity_);
    copy_vector(external_wrench, external_wrench_);
    copy_vector(desired_wrench, desired_wrench_);
    copy_matrix(jacobian, jacobian_);
    dt_[0] = dt_seconds;
    status_[0] = 0;
    if (!mass_workspace_.step(data.mass, jacobian)) {
      return false;
    }
    const controllers::Result result = controller_.step(inputs_);
    const CartesianVector wrench = map_vector(wrench_);
    torque = jacobian.transpose() * wrench;
    return result.ok() && safe_status(status_[0]) && wrench.allFinite() && torque.allFinite();
  }

  void print_observation() const {}

 private:
  controllers::CartesianFrame frame_;
  double target_force_newtons_;
  Eigen::Matrix4d desired_pose_{Eigen::Matrix4d::Identity()};
  CartesianMassWorkspace mass_workspace_;
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kCartesianDim> proportional_gain_{};
  std::array<double, kCartesianDim> differentiator_gain_{};
  std::array<double, kCartesianDim> integral_gain_{};
  std::array<double, kCartesianDim> initial_output_{};
  std::array<double, kCartesianDim> maximum_distance_{};
  std::array<double, kCartesianDim> shaping_distance_{};
  std::array<double, kCartesianDim> pose_error_{};
  std::array<double, kCartesianDim> velocity_error_{};
  std::array<double, kCartesianDim> desired_acceleration_{};
  std::array<double, kJointCount> joint_velocity_{};
  std::array<double, kCartesianDim * kJointCount> jacobian_{};
  std::array<double, kCartesianDim> external_wrench_{};
  std::array<double, kCartesianDim> desired_wrench_{};
  std::array<double, kCartesianDim> feedforward_wrench_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> wrench_{};
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
  std::array<std::uint64_t, 1> parameter_status_{};
  std::array<std::uint64_t, 1> status_{};
  controllers::CartesianUnifiedForceImpedanceController::Inputs inputs_{};
  controllers::CartesianUnifiedForceImpedanceController controller_;
};

}  // namespace rizon_example
