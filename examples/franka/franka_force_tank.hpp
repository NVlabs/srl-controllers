// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#pragma once
#include <eigen3/Eigen/Dense>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "nvidia/srl/controllers/controllers.hpp"

namespace franka_example {
namespace force_tank {
namespace controllers = nvidia::srl::controllers;
// Fixed-pose, base-frame force/impedance composition. Prepare before connection;
// steps use caller-owned storage and measured positive periods in seconds.
constexpr double kTargetForceNewtons = -10.0;
constexpr double kForceRampSeconds = 2.0;
constexpr double kDampingRechargeFraction = 0.5;
constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;

class ForceTankWorkspace {
 public:
  ForceTankWorkspace() {
    const std::array<double, kCartesianDim> stiffness{2000, 2000, 50, 100, 100, 100};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = stiffness[axis];
      damping_factors_[axis] = 0.7;
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
    config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
    config.use_shaping_function = false;

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

    const controllers::CartesianImpedanceController::Parameters impedance_parameters{
        controllers::shared(controllers::read_view(stiffness_, controllers::Device::kCpu, 6, 6)),
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, 6))};
    const controllers::CartesianForceController::Parameters force_parameters{
        controllers::shared(
            controllers::read_view(proportional_gain_, controllers::Device::kCpu, 6)),
        controllers::shared(
            controllers::read_view(differentiator_gain_, controllers::Device::kCpu, 6)),
        controllers::shared(controllers::read_view(integral_gain_, controllers::Device::kCpu, 6)),
        controllers::shared(controllers::read_view(initial_output_, controllers::Device::kCpu, 6)),
        {},
        {}};
    const controllers::CartesianImpedanceController::Storage impedance_storage{
        {controllers::write_view(impedance_wrench_, controllers::Device::kCpu, 1, 6),
         controllers::write_status(parameter_status_, controllers::Device::kCpu),
         controllers::write_status(impedance_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(last_valid_wrench_, controllers::Device::kCpu, 1, 6),
             controllers::write_view(safety_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(hold_count_, controllers::Device::kCpu, 1))},
        {controllers::write_view(previous_cartesian_mass_, controllers::Device::kCpu, 1, 6, 6),
         controllers::write_view(previous_jacobian_, controllers::Device::kCpu, 1, 6, 7),
         controllers::write_view(inertial_initialized_, controllers::Device::kCpu, 1)}};
    const controllers::CartesianDampingDoubleDiagonalization::Parameters damping_parameters{
        controllers::shared(
            controllers::read_view(damping_factors_, controllers::Device::kCpu, 6))};
    const controllers::CartesianDampingDoubleDiagonalization::Storage damping_storage{
        {controllers::write_view(damping_wrench_, controllers::Device::kCpu, 1, 6),
         controllers::write_status(damping_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(damping_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(damping_last_valid_, controllers::Device::kCpu, 1, 6),
             controllers::write_view(damping_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(damping_hold_count_, controllers::Device::kCpu, 1))},
        {},
        {}};
    const controllers::CartesianForceController::Storage force_storage{
        {controllers::write_view(force_wrench_, controllers::Device::kCpu, 1, 6),
         controllers::write_status(force_parameter_status_, controllers::Device::kCpu),
         controllers::write_status(force_status_, controllers::Device::kCpu),
         controllers::make_safety_state(
             controllers::write_view(force_last_valid_, controllers::Device::kCpu, 1, 6),
             controllers::write_view(force_valid_, controllers::Device::kCpu, 1),
             controllers::write_view(force_hold_count_, controllers::Device::kCpu, 1))},
        {controllers::write_view(differentiator_previous_error_, controllers::Device::kCpu, 1, 6),
         controllers::write_view(differentiator_initialized_, controllers::Device::kCpu, 1),
         controllers::write_view(integrator_previous_error_, controllers::Device::kCpu, 1, 6),
         controllers::write_view(integrated_error_, controllers::Device::kCpu, 1, 6),
         controllers::write_view(integrator_initialized_, controllers::Device::kCpu, 1)}};
    const controllers::SignalBlockConfig tank_config{
        controllers::Device::kCpu, controllers::DType::kFloat64, 1, 1};
    const controllers::EnergyTank::Parameters tank_parameters{
        controllers::shared(controllers::read_view(upper_energy_, controllers::Device::kCpu)),
        controllers::shared(controllers::read_view(lower_energy_, controllers::Device::kCpu)),
        controllers::shared(controllers::read_view(initial_energy_, controllers::Device::kCpu)),
        controllers::shared(controllers::read_view(empty_width_, controllers::Device::kCpu)),
        controllers::shared(controllers::read_view(power_noise_, controllers::Device::kCpu))};
    const controllers::EnergyTank::Storage tank_storage{
        controllers::write_view(scale_, controllers::Device::kCpu, 1),
        controllers::write_view(filtered_power_, controllers::Device::kCpu, 1),
        controllers::write_status(tank_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(tank_status_, controllers::Device::kCpu),
        {controllers::write_view(tank_state_, controllers::Device::kCpu, 1)},
        {}};

    mass_inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    impedance_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, 6),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, 6),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, 6),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, 7),
        controllers::read_view(cartesian_mass_, controllers::Device::kCpu, 1, 6, 6),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, 6, 7),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    force_inputs_ = {controllers::read_view(external_wrench_, controllers::Device::kCpu, 1, 6),
                     controllers::read_view(desired_wrench_, controllers::Device::kCpu, 1, 6),
                     controllers::read_view(feedforward_wrench_, controllers::Device::kCpu, 1, 6),
                     controllers::read_view(pose_error_, controllers::Device::kCpu, 1, 6),
                     controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    damping_inputs_ = {controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, 6),
                      controllers::read_view(cartesian_mass_, controllers::Device::kCpu, 1, 6, 6),
                      controllers::read_view(stiffness_, controllers::Device::kCpu, 1, 6, 6)};
    tank_inputs_ = {controllers::read_view(power_, controllers::Device::kCpu, 1),
                    controllers::read_view(additional_power_, controllers::Device::kCpu, 1),
                    controllers::read_view(dt_, controllers::Device::kCpu, 1)};

    controllers::Result result = mass_controller_.prepare(config, mass_parameters, mass_storage);
    if (!result.ok()) {
      return result;
    }
    result = impedance_.prepare(config, impedance_parameters, impedance_storage);
    if (!result.ok()) return result;
    result = damping_.prepare(config, damping_parameters, damping_storage);
    if (!result.ok()) return result;
    result = force_.prepare(config, force_parameters, force_storage);
    if (!result.ok()) return result;
    result = tank_.prepare(tank_config, tank_parameters, tank_storage);
    if (!result.ok()) {
      return result;
    }
    result = mass_controller_.reset();
    if (!result.ok()) {
      return result;
    }
    result = impedance_.reset();
    if (!result.ok()) return result;
    result = damping_.reset();
    if (!result.ok()) return result;
    result = force_.reset();
    if (!result.ok()) return result;
    return tank_.reset();
  }

  bool step(const Eigen::Vector<double, kCartesianDim>& pose_error,
            const Eigen::Vector<double, kJointCount>& joint_velocity,
            const Eigen::Matrix<double, kJointCount, kJointCount>& mass,
            const Eigen::Matrix<double, kCartesianDim, kJointCount>& jacobian,
            const Eigen::Vector<double, kCartesianDim>& external_wrench,
            const Eigen::Vector<double, kCartesianDim>& desired_wrench,
            const double dt_seconds) {
    const Eigen::Vector<double, kCartesianDim> twist = jacobian * joint_velocity;
    for (std::size_t row = 0; row < kJointCount; ++row) {
      joint_velocity_[row] = joint_velocity[static_cast<Eigen::Index>(row)];
      for (std::size_t column = 0; column < kJointCount; ++column) {
        joint_mass_[row * kJointCount + column] =
            mass(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      pose_error_[row] = pose_error[static_cast<Eigen::Index>(row)];
      velocity_error_[row] = -twist[static_cast<Eigen::Index>(row)];
      external_wrench_[row] = external_wrench[static_cast<Eigen::Index>(row)];
      desired_wrench_[row] = desired_wrench[static_cast<Eigen::Index>(row)];
      for (std::size_t column = 0; column < kJointCount; ++column) {
        jacobian_[row * kJointCount + column] =
            jacobian(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
    dt_[0] = dt_seconds;
    commanded_wrench_.fill(0.0);
    if (!valid_step(mass_controller_.step(mass_inputs_), mass_status_[0]) ||
        !valid_step(impedance_.step(impedance_inputs_), impedance_status_[0]) ||
        !valid_step(damping_.step(damping_inputs_), damping_status_[0]) ||
        !valid_step(force_.step(force_inputs_), force_status_[0]))
      return false;
    // Damping wrench is -D*v. Harvest only half its dissipated power, leaving
    // the remaining half dissipative; do not count spring or inertial work.
    const double dissipated_power =
        -twist.dot(Eigen::Map<const Eigen::Vector<double, 6>>(damping_wrench_.data()));
    if (!std::isfinite(dissipated_power) || dissipated_power < 0.0) return false;
    additional_power_[0] = kDampingRechargeFraction * dissipated_power;
    power_[0] = -twist.dot(Eigen::Map<const Eigen::Vector<double, 6>>(force_wrench_.data()));
    if (!valid_step(tank_.step(tank_inputs_), tank_status_[0])) return false;
    for (std::size_t axis = 0; axis < 6; ++axis) {
      commanded_wrench_[axis] = impedance_wrench_[axis] + scale_[0] * force_wrench_[axis];
      if (!std::isfinite(commanded_wrench_[axis])) {
        commanded_wrench_.fill(0.0);
        return false;
      }
    }
    return true;
  }

  const std::array<double, kCartesianDim>& wrench() const { return commanded_wrench_; }

  const std::array<double, kCartesianDim>& impedance_wrench() const { return impedance_wrench_; }
  const std::array<double, kCartesianDim>& raw_force_wrench() const { return force_wrench_; }

  double energy_joules() const { return 0.5 * tank_state_[0] * tank_state_[0]; }
  double scale() const { return scale_[0]; }
  double power_watts() const { return power_[0]; }
  double damping_recharge_watts() const { return additional_power_[0]; }

 private:
  static bool valid_step(const controllers::Result result, const std::uint64_t& status) {
    return result.ok() && !controllers::has_status_flag(status, controllers::StatusFlag::kSafeMode);
  }
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim> damping_factors_{};
  std::array<double, kCartesianDim> proportional_gain_{};
  std::array<double, kCartesianDim> differentiator_gain_{};
  std::array<double, kCartesianDim> integral_gain_{};
  std::array<double, kCartesianDim> initial_output_{};
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
  std::array<double, kCartesianDim> damping_wrench_{}, damping_last_valid_{};
  std::array<std::uint8_t, 1> damping_valid_{};
  std::array<std::uint64_t, 1> damping_parameter_status_{}, damping_status_{};
  std::array<std::uint64_t, 1> damping_hold_count_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  std::array<std::uint64_t, 1> force_status_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> safety_valid_{};
  std::array<std::uint64_t, 1> hold_count_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> parameter_status_{};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  std::array<double, 6> force_last_valid_{};
  std::array<std::uint8_t, 1> force_valid_{};
  std::array<std::uint64_t, 1> force_hold_count_{}, force_parameter_status_{};
  std::array<double, 1> upper_energy_{1.0}, lower_energy_{0.01}, initial_energy_{1.0};
  std::array<double, 1> empty_width_{0.1}, power_noise_{}, additional_power_{};
  std::array<double, 1> power_{}, scale_{}, filtered_power_{}, tank_state_{};
  std::array<std::uint64_t, 1> tank_parameter_status_{}, tank_status_{};
  controllers::CartesianImpedanceController::Inputs impedance_inputs_{};
  controllers::CartesianForceController::Inputs force_inputs_{};
  controllers::CartesianDampingDoubleDiagonalization::Inputs damping_inputs_{};
  controllers::EnergyTank::Inputs tank_inputs_{};
  controllers::OperationalSpaceMass mass_controller_;
  controllers::CartesianImpedanceController impedance_;
  controllers::CartesianDampingDoubleDiagonalization damping_;
  controllers::CartesianForceController force_;
  controllers::EnergyTank tank_;
};

}  // namespace force_tank
}  // namespace franka_example
