/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file franka_admittance.hpp
 * @brief Filtered admittance state for the Franka Cartesian velocity example.
 */

#pragma once

#include <eigen3/Eigen/Core>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>

#include "nvidia/srl/controllers/controllers.hpp"

namespace franka_example {

namespace controllers = nvidia::srl::controllers;
constexpr std::size_t kJointCount = 7;
constexpr std::size_t kCartesianDim = 6;

template <std::size_t Size>
void print_trace_array(std::ostream& output, const std::array<double, Size>& values) {
  output << '[';
  for (std::size_t index = 0; index < Size; ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

struct AdmittanceStepTrace {
  std::array<double, kCartesianDim> external_wrench{};
  std::array<double, kCartesianDim> feedforward_wrench{};
  double dt_seconds{};
  std::array<double, kCartesianDim> input_pose_offset{};
  std::array<double, kCartesianDim> input_twist{};
  std::array<double, kCartesianDim> input_pose_previous_signal{};
  std::array<double, kCartesianDim> input_twist_previous_signal{};
  std::uint8_t input_pose_initialized{};
  std::uint8_t input_twist_initialized{};
  std::array<double, kCartesianDim> input_last_valid_acceleration{};
  std::uint8_t input_safety_valid{};
  std::uint64_t input_hold_count{};
  std::array<double, kCartesianDim> acceleration{};
  std::array<double, kCartesianDim> twist{};
  std::array<double, kCartesianDim> pose_offset{};
  std::array<double, kCartesianDim> pose_previous_signal{};
  std::array<double, kCartesianDim> twist_previous_signal{};
  std::uint8_t pose_initialized{};
  std::uint8_t twist_initialized{};
  std::uint64_t parameter_status{};
  std::uint64_t status{};
  std::array<double, kCartesianDim> last_valid_acceleration{};
  std::uint8_t safety_valid{};
  std::uint64_t hold_count{};
};

inline void print_admittance_step(std::ostream& output, const AdmittanceStepTrace& trace) {
  output << " controller_input_external_wrench_O_N_Nm=";
  print_trace_array(output, trace.external_wrench);
  output << " controller_input_feedforward_wrench_O_N_Nm=";
  print_trace_array(output, trace.feedforward_wrench);
  output << " controller_input_dt_s=" << trace.dt_seconds << " controller_input_pose_offset=";
  print_trace_array(output, trace.input_pose_offset);
  output << " controller_input_twist=";
  print_trace_array(output, trace.input_twist);
  output << " controller_input_pose_previous_signal=";
  print_trace_array(output, trace.input_pose_previous_signal);
  output << " controller_input_twist_previous_signal=";
  print_trace_array(output, trace.input_twist_previous_signal);
  output << " controller_input_pose_initialized="
         << static_cast<int>(trace.input_pose_initialized);
  output << " controller_input_twist_initialized="
         << static_cast<int>(trace.input_twist_initialized);
  output << " controller_input_safety_last_valid_acceleration=";
  print_trace_array(output, trace.input_last_valid_acceleration);
  output << " controller_input_safety_valid=" << static_cast<int>(trace.input_safety_valid)
         << " controller_input_safety_hold_count=" << trace.input_hold_count
         << " controller_output_acceleration=";
  print_trace_array(output, trace.acceleration);
  output << " controller_output_twist_O_m_s_rad_s=";
  print_trace_array(output, trace.twist);
  output << " controller_output_pose_offset_O_m_rad=";
  print_trace_array(output, trace.pose_offset);
  output << " controller_output_pose_previous_signal=";
  print_trace_array(output, trace.pose_previous_signal);
  output << " controller_output_twist_previous_signal=";
  print_trace_array(output, trace.twist_previous_signal);
  output << " controller_output_pose_initialized=" << static_cast<int>(trace.pose_initialized)
         << " controller_output_twist_initialized=" << static_cast<int>(trace.twist_initialized)
         << " controller_output_parameter_status=" << trace.parameter_status
         << " controller_output_status=" << trace.status
         << " controller_safety_last_valid_acceleration=";
  print_trace_array(output, trace.last_valid_acceleration);
  output << " controller_safety_valid=" << static_cast<int>(trace.safety_valid)
         << " controller_safety_hold_count=" << trace.hold_count;
}

class WrenchLowPassFilter {
 public:
  explicit WrenchLowPassFilter(const double cutoff_hertz) : cutoff_hertz_(cutoff_hertz) {}

  Eigen::Vector<double, 6> update(const Eigen::Vector<double, 6>& input, const double dt_seconds) {
    constexpr double kTwoPi = 6.28318530717958647692;
    const double time_constant = 1.0 / (kTwoPi * cutoff_hertz_);
    const double alpha = dt_seconds / (time_constant + dt_seconds);
    output_ += alpha * (input - output_);
    return output_;
  }

  void reset() { output_.setZero(); }

 private:
  double cutoff_hertz_;
  Eigen::Vector<double, 6> output_{Eigen::Vector<double, 6>::Zero()};
};

class AdmittanceWorkspace {
 public:
  AdmittanceWorkspace() {
    const std::array<double, kCartesianDim> admittance_stiffness{500, 500, 500, 50, 50, 50};
#if defined(NVIDIA_SRL_FRANKA_FER)
    const std::array<double, kCartesianDim> virtual_mass{29.25, 29.25, 29.25, 3.9, 3.9, 3.9};
#else
    const std::array<double, kCartesianDim> virtual_mass{1.95, 1.95, 1.95, 1.95, 1.95, 1.95};
#endif
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      admittance_stiffness_[axis * kCartesianDim + axis] = admittance_stiffness[axis];
      virtual_mass_[axis * kCartesianDim + axis] = virtual_mass[axis];
      // Twice critical damping for each independent virtual mass-spring axis.
      admittance_damping_[axis * kCartesianDim + axis] =
          4.0 * std::sqrt(virtual_mass_[axis * kCartesianDim + axis] * admittance_stiffness[axis]);
      force_filter_threshold_[axis] = 0.0;
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               kJointCount,
                                               {},
                                               controllers::CartesianFrame::kRobotBase};

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
        {controllers::write_view(
             commanded_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
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
    if (!result.ok()) {
      return result;
    }
    wrench_filter_.reset();
    last_step_invoked_ = false;
    return admittance_controller_.reset();
  }

  bool step_admittance(const Eigen::Vector<double, kCartesianDim>& external_wrench,
                       const double dt_seconds) {
    last_step_invoked_ = false;
    if (!external_wrench.allFinite() || !std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
      return false;
    }
    const auto filtered_wrench = wrench_filter_.update(external_wrench, dt_seconds);
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      external_wrench_[axis] = -filtered_wrench[static_cast<Eigen::Index>(axis)];
    }
    dt_[0] = dt_seconds;
    admittance_status_[0] = 0;
    trace_input_pose_offset_ = pose_offset_;
    trace_input_twist_ = twist_;
    trace_input_pose_previous_signal_ = pose_previous_signal_;
    trace_input_twist_previous_signal_ = twist_previous_signal_;
    trace_input_pose_initialized_ = pose_integrator_initialized_[0];
    trace_input_twist_initialized_ = twist_integrator_initialized_[0];
    trace_input_last_valid_acceleration_ = last_valid_acceleration_;
    trace_input_safety_valid_ = admittance_safety_valid_[0];
    trace_input_hold_count_ = admittance_hold_count_[0];
    last_step_invoked_ = true;
    const controllers::Result result = admittance_controller_.step(admittance_inputs_);
    return result.ok() &&
           !controllers::has_status_flag(admittance_status_[0], controllers::StatusFlag::kSafeMode);
  }

  const std::array<double, kCartesianDim>& acceleration() const { return commanded_acceleration_; }
  const std::array<double, kCartesianDim>& twist() const { return twist_; }
  const std::array<double, kCartesianDim>& pose_offset() const { return pose_offset_; }
  bool last_step_invoked() const { return last_step_invoked_; }

  void print_configuration(std::ostream& output) const {
    output << "admittance_configuration device=cpu dtype=float64 batch_size=1 joint_dim=7"
           << " frame=robot_base"
           << " wrench_filter_cutoff_hz=7 stiffness_matrix_row_major=";
    print_trace_array(output, admittance_stiffness_);
    output << " damping_matrix_row_major=";
    print_trace_array(output, admittance_damping_);
    output << " virtual_mass_matrix_row_major=";
    print_trace_array(output, virtual_mass_);
    output << " force_filter_threshold=";
    print_trace_array(output, force_filter_threshold_);
    output << " acceleration_limit=unset twist_limit=unset pose_limit=unset";
  }

  AdmittanceStepTrace last_step_trace() const noexcept {
    AdmittanceStepTrace trace;
    trace.external_wrench = external_wrench_;
    trace.feedforward_wrench = feedforward_wrench_;
    trace.dt_seconds = dt_[0];
    trace.input_pose_offset = trace_input_pose_offset_;
    trace.input_twist = trace_input_twist_;
    trace.input_pose_previous_signal = trace_input_pose_previous_signal_;
    trace.input_twist_previous_signal = trace_input_twist_previous_signal_;
    trace.input_pose_initialized = trace_input_pose_initialized_;
    trace.input_twist_initialized = trace_input_twist_initialized_;
    trace.input_last_valid_acceleration = trace_input_last_valid_acceleration_;
    trace.input_safety_valid = trace_input_safety_valid_;
    trace.input_hold_count = trace_input_hold_count_;
    trace.acceleration = commanded_acceleration_;
    trace.twist = twist_;
    trace.pose_offset = pose_offset_;
    trace.pose_previous_signal = pose_previous_signal_;
    trace.twist_previous_signal = twist_previous_signal_;
    trace.pose_initialized = pose_integrator_initialized_[0];
    trace.twist_initialized = twist_integrator_initialized_[0];
    trace.parameter_status = admittance_parameter_status_[0];
    trace.status = admittance_status_[0];
    trace.last_valid_acceleration = last_valid_acceleration_;
    trace.safety_valid = admittance_safety_valid_[0];
    trace.hold_count = admittance_hold_count_[0];
    return trace;
  }

 private:
  WrenchLowPassFilter wrench_filter_{7.0};
  std::array<double, kCartesianDim * kCartesianDim> admittance_stiffness_{};
  std::array<double, kCartesianDim * kCartesianDim> admittance_damping_{};
  std::array<double, kCartesianDim * kCartesianDim> virtual_mass_{};
  std::array<double, kCartesianDim> force_filter_threshold_{};
  std::array<double, kCartesianDim> external_wrench_{};
  std::array<double, kCartesianDim> feedforward_wrench_{};
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> commanded_acceleration_{};
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
  std::array<double, kCartesianDim> trace_input_pose_offset_{};
  std::array<double, kCartesianDim> trace_input_twist_{};
  std::array<double, kCartesianDim> trace_input_pose_previous_signal_{};
  std::array<double, kCartesianDim> trace_input_twist_previous_signal_{};
  std::array<double, kCartesianDim> trace_input_last_valid_acceleration_{};
  std::uint8_t trace_input_pose_initialized_{0};
  std::uint8_t trace_input_twist_initialized_{0};
  std::uint8_t trace_input_safety_valid_{0};
  std::uint64_t trace_input_hold_count_{0};
  bool last_step_invoked_{false};
  controllers::CartesianAdmittancePoseController::Inputs admittance_inputs_{};
  controllers::CartesianAdmittancePoseController admittance_controller_;
};

struct CartesianImpedanceStepTrace {
  std::array<double, kCartesianDim> pose_error{};
  std::array<double, kCartesianDim> velocity_error{};
  std::array<double, kCartesianDim> desired_acceleration{};
  std::array<double, kJointCount> joint_velocity{};
  std::array<double, kJointCount * kJointCount> joint_mass{};
  std::array<double, kCartesianDim * kCartesianDim> cartesian_mass{};
  std::array<double, kCartesianDim * kJointCount> jacobian{};
  double dt_seconds{};
  std::array<double, kCartesianDim> commanded_wrench{};
  std::uint64_t mass_parameter_status{};
  std::uint64_t mass_status{};
  std::uint64_t impedance_parameter_status{};
  std::uint64_t impedance_status{};
};

inline void print_cartesian_impedance_step(std::ostream& output,
                                           const CartesianImpedanceStepTrace& trace) {
  output << " impedance_input_pose_error_O_m_rad=";
  print_trace_array(output, trace.pose_error);
  output << " impedance_input_velocity_error_O_m_s_rad_s=";
  print_trace_array(output, trace.velocity_error);
  output << " impedance_input_desired_acceleration_O_m_s2_rad_s2=";
  print_trace_array(output, trace.desired_acceleration);
  output << " impedance_input_joint_velocity_rad_s=";
  print_trace_array(output, trace.joint_velocity);
  output << " impedance_mass_input_joint_mass_row_major=";
  print_trace_array(output, trace.joint_mass);
  output << " impedance_input_cartesian_mass_row_major=";
  print_trace_array(output, trace.cartesian_mass);
  output << " impedance_input_jacobian_row_major=";
  print_trace_array(output, trace.jacobian);
  output << " impedance_input_dt_s=" << trace.dt_seconds
         << " impedance_output_wrench_O_N_Nm=";
  print_trace_array(output, trace.commanded_wrench);
  output << " impedance_mass_parameter_status=" << trace.mass_parameter_status
         << " impedance_mass_status=" << trace.mass_status
         << " impedance_parameter_status=" << trace.impedance_parameter_status
         << " impedance_status=" << trace.impedance_status;
}

class CartesianImpedanceWorkspace {
 public:
  CartesianImpedanceWorkspace() {
    constexpr std::array<double, kCartesianDim> stiffness{2000, 2000, 2000, 100, 100, 100};
    for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
      stiffness_[axis * kCartesianDim + axis] = stiffness[axis];
      damping_factors_[axis] = 0.7;
    }
  }

  controllers::Result prepare() {
    const controllers::ControllerConfig config{controllers::Device::kCpu,
                                               controllers::DType::kFloat64,
                                               1,
                                               kJointCount,
                                               {},
                                               controllers::CartesianFrame::kRobotBase};
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
        controllers::shared(controllers::read_view(
            stiffness_, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            damping_factors_, controllers::Device::kCpu, kCartesianDim))};
    const controllers::CartesianImpedanceController::Storage impedance_storage{
        controllers::write_view(commanded_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_status(impedance_parameter_status_, controllers::Device::kCpu),
        controllers::write_status(impedance_status_, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(
                last_valid_wrench_, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(impedance_safety_valid_, controllers::Device::kCpu, 1),
            controllers::write_view(impedance_hold_count_, controllers::Device::kCpu, 1)),
        {controllers::write_view(
             previous_cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
         controllers::write_view(
             previous_jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
         controllers::write_view(impedance_initialized_, controllers::Device::kCpu, 1)}};
    mass_inputs_ = {
        controllers::read_view(joint_mass_, controllers::Device::kCpu, 1, kJointCount, kJointCount),
        controllers::read_view(
            jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount)};
    impedance_inputs_ = {
        controllers::read_view(pose_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(velocity_error_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(desired_acceleration_, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::read_view(joint_velocity_, controllers::Device::kCpu, 1, kJointCount),
        controllers::read_view(cartesian_mass_, controllers::Device::kCpu, 1, kCartesianDim,
                               kCartesianDim),
        controllers::read_view(jacobian_, controllers::Device::kCpu, 1, kCartesianDim, kJointCount),
        controllers::read_view(dt_, controllers::Device::kCpu, 1)};
    controllers::Result result = mass_controller_.prepare(config, mass_parameters, mass_storage);
    if (!result.ok()) return result;
    result = impedance_controller_.prepare(config, impedance_parameters, impedance_storage);
    if (!result.ok()) return result;
    result = mass_controller_.reset();
    if (!result.ok()) return result;
    return impedance_controller_.reset();
  }

  bool step(const Eigen::Vector<double, kCartesianDim>& pose_error,
            const Eigen::Vector<double, kCartesianDim>& velocity_error,
            const Eigen::Vector<double, kCartesianDim>& desired_acceleration,
            const Eigen::Vector<double, kJointCount>& joint_velocity,
            const Eigen::Matrix<double, kJointCount, kJointCount>& mass,
            const Eigen::Matrix<double, kCartesianDim, kJointCount>& jacobian,
            const double dt_seconds) {
    last_step_invoked_ = false;
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
      desired_acceleration_[row] = desired_acceleration[static_cast<Eigen::Index>(row)];
      for (std::size_t column = 0; column < kJointCount; ++column) {
        jacobian_[row * kJointCount + column] =
            jacobian(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
      }
    }
    dt_[0] = dt_seconds;
    mass_status_[0] = 0;
    impedance_status_[0] = 0;
    const controllers::Result mass_result = mass_controller_.step(mass_inputs_);
    if (!mass_result.ok() ||
        controllers::has_status_flag(mass_status_[0], controllers::StatusFlag::kSafeMode)) {
      return false;
    }
    last_step_invoked_ = true;
    const controllers::Result result = impedance_controller_.step(impedance_inputs_);
    return result.ok() &&
           !controllers::has_status_flag(impedance_status_[0], controllers::StatusFlag::kSafeMode);
  }

  const std::array<double, kCartesianDim>& wrench() const noexcept { return commanded_wrench_; }
  bool last_step_invoked() const noexcept { return last_step_invoked_; }

  CartesianImpedanceStepTrace last_step_trace() const noexcept {
    CartesianImpedanceStepTrace trace;
    trace.pose_error = pose_error_;
    trace.velocity_error = velocity_error_;
    trace.desired_acceleration = desired_acceleration_;
    trace.joint_velocity = joint_velocity_;
    trace.joint_mass = joint_mass_;
    trace.cartesian_mass = cartesian_mass_;
    trace.jacobian = jacobian_;
    trace.dt_seconds = dt_[0];
    trace.commanded_wrench = commanded_wrench_;
    trace.mass_parameter_status = mass_parameter_status_[0];
    trace.mass_status = mass_status_[0];
    trace.impedance_parameter_status = impedance_parameter_status_[0];
    trace.impedance_status = impedance_status_[0];
    return trace;
  }

  void print_configuration(std::ostream& output) const {
    output << " cartesian_impedance_stiffness_matrix_row_major=";
    print_trace_array(output, stiffness_);
    output << " cartesian_impedance_damping_factors=";
    print_trace_array(output, damping_factors_);
    output << " cartesian_mass_regularization=" << regularization_[0];
  }

 private:
  std::array<double, kCartesianDim * kCartesianDim> stiffness_{};
  std::array<double, kCartesianDim> damping_factors_{};
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
  std::array<double, 1> dt_{};
  std::array<double, kCartesianDim> commanded_wrench_{};
  std::array<double, kCartesianDim> last_valid_wrench_{};
  std::array<std::uint8_t, 1> impedance_safety_valid_{};
  std::array<std::uint64_t, 1> impedance_hold_count_{};
  std::array<double, kCartesianDim * kCartesianDim> previous_cartesian_mass_{};
  std::array<double, kCartesianDim * kJointCount> previous_jacobian_{};
  std::array<std::uint8_t, 1> impedance_initialized_{};
  std::array<std::uint64_t, 1> mass_parameter_status_{};
  std::array<std::uint64_t, 1> mass_status_{};
  std::array<std::uint64_t, 1> impedance_parameter_status_{};
  std::array<std::uint64_t, 1> impedance_status_{};
  bool last_step_invoked_{false};
  controllers::OperationalSpaceMass::Inputs mass_inputs_{};
  controllers::CartesianImpedanceController::Inputs impedance_inputs_{};
  controllers::OperationalSpaceMass mass_controller_;
  controllers::CartesianImpedanceController impedance_controller_;
};

}  // namespace franka_example
