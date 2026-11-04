/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file controllers_test.cpp
 * @brief Tests the platform-neutral controller lifecycle.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "./controller_test_support.hpp"
#include "nvidia/srl/controllers/controllers.hpp"
#include "nvidia/srl/controllers/example_specs.hpp"

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace {
namespace controllers = nvidia::srl::controllers;

constexpr std::size_t kJointCount = 2;
constexpr std::size_t kCartesianDim = 6;

template <typename T, std::size_t Rank>
controllers::TensorView input_view(T* data, const std::array<std::int64_t, Rank>& shape) noexcept {
  return controllers::read_view(data, controllers::Device::kCpu, shape);
}

struct JointFixture {
  std::array<double, 4> stiffness{100.0, 0.0, 0.0, 200.0};
  std::array<double, 2> damping{1.0, 1.0};
  std::array<double, 2> position_error{0.1, -0.1};
  std::array<double, 2> velocity_error{};
  std::array<double, 2> desired_acceleration{};
  std::array<double, 4> mass{1.0, 0.0, 0.0, 1.0};
  std::array<double, 2> torque{};
  std::array<double, 2> last_valid_torque{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};

  controllers::JointImpedanceController controller;

  controllers::Result prepare(const bool alias_safety_output = false) {
    const auto batch_vector = std::array<std::int64_t, 2>{1, kJointCount};
    const auto batch_matrix = std::array<std::int64_t, 3>{1, kJointCount, kJointCount};
    const controllers::JointImpedanceController::Parameters parameters{
        {input_view(stiffness.data(), batch_matrix), controllers::ParameterBatching::kPerBatch},
        {input_view(damping.data(), batch_vector), controllers::ParameterBatching::kPerBatch}};
    const controllers::JointImpedanceController::Storage storage{
        controllers::write_view(torque, controllers::Device::kCpu, batch_vector),
        {parameter_status.data(), parameter_status.size(), controllers::Device::kCpu},
        {status.data(), status.size(), controllers::Device::kCpu},
        {controllers::write_view(alias_safety_output ? torque.data() : last_valid_torque.data(),
                                 controllers::Device::kCpu,
                                 batch_vector),
         controllers::write_view(valid, controllers::Device::kCpu, std::array<std::int64_t, 1>{1}),
         controllers::write_view(
             hold_count, controllers::Device::kCpu, std::array<std::int64_t, 1>{1})}};
    return controller.prepare({controllers::Device::kCpu,
                               controllers::DType::kFloat64,
                               1,
                               kJointCount,
                               {controllers::SafeOutputPolicy::kZero, 0}},
                              parameters,
                              storage);
  }

  controllers::Result step() {
    const auto batch_vector = std::array<std::int64_t, 2>{1, kJointCount};
    const auto batch_matrix = std::array<std::int64_t, 3>{1, kJointCount, kJointCount};
    return controller.step({input_view(position_error.data(), batch_vector),
                            input_view(velocity_error.data(), batch_vector),
                            input_view(desired_acceleration.data(), batch_vector),
                            input_view(mass.data(), batch_matrix)});
  }
};

template <std::size_t OutputDim, typename T = double, std::size_t BatchSize = 1>
struct OutputFixture {
  std::array<T, BatchSize * OutputDim> command{};
  std::array<T, BatchSize * OutputDim> last_valid{};
  std::array<std::uint8_t, BatchSize> valid{};
  std::array<std::uint64_t, BatchSize> hold_count{};
  std::array<std::uint64_t, BatchSize> parameter_status{};
  std::array<std::uint64_t, BatchSize> status{};

  controllers::ControllerOutputStorage view() {
    return {
        controllers::write_view(command, controllers::Device::kCpu, BatchSize, OutputDim),
        controllers::write_status(parameter_status, controllers::Device::kCpu),
        controllers::write_status(status, controllers::Device::kCpu),
        controllers::make_safety_state(
            controllers::write_view(last_valid, controllers::Device::kCpu, BatchSize, OutputDim),
            controllers::write_view(valid, controllers::Device::kCpu, BatchSize),
            controllers::write_view(hold_count, controllers::Device::kCpu, BatchSize))};
  }
};

template <std::size_t JointDim>
struct InertialFixture {
  std::array<double, kCartesianDim * kCartesianDim> previous_mass{};
  std::array<double, kCartesianDim * JointDim> previous_jacobian{};
  std::array<std::uint8_t, 1> initialized{};

  controllers::InertialStateStorage view() {
    return {controllers::write_view(
                previous_mass, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
            controllers::write_view(
                previous_jacobian, controllers::Device::kCpu, 1, kCartesianDim, JointDim),
            controllers::write_view(initialized, controllers::Device::kCpu, 1)};
  }
};

struct ForceStateFixture {
  std::array<double, kCartesianDim> differentiator_previous{};
  std::array<std::uint8_t, 1> differentiator_initialized{};
  std::array<double, kCartesianDim> integrator_previous{};
  std::array<double, kCartesianDim> integrated_error{};
  std::array<std::uint8_t, 1> integrator_initialized{};

  controllers::ForceControllerStateStorage view() {
    return {
        controllers::write_view(
            differentiator_previous, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(differentiator_initialized, controllers::Device::kCpu, 1),
        controllers::write_view(integrator_previous, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(integrated_error, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(integrator_initialized, controllers::Device::kCpu, 1)};
  }
};

struct AdmittanceStateFixture {
  std::array<double, kCartesianDim> pose{};
  std::array<double, kCartesianDim> twist{};
  std::array<double, kCartesianDim> pose_previous{};
  std::array<double, kCartesianDim> twist_previous{};
  std::array<std::uint8_t, 1> pose_initialized{};
  std::array<std::uint8_t, 1> twist_initialized{};

  controllers::AdmittanceStateStorage view() {
    return {controllers::write_view(pose, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(twist, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(pose_previous, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(twist_previous, controllers::Device::kCpu, 1, kCartesianDim),
            controllers::write_view(pose_initialized, controllers::Device::kCpu, 1),
            controllers::write_view(twist_initialized, controllers::Device::kCpu, 1)};
  }
};

void set_identity(std::array<double, kCartesianDim * kCartesianDim>* matrix) {
  matrix->fill(0.0);
  for (std::size_t index = 0; index < kCartesianDim; ++index) {
    (*matrix)[index * kCartesianDim + index] = 1.0;
  }
}

TEST(ControllersTest, ViewHelpersPreserveMetadataAndRejectContainerSizeMismatch) {
  const std::array<double, 4> input{1.0, 2.0, 3.0, 4.0};
  std::array<double, 4> output{};
  std::array<std::uint64_t, 2> status{};

  const controllers::TensorView input_view =
      controllers::read_view(input, controllers::Device::kCpu, 2, 2);
  const controllers::MutableTensorView output_view =
      controllers::write_view(output, controllers::Device::kCpu, 2, 2);
  const controllers::MutableStatusView status_view =
      controllers::write_status(status, controllers::Device::kCpu);

  EXPECT_EQ(input_view.data, input.data());
  EXPECT_EQ(input_view.dtype, controllers::DType::kFloat64);
  EXPECT_EQ(input_view.device, controllers::Device::kCpu);
  EXPECT_EQ(input_view.layout, controllers::Layout::kRowMajor);
  EXPECT_EQ(input_view.rank, 2);
  EXPECT_EQ(input_view.shape[0], 2);
  EXPECT_EQ(input_view.shape[1], 2);
  EXPECT_EQ(input_view.strides[0], 2);
  EXPECT_EQ(input_view.strides[1], 1);
  EXPECT_EQ(output_view.data, output.data());
  EXPECT_EQ(status_view.flags, status.data());
  EXPECT_EQ(status_view.size, status.size());

  const controllers::TensorView mismatched =
      controllers::read_view(input, controllers::Device::kCpu, 1, 3);
  EXPECT_EQ(mismatched.data, nullptr);
  EXPECT_EQ(mismatched.device, controllers::Device::kUnknown);

  const controllers::ParameterTensor shared = controllers::shared(input_view);
  const controllers::ParameterTensor per_batch = controllers::per_batch(input_view);
  EXPECT_EQ(shared.batching, controllers::ParameterBatching::kShared);
  EXPECT_EQ(per_batch.batching, controllers::ParameterBatching::kPerBatch);
}

TEST(ControllersTest, CpuLifecycleProducesExpectedTorque) {
  JointFixture fixture;
  ASSERT_TRUE(fixture.prepare().ok());
  ASSERT_EQ(fixture.parameter_status[0], 0U);
  ASSERT_TRUE(fixture.controller.reset().ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_DOUBLE_EQ(fixture.torque[0], 10.0);
  EXPECT_DOUBLE_EQ(fixture.torque[1], -20.0);
  EXPECT_EQ(fixture.status[0], 0U);
}

TEST(ControllersTest, StaticJointImpedanceRetainsSixtyFourJointSupport) {
  controllers::JointBlockConfig config;
  config.batch_size = 1;
  config.joint_dim = 64;
  config.mode = controllers::ComponentMode::kDiagonal;
  std::array<double, 64> gains;
  std::array<double, 64> errors;
  gains.fill(2.0);
  errors.fill(1.0);
  OutputFixture<64> output;
  controllers::StaticJointImpedanceController controller;
  controllers::Requirements requirements;
  ASSERT_TRUE(controller.requirements(config, &requirements).ok());
  ASSERT_EQ(requirements.outputs[0].shape[1], 64);
  const auto parameter =
      controllers::shared(controllers::read_view(gains, controllers::Device::kCpu, 64));
  ASSERT_TRUE(controller.prepare(config, {parameter, parameter}, output.view()).ok());
  const auto input = controllers::read_view(errors, controllers::Device::kCpu, 1, 64);
  ASSERT_TRUE(controller.step({input, input}).ok());
  for (const double torque : output.command) EXPECT_DOUBLE_EQ(torque, 4.0);
  config.joint_dim = 65;
  EXPECT_FALSE(controller.requirements(config, &requirements).ok());
  EXPECT_EQ(requirements.output_count, 0U);
}

TEST(ControllersTest, StaticControllersRejectStatusOverlapBeforeParameterWrites) {
  controllers::JointBlockConfig config;
  config.batch_size = 1;
  config.joint_dim = 2;
  OutputFixture<2> joint_output;
  joint_output.status[0] = 123;
  auto joint_storage = joint_output.view();
  joint_storage.parameter_status.flags = joint_storage.status.flags;
  controllers::StaticJointImpedanceController joint;
  EXPECT_EQ(joint.prepare(config, {}, joint_storage).code, controllers::ResultCode::kInvalidView);
  EXPECT_EQ(joint_output.status[0], 123U);
  controllers::ControllerConfig cartesian_config = config;
  cartesian_config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  OutputFixture<6> cartesian_output;
  cartesian_output.status[0] = 456;
  auto cartesian_storage = cartesian_output.view();
  cartesian_storage.parameter_status.flags = cartesian_storage.status.flags;
  controllers::StaticCartesianImpedanceController cartesian;
  EXPECT_EQ(cartesian.prepare(cartesian_config, {}, cartesian_storage).code,
            controllers::ResultCode::kInvalidView);
  EXPECT_EQ(cartesian_output.status[0], 456U);
}

TEST(ControllersTest, ForceAdmittanceRejectsCrossStateOverlapBeforeParameterWrites) {
  controllers::ControllerConfig config;
  config.batch_size = 1;
  config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  OutputFixture<6> output;
  ForceStateFixture force;
  AdmittanceStateFixture admittance;
  output.parameter_status[0] = 123;
  controllers::CartesianUnifiedForceAdmittanceController::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
  storage.state = {force.view(), admittance.view()};
  storage.state.admittance.pose_offset = storage.state.force.integrated_error;
  controllers::CartesianUnifiedForceAdmittanceController controller;
  EXPECT_EQ(controller.prepare(config, {}, storage).code, controllers::ResultCode::kInvalidView);
  EXPECT_EQ(output.parameter_status[0], 123U);
}

TEST(ControllersTest, StaticControllerRejectsUnalignedCppInputBeforeWriting) {
  controllers::JointBlockConfig config;
  config.batch_size = 1;
  config.joint_dim = 2;
  config.mode = controllers::ComponentMode::kDiagonal;
  std::array<double, 2> gains{1, 1};
  OutputFixture<2> output;
  output.command.fill(123.0);
  controllers::StaticJointImpedanceController controller;
  const auto parameter =
      controllers::shared(controllers::read_view(gains, controllers::Device::kCpu, 2));
  ASSERT_TRUE(controller.prepare(config, {parameter, parameter}, output.view()).ok());
  alignas(double) std::array<unsigned char, 2 * sizeof(double) + 1> backing{};
  const auto invalid = controllers::read_view(
      reinterpret_cast<const double*>(backing.data() + 1), controllers::Device::kCpu, 1, 2);
  EXPECT_EQ(controller.step({invalid, invalid}).code, controllers::ResultCode::kInvalidView);
  EXPECT_EQ(output.command, (std::array<double, 2>{123.0, 123.0}));
  const auto wrong_device = controllers::read_view(gains, controllers::Device::kCuda, 1, 2);
  EXPECT_EQ(controller.step({wrong_device, wrong_device}).code,
            controllers::ResultCode::kInvalidView);
  EXPECT_EQ(output.command, (std::array<double, 2>{123.0, 123.0}));
}

TEST(ControllersTest, StaticImpedanceUsesDesiredVelocityError) {
  controllers::JointBlockConfig joint_config;
  joint_config.device = controllers::Device::kCpu;
  joint_config.dtype = controllers::DType::kFloat64;
  joint_config.batch_size = 1;
  joint_config.joint_dim = 2;
  joint_config.mode = controllers::ComponentMode::kDiagonal;
  std::array<double, 2> stiffness{10.0, 20.0};
  std::array<double, 2> damping{2.0, 4.0};
  std::array<double, 2> position{1.0, -1.0};
  std::array<double, 2> velocity{0.5, 0.25};
  OutputFixture<2> output;
  controllers::StaticJointImpedanceController joint;
  ASSERT_TRUE(
      joint
          .prepare(
              joint_config,
              {controllers::shared(controllers::read_view(stiffness, controllers::Device::kCpu, 2)),
               controllers::shared(controllers::read_view(damping, controllers::Device::kCpu, 2))},
              output.view())
          .ok());
  ASSERT_TRUE(joint
                  .step({controllers::read_view(position, controllers::Device::kCpu, 1, 2),
                         controllers::read_view(velocity, controllers::Device::kCpu, 1, 2)})
                  .ok());
  EXPECT_DOUBLE_EQ(output.command[0], 11.0);
  EXPECT_DOUBLE_EQ(output.command[1], -19.0);

  controllers::ControllerConfig cartesian_config;
  cartesian_config.device = controllers::Device::kCpu;
  cartesian_config.dtype = controllers::DType::kFloat64;
  cartesian_config.batch_size = 1;
  cartesian_config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  std::array<double, 36> stiffness_matrix{};
  std::array<double, 36> damping_matrix{};
  set_identity(&stiffness_matrix);
  set_identity(&damping_matrix);
  OutputFixture<6> cartesian_output;
  controllers::StaticCartesianImpedanceController cartesian;
  ASSERT_TRUE(cartesian
                  .prepare(cartesian_config,
                           {controllers::shared(controllers::read_view(
                                stiffness_matrix, controllers::Device::kCpu, 6, 6)),
                            controllers::shared(controllers::read_view(
                                damping_matrix, controllers::Device::kCpu, 6, 6))},
                           cartesian_output.view())
                  .ok());
  std::array<double, 6> pose{1, 2, 3, 4, 5, 6};
  std::array<double, 6> twist{6, 5, 4, 3, 2, 1};
  ASSERT_TRUE(cartesian
                  .step({controllers::read_view(pose, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(twist, controllers::Device::kCpu, 1, 6)})
                  .ok());
  for (double value : cartesian_output.command) EXPECT_DOUBLE_EQ(value, 7.0);
}

TEST(ControllersTest, UnifiedForceAdmittanceUsesExternalMinusDesiredAndAtomicState) {
  controllers::ControllerConfig config;
  config.device = controllers::Device::kCpu;
  config.dtype = controllers::DType::kFloat64;
  config.batch_size = 1;
  config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  std::array<double, 36> zero_matrix{};
  std::array<double, 36> mass{};
  set_identity(&mass);
  std::array<double, 6> zero{};
  std::array<double, 6> one{1, 1, 1, 1, 1, 1};
  std::array<double, 6> large{100, 100, 100, 100, 100, 100};
  std::array<double, 6> lower{-10, -10, -10, -10, -10, -10};
  std::array<double, 6> upper{10, 10, 10, 10, 10, 10};
  std::array<double, 6> deadzone{0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  auto parameter = [](auto& values) {
    return controllers::shared(controllers::read_view(values,
                                                      controllers::Device::kCpu,
                                                      values.size() == 36
                                                          ? std::array<std::int64_t, 2>{6, 6}
                                                          : std::array<std::int64_t, 2>{1, 6}));
  };
  const auto vector_parameter = [](auto& values) {
    return controllers::shared(controllers::read_view(values, controllers::Device::kCpu, 6));
  };
  controllers::CartesianUnifiedForceAdmittanceController::Parameters parameters{
      parameter(zero_matrix),
      parameter(zero_matrix),
      parameter(mass),
      vector_parameter(large),
      vector_parameter(large),
      vector_parameter(large),
      {vector_parameter(one),
       vector_parameter(zero),
       vector_parameter(zero),
       vector_parameter(zero),
       vector_parameter(lower),
       vector_parameter(upper),
       vector_parameter(deadzone)}};
  OutputFixture<6> output;
  ForceStateFixture force;
  AdmittanceStateFixture admittance;
  controllers::CartesianUnifiedForceAdmittanceController controller;
  controllers::CartesianUnifiedForceAdmittanceController::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
  storage.state = {force.view(), admittance.view()};
  ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
  std::array<double, 6> desired{1, 1, 1, 1, 1, 1};
  std::array<double, 6> external{};
  std::array<double, 1> dt{1.0};
  ASSERT_TRUE(controller
                  .step({controllers::read_view(desired, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(external, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(zero, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(dt, controllers::Device::kCpu, 1)})
                  .ok());
  for (double value : output.command) EXPECT_DOUBLE_EQ(value, -0.5);
  EXPECT_DOUBLE_EQ(admittance.twist[0], -0.5);
  const auto pose_before_failure = admittance.pose;
  const auto force_before_failure = force.differentiator_previous;
  dt[0] = 0.0;
  ASSERT_TRUE(controller
                  .step({controllers::read_view(desired, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(external, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(zero, controllers::Device::kCpu, 1, 6),
                         controllers::read_view(dt, controllers::Device::kCpu, 1)})
                  .ok());
  EXPECT_EQ(admittance.pose, pose_before_failure);
  EXPECT_EQ(force.differentiator_previous, force_before_failure);
  EXPECT_TRUE(controllers::has_status_flag(output.status[0], controllers::StatusFlag::kInvalidDt));
}

TEST(ControllersTest, TaskWrenchMappingProducesFinalFilteredJointTorque) {
  constexpr std::size_t batch_size = 2;
  constexpr std::size_t joint_dim = 2;
  constexpr std::size_t task_dim = 2;
  const controllers::ControllerConfig config{controllers::Device::kCpu,
                                             controllers::DType::kFloat64,
                                             batch_size,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase,
                                             task_dim};
  std::array<double, 4> wrench{1.0, 2.0, 3.0, 4.0};
  std::array<double, 8> jacobian{1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0};
  std::array<double, 4> bias{0.5, -0.5, 0.5, -0.5};
  std::array<double, 4> feedback{1.0, 1.0, 1.0, 1.0};
  std::array<std::uint64_t, batch_size> task_status{};
  std::array<std::uint64_t, batch_size> feedback_status{};
  std::array<double, batch_size> dt{0.1, 0.1};
  std::array<double, joint_dim> lower{-100.0, -100.0};
  std::array<double, joint_dim> upper{100.0, 100.0};
  std::array<double, joint_dim> maximum_rate{1000.0, 1000.0};
  std::array<double, 4> unfiltered{};
  std::array<double, 4> torque{};
  std::array<double, 4> previous{};
  std::array<double, 4> last_valid{};
  std::array<std::uint8_t, batch_size> initialized{};
  std::array<std::uint8_t, batch_size> valid{};
  std::array<std::uint64_t, batch_size> hold_count{};
  std::array<std::uint64_t, batch_size> parameter_status{};
  std::array<std::uint64_t, batch_size> output_status{};

  controllers::TaskWrenchToJointTorqueController controller;
  ASSERT_TRUE(
      controller
          .prepare(
              config,
              {controllers::shared(
                   controllers::read_view(lower, controllers::Device::kCpu, joint_dim)),
               controllers::shared(
                   controllers::read_view(upper, controllers::Device::kCpu, joint_dim)),
               controllers::shared(
                   controllers::read_view(maximum_rate, controllers::Device::kCpu, joint_dim))},
              {controllers::write_view(
                   unfiltered, controllers::Device::kCpu, batch_size, joint_dim),
               {{controllers::write_view(torque, controllers::Device::kCpu, batch_size, joint_dim),
                 controllers::write_status(parameter_status, controllers::Device::kCpu),
                 controllers::write_status(output_status, controllers::Device::kCpu),
                 controllers::make_safety_state(
                     controllers::write_view(
                         last_valid, controllers::Device::kCpu, batch_size, joint_dim),
                     controllers::write_view(valid, controllers::Device::kCpu, batch_size),
                     controllers::write_view(hold_count, controllers::Device::kCpu, batch_size))},
                {controllers::write_view(
                     previous, controllers::Device::kCpu, batch_size, joint_dim),
                 controllers::write_view(initialized, controllers::Device::kCpu, batch_size)}}})
          .ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(wrench, controllers::Device::kCpu, batch_size, task_dim),
                 controllers::read_view(
                     jacobian, controllers::Device::kCpu, batch_size, task_dim, joint_dim),
                 controllers::read_view(bias, controllers::Device::kCpu, batch_size, joint_dim),
                 controllers::read_view(feedback, controllers::Device::kCpu, batch_size, joint_dim),
                 {task_status.data(), task_status.size(), controllers::Device::kCpu},
                 {feedback_status.data(), feedback_status.size(), controllers::Device::kCpu},
                 controllers::read_view(dt, controllers::Device::kCpu, batch_size)})
          .ok());
  EXPECT_DOUBLE_EQ(torque[0], 2.5);
  EXPECT_DOUBLE_EQ(torque[1], 2.5);
  EXPECT_DOUBLE_EQ(torque[2], 7.5);
  EXPECT_DOUBLE_EQ(torque[3], 12.5);

  task_status[1] = controllers::status_bit(controllers::StatusFlag::kSafeMode);
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(wrench, controllers::Device::kCpu, batch_size, task_dim),
                 controllers::read_view(
                     jacobian, controllers::Device::kCpu, batch_size, task_dim, joint_dim),
                 controllers::read_view(bias, controllers::Device::kCpu, batch_size, joint_dim),
                 controllers::read_view(feedback, controllers::Device::kCpu, batch_size, joint_dim),
                 {task_status.data(), task_status.size(), controllers::Device::kCpu},
                 {feedback_status.data(), feedback_status.size(), controllers::Device::kCpu},
                 controllers::read_view(dt, controllers::Device::kCpu, batch_size)})
          .ok());
  EXPECT_DOUBLE_EQ(torque[2], 0.0);
  EXPECT_DOUBLE_EQ(torque[3], 0.0);
  EXPECT_TRUE(controllers::has_status_flag(output_status[1], controllers::StatusFlag::kSafeMode));
}

TEST(ControllersTest, ExampleRegistryHasOneCanonicalLauncherPerHardwareTarget) {
  const auto& specs = controllers::controller_example_specs();
  ASSERT_EQ(specs.size(), 20U);
  for (std::size_t first = 0; first < specs.size(); ++first) {
    EXPECT_NE(specs[first].hardware_target, nullptr);
    EXPECT_NE(specs[first].isaac_launcher, nullptr);
    EXPECT_GT(specs[first].nominal_period_seconds, 0.0);
    EXPECT_EQ(controllers::find_controller_example_spec(specs[first].hardware_target),
              &specs[first]);
    for (std::size_t second = first + 1; second < specs.size(); ++second) {
      EXPECT_STRNE(specs[first].hardware_target, specs[second].hardware_target);
      EXPECT_STRNE(specs[first].isaac_launcher, specs[second].isaac_launcher);
    }
  }
  EXPECT_EQ(controllers::find_controller_example_spec("unknown"), nullptr);
}

TEST(ControllersTest, FullExampleRegistryHasOneCanonicalLauncherPerHardwareTarget) {
  const auto specs = controllers::all_controller_example_specs();
  ASSERT_EQ(specs.count, 22U);
  for (std::size_t first = 0; first < specs.count; ++first) {
    EXPECT_NE(specs.data[first].hardware_target, nullptr);
    EXPECT_NE(specs.data[first].isaac_launcher, nullptr);
    EXPECT_GT(specs.data[first].nominal_period_seconds, 0.0);
    const auto* found =
        controllers::find_controller_example_spec(specs.data[first].hardware_target);
    ASSERT_NE(found, nullptr);
    EXPECT_STREQ(found->isaac_launcher, specs.data[first].isaac_launcher);
    for (std::size_t second = first + 1; second < specs.count; ++second) {
      EXPECT_STRNE(specs.data[first].hardware_target, specs.data[second].hardware_target);
      EXPECT_STRNE(specs.data[first].isaac_launcher, specs.data[second].isaac_launcher);
    }
  }
  EXPECT_EQ(controllers::find_controller_example_spec("unknown"), nullptr);
}

TEST(ControllersTest, FullExampleRegistryRetainsFixedViewAndAddsRecentCompanions) {
  const auto full = controllers::all_controller_example_specs();
  ASSERT_EQ(full.count, 22U);
  ASSERT_EQ(controllers::controller_example_specs().size(), 20U);
  const auto& legacy = controllers::controller_example_specs();
  for (std::size_t index = 0; index < legacy.size(); ++index) {
    EXPECT_STREQ(full.data[index].hardware_target, legacy[index].hardware_target);
    EXPECT_STREQ(full.data[index].isaac_launcher, legacy[index].isaac_launcher);
  }
  const auto* tank = controllers::find_controller_example_spec(
      "franka_cartesian_force_impedance_controller_with_tank");
  ASSERT_NE(tank, nullptr);
  EXPECT_EQ(tank->controller_kind,
            controllers::ExampleControllerKind::kCartesianForceImpedanceWithTankTorque);
  EXPECT_EQ(tank, &legacy[18]);
}

TEST(ControllersTest, OnlyUrExamplesSelectStaticImpedance) {
  for (const auto& spec : controllers::controller_example_specs()) {
    if (spec.controller_kind == controllers::ExampleControllerKind::kStaticJointImpedance ||
        spec.controller_kind ==
            controllers::ExampleControllerKind::kStaticCartesianImpedanceTorque) {
      EXPECT_EQ(spec.robot_family, controllers::ExampleRobotFamily::kUniversalRobots)
          << spec.hardware_target;
    }
  }
  const auto* joint =
      controllers::find_controller_example_spec("franka_joint_impedance_controller");
  const auto* cartesian =
      controllers::find_controller_example_spec("franka_cartesian_impedance_controller");
  ASSERT_NE(joint, nullptr);
  ASSERT_NE(cartesian, nullptr);
  EXPECT_EQ(joint->controller_kind, controllers::ExampleControllerKind::kJointImpedance);
  EXPECT_EQ(cartesian->controller_kind,
            controllers::ExampleControllerKind::kCartesianImpedanceTorque);
}

TEST(ControllersTest, RemainingPublicControllersExecuteCpuLifecycle) {
  constexpr std::size_t joint_dim = kCartesianDim;
  const controllers::ControllerConfig config{controllers::Device::kCpu,
                                             controllers::DType::kFloat64,
                                             1,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase};
  controllers::CompositeControllerConfig composite_config;
  static_cast<controllers::ControllerConfig&>(composite_config) = config;

  std::array<double, kCartesianDim * kCartesianDim> identity{};
  set_identity(&identity);
  std::array<double, kCartesianDim> zeros{};
  std::array<double, kCartesianDim> ones{};
  std::array<double, kCartesianDim> twos{};
  std::array<double, kCartesianDim> negative_ones{};
  std::array<double, kCartesianDim> rates{};
  ones.fill(1.0);
  twos.fill(2.0);
  negative_ones.fill(-1.0);
  rates.fill(10.0);
  std::array<double, 1> dt{0.01};

  {
    OutputFixture<kCartesianDim> output;
    std::array<double, kCartesianDim> previous_command{};
    std::array<std::uint8_t, 1> initialized{};
    controllers::TorqueFilter::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = {
        controllers::write_view(previous_command, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(initialized, controllers::Device::kCpu, 1)};
    const controllers::TorqueFilter::Parameters parameters{
        controllers::shared(
            controllers::read_view(negative_ones, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(rates, controllers::Device::kCpu, kCartesianDim))};
    controllers::TorqueFilter controller;
    ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(twos, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    for (const double value : output.command) {
      EXPECT_DOUBLE_EQ(value, 1.0);
    }
    EXPECT_TRUE(
        controllers::has_status_flag(output.status[0], controllers::StatusFlag::kSaturation));
  }

  {
    OutputFixture<joint_dim> output;
    const controllers::NullspaceController::Parameters parameters{
        controllers::shared(
            controllers::read_view(identity, controllers::Device::kCpu, joint_dim, joint_dim)),
        controllers::shared(
            controllers::read_view(identity, controllers::Device::kCpu, joint_dim, joint_dim))};
    controllers::NullspaceController controller;
    ASSERT_TRUE(controller.prepare(config, parameters, output.view()).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(controller
                    .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, joint_dim),
                           controllers::read_view(zeros, controllers::Device::kCpu, 1, joint_dim),
                           controllers::read_view(
                               identity, controllers::Device::kCpu, 1, joint_dim, joint_dim),
                           controllers::read_view(
                               identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim)})
                    .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }

  {
    std::array<double, 1> regularization{1.0e-6};
    std::array<double, kCartesianDim * kCartesianDim> cartesian_mass{};
    std::array<double, joint_dim * joint_dim> joint_factor{};
    std::array<double, joint_dim * kCartesianDim> solve_rhs{};
    std::array<double, kCartesianDim * kCartesianDim> cartesian_inverse{};
    std::array<std::uint64_t, 1> parameter_status{};
    std::array<std::uint64_t, 1> status{};
    const controllers::OperationalSpaceMass::Storage storage{
        controllers::write_view(
            cartesian_mass, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::write_status(parameter_status, controllers::Device::kCpu),
        controllers::write_status(status, controllers::Device::kCpu),
        {controllers::write_view(joint_factor, controllers::Device::kCpu, 1, joint_dim, joint_dim),
         controllers::write_view(solve_rhs, controllers::Device::kCpu, 1, joint_dim, kCartesianDim),
         controllers::write_view(
             cartesian_inverse, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim)}};
    const controllers::OperationalSpaceMass::Parameters parameters{
        controllers::shared(controllers::read_view(regularization, controllers::Device::kCpu, 1)),
        {}};
    controllers::OperationalSpaceMass controller;
    ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(controller
                    .step({controllers::read_view(
                               identity, controllers::Device::kCpu, 1, joint_dim, joint_dim),
                           controllers::read_view(
                               identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim)})
                    .ok());
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t column = 0; column < kCartesianDim; ++column) {
        const double expected = row == column ? 1.0 / (1.0 + regularization[0]) : 0.0;
        EXPECT_NEAR(cartesian_mass[row * kCartesianDim + column], expected, 1.0e-12);
      }
    }
    EXPECT_EQ(status[0], 0U);
  }

  {
    OutputFixture<kCartesianDim> output;
    InertialFixture<joint_dim> inertial;
    controllers::CartesianImpedanceController::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = inertial.view();
    const controllers::CartesianImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim))};
    controllers::CartesianImpedanceController controller;
    ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, joint_dim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }

  {
    OutputFixture<kCartesianDim> output;
    std::array<double, kCartesianDim> pose_offset{};
    std::array<double, kCartesianDim> twist{};
    std::array<double, kCartesianDim> pose_previous{};
    std::array<double, kCartesianDim> twist_previous{};
    std::array<std::uint8_t, 1> pose_initialized{};
    std::array<std::uint8_t, 1> twist_initialized{};
    controllers::CartesianAdmittancePoseController::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = {
        controllers::write_view(pose_offset, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(twist, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(pose_previous, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(twist_previous, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(pose_initialized, controllers::Device::kCpu, 1),
        controllers::write_view(twist_initialized, controllers::Device::kCpu, 1)};
    const controllers::CartesianAdmittancePoseController::Parameters parameters{
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(
            controllers::read_view(zeros, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim))};
    controllers::CartesianAdmittancePoseController controller;
    ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(twist, zeros);
    EXPECT_EQ(pose_offset, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }

  {
    OutputFixture<kCartesianDim> output;
    const controllers::OperationalSpaceInverseDynamicsController::Parameters parameters{
        controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim))};
    controllers::OperationalSpaceInverseDynamicsController controller;
    ASSERT_TRUE(controller.prepare(config, parameters, output.view()).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(ones, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(twos, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(ones, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(twos, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim),
                   controllers::read_view(ones, controllers::Device::kCpu, 1, joint_dim)})
            .ok());
    for (const double torque : output.command) {
      EXPECT_DOUBLE_EQ(torque, 3.0);
    }
    EXPECT_EQ(output.status[0], 0U);
  }

  std::array<double, kCartesianDim> force_proportional{};
  std::array<double, kCartesianDim> force_differentiator{};
  std::array<double, kCartesianDim> force_integral{};
  force_proportional.fill(1.0);
  const controllers::ForceControllerParameters force_parameters{
      controllers::shared(
          controllers::read_view(force_proportional, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(
          controllers::read_view(force_differentiator, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(
          controllers::read_view(force_integral, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(controllers::read_view(zeros, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(controllers::read_view(twos, controllers::Device::kCpu, kCartesianDim)),
      controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim))};

  {
    OutputFixture<kCartesianDim> output;
    ForceStateFixture force_state;
    controllers::CartesianForceController::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = force_state.view();
    controllers::CartesianForceController controller;
    ASSERT_TRUE(controller.prepare(composite_config, force_parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }

  {
    OutputFixture<kCartesianDim> output;
    InertialFixture<joint_dim> inertial;
    ForceStateFixture force_state;
    std::array<double, kCartesianDim> impedance_wrench{};
    std::array<double, kCartesianDim> force_wrench{};
    std::array<std::uint64_t, 1> impedance_status{};
    std::array<std::uint64_t, 1> force_status{};
    controllers::CartesianUnifiedForceImpedanceController::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = {inertial.view(), force_state.view()};
    storage.scratch = {
        controllers::write_view(impedance_wrench, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(force_wrench, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(impedance_status, controllers::Device::kCpu, 1),
        controllers::write_view(force_status, controllers::Device::kCpu, 1)};
    const controllers::CartesianUnifiedForceImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim)),
        force_parameters};
    controllers::CartesianUnifiedForceImpedanceController controller;
    ASSERT_TRUE(controller.prepare(composite_config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, joint_dim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }

  {
    OutputFixture<kCartesianDim> output;
    InertialFixture<joint_dim> inertial;
    std::array<double, kCartesianDim * kCartesianDim> current_stiffness{};
    std::array<double, kCartesianDim * kCartesianDim> previous_stiffness{};
    std::array<std::uint8_t, 1> stiffness_initialized{};
    std::array<double, kCartesianDim> stiffness_wrench{};
    std::array<double, kCartesianDim> damping_wrench{};
    std::array<double, kCartesianDim> inertial_wrench{};
    std::array<std::uint64_t, 1> stiffness_status{};
    std::array<std::uint64_t, 1> damping_status{};
    std::array<std::uint64_t, 1> inertial_status{};
    controllers::VariableCartesianImpedanceController::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = {
        controllers::write_view(
            current_stiffness, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::write_view(
            previous_stiffness, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
        controllers::write_view(stiffness_initialized, controllers::Device::kCpu, 1),
        inertial.view()};
    storage.scratch = {
        controllers::write_view(stiffness_wrench, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(damping_wrench, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(inertial_wrench, controllers::Device::kCpu, 1, kCartesianDim),
        controllers::write_view(stiffness_status, controllers::Device::kCpu, 1),
        controllers::write_view(damping_status, controllers::Device::kCpu, 1),
        controllers::write_view(inertial_status, controllers::Device::kCpu, 1)};
    const controllers::VariableCartesianImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(
            identity, controllers::Device::kCpu, kCartesianDim, kCartesianDim)),
        controllers::shared(controllers::read_view(twos, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(zeros, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(rates, controllers::Device::kCpu, kCartesianDim)),
        controllers::shared(
            controllers::read_view(ones, controllers::Device::kCpu, kCartesianDim))};
    controllers::VariableCartesianImpedanceController controller;
    ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
    ASSERT_TRUE(controller.reset().ok());
    ASSERT_TRUE(
        controller
            .step({controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, kCartesianDim),
                   controllers::read_view(zeros, controllers::Device::kCpu, 1, joint_dim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, kCartesianDim),
                   controllers::read_view(
                       identity, controllers::Device::kCpu, 1, kCartesianDim, joint_dim),
                   controllers::read_view(dt, controllers::Device::kCpu, 1)})
            .ok());
    EXPECT_EQ(output.command, zeros);
    EXPECT_EQ(output.status[0], 0U);
  }
}

TEST(ControllersTest, NullspaceTorqueUsesDynamicallyConsistentTransposeProjector) {
  constexpr std::size_t joint_dim = 7;
  std::array<double, joint_dim * joint_dim> identity{};
  std::array<double, joint_dim * joint_dim> mass{};
  std::array<double, kCartesianDim * joint_dim> jacobian{};
  std::array<double, joint_dim> position_error{};
  std::array<double, joint_dim> velocity{};
  for (std::size_t joint = 0; joint < joint_dim; ++joint) {
    identity[joint * joint_dim + joint] = 1.0;
    mass[joint * joint_dim + joint] = static_cast<double>(joint + 1U);
    position_error[joint] = 1.0;
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    jacobian[row * joint_dim + row] = 1.0;
    jacobian[row * joint_dim + (joint_dim - 1U)] = 0.25 * static_cast<double>(row + 1U);
  }

  OutputFixture<joint_dim> output;
  const controllers::ControllerConfig config{controllers::Device::kCpu,
                                             controllers::DType::kFloat64,
                                             1,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase};
  const controllers::NullspaceController::Parameters parameters{
      controllers::shared(
          controllers::read_view(identity, controllers::Device::kCpu, joint_dim, joint_dim)),
      controllers::shared(
          controllers::read_view(identity, controllers::Device::kCpu, joint_dim, joint_dim))};
  controllers::NullspaceController controller;
  ASSERT_TRUE(controller.prepare(config, parameters, output.view()).ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(position_error, controllers::Device::kCpu, 1, joint_dim),
                 controllers::read_view(velocity, controllers::Device::kCpu, 1, joint_dim),
                 controllers::read_view(mass, controllers::Device::kCpu, 1, joint_dim, joint_dim),
                 controllers::read_view(
                     jacobian, controllers::Device::kCpu, 1, kCartesianDim, joint_dim)})
          .ok());

  double output_norm = 0.0;
  for (const double torque : output.command) {
    output_norm += torque * torque;
  }
  EXPECT_GT(output_norm, 1.0e-12);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    double task_acceleration = 0.0;
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      task_acceleration += jacobian[row * joint_dim + joint] * output.command[joint] /
                           mass[joint * joint_dim + joint];
    }
    EXPECT_NEAR(task_acceleration, 0.0, 1.0e-10);
  }
  EXPECT_EQ(output.status[0], 0U);
}

TEST(ControllersTest, FailedRepreparationLeavesControllerUnprepared) {
  JointFixture fixture;
  ASSERT_TRUE(fixture.prepare().ok());

  const controllers::Result prepare_result = fixture.prepare(true);
  ASSERT_EQ(prepare_result.code, controllers::ResultCode::kInvalidView);
  const controllers::Result step_result = fixture.step();

  EXPECT_EQ(step_result.code, controllers::ResultCode::kInvalidArgument);
  EXPECT_STREQ(step_result.message, "controller is not prepared");
}

TEST(ControllersTest, ImmediateParameterUpdateFailureLeavesControllerUnprepared) {
  JointFixture fixture;
  ASSERT_TRUE(fixture.prepare().ok());
  std::array<double, 3> invalid_stiffness{};
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::per_batch(
          controllers::read_view(invalid_stiffness, controllers::Device::kCpu, 3)),
      controllers::per_batch(
          controllers::read_view(fixture.damping, controllers::Device::kCpu, 1, kJointCount))};

  const controllers::Result update_result = fixture.controller.apply_parameters(parameters);
  ASSERT_EQ(update_result.code, controllers::ResultCode::kInvalidView);
  const controllers::Result step_result = fixture.step();

  EXPECT_EQ(step_result.code, controllers::ResultCode::kInvalidArgument);
  EXPECT_STREQ(step_result.message, "controller is not prepared");
}

TEST(ControllersTest, RowLocalParameterFailureKeepsLifecycleUsableAndSafe) {
  JointFixture fixture;
  ASSERT_TRUE(fixture.prepare().ok());
  fixture.stiffness[0] = std::numeric_limits<double>::quiet_NaN();
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::per_batch(controllers::read_view(
          fixture.stiffness, controllers::Device::kCpu, 1, kJointCount, kJointCount)),
      controllers::per_batch(
          controllers::read_view(fixture.damping, controllers::Device::kCpu, 1, kJointCount))};

  ASSERT_TRUE(fixture.controller.apply_parameters(parameters).ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_TRUE(controllers::has_status_flag(fixture.parameter_status[0],
                                           controllers::StatusFlag::kInvalidParameter));
  EXPECT_TRUE(controllers::has_status_flag(fixture.status[0], controllers::StatusFlag::kSafeMode));
  EXPECT_EQ(fixture.torque, (std::array<double, 2>{0.0, 0.0}));
}

TEST(ControllersTest, NonFiniteRowUsesConfiguredSafeOutput) {
  JointFixture fixture;
  ASSERT_TRUE(fixture.prepare().ok());
  ASSERT_TRUE(fixture.controller.reset({}).ok());
  ASSERT_TRUE(fixture.step().ok());
  ASSERT_NE(fixture.torque[0], 0.0);

  fixture.position_error[0] = std::numeric_limits<double>::quiet_NaN();
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_DOUBLE_EQ(fixture.torque[0], 0.0);
  EXPECT_DOUBLE_EQ(fixture.torque[1], 0.0);
  EXPECT_TRUE(
      controllers::has_status_flag(fixture.status[0], controllers::StatusFlag::kNonFiniteInput));
  EXPECT_TRUE(controllers::has_status_flag(fixture.status[0], controllers::StatusFlag::kSafeMode));
}

TEST(ControllersTest, PreparationRejectsAliasedSafetyHistory) {
  JointFixture fixture;
  const controllers::Result result = fixture.prepare(true);

  EXPECT_EQ(result.code, controllers::ResultCode::kInvalidView);
  EXPECT_STREQ(result.message, "last_valid_output must not alias the command output");
}

TEST(ControllersTest, PreparationRejectsPartialSafetyOverlapWithoutWritingStorage) {
  constexpr auto device = controllers::Device::kCpu;
  for (const bool reverse : {false, true}) {
    JointFixture fixture;
    OutputFixture<2> output;
    std::array<double, 3> backing{3.0, 4.0, 5.0};
    auto storage = output.view();
    storage.command = controllers::write_view(backing.data() + (reverse ? 1 : 0), device, 1, 2);
    storage.safety.last_valid_output =
        controllers::write_view(backing.data() + (reverse ? 0 : 1), device, 1, 2);
    output.parameter_status[0] = 123U;
    const controllers::JointImpedanceController::Parameters parameters{
        controllers::shared(controllers::read_view(fixture.stiffness, device, 2, 2)),
        controllers::shared(controllers::read_view(fixture.damping, device, 2))};
    const auto result = fixture.controller.prepare(
        {device, controllers::DType::kFloat64, 1, 2, {}}, parameters, storage);
    EXPECT_EQ(result.code, controllers::ResultCode::kInvalidView);
    EXPECT_EQ(backing, (std::array<double, 3>{3.0, 4.0, 5.0}));
    EXPECT_EQ(output.parameter_status[0], 123U);
  }
}

TEST(ControllersTest, PreparationAcceptsAdjacentSafetyBuffers) {
  constexpr auto device = controllers::Device::kCpu;
  JointFixture fixture;
  OutputFixture<2> output;
  std::array<double, 4> backing{};
  auto storage = output.view();
  storage.command = controllers::write_view(backing.data(), device, 1, 2);
  storage.safety.last_valid_output = controllers::write_view(backing.data() + 2, device, 1, 2);
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::shared(controllers::read_view(fixture.stiffness, device, 2, 2)),
      controllers::shared(controllers::read_view(fixture.damping, device, 2))};
  ASSERT_TRUE(fixture.controller
                  .prepare({device, controllers::DType::kFloat64, 1, 2, {}}, parameters, storage)
                  .ok());
  ASSERT_TRUE(fixture.controller.reset().ok());
  ASSERT_TRUE(fixture.step().ok());
  EXPECT_EQ(backing, (std::array<double, 4>{10.0, -20.0, 10.0, -20.0}));
}

TEST(ControllersTest, PreparationRejectsOverlappingStatusAndHistory) {
  constexpr auto device = controllers::Device::kCpu;
  JointFixture fixture;
  OutputFixture<2> output;
  auto storage = output.view();
  storage.status = storage.parameter_status;
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::shared(controllers::read_view(fixture.stiffness, device, 2, 2)),
      controllers::shared(controllers::read_view(fixture.damping, device, 2))};
  EXPECT_EQ(fixture.controller
                .prepare({device, controllers::DType::kFloat64, 1, 2, {}}, parameters, storage)
                .code,
            controllers::ResultCode::kInvalidView);
  storage = output.view();
  storage.safety.hold_count = controllers::write_view(output.status, device, 1);
  EXPECT_EQ(fixture.controller
                .prepare({device, controllers::DType::kFloat64, 1, 2, {}}, parameters, storage)
                .code,
            controllers::ResultCode::kInvalidView);
}

TEST(ControllersTest, CartesianPreparationRequiresAnExplicitFrame) {
  controllers::Requirements requirements;
  controllers::ControllerConfig config{
      controllers::Device::kCpu, controllers::DType::kFloat64, 1, 7, {}};
  const controllers::Result missing_frame =
      controllers::CartesianImpedanceController::requirements(config, &requirements);
  EXPECT_EQ(missing_frame.code, controllers::ResultCode::kInvalidArgument);

  config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  ASSERT_TRUE(controllers::CartesianImpedanceController::requirements(config, &requirements).ok());
  ASSERT_EQ(requirements.output_count, 3U);
  EXPECT_STREQ(requirements.outputs[0].frame, "robot_base");
  EXPECT_STREQ(requirements.outputs[1].name, "parameter_status");
  EXPECT_STREQ(requirements.outputs[2].name, "status");
  ASSERT_EQ(requirements.state_count, 6U);
  EXPECT_STREQ(requirements.state[3].name, "last_valid_output");
  EXPECT_STREQ(requirements.state[4].name, "valid_mask");
  EXPECT_STREQ(requirements.state[5].name, "hold_count");
}

TEST(ControllersTest, RequirementsDescribeAllCallerOwnedStorageRoles) {
  const controllers::ControllerConfig joint_config{
      controllers::Device::kCpu, controllers::DType::kFloat64, 2, 7, {}};
  controllers::ControllerConfig cartesian_config = joint_config;
  cartesian_config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  controllers::CompositeControllerConfig composite_config;
  static_cast<controllers::ControllerConfig&>(composite_config) = cartesian_config;

  const auto expect_requirements = [](const controllers::Requirements& requirements,
                                      const std::size_t state_count,
                                      const std::size_t scratch_count,
                                      const bool has_safety) {
    EXPECT_EQ(requirements.output_count, 3U);
    EXPECT_STREQ(requirements.outputs[1].name, "parameter_status");
    EXPECT_STREQ(requirements.outputs[2].name, "status");
    EXPECT_EQ(requirements.state_count, state_count);
    EXPECT_EQ(requirements.scratch_count, scratch_count);
    if (has_safety) {
      ASSERT_GE(requirements.state_count, 3U);
      EXPECT_STREQ(requirements.state[requirements.state_count - 3].name, "last_valid_output");
      EXPECT_STREQ(requirements.state[requirements.state_count - 2].name, "valid_mask");
      EXPECT_STREQ(requirements.state[requirements.state_count - 1].name, "hold_count");
    }
  };

  controllers::Requirements requirements;
  ASSERT_TRUE(controllers::TorqueFilter::requirements(joint_config, &requirements).ok());
  expect_requirements(requirements, 5, 0, true);
  ASSERT_TRUE(
      controllers::JointImpedanceController::requirements(joint_config, &requirements).ok());
  expect_requirements(requirements, 3, 0, true);
  ASSERT_TRUE(controllers::NullspaceController::requirements(joint_config, &requirements).ok());
  expect_requirements(requirements, 3, 0, true);
  ASSERT_TRUE(
      controllers::OperationalSpaceMass::requirements(cartesian_config, &requirements).ok());
  expect_requirements(requirements, 0, 3, false);
  ASSERT_TRUE(
      controllers::CartesianImpedanceController::requirements(cartesian_config, &requirements)
          .ok());
  expect_requirements(requirements, 6, 0, true);
  ASSERT_TRUE(
      controllers::CartesianAdmittancePoseController::requirements(cartesian_config, &requirements)
          .ok());
  expect_requirements(requirements, 9, 0, true);
  ASSERT_TRUE(
      controllers::OperationalSpaceController::requirements(cartesian_config, &requirements).ok());
  expect_requirements(requirements, 3, 0, true);
  ASSERT_TRUE(
      controllers::CartesianForceController::requirements(composite_config, &requirements).ok());
  expect_requirements(requirements, 8, 0, true);
  ASSERT_TRUE(controllers::CartesianUnifiedForceImpedanceController::requirements(composite_config,
                                                                                  &requirements)
                  .ok());
  expect_requirements(requirements, 11, 4, true);
  ASSERT_TRUE(controllers::VariableCartesianImpedanceController::requirements(cartesian_config,
                                                                              &requirements)
                  .ok());
  expect_requirements(requirements, 9, 6, true);
}

template <typename T>
class ControllersBehaviorTest : public testing::Test {};
using BehaviorScalars = testing::Types<float, double>;
TYPED_TEST_SUITE(ControllersBehaviorTest, BehaviorScalars);

template <typename T>
constexpr double behavior_tolerance() {
  return std::is_same<T, float>::value ? 2.0e-5 : 2.0e-11;
}

TYPED_TEST(ControllersBehaviorTest, MixedBatchSafetyTransitionsAndSelectiveReset) {
  using T = TypeParam;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  constexpr std::size_t batch = 3;
  const std::array<T, 4> stiffness{T{4}, T{0}, T{0}, T{9}};
  const std::array<T, 2> damping{T{1}, T{1}};
  const std::array<T, 6> velocity{T{0.5}, T{-0.5}, T{0.5}, T{-0.5}, T{0.5}, T{-0.5}};
  const std::array<T, 6> acceleration{T{1}, T{2}, T{1}, T{2}, T{1}, T{2}};
  const std::array<T, 12> mass{
      T{1}, T{0}, T{0}, T{1}, T{1}, T{0}, T{0}, T{1}, T{1}, T{0}, T{0}, T{1}};
  const std::uint64_t failure = controllers::status_bit(controllers::StatusFlag::kNonFiniteInput) |
                                controllers::status_bit(controllers::StatusFlag::kSafeMode);
  const std::array<controllers::SafeOutputPolicy, 2> policies{
      controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid};
  for (const controllers::SafeOutputPolicy policy : policies) {
    SCOPED_TRACE(static_cast<int>(policy));
    OutputFixture<2, T, batch> output;
    std::array<T, 6> position{T{1}, T{-1}, T{2}, T{-2}, T{3}, T{-3}};
    controllers::JointImpedanceController controller;
    ASSERT_TRUE(controller
                    .prepare({cpu, controllers::dtype_for<T>(), batch, 2, {policy, 2}},
                             {controllers::shared(controllers::read_view(stiffness, cpu, 2, 2)),
                              controllers::shared(controllers::read_view(damping, cpu, 2))},
                             output.view())
                    .ok());
    ASSERT_TRUE(controller.reset().ok());
    const auto step = [&]() {
      return controller.step({controllers::read_view(position, cpu, batch, 2),
                              controllers::read_view(velocity, cpu, batch, 2),
                              controllers::read_view(acceleration, cpu, batch, 2),
                              controllers::read_view(mass, cpu, batch, 2, 2)});
    };
    const auto expect_row = [&](std::size_t row,
                                T first,
                                T second,
                                std::uint64_t flags,
                                std::uint8_t valid,
                                std::uint64_t count) {
      EXPECT_NEAR(output.command[2 * row], first, behavior_tolerance<T>());
      EXPECT_NEAR(output.command[2 * row + 1], second, behavior_tolerance<T>());
      EXPECT_EQ(output.status[row], flags);
      EXPECT_EQ(output.parameter_status[row], 0U);
      EXPECT_EQ(output.valid[row], valid);
      EXPECT_EQ(output.hold_count[row], count);
    };
    position[0] = std::numeric_limits<T>::quiet_NaN();
    ASSERT_TRUE(step().ok());
    expect_row(0,
               T{0},
               T{0},
               failure | controllers::status_bit(controllers::StatusFlag::kNoValidOutput),
               0U,
               0U);
    expect_row(1, T{11}, T{-19}, 0U, 1U, 0U);
    expect_row(2, T{15}, T{-28}, 0U, 1U, 0U);
    position[0] = T{1};
    ASSERT_TRUE(step().ok());
    expect_row(0, T{7}, T{-10}, 0U, 1U, 0U);
    const auto history = output.last_valid;
    position[0] = std::numeric_limits<T>::infinity();
    for (std::uint64_t count = 1; count <= 3; ++count) {
      SCOPED_TRACE(count);
      position[2] += T{1};
      ASSERT_TRUE(step().ok());
      const bool held = policy == controllers::SafeOutputPolicy::kHoldLastValid && count <= 2;
      const std::uint64_t flags =
          failure | (policy == controllers::SafeOutputPolicy::kZero
                         ? 0U
                         : controllers::status_bit(held ? controllers::StatusFlag::kOutputHeld
                                                        : controllers::StatusFlag::kHoldExpired));
      expect_row(0, held ? T{7} : T{0}, held ? T{-10} : T{0}, flags, 1U, count);
      expect_row(1, T{11} + T{4} * static_cast<T>(count), T{-19}, 0U, 1U, 0U);
      expect_row(2, T{15}, T{-28}, 0U, 1U, 0U);
      EXPECT_EQ(output.last_valid[0], history[0]);
      EXPECT_EQ(output.last_valid[1], history[1]);
    }
    position[0] = T{2};
    ASSERT_TRUE(step().ok());
    expect_row(0, T{11}, T{-10}, 0U, 1U, 0U);
    EXPECT_EQ(output.last_valid[0], T{11});
    const auto before_reset = output;
    const std::array<std::uint8_t, batch> mask{1U, 0U, 0U};
    ASSERT_TRUE(controller.reset({controllers::read_view(mask, cpu, batch)}).ok());
    EXPECT_EQ(output.valid[0], 0U);
    EXPECT_EQ(output.hold_count[0], 0U);
    EXPECT_EQ(output.last_valid[0], T{0});
    EXPECT_EQ(output.last_valid[1], T{0});
    EXPECT_TRUE(
        controllers::has_status_flag(output.status[0], controllers::StatusFlag::kResetApplied));
    for (std::size_t row = 1; row < batch; ++row) {
      expect_row(row,
                 before_reset.command[2 * row],
                 before_reset.command[2 * row + 1],
                 before_reset.status[row],
                 before_reset.valid[row],
                 before_reset.hold_count[row]);
      EXPECT_EQ(output.last_valid[2 * row], before_reset.last_valid[2 * row]);
      EXPECT_EQ(output.last_valid[2 * row + 1], before_reset.last_valid[2 * row + 1]);
    }
    position[0] = std::numeric_limits<T>::quiet_NaN();
    ASSERT_TRUE(step().ok());
    expect_row(0,
               T{0},
               T{0},
               failure | controllers::status_bit(controllers::StatusFlag::kNoValidOutput),
               0U,
               0U);
    expect_row(1, T{23}, T{-19}, 0U, 1U, 0U);
    expect_row(2, T{15}, T{-28}, 0U, 1U, 0U);
  }
}

TYPED_TEST(ControllersBehaviorTest, StatefulFilterRateLimitsInPlaceAndResetsOneRow) {
  using T = TypeParam;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  OutputFixture<2, T, 2> output;
  std::array<T, 4> previous{};
  std::array<std::uint8_t, 2> initialized{};
  const std::array<T, 2> lower{T{-2}, T{-3}}, upper{T{2}, T{3}}, rate{T{1}, T{2}};
  std::array<T, 2> dt{T{0.25}, T{0.5}};
  controllers::TorqueFilter::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
  storage.state = {controllers::write_view(previous, cpu, 2, 2),
                   controllers::write_view(initialized, cpu, 2)};
  controllers::TorqueFilter controller;
  ASSERT_TRUE(controller
                  .prepare({cpu,
                            controllers::dtype_for<T>(),
                            2,
                            2,
                            {controllers::SafeOutputPolicy::kHoldLastValid, 2}},
                           {controllers::shared(controllers::read_view(lower, cpu, 2)),
                            controllers::shared(controllers::read_view(upper, cpu, 2)),
                            controllers::shared(controllers::read_view(rate, cpu, 2))},
                           storage)
                  .ok());
  ASSERT_TRUE(controller.reset().ok());
  const auto step = [&]() {
    return controller.step(
        {controllers::read_view(output.command, cpu, 2, 2), controllers::read_view(dt, cpu, 2)});
  };
  output.command = {T{1}, T{-1}, T{-1}, T{1}};
  ASSERT_TRUE(step().ok());
  EXPECT_EQ(previous, output.command);
  output.command = {T{10}, T{-10}, T{10}, T{-10}};
  ASSERT_TRUE(step().ok());
  const std::array<T, 4> limited{T{1.25}, T{-1.5}, T{-0.5}, T{0}};
  EXPECT_EQ(output.command, limited);
  EXPECT_EQ(previous, limited);
  for (std::size_t row = 0; row < 2; ++row) {
    EXPECT_EQ(output.status[row], controllers::status_bit(controllers::StatusFlag::kSaturation));
    EXPECT_EQ(output.valid[row], 1U);
    EXPECT_EQ(output.hold_count[row], 0U);
    EXPECT_EQ(initialized[row], 1U);
  }
  dt[0] = T{0};
  output.command = {T{-1}, T{1}, T{-0.5}, T{0}};
  ASSERT_TRUE(step().ok());
  EXPECT_EQ(output.command, limited);
  EXPECT_EQ(previous, limited);
  EXPECT_EQ(output.hold_count[0], 1U);
  EXPECT_EQ(output.status[0],
            controllers::status_bit(controllers::StatusFlag::kInvalidDt) |
                controllers::status_bit(controllers::StatusFlag::kSafeMode) |
                controllers::status_bit(controllers::StatusFlag::kOutputHeld));
  EXPECT_EQ(output.status[1], 0U);
  const std::array<std::uint8_t, 2> mask{1U, 0U};
  ASSERT_TRUE(controller.reset({controllers::read_view(mask, cpu, 2)}).ok());
  EXPECT_EQ(initialized[0], 0U);
  EXPECT_EQ(initialized[1], 1U);
  EXPECT_EQ(previous[2], limited[2]);
  EXPECT_EQ(previous[3], limited[3]);
  EXPECT_EQ(output.valid[0], 0U);
  EXPECT_EQ(output.valid[1], 1U);
  EXPECT_EQ(output.hold_count[0], 0U);
  dt[0] = T{0.25};
  output.command = {T{-2}, T{3}, T{2}, T{-3}};
  ASSERT_TRUE(step().ok());
  const std::array<T, 4> recovered{T{-2}, T{3}, T{0}, T{-1}};
  EXPECT_EQ(output.command, recovered);
  EXPECT_EQ(previous, recovered);
  EXPECT_EQ(output.status[0], 0U);
  EXPECT_EQ(output.status[1], controllers::status_bit(controllers::StatusFlag::kSaturation));
  EXPECT_EQ(output.valid[0], 1U);
  EXPECT_EQ(output.hold_count[0], 0U);
}

TYPED_TEST(ControllersBehaviorTest, ForceDerivativeAndIntegralHistoryStayRowLocal) {
  using T = TypeParam;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  controllers::CartesianForceController::Config config;
  config.dtype = controllers::dtype_for<T>();
  config.batch_size = 2;
  config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
  config.safety = {controllers::SafeOutputPolicy::kHoldLastValid, 2};
  controllers::Requirements requirements;
  ASSERT_TRUE(controllers::CartesianForceController::requirements(config, &requirements).ok());
  controllers::test_support::DescriptorStorage<T> buffers(requirements);
  controllers::CartesianForceController::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) =
      controllers::test_support::bind_output(buffers, "", requirements.outputs[0].name);
  storage.state = {buffers.view("differentiator_previous_error"),
                   buffers.view("differentiator_initialized"),
                   buffers.view("integrator_previous_error"),
                   buffers.view("integrated_error"),
                   buffers.view("integrator_initialized")};
  ASSERT_TRUE(buffers.all_bound_once());
  std::array<T, 6> proportional, derivative, integral_gain, initial{};
  proportional.fill(T{2});
  derivative.fill(T{0.5});
  integral_gain.fill(T{0.25});
  controllers::CartesianForceController controller;
  ASSERT_TRUE(controller
                  .prepare(config,
                           {controllers::shared(controllers::read_view(proportional, cpu, 6)),
                            controllers::shared(controllers::read_view(derivative, cpu, 6)),
                            controllers::shared(controllers::read_view(integral_gain, cpu, 6)),
                            controllers::shared(controllers::read_view(initial, cpu, 6)),
                            {},
                            {}},
                           storage)
                  .ok());
  ASSERT_TRUE(controller.reset().ok());
  std::array<T, 12> external, desired, feedforward, pose{};
  external.fill(T{1});
  feedforward.fill(T{0.5});
  std::array<T, 2> dt{T{0.25}, T{0.5}};
  std::array<double, 2> previous_error{}, integrated{}, previous_output{};
  std::array<bool, 2> initialized{};
  auto& command = buffers.template values<T>(requirements.outputs[0].name);
  auto& status = buffers.template values<std::uint64_t>("status");
  auto& valid = buffers.template values<std::uint8_t>("valid_mask");
  auto& count = buffers.template values<std::uint64_t>("hold_count");
  for (int step = 0; step < 5; ++step) {
    SCOPED_TRACE(step);
    desired.fill(static_cast<T>(3 + 2 * step));
    dt[0] = step == 2 ? T{0} : T{0.25};
    if (step == 4) {
      const std::array<std::uint8_t, 2> mask{1U, 0U};
      ASSERT_TRUE(controller.reset({controllers::read_view(mask, cpu, 2)}).ok());
      initialized[0] = false;
      integrated[0] = 0;
      EXPECT_EQ(valid[0], 0U);
      EXPECT_EQ(valid[1], 1U);
      EXPECT_EQ(buffers.template values<std::uint8_t>("integrator_initialized")[0], 0U);
      EXPECT_EQ(buffers.template values<std::uint8_t>("integrator_initialized")[1], 1U);
    }
    ASSERT_TRUE(controller
                    .step({controllers::read_view(external, cpu, 2, 6),
                           controllers::read_view(desired, cpu, 2, 6),
                           controllers::read_view(feedforward, cpu, 2, 6),
                           controllers::read_view(pose, cpu, 2, 6),
                           controllers::read_view(dt, cpu, 2)})
                    .ok());
    for (std::size_t row = 0; row < 2; ++row) {
      const bool failed = row == 0 && step == 2;
      const double error = desired[6 * row] - 1.0;
      double expected = previous_output[row];
      if (!failed) {
        const double differential = initialized[row] ? (error - previous_error[row]) / dt[row] : 0;
        if (initialized[row]) integrated[row] += 0.5 * (error + previous_error[row]) * dt[row];
        expected = 2 * error + 0.5 * differential + 0.25 * integrated[row] + desired[6 * row] + 0.5;
        previous_error[row] = error;
        previous_output[row] = expected;
        initialized[row] = true;
      }
      EXPECT_EQ(status[row],
                failed ? controllers::status_bit(controllers::StatusFlag::kInvalidDt) |
                             controllers::status_bit(controllers::StatusFlag::kSafeMode) |
                             controllers::status_bit(controllers::StatusFlag::kOutputHeld)
                       : 0U);
      EXPECT_EQ(valid[row], 1U);
      EXPECT_EQ(count[row], failed ? 1U : 0U);
      for (std::size_t axis = 0; axis < 6; ++axis) {
        const std::size_t index = 6 * row + axis;
        EXPECT_NEAR(command[index], expected, behavior_tolerance<T>());
        EXPECT_NEAR(buffers.template values<T>("integrated_error")[index],
                    integrated[row],
                    behavior_tolerance<T>());
        EXPECT_EQ(buffers.template values<T>("differentiator_previous_error")[index],
                  previous_error[row]);
        EXPECT_EQ(buffers.template values<T>("integrator_previous_error")[index],
                  previous_error[row]);
      }
    }
  }
}

TYPED_TEST(ControllersBehaviorTest, AdmittanceNondiagonalMassMatchesStatefulTrapezoidReference) {
  using T = TypeParam;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  const controllers::ControllerConfig config{cpu,
                                             controllers::dtype_for<T>(),
                                             2,
                                             6,
                                             {controllers::SafeOutputPolicy::kHoldLastValid, 2},
                                             controllers::CartesianFrame::kRobotBase};
  controllers::Requirements requirements;
  ASSERT_TRUE(
      controllers::CartesianAdmittancePoseController::requirements(config, &requirements).ok());
  controllers::test_support::DescriptorStorage<T> buffers(requirements);
  controllers::CartesianAdmittancePoseController::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) =
      controllers::test_support::bind_output(buffers, "", requirements.outputs[0].name);
  storage.state = {buffers.view("admittance_pose"),
                   buffers.view("admittance_twist"),
                   buffers.view("pose_previous_signal"),
                   buffers.view("twist_previous_signal"),
                   buffers.view("pose_integrator_initialized"),
                   buffers.view("twist_integrator_initialized")};
  ASSERT_TRUE(buffers.all_bound_once());
  std::array<T, 36> stiffness{}, damping{}, mass{};
  std::array<T, 6> threshold{}, limit;
  limit.fill(T{100});
  for (std::size_t axis = 0; axis < 6; ++axis) {
    stiffness[7 * axis] = T{1};
    damping[7 * axis] = T{0.5};
    mass[7 * axis] = axis % 2 == 0 ? T{2} : T{1};
    mass[6 * axis + (axis ^ 1U)] = T{0.5};
  }
  controllers::CartesianAdmittancePoseController controller;
  ASSERT_TRUE(controller
                  .prepare(config,
                           {controllers::shared(controllers::read_view(stiffness, cpu, 6, 6)),
                            controllers::shared(controllers::read_view(damping, cpu, 6, 6)),
                            controllers::shared(controllers::read_view(mass, cpu, 6, 6)),
                            controllers::shared(controllers::read_view(threshold, cpu, 6)),
                            controllers::shared(controllers::read_view(limit, cpu, 6)),
                            controllers::shared(controllers::read_view(limit, cpu, 6)),
                            controllers::shared(controllers::read_view(limit, cpu, 6))},
                           storage)
                  .ok());
  ASSERT_TRUE(controller.reset().ok());
  std::array<T, 12> external, feedforward;
  external.fill(T{1});
  feedforward.fill(T{0.5});
  std::array<T, 2> dt{T{0.125}, T{0.25}};
  std::array<double, 12> pose{}, twist{}, previous_acceleration{}, last_output{};
  std::array<bool, 2> initialized{};
  auto& command = buffers.template values<T>(requirements.outputs[0].name);
  for (int step = 0; step < 5; ++step) {
    SCOPED_TRACE(step);
    dt[0] = step == 2 ? T{0} : T{0.125};
    if (step == 4) {
      const std::array<std::uint8_t, 2> mask{1U, 0U};
      ASSERT_TRUE(controller.reset({controllers::read_view(mask, cpu, 2)}).ok());
      initialized[0] = false;
      std::fill_n(pose.begin(), 6, 0.0);
      std::fill_n(twist.begin(), 6, 0.0);
      EXPECT_EQ(buffers.template values<std::uint8_t>("valid_mask")[0], 0U);
      EXPECT_EQ(buffers.template values<std::uint8_t>("valid_mask")[1], 1U);
    }
    ASSERT_TRUE(controller
                    .step({controllers::read_view(external, cpu, 2, 6),
                           controllers::read_view(feedforward, cpu, 2, 6),
                           controllers::read_view(dt, cpu, 2)})
                    .ok());
    for (std::size_t row = 0; row < 2; ++row) {
      const bool failed = row == 0 && step == 2;
      EXPECT_EQ(buffers.template values<std::uint64_t>("status")[row],
                failed ? controllers::status_bit(controllers::StatusFlag::kInvalidDt) |
                             controllers::status_bit(controllers::StatusFlag::kSafeMode) |
                             controllers::status_bit(controllers::StatusFlag::kOutputHeld)
                       : 0U);
      EXPECT_EQ(buffers.template values<std::uint8_t>("valid_mask")[row], 1U);
      EXPECT_EQ(buffers.template values<std::uint64_t>("hold_count")[row], failed ? 1U : 0U);
      for (std::size_t pair = 0; pair < 3; ++pair) {
        const std::size_t first = 6 * row + 2 * pair;
        const double f0 = 1.5 - pose[first] - 0.5 * twist[first];
        const double f1 = 1.5 - pose[first + 1] - 0.5 * twist[first + 1];
        const std::array<double, 2> acceleration{(f0 - 0.5 * f1) / 1.75,
                                                 (2 * f1 - 0.5 * f0) / 1.75};
        for (std::size_t axis = 0; axis < 2; ++axis) {
          const std::size_t index = first + axis;
          if (!failed) {
            if (initialized[row]) {
              const double old_twist = twist[index];
              twist[index] += 0.5 * dt[row] * (acceleration[axis] + previous_acceleration[index]);
              pose[index] += 0.5 * dt[row] * (twist[index] + old_twist);
            }
            previous_acceleration[index] = acceleration[axis];
            last_output[index] = acceleration[axis];
          }
          EXPECT_NEAR(command[index], last_output[index], behavior_tolerance<T>());
          EXPECT_NEAR(buffers.template values<T>("admittance_pose")[index],
                      pose[index],
                      behavior_tolerance<T>());
          EXPECT_NEAR(buffers.template values<T>("admittance_twist")[index],
                      twist[index],
                      behavior_tolerance<T>());
          EXPECT_NEAR(buffers.template values<T>("twist_previous_signal")[index],
                      previous_acceleration[index],
                      behavior_tolerance<T>());
        }
      }
      initialized[row] = true;
    }
  }
}

TYPED_TEST(ControllersBehaviorTest, OperationalMassMatchesNondiagonalAndIllConditionedResiduals) {
  using T = TypeParam;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  const controllers::ControllerConfig config{
      cpu, controllers::dtype_for<T>(), 3, 2, {}, controllers::CartesianFrame::kRobotBase, 2};
  controllers::Requirements requirements;
  ASSERT_TRUE(controllers::OperationalSpaceMass::requirements(config, &requirements).ok());
  controllers::test_support::DescriptorStorage<T> buffers(requirements);
  const controllers::OperationalSpaceMass::Storage storage{
      buffers.view("cartesian_mass_matrix"),
      buffers.status("parameter_status"),
      buffers.status("status"),
      {buffers.view("joint_mass_factor"),
       buffers.view("mass_inverse_jacobian_transpose"),
       buffers.view("cartesian_inverse_mass_factor")}};
  ASSERT_TRUE(buffers.all_bound_once());
  const std::array<T, 1> regularization{T{0.01}};
  std::array<T, 12> mass{
      T{4}, T{1}, T{1}, T{3}, T{1}, T{0.999}, T{0.999}, T{1}, T{-1}, T{0}, T{0}, T{1}};
  const std::array<T, 12> jacobian{
      T{1}, T{0}, T{0}, T{1}, T{1}, T{0}, T{0}, T{1}, T{1}, T{0}, T{0}, T{1}};
  controllers::OperationalSpaceMass controller;
  ASSERT_TRUE(
      controller
          .prepare(config,
                   {controllers::shared(controllers::read_view(regularization, cpu, 1)), {}},
                   storage)
          .ok());
  ASSERT_TRUE(controller.reset().ok());
  const auto step = [&]() {
    buffers.poison_scratch();
    return controller.step({controllers::read_view(mass, cpu, 3, 2, 2),
                            controllers::read_view(jacobian, cpu, 3, 2, 2)});
  };
  ASSERT_TRUE(step().ok());
  auto& actual = buffers.template values<T>("cartesian_mass_matrix");
  auto& status = buffers.template values<std::uint64_t>("status");
  for (std::size_t batch = 0; batch < 2; ++batch) {
    SCOPED_TRACE(batch);
    const double a = mass[4 * batch], b = mass[4 * batch + 1], d = mass[4 * batch + 3];
    const double determinant = a * d - b * b;
    const double r = regularization[0];
    const double denominator = 1.0 + r * (a + d) + r * r * determinant;
    const std::array<double, 4> expected{(a + r * determinant) / denominator,
                                         b / denominator,
                                         b / denominator,
                                         (d + r * determinant) / denominator};
    const std::array<double, 4> inverse{
        d / determinant + r, -b / determinant, -b / determinant, a / determinant + r};
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t col = 0; col < 2; ++col) {
        EXPECT_NEAR(actual[4 * batch + 2 * row + col],
                    expected[2 * row + col],
                    behavior_tolerance<T>() * 5);
        double residual = 0;
        for (std::size_t k = 0; k < 2; ++k) {
          residual += inverse[2 * row + k] * actual[4 * batch + 2 * k + col];
        }
        EXPECT_NEAR(residual, row == col ? 1.0 : 0.0, behavior_tolerance<T>() * 100);
      }
    }
    EXPECT_EQ(status[batch], 0U);
  }
  EXPECT_TRUE(controllers::has_status_flag(status[2], controllers::StatusFlag::kSafeMode));
  for (std::size_t i = 8; i < 12; ++i) EXPECT_EQ(actual[i], T{0});
  const auto before = actual;
  mass[8] = T{1};
  ASSERT_TRUE(step().ok());
  EXPECT_EQ(status[2], 0U);
  EXPECT_NEAR(actual[8], 1.0 / (1.0 + regularization[0]), behavior_tolerance<T>());
  EXPECT_EQ(actual[9], T{0});
  EXPECT_EQ(actual[10], T{0});
  EXPECT_NEAR(actual[11], actual[8], behavior_tolerance<T>());
  for (std::size_t i = 0; i < 8; ++i) EXPECT_EQ(actual[i], before[i]);
}

template <typename T, typename Controller>
void check_composite_mixed_batch_recovery() {
  controllers::test_support::CompositeFixture<T, Controller> fixture(2);
  controllers::test_support::CompositeFixture<T, Controller> reference(2);
  ASSERT_TRUE(reference.prepare().ok());
  ASSERT_TRUE(reference.reset().ok());
  ASSERT_TRUE(reference.step().ok());
  const auto& reference_command = reference.buffers.template values<T>("filtered_command");
  ASSERT_TRUE(fixture.buffers.all_bound_once());
  ASSERT_TRUE(fixture.prepare().ok());
  ASSERT_TRUE(fixture.reset().ok());
  fixture.buffers.poison_scratch();
  ASSERT_TRUE(fixture.step().ok());
  auto& command = fixture.buffers.template values<T>("filtered_command");
  auto& status = fixture.buffers.template values<std::uint64_t>("status");
  auto& valid = fixture.buffers.template values<std::uint8_t>("valid_mask");
  auto& count = fixture.buffers.template values<std::uint64_t>("hold_count");
  const auto baseline = command;
  for (const T value : baseline) EXPECT_TRUE(std::isfinite(value));
  EXPECT_GT(std::abs(baseline[0]), T{0.01});
  EXPECT_EQ(status[0], 0U);
  EXPECT_EQ(status[1], 0U);
  fixture.joint_bias[0] = std::numeric_limits<T>::quiet_NaN();
  for (std::uint64_t failure = 1; failure <= 3; ++failure) {
    fixture.buffers.poison_scratch();
    ASSERT_TRUE(reference.step().ok());
    ASSERT_TRUE(fixture.step().ok());
    EXPECT_TRUE(controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode));
    EXPECT_TRUE(controllers::has_status_flag(status[0],
                                             failure <= 2 ? controllers::StatusFlag::kOutputHeld
                                                          : controllers::StatusFlag::kHoldExpired));
    EXPECT_EQ(valid[0], 1U);
    EXPECT_EQ(count[0], failure);
    EXPECT_EQ(status[1], 0U);
    EXPECT_EQ(valid[1], 1U);
    EXPECT_EQ(count[1], 0U);
    for (std::size_t i = 0; i < fixture.config.joint_dim; ++i) {
      EXPECT_EQ(command[i], failure <= 2 ? baseline[i] : T{0});
      EXPECT_EQ(command[fixture.config.joint_dim + i],
                reference_command[fixture.config.joint_dim + i]);
    }
  }
  fixture.joint_bias[0] = T{0.1};
  ASSERT_TRUE(fixture.step().ok());
  EXPECT_EQ(status[0], 0U);
  EXPECT_EQ(valid[0], 1U);
  EXPECT_EQ(count[0], 0U);
  const auto before_reset = command;
  const std::array<std::uint8_t, 2> mask{1U, 0U};
  ASSERT_TRUE(
      fixture.controller.reset({controllers::read_view(mask, controllers::Device::kCpu, 2)}).ok());
  EXPECT_EQ(valid[0], 0U);
  EXPECT_EQ(valid[1], 1U);
  EXPECT_EQ(count[0], 0U);
  EXPECT_EQ(count[1], 0U);
  EXPECT_TRUE(controllers::has_status_flag(status[0], controllers::StatusFlag::kResetApplied));
  EXPECT_EQ(status[1], 0U);
  for (std::size_t i = fixture.config.joint_dim; i < command.size(); ++i) {
    EXPECT_EQ(command[i], before_reset[i]);
  }
  ASSERT_TRUE(fixture.step().ok());
  EXPECT_EQ(valid[0], 1U);
  EXPECT_EQ(status[0], 0U);
  for (std::size_t i = 0; i < fixture.config.joint_dim; ++i) {
    EXPECT_NEAR(command[i], baseline[i], behavior_tolerance<T>());
  }
}

TYPED_TEST(ControllersBehaviorTest, NonzeroCompositesRecoverAndResetSelectively) {
  check_composite_mixed_batch_recovery<TypeParam,
                                       controllers::CartesianImpedanceTorqueController>();
  check_composite_mixed_batch_recovery<TypeParam,
                                       controllers::VariableCartesianImpedanceTorqueController>();
  check_composite_mixed_batch_recovery<
      TypeParam,
      controllers::CartesianUnifiedForceImpedanceTorqueController>();
  check_composite_mixed_batch_recovery<TypeParam,
                                       controllers::BimanualCartesianImpedanceTorqueController>();
}

#ifndef NVIDIASRLCONTROLLERS_ENABLE_CUDA
TEST(ControllersTest, CpuOnlyBuildRejectsCudaDuringRequirements) {
  controllers::Requirements requirements;
  const controllers::Result result = controllers::JointImpedanceController::requirements(
      {controllers::Device::kCuda, controllers::DType::kFloat32, 1, 7, {}}, &requirements);
  EXPECT_EQ(result.code, controllers::ResultCode::kBackendUnavailable);
}
#else
struct CudaTestResources {
  cudaStream_t stream{nullptr};
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t executable{nullptr};
  bool capturing{false};
  std::vector<void*> allocations;

  CudaTestResources() = default;
  CudaTestResources(const CudaTestResources&) = delete;
  CudaTestResources& operator=(const CudaTestResources&) = delete;
  ~CudaTestResources() {
    if (capturing) cudaStreamEndCapture(stream, &graph);
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (executable != nullptr) cudaGraphExecDestroy(executable);
    if (graph != nullptr) cudaGraphDestroy(graph);
    for (void* pointer : allocations) cudaFree(pointer);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  template <typename T>
  cudaError_t allocate(T** pointer, std::size_t count) {
    const cudaError_t result = cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T));
    if (result == cudaSuccess) allocations.push_back(*pointer);
    return result;
  }
};

template <typename T>
void check_cuda_graph_replay() {
  constexpr controllers::Device cuda = controllers::Device::kCuda;
  constexpr controllers::Device cpu = controllers::Device::kCpu;
  CudaTestResources resources;
  ASSERT_EQ(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), cudaSuccess);
  const controllers::ExecutionContext context{resources.stream};
  T* scalars = nullptr;
  std::uint64_t* words = nullptr;
  std::uint8_t* flags = nullptr;
  ASSERT_EQ(resources.allocate(&scalars, 20), cudaSuccess);
  ASSERT_EQ(resources.allocate(&words, 6), cudaSuccess);
  ASSERT_EQ(resources.allocate(&flags, 4), cudaSuccess);
  T* command = scalars;
  T* history = scalars + 4;
  T* previous = scalars + 8;
  T* lower = scalars + 12;
  T* upper = scalars + 14;
  T* rate = scalars + 16;
  T* dt = scalars + 18;
  const std::array<T, 8> host_parameters{T{-3}, T{-3}, T{3}, T{3}, T{1}, T{2}, T{0.25}, T{0.5}};
  ASSERT_EQ(cudaMemcpyAsync(lower,
                            host_parameters.data(),
                            sizeof(host_parameters),
                            cudaMemcpyHostToDevice,
                            resources.stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(words, 0, 6 * sizeof(std::uint64_t), resources.stream), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(flags, 0, 4 * sizeof(std::uint8_t), resources.stream), cudaSuccess);
  controllers::TorqueFilter::Storage storage;
  static_cast<controllers::ControllerOutputStorage&>(storage) = {
      controllers::write_view(command, cuda, 2, 2),
      {words, 2, cuda},
      {words + 2, 2, cuda},
      {controllers::write_view(history, cuda, 2, 2),
       controllers::write_view(flags, cuda, 2),
       controllers::write_view(words + 4, cuda, 2)}};
  storage.state = {controllers::write_view(previous, cuda, 2, 2),
                   controllers::write_view(flags + 2, cuda, 2)};
  const controllers::TorqueFilter::Parameters parameters{
      controllers::shared(controllers::read_view(lower, cuda, 2)),
      controllers::shared(controllers::read_view(upper, cuda, 2)),
      controllers::shared(controllers::read_view(rate, cuda, 2))};
  const controllers::ControllerConfig config{
      cuda, controllers::dtype_for<T>(), 2, 2, {controllers::SafeOutputPolicy::kHoldLastValid, 2}};
  controllers::TorqueFilter controller;
  for (const bool reverse : {false, true}) {
    auto overlapping = storage;
    overlapping.command = controllers::write_view(command + (reverse ? 1 : 0), cuda, 2, 2);
    overlapping.safety.last_valid_output =
        controllers::write_view(command + (reverse ? 0 : 1), cuda, 2, 2);
    EXPECT_EQ(controller.prepare(config, parameters, overlapping, context).code,
              controllers::ResultCode::kInvalidView);
  }
  ASSERT_TRUE(controller.prepare(config, parameters, storage, context).ok());
  ASSERT_TRUE(controller.reset({}, context).ok());
  ASSERT_EQ(cudaStreamSynchronize(resources.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  resources.capturing = true;
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(command, cuda, 2, 2), controllers::read_view(dt, cuda, 2)},
                context)
          .ok());
  const cudaError_t capture_result = cudaStreamEndCapture(resources.stream, &resources.graph);
  resources.capturing = false;
  ASSERT_EQ(capture_result, cudaSuccess);
  ASSERT_NE(resources.graph, nullptr);
  std::size_t node_count = 0;
  ASSERT_EQ(cudaGraphGetNodes(resources.graph, nullptr, &node_count), cudaSuccess);
  EXPECT_GT(node_count, 0U);
  ASSERT_EQ(cudaGraphInstantiate(&resources.executable, resources.graph, nullptr, nullptr, 0),
            cudaSuccess);
  OutputFixture<2, T, 2> reference;
  std::array<T, 4> reference_previous{};
  std::array<std::uint8_t, 2> reference_initialized{};
  controllers::TorqueFilter::Storage reference_storage;
  static_cast<controllers::ControllerOutputStorage&>(reference_storage) = reference.view();
  reference_storage.state = {controllers::write_view(reference_previous, cpu, 2, 2),
                             controllers::write_view(reference_initialized, cpu, 2)};
  auto reference_config = config;
  reference_config.device = cpu;
  controllers::TorqueFilter reference_controller;
  ASSERT_TRUE(
      reference_controller
          .prepare(
              reference_config,
              {controllers::shared(controllers::read_view(host_parameters.data(), cpu, 2)),
               controllers::shared(controllers::read_view(host_parameters.data() + 2, cpu, 2)),
               controllers::shared(controllers::read_view(host_parameters.data() + 4, cpu, 2))},
              reference_storage)
          .ok());
  ASSERT_TRUE(reference_controller.reset().ok());
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const std::array<std::array<T, 4>, 6> requests{{{T{1}, T{-1}, T{-1}, T{1}},
                                                  {T{10}, T{-10}, T{10}, T{-10}},
                                                  {nan, T{0}, T{0}, T{0}},
                                                  {nan, T{0}, T{1}, T{-1}},
                                                  {nan, T{0}, T{2}, T{-2}},
                                                  {T{-2}, T{2}, T{0}, T{0}}}};
  const std::array<std::array<T, 4>, 6> expected{{{T{1}, T{-1}, T{-1}, T{1}},
                                                  {T{1.25}, T{-1.5}, T{-0.5}, T{0}},
                                                  {T{1.25}, T{-1.5}, T{0}, T{0}},
                                                  {T{1.25}, T{-1.5}, T{0.5}, T{-1}},
                                                  {T{0}, T{0}, T{1}, T{-2}},
                                                  {T{1}, T{-1}, T{0.5}, T{-1}}}};
  for (std::size_t replay = 0; replay < requests.size(); ++replay) {
    SCOPED_TRACE(replay);
    reference.command = requests[replay];
    ASSERT_TRUE(reference_controller
                    .step({controllers::read_view(reference.command, cpu, 2, 2),
                           controllers::read_view(host_parameters.data() + 6, cpu, 2)})
                    .ok());
    ASSERT_EQ(cudaMemcpyAsync(command,
                              requests[replay].data(),
                              sizeof(requests[replay]),
                              cudaMemcpyHostToDevice,
                              resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(resources.executable, resources.stream), cudaSuccess);
    std::array<T, 12> actual{};
    std::array<std::uint64_t, 6> actual_words{};
    std::array<std::uint8_t, 4> actual_flags{};
    ASSERT_EQ(cudaMemcpyAsync(
                  actual.data(), scalars, sizeof(actual), cudaMemcpyDeviceToHost, resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_words.data(),
                              words,
                              sizeof(actual_words),
                              cudaMemcpyDeviceToHost,
                              resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(actual_flags.data(),
                              flags,
                              sizeof(actual_flags),
                              cudaMemcpyDeviceToHost,
                              resources.stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(resources.stream), cudaSuccess);
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_NEAR(actual[i], expected[replay][i], behavior_tolerance<T>());
      EXPECT_EQ(actual[i], reference.command[i]);
      EXPECT_EQ(actual[4 + i], reference.last_valid[i]);
      EXPECT_EQ(actual[8 + i], reference_previous[i]);
    }
    for (std::size_t row = 0; row < 2; ++row) {
      EXPECT_EQ(actual_words[row], reference.parameter_status[row]);
      EXPECT_EQ(actual_words[2 + row], reference.status[row]);
      EXPECT_EQ(actual_words[4 + row], reference.hold_count[row]);
      EXPECT_EQ(actual_flags[row], reference.valid[row]);
      EXPECT_EQ(actual_flags[2 + row], reference_initialized[row]);
    }
  }
}

TEST(ControllersTest, CudaGraphReplaysOnCallerStream) {
  check_cuda_graph_replay<float>();
  check_cuda_graph_replay<double>();
}

TEST(ControllersTest, CudaUsesTheSameLifecycleOnACallerStream) {
  constexpr std::size_t batch_size = 2;
  const std::array<double, 4> host_stiffness{100.0, 0.0, 0.0, 200.0};
  const std::array<double, 2> host_damping{1.0, 1.0};
  const std::array<double, 4> host_position_error{0.1, -0.1, 0.2, -0.2};
  const std::array<double, 4> host_velocity_error{};
  const std::array<double, 4> host_desired_acceleration{};
  const std::array<double, 8> host_mass{1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  std::array<double, 4> host_torque{};
  std::array<std::uint64_t, batch_size> host_status{};

  double* stiffness = nullptr;
  double* damping = nullptr;
  double* position_error = nullptr;
  double* velocity_error = nullptr;
  double* desired_acceleration = nullptr;
  double* mass = nullptr;
  double* torque = nullptr;
  double* last_valid_torque = nullptr;
  std::uint8_t* valid = nullptr;
  std::uint64_t* hold_count = nullptr;
  std::uint64_t* parameter_status = nullptr;
  std::uint64_t* status = nullptr;
  cudaStream_t stream = nullptr;

  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&stiffness, sizeof(host_stiffness)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&damping, sizeof(host_damping)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&position_error, sizeof(host_position_error)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&velocity_error, sizeof(host_velocity_error)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&desired_acceleration, sizeof(host_desired_acceleration)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&mass, sizeof(host_mass)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&torque, sizeof(host_torque)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&last_valid_torque, sizeof(host_torque)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&valid, batch_size * sizeof(std::uint8_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&hold_count, batch_size * sizeof(std::uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&parameter_status, batch_size * sizeof(std::uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&status, batch_size * sizeof(std::uint64_t)), cudaSuccess);

  ASSERT_EQ(
      cudaMemcpyAsync(
          stiffness, host_stiffness.data(), sizeof(host_stiffness), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(
                damping, host_damping.data(), sizeof(host_damping), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(position_error,
                            host_position_error.data(),
                            sizeof(host_position_error),
                            cudaMemcpyHostToDevice,
                            stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(velocity_error,
                            host_velocity_error.data(),
                            sizeof(host_velocity_error),
                            cudaMemcpyHostToDevice,
                            stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(desired_acceleration,
                            host_desired_acceleration.data(),
                            sizeof(host_desired_acceleration),
                            cudaMemcpyHostToDevice,
                            stream),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(mass, host_mass.data(), sizeof(host_mass), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(parameter_status, 0, batch_size * sizeof(std::uint64_t), stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(status, 0, batch_size * sizeof(std::uint64_t), stream), cudaSuccess);

  const auto batch_vector = std::array<std::int64_t, 2>{batch_size, kJointCount};
  const auto batch_matrix = std::array<std::int64_t, 3>{batch_size, kJointCount, kJointCount};
  const auto shared_matrix = std::array<std::int64_t, 2>{kJointCount, kJointCount};
  const auto shared_vector = std::array<std::int64_t, 1>{kJointCount};
  const auto batch = std::array<std::int64_t, 1>{batch_size};
  controllers::JointImpedanceController controller;
  const controllers::JointImpedanceController::Parameters parameters{
      {controllers::read_view(stiffness, controllers::Device::kCuda, shared_matrix),
       controllers::ParameterBatching::kShared},
      {controllers::read_view(damping, controllers::Device::kCuda, shared_vector),
       controllers::ParameterBatching::kShared}};
  const controllers::JointImpedanceController::Storage storage{
      controllers::write_view(torque, controllers::Device::kCuda, batch_vector),
      {parameter_status, batch_size, controllers::Device::kCuda},
      {status, batch_size, controllers::Device::kCuda},
      {controllers::write_view(last_valid_torque, controllers::Device::kCuda, batch_vector),
       controllers::write_view(valid, controllers::Device::kCuda, batch),
       controllers::write_view(hold_count, controllers::Device::kCuda, batch)}};
  const controllers::ExecutionContext context{stream};
  ASSERT_TRUE(controller
                  .prepare({controllers::Device::kCuda,
                            controllers::DType::kFloat64,
                            batch_size,
                            kJointCount,
                            {}},
                           parameters,
                           storage,
                           context)
                  .ok());
  ASSERT_TRUE(controller.reset({}, context).ok());
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(position_error, controllers::Device::kCuda, batch_vector),
                 controllers::read_view(velocity_error, controllers::Device::kCuda, batch_vector),
                 controllers::read_view(
                     desired_acceleration, controllers::Device::kCuda, batch_vector),
                 controllers::read_view(mass, controllers::Device::kCuda, batch_matrix)},
                context)
          .ok());
  ASSERT_EQ(cudaMemcpyAsync(
                host_torque.data(), torque, sizeof(host_torque), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(
                host_status.data(), status, sizeof(host_status), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(host_status, (std::array<std::uint64_t, batch_size>{0, 0}));
  EXPECT_EQ(host_torque, (std::array<double, 4>{10.0, -20.0, 20.0, -40.0}));

  EXPECT_EQ(cudaFree(status), cudaSuccess);
  EXPECT_EQ(cudaFree(parameter_status), cudaSuccess);
  EXPECT_EQ(cudaFree(hold_count), cudaSuccess);
  EXPECT_EQ(cudaFree(valid), cudaSuccess);
  EXPECT_EQ(cudaFree(last_valid_torque), cudaSuccess);
  EXPECT_EQ(cudaFree(torque), cudaSuccess);
  EXPECT_EQ(cudaFree(mass), cudaSuccess);
  EXPECT_EQ(cudaFree(desired_acceleration), cudaSuccess);
  EXPECT_EQ(cudaFree(velocity_error), cudaSuccess);
  EXPECT_EQ(cudaFree(position_error), cudaSuccess);
  EXPECT_EQ(cudaFree(damping), cudaSuccess);
  EXPECT_EQ(cudaFree(stiffness), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(ControllersTest, CudaOperationalSpaceControllerComputesJointTorqueEquation) {
  constexpr std::size_t joint_dim = 7;
  constexpr std::size_t scalar_count = 4 * kCartesianDim + kCartesianDim * kCartesianDim +
                                       kCartesianDim * joint_dim + 3 * joint_dim +
                                       2 * kCartesianDim;
  double* values = nullptr;
  std::uint64_t* words = nullptr;
  std::uint8_t* valid = nullptr;
  ASSERT_EQ(cudaMallocManaged(&values, scalar_count * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&words, 3 * sizeof(std::uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&valid, sizeof(std::uint8_t)), cudaSuccess);
  std::fill_n(values, scalar_count, 0.0);
  std::fill_n(words, 3, 0U);
  valid[0] = 0U;

  std::size_t offset = 0;
  double* pose_error = values + offset;
  offset += kCartesianDim;
  double* velocity_error = values + offset;
  offset += kCartesianDim;
  double* desired_acceleration = values + offset;
  offset += kCartesianDim;
  double* jacobian_dot_joint_velocity = values + offset;
  offset += kCartesianDim;
  double* cartesian_mass = values + offset;
  offset += kCartesianDim * kCartesianDim;
  double* jacobian = values + offset;
  offset += kCartesianDim * joint_dim;
  double* bias = values + offset;
  offset += joint_dim;
  double* torque = values + offset;
  offset += joint_dim;
  double* last_valid = values + offset;
  offset += joint_dim;
  double* position_gain = values + offset;
  offset += kCartesianDim;
  double* velocity_gain = values + offset;

  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    pose_error[axis] = 1.0;
    velocity_error[axis] = 2.0;
    desired_acceleration[axis] = 1.0;
    jacobian_dot_joint_velocity[axis] = 2.0;
    position_gain[axis] = 1.0;
    velocity_gain[axis] = 1.0;
    cartesian_mass[axis * kCartesianDim + axis] = 1.0;
    jacobian[axis * joint_dim + axis] = 1.0;
  }
  std::fill_n(bias, joint_dim, 1.0);

  const controllers::ControllerConfig config{controllers::Device::kCuda,
                                             controllers::DType::kFloat64,
                                             1,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase};
  const controllers::OperationalSpaceController::Parameters parameters{
      controllers::shared(
          controllers::read_view(position_gain, controllers::Device::kCuda, kCartesianDim)),
      controllers::shared(
          controllers::read_view(velocity_gain, controllers::Device::kCuda, kCartesianDim))};
  const controllers::OperationalSpaceController::Storage storage{
      controllers::write_view(torque, controllers::Device::kCuda, 1, joint_dim),
      {words, 1, controllers::Device::kCuda},
      {words + 1, 1, controllers::Device::kCuda},
      controllers::make_safety_state(
          controllers::write_view(last_valid, controllers::Device::kCuda, 1, joint_dim),
          controllers::write_view(valid, controllers::Device::kCuda, 1),
          controllers::write_view(words + 2, controllers::Device::kCuda, 1))};
  controllers::OperationalSpaceController controller;
  ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(
      controller
          .step(
              {controllers::read_view(pose_error, controllers::Device::kCuda, 1, kCartesianDim),
               controllers::read_view(velocity_error, controllers::Device::kCuda, 1, kCartesianDim),
               controllers::read_view(
                   desired_acceleration, controllers::Device::kCuda, 1, kCartesianDim),
               controllers::read_view(
                   jacobian_dot_joint_velocity, controllers::Device::kCuda, 1, kCartesianDim),
               controllers::read_view(
                   cartesian_mass, controllers::Device::kCuda, 1, kCartesianDim, kCartesianDim),
               controllers::read_view(
                   jacobian, controllers::Device::kCuda, 1, kCartesianDim, joint_dim),
               controllers::read_view(bias, controllers::Device::kCuda, 1, joint_dim)})
          .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t joint = 0; joint < joint_dim; ++joint) {
    EXPECT_DOUBLE_EQ(torque[joint], joint < kCartesianDim ? 3.0 : 1.0);
  }
  EXPECT_EQ(words[1], 0U);

  EXPECT_EQ(cudaFree(valid), cudaSuccess);
  EXPECT_EQ(cudaFree(words), cudaSuccess);
  EXPECT_EQ(cudaFree(values), cudaSuccess);
}

TEST(ControllersTest, CudaNullspaceTorqueUsesTransposeProjector) {
  constexpr std::size_t joint_dim = 7;
  constexpr std::size_t scalar_count = 2 * joint_dim + joint_dim * joint_dim +
                                       kCartesianDim * joint_dim + 2 * joint_dim * joint_dim +
                                       2 * joint_dim;
  double* values = nullptr;
  std::uint64_t* words = nullptr;
  std::uint8_t* valid = nullptr;
  ASSERT_EQ(cudaMallocManaged(&values, scalar_count * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&words, 3 * sizeof(std::uint64_t)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&valid, sizeof(std::uint8_t)), cudaSuccess);
  std::fill_n(values, scalar_count, 0.0);
  std::fill_n(words, 3, 0U);
  valid[0] = 0U;

  std::size_t offset = 0;
  double* position_error = values + offset;
  offset += joint_dim;
  double* velocity = values + offset;
  offset += joint_dim;
  double* mass = values + offset;
  offset += joint_dim * joint_dim;
  double* jacobian = values + offset;
  offset += kCartesianDim * joint_dim;
  double* stiffness = values + offset;
  offset += joint_dim * joint_dim;
  double* damping = values + offset;
  offset += joint_dim * joint_dim;
  double* torque = values + offset;
  offset += joint_dim;
  double* last_valid = values + offset;

  for (std::size_t joint = 0; joint < joint_dim; ++joint) {
    position_error[joint] = 1.0;
    mass[joint * joint_dim + joint] = static_cast<double>(joint + 1U);
    stiffness[joint * joint_dim + joint] = 1.0;
    damping[joint * joint_dim + joint] = 1.0;
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    jacobian[row * joint_dim + row] = 1.0;
    jacobian[row * joint_dim + (joint_dim - 1U)] = 0.25 * static_cast<double>(row + 1U);
  }

  const controllers::ControllerConfig config{controllers::Device::kCuda,
                                             controllers::DType::kFloat64,
                                             1,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase};
  const controllers::NullspaceController::Parameters parameters{
      controllers::shared(
          controllers::read_view(stiffness, controllers::Device::kCuda, joint_dim, joint_dim)),
      controllers::shared(
          controllers::read_view(damping, controllers::Device::kCuda, joint_dim, joint_dim))};
  const controllers::NullspaceController::Storage storage{
      controllers::write_view(torque, controllers::Device::kCuda, 1, joint_dim),
      {words, 1, controllers::Device::kCuda},
      {words + 1, 1, controllers::Device::kCuda},
      controllers::make_safety_state(
          controllers::write_view(last_valid, controllers::Device::kCuda, 1, joint_dim),
          controllers::write_view(valid, controllers::Device::kCuda, 1),
          controllers::write_view(words + 2, controllers::Device::kCuda, 1))};
  controllers::NullspaceController controller;
  ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(position_error, controllers::Device::kCuda, 1, joint_dim),
                 controllers::read_view(velocity, controllers::Device::kCuda, 1, joint_dim),
                 controllers::read_view(mass, controllers::Device::kCuda, 1, joint_dim, joint_dim),
                 controllers::read_view(
                     jacobian, controllers::Device::kCuda, 1, kCartesianDim, joint_dim)})
          .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    double task_acceleration = 0.0;
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      task_acceleration +=
          jacobian[row * joint_dim + joint] * torque[joint] / mass[joint * joint_dim + joint];
    }
    EXPECT_NEAR(task_acceleration, 0.0, 1.0e-10);
  }
  EXPECT_EQ(words[1], 0U);

  EXPECT_EQ(cudaFree(valid), cudaSuccess);
  EXPECT_EQ(cudaFree(words), cudaSuccess);
  EXPECT_EQ(cudaFree(values), cudaSuccess);
}

TEST(ControllersTest, CudaOperationalSpaceMassAppliesRegularization) {
  constexpr std::size_t joint_dim = kCartesianDim;
  constexpr std::size_t matrix_elements = kCartesianDim * kCartesianDim;
  constexpr std::size_t scalar_count = 1 + 6 * matrix_elements;
  double* values = nullptr;
  std::uint64_t* words = nullptr;
  ASSERT_EQ(cudaMallocManaged(&values, scalar_count * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(&words, 2 * sizeof(std::uint64_t)), cudaSuccess);
  std::fill_n(values, scalar_count, 0.0);
  std::fill_n(words, 2, 0U);

  std::size_t offset = 0;
  double* regularization = values + offset++;
  double* mass = values + offset;
  offset += matrix_elements;
  double* jacobian = values + offset;
  offset += matrix_elements;
  double* cartesian_mass = values + offset;
  offset += matrix_elements;
  double* joint_factor = values + offset;
  offset += matrix_elements;
  double* solve_rhs = values + offset;
  offset += matrix_elements;
  double* cartesian_inverse = values + offset;
  regularization[0] = 1.0e-6;
  for (std::size_t axis = 0; axis < kCartesianDim; ++axis) {
    mass[axis * joint_dim + axis] = 1.0;
    jacobian[axis * joint_dim + axis] = 1.0;
  }

  const controllers::ControllerConfig config{controllers::Device::kCuda,
                                             controllers::DType::kFloat64,
                                             1,
                                             joint_dim,
                                             {},
                                             controllers::CartesianFrame::kRobotBase};
  const controllers::OperationalSpaceMass::Parameters parameters{
      controllers::shared(controllers::read_view(regularization, controllers::Device::kCuda, 1)),
      {}};
  const controllers::OperationalSpaceMass::Storage storage{
      controllers::write_view(
          cartesian_mass, controllers::Device::kCuda, 1, kCartesianDim, kCartesianDim),
      {words, 1, controllers::Device::kCuda},
      {words + 1, 1, controllers::Device::kCuda},
      {controllers::write_view(joint_factor, controllers::Device::kCuda, 1, joint_dim, joint_dim),
       controllers::write_view(solve_rhs, controllers::Device::kCuda, 1, joint_dim, kCartesianDim),
       controllers::write_view(
           cartesian_inverse, controllers::Device::kCuda, 1, kCartesianDim, kCartesianDim)}};
  controllers::OperationalSpaceMass controller;
  ASSERT_TRUE(controller.prepare(config, parameters, storage).ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(
      controller
          .step({controllers::read_view(mass, controllers::Device::kCuda, 1, joint_dim, joint_dim),
                 controllers::read_view(
                     jacobian, controllers::Device::kCuda, 1, kCartesianDim, joint_dim)})
          .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      const double expected = row == col ? 1.0 / (1.0 + regularization[0]) : 0.0;
      EXPECT_NEAR(cartesian_mass[row * kCartesianDim + col], expected, 1.0e-12);
    }
  }
  EXPECT_EQ(words[1], 0U);

  EXPECT_EQ(cudaFree(words), cudaSuccess);
  EXPECT_EQ(cudaFree(values), cudaSuccess);
}
#endif

}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
