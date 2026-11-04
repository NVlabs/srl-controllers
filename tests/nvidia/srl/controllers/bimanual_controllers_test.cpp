/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file bimanual_controllers_test.cpp
 * @brief Tests for coupled 12D bimanual task kinematics and dynamics.
 */

#include <gtest/gtest.h>

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/controllers.hpp"

namespace {
namespace controllers = nvidia::srl::controllers;

constexpr std::size_t kArmJointDim = 7;
constexpr std::size_t kJointDim = 14;
constexpr std::size_t kSpatialDim = 6;
constexpr std::size_t kTaskDim = 12;
constexpr double kPi = 3.14159265358979323846;

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
template <typename T>
class CudaManagedBuffer {
 public:
  explicit CudaManagedBuffer(const std::size_t size) : size_(size) {
    EXPECT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&data_), size_ * sizeof(T)), cudaSuccess);
  }
  CudaManagedBuffer(const CudaManagedBuffer&) = delete;
  CudaManagedBuffer& operator=(const CudaManagedBuffer&) = delete;
  ~CudaManagedBuffer() {
    if (data_ != nullptr) {
      EXPECT_EQ(cudaFree(data_), cudaSuccess);
    }
  }

  T* data() noexcept { return data_; }

  template <std::size_t Size>
  void assign(const std::array<T, Size>& values) {
    ASSERT_EQ(Size, size_);
    std::copy(values.begin(), values.end(), data_);
  }

 private:
  T* data_{nullptr};
  std::size_t size_{0};
};
#endif

template <typename T, std::size_t Size>
controllers::TensorView read_cpu(const std::array<T, Size>& values, const std::int64_t first) {
  return controllers::read_view(values, controllers::Device::kCpu, first);
}

template <typename T, std::size_t Size>
controllers::TensorView read_cpu(const std::array<T, Size>& values,
                                 const std::int64_t first,
                                 const std::int64_t second) {
  return controllers::read_view(values, controllers::Device::kCpu, first, second);
}

template <typename T, std::size_t Size>
controllers::TensorView read_cpu(const std::array<T, Size>& values,
                                 const std::int64_t first,
                                 const std::int64_t second,
                                 const std::int64_t third) {
  return controllers::read_view(values, controllers::Device::kCpu, first, second, third);
}

template <typename T, std::size_t Size>
controllers::MutableTensorView write_cpu(std::array<T, Size>& values, const std::int64_t first) {
  return controllers::write_view(values, controllers::Device::kCpu, first);
}

template <typename T, std::size_t Size>
controllers::MutableTensorView write_cpu(std::array<T, Size>& values,
                                         const std::int64_t first,
                                         const std::int64_t second) {
  return controllers::write_view(values, controllers::Device::kCpu, first, second);
}

template <typename T, std::size_t Size>
controllers::MutableTensorView write_cpu(std::array<T, Size>& values,
                                         const std::int64_t first,
                                         const std::int64_t second,
                                         const std::int64_t third) {
  return controllers::write_view(values, controllers::Device::kCpu, first, second, third);
}

std::array<double, 16> identity_pose() {
  return {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
}

std::array<double, 16> z_rotation_pose(const double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {cosine, -sine, 0.0, 0.0, sine, cosine, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
}

template <std::size_t Dimension>
std::array<double, Dimension * Dimension> identity_matrix() {
  std::array<double, Dimension * Dimension> values{};
  for (std::size_t index = 0; index < Dimension; ++index) {
    values[index * Dimension + index] = 1.0;
  }
  return values;
}

template <std::size_t TaskDimension, std::size_t JointDimension>
struct ImpedanceStorageFixture {
  std::array<double, TaskDimension> command{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  std::array<double, TaskDimension> last_valid{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<double, TaskDimension * TaskDimension> previous_mass{};
  std::array<double, TaskDimension * JointDimension> previous_jacobian{};
  std::array<std::uint8_t, 1> initialized{};

  controllers::CartesianImpedanceController::Storage storage() {
    return {
        {write_cpu(command, 1, TaskDimension),
         controllers::write_status(parameter_status, controllers::Device::kCpu),
         controllers::write_status(status, controllers::Device::kCpu),
         {write_cpu(last_valid, 1, TaskDimension), write_cpu(valid, 1), write_cpu(hold_count, 1)}},
        {write_cpu(previous_mass, 1, TaskDimension, TaskDimension),
         write_cpu(previous_jacobian, 1, TaskDimension, JointDimension),
         write_cpu(initialized, 1)}};
  }
};

struct KinematicsFixture {
  std::array<double, 16> desired_absolute_pose{identity_pose()};
  std::array<double, 16> arm1_tcp_pose{identity_pose()};
  std::array<double, 16> arm2_tcp_pose{identity_pose()};
  std::array<double, 16> desired_relative_pose{identity_pose()};
  std::array<double, kSpatialDim * kArmJointDim> arm1_jacobian{};
  std::array<double, kSpatialDim * kArmJointDim> arm2_jacobian{};
  std::array<double, kTaskDim> task_error{};
  std::array<double, kTaskDim * kJointDim> task_jacobian{};
  std::array<std::uint64_t, 1> status{};
  controllers::BimanualTaskKinematics kinematics;

  controllers::BimanualTaskConfig config() const {
    controllers::BimanualTaskConfig config;
    config.device = controllers::Device::kCpu;
    config.dtype = controllers::DType::kFloat64;
    config.batch_size = 1;
    config.joint_dim = kJointDim;
    config.cartesian_frame = controllers::CartesianFrame::kMainFrameOrientation;
    config.task_dim = kTaskDim;
    return config;
  }

  controllers::Result prepare(const controllers::BimanualTaskConfig& config) {
    return kinematics.prepare(config,
                              {write_cpu(task_error, 1, kTaskDim),
                               write_cpu(task_jacobian, 1, kTaskDim, kJointDim),
                               controllers::write_status(status, controllers::Device::kCpu)});
  }

  controllers::Result step() {
    return kinematics.step({read_cpu(desired_absolute_pose, 1, 4, 4),
                            read_cpu(arm1_tcp_pose, 1, 4, 4),
                            read_cpu(arm2_tcp_pose, 1, 4, 4),
                            read_cpu(desired_relative_pose, 1, 4, 4),
                            read_cpu(arm1_jacobian, 1, kSpatialDim, kArmJointDim),
                            read_cpu(arm2_jacobian, 1, kSpatialDim, kArmJointDim)});
  }
};

TEST(BimanualTaskKinematicsTest, UsesFullBaseTransformAndIgnoresMainFrameTranslation) {
  KinematicsFixture fixture;
  fixture.arm1_tcp_pose[3] = 0.5;
  fixture.desired_relative_pose[3] = 0.4;
  fixture.arm2_tcp_pose[3] = -0.3;

  controllers::BimanualTaskConfig config = fixture.config();
  config.coordination_mode = controllers::BimanualCoordinationMode::kCoupledRigidTransform;
  config.frames.arm1_T_arm2[3] = 1.0;
  config.frames.main_frame = controllers::BimanualMainFrame::kExplicit;
  config.frames.reference_T_main[3] = 42.0;
  config.frames.reference_T_main[7] = -17.0;

  ASSERT_TRUE(fixture.prepare(config).ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_NEAR(fixture.task_error[6], 0.2, 1.0e-12);
  EXPECT_EQ(fixture.status[0], 0U);

  const std::array<double, kTaskDim> first_error = fixture.task_error;
  const std::array<double, kTaskDim * kJointDim> first_jacobian = fixture.task_jacobian;
  config.frames.reference_T_main[3] = -100.0;
  config.frames.reference_T_main[7] = 300.0;
  ASSERT_TRUE(fixture.prepare(config).ok());
  ASSERT_TRUE(fixture.step().ok());
  EXPECT_EQ(fixture.task_error, first_error);
  EXPECT_EQ(fixture.task_jacobian, first_jacobian);
}

TEST(BimanualTaskKinematicsTest, RotatesBothTasksIntoMainAxesAndIncludesLeverArm) {
  KinematicsFixture fixture;
  fixture.desired_absolute_pose[3] = 1.0;
  fixture.arm1_tcp_pose[3] = 0.0;
  fixture.arm2_tcp_pose[3] = 1.0;
  fixture.arm1_jacobian[5 * kArmJointDim] = 1.0;

  controllers::BimanualTaskConfig config = fixture.config();
  config.coordination_mode = controllers::BimanualCoordinationMode::kCoupledRigidTransform;
  config.frames.main_frame = controllers::BimanualMainFrame::kExplicit;
  config.frames.reference_T_main = {
      0.0, -1.0, 0.0, 9.0, 1.0, 0.0, 0.0, -4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0};

  ASSERT_TRUE(fixture.prepare(config).ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_NEAR(fixture.task_error[0], 0.0, 1.0e-12);
  EXPECT_NEAR(fixture.task_error[1], -0.5, 1.0e-12);
  EXPECT_NEAR(fixture.task_jacobian[6 * kJointDim], -1.0, 1.0e-12);
  EXPECT_NEAR(fixture.task_jacobian[11 * kJointDim], -1.0, 1.0e-12);
}

TEST(BimanualTaskKinematicsTest, DecouplesRelativeTranslationFromArmRotation) {
  KinematicsFixture fixture;
  fixture.arm1_tcp_pose[3] = -0.5;
  fixture.arm2_tcp_pose[3] = 0.5;
  fixture.desired_relative_pose[3] = 1.0;
  fixture.arm1_jacobian[5 * kArmJointDim] = 1.0;
  fixture.arm2_jacobian[0] = 2.0;

  ASSERT_TRUE(fixture.prepare(fixture.config()).ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_NEAR(fixture.task_error[6], 0.0, 1.0e-12);
  EXPECT_NEAR(fixture.task_jacobian[6 * kJointDim], 0.0, 1.0e-12);
  EXPECT_NEAR(fixture.task_jacobian[6 * kJointDim + kArmJointDim], 2.0, 1.0e-12);
  EXPECT_NEAR(fixture.task_jacobian[11 * kJointDim], -1.0, 1.0e-12);
}

TEST(BimanualTaskKinematicsTest, DecoupledTranslationIsIsotropicAcrossAllAxes) {
  KinematicsFixture fixture;
  fixture.arm1_tcp_pose[3] = 0.1;
  fixture.arm1_tcp_pose[7] = 0.2;
  fixture.arm1_tcp_pose[11] = 0.3;
  fixture.arm2_tcp_pose[3] = 0.4;
  fixture.arm2_tcp_pose[7] = 0.8;
  fixture.arm2_tcp_pose[11] = 1.2;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    fixture.desired_absolute_pose[axis * 4 + 3] =
        0.5 * (fixture.arm1_tcp_pose[axis * 4 + 3] + fixture.arm2_tcp_pose[axis * 4 + 3]);
    fixture.desired_relative_pose[axis * 4 + 3] =
        fixture.arm2_tcp_pose[axis * 4 + 3] - fixture.arm1_tcp_pose[axis * 4 + 3] + 0.05;
    fixture.arm1_jacobian[axis * kArmJointDim + axis] = 1.0;
    fixture.arm2_jacobian[axis * kArmJointDim + axis] = 1.0;
  }
  ASSERT_TRUE(fixture.prepare(fixture.config()).ok());
  ASSERT_TRUE(fixture.step().ok());

  for (std::size_t axis = 0; axis < 3; ++axis) {
    EXPECT_NEAR(fixture.task_error[kSpatialDim + axis], 0.05, 1.0e-12);
    EXPECT_DOUBLE_EQ(fixture.task_jacobian[(kSpatialDim + axis) * kJointDim + axis], -1.0);
    EXPECT_DOUBLE_EQ(fixture.task_jacobian[(kSpatialDim + axis) * kJointDim + kArmJointDim + axis],
                     1.0);
  }
}

TEST(BimanualTaskKinematicsTest, TaskJacobianAndTransposePreserveVirtualWork) {
  KinematicsFixture fixture;
  for (std::size_t axis = 0; axis < kSpatialDim; ++axis) {
    fixture.arm1_jacobian[axis * kArmJointDim + axis] = 0.7 + 0.1 * axis;
    fixture.arm2_jacobian[axis * kArmJointDim + axis] = 1.3 - 0.05 * axis;
  }
  ASSERT_TRUE(fixture.prepare(fixture.config()).ok());
  ASSERT_TRUE(fixture.step().ok());
  std::array<double, kJointDim> joint_displacement{};
  std::array<double, kTaskDim> wrench{};
  for (std::size_t joint = 0; joint < kJointDim; ++joint) {
    joint_displacement[joint] = 0.01 * static_cast<double>(joint + 1);
  }
  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    wrench[axis] = -0.2 + 0.03 * static_cast<double>(axis);
  }
  double task_work = 0.0;
  double joint_work = 0.0;
  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    double task_displacement = 0.0;
    for (std::size_t joint = 0; joint < kJointDim; ++joint) {
      task_displacement +=
          fixture.task_jacobian[axis * kJointDim + joint] * joint_displacement[joint];
    }
    task_work += wrench[axis] * task_displacement;
  }
  for (std::size_t joint = 0; joint < kJointDim; ++joint) {
    double joint_torque = 0.0;
    for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
      joint_torque += fixture.task_jacobian[axis * kJointDim + joint] * wrench[axis];
    }
    joint_work += joint_torque * joint_displacement[joint];
  }
  EXPECT_NEAR(task_work, joint_work, 1.0e-14);
}

TEST(BimanualTaskKinematicsTest, AngularJacobianMatchesErrorDifferentialAtFiniteError) {
  constexpr double kArm1Angle = 0.4;
  constexpr double kArm2Angle = -0.2;
  constexpr double kAbsoluteError = 0.3;
  constexpr double kRelativeError = -0.25;
  constexpr double kStep = 1.0e-7;
  KinematicsFixture nominal;
  nominal.arm1_tcp_pose = z_rotation_pose(kArm1Angle);
  nominal.arm2_tcp_pose = z_rotation_pose(kArm2Angle);
  nominal.desired_absolute_pose = z_rotation_pose(0.5 * (kArm1Angle + kArm2Angle) + kAbsoluteError);
  nominal.desired_relative_pose = z_rotation_pose(kArm2Angle - kArm1Angle + kRelativeError);
  nominal.arm1_jacobian[5 * kArmJointDim] = 1.0;
  ASSERT_TRUE(nominal.prepare(nominal.config()).ok());
  ASSERT_TRUE(nominal.step().ok());

  KinematicsFixture perturbed;
  perturbed.arm1_tcp_pose = z_rotation_pose(kArm1Angle + kStep);
  perturbed.arm2_tcp_pose = nominal.arm2_tcp_pose;
  perturbed.desired_absolute_pose = nominal.desired_absolute_pose;
  perturbed.desired_relative_pose = nominal.desired_relative_pose;
  perturbed.arm1_jacobian = nominal.arm1_jacobian;
  ASSERT_TRUE(perturbed.prepare(perturbed.config()).ok());
  ASSERT_TRUE(perturbed.step().ok());

  const double absolute_error_rate = (perturbed.task_error[5] - nominal.task_error[5]) / kStep;
  const double relative_error_rate = (perturbed.task_error[11] - nominal.task_error[11]) / kStep;
  EXPECT_NEAR(absolute_error_rate, -nominal.task_jacobian[5 * kJointDim], 1.0e-7);
  EXPECT_NEAR(relative_error_rate, -nominal.task_jacobian[11 * kJointDim], 1.0e-7);
}

TEST(BimanualTaskKinematicsTest, UsesCanonicalSo3ErrorsForBothTasks) {
  KinematicsFixture fixture;
  fixture.desired_absolute_pose = {
      0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  fixture.desired_relative_pose = {
      0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  ASSERT_TRUE(fixture.prepare(fixture.config()).ok());
  ASSERT_TRUE(fixture.step().ok());

  EXPECT_NEAR(fixture.task_error[5], 0.5 * kPi, 1.0e-12);
  EXPECT_NEAR(fixture.task_error[11], -0.5 * kPi, 1.0e-12);
}

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
TEST(BimanualTaskKinematicsTest, RunsBothCoordinationModesOnCuda) {
  std::array<double, 16> host_desired_absolute_pose = identity_pose();
  std::array<double, 16> host_arm1_tcp_pose = identity_pose();
  std::array<double, 16> host_arm2_tcp_pose = identity_pose();
  const std::array<double, 16> host_desired_relative_pose = identity_pose();
  host_desired_absolute_pose[3] = 1.0;
  host_arm2_tcp_pose[3] = 1.0;
  std::array<double, kSpatialDim * kArmJointDim> host_arm1_jacobian{};
  const std::array<double, kSpatialDim * kArmJointDim> host_arm2_jacobian{};
  host_arm1_jacobian[5 * kArmJointDim] = 1.0;
  const std::array<double, kTaskDim> host_zero_error{};
  const std::array<double, kTaskDim * kJointDim> host_zero_task_jacobian{};
  const std::array<std::uint64_t, 1> host_zero_status{};

  CudaManagedBuffer<double> desired_absolute_pose(16);
  CudaManagedBuffer<double> arm1_tcp_pose(16);
  CudaManagedBuffer<double> arm2_tcp_pose(16);
  CudaManagedBuffer<double> desired_relative_pose(16);
  CudaManagedBuffer<double> arm1_jacobian(host_arm1_jacobian.size());
  CudaManagedBuffer<double> arm2_jacobian(host_arm2_jacobian.size());
  CudaManagedBuffer<double> task_error(host_zero_error.size());
  CudaManagedBuffer<double> task_jacobian(host_zero_task_jacobian.size());
  CudaManagedBuffer<std::uint64_t> status(1);
  desired_absolute_pose.assign(host_desired_absolute_pose);
  arm1_tcp_pose.assign(host_arm1_tcp_pose);
  arm2_tcp_pose.assign(host_arm2_tcp_pose);
  desired_relative_pose.assign(host_desired_relative_pose);
  arm1_jacobian.assign(host_arm1_jacobian);
  arm2_jacobian.assign(host_arm2_jacobian);
  task_error.assign(host_zero_error);
  task_jacobian.assign(host_zero_task_jacobian);
  status.assign(host_zero_status);

  const std::array<std::int64_t, 3> pose_shape{1, 4, 4};
  const std::array<std::int64_t, 3> arm_jacobian_shape{1, kSpatialDim, kArmJointDim};
  const std::array<std::int64_t, 2> task_error_shape{1, kTaskDim};
  const std::array<std::int64_t, 3> task_jacobian_shape{1, kTaskDim, kJointDim};
  controllers::BimanualTaskConfig config;
  config.device = controllers::Device::kCuda;
  config.dtype = controllers::DType::kFloat64;
  config.batch_size = 1;
  config.joint_dim = kJointDim;
  config.cartesian_frame = controllers::CartesianFrame::kMainFrameOrientation;
  config.task_dim = kTaskDim;
  config.frames.main_frame = controllers::BimanualMainFrame::kExplicit;
  config.coordination_mode = controllers::BimanualCoordinationMode::kCoupledRigidTransform;
  config.frames.reference_T_main = {
      0.0, -1.0, 0.0, 9.0, 1.0, 0.0, 0.0, -4.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0};
  controllers::BimanualTaskKinematics kinematics;
  ASSERT_TRUE(
      kinematics
          .prepare(config,
                   {controllers::write_view(
                        task_error.data(), controllers::Device::kCuda, task_error_shape),
                    controllers::write_view(
                        task_jacobian.data(), controllers::Device::kCuda, task_jacobian_shape),
                    {status.data(), 1, controllers::Device::kCuda}})
          .ok());
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  const controllers::ExecutionContext context{stream};
  ASSERT_TRUE(kinematics.reset({}, context).ok());
  const controllers::Result step_result = kinematics.step(
      {controllers::read_view(desired_absolute_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm1_tcp_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm2_tcp_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(desired_relative_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm1_jacobian.data(), controllers::Device::kCuda, arm_jacobian_shape),
       controllers::read_view(
           arm2_jacobian.data(), controllers::Device::kCuda, arm_jacobian_shape)},
      context);
  ASSERT_TRUE(step_result.ok()) << step_result.message;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_NEAR(task_error.data()[0], 0.0, 1.0e-12);
  EXPECT_NEAR(task_error.data()[1], -0.5, 1.0e-12);
  EXPECT_NEAR(task_jacobian.data()[6 * kJointDim], -1.0, 1.0e-12);
  EXPECT_NEAR(task_jacobian.data()[11 * kJointDim], -1.0, 1.0e-12);
  EXPECT_EQ(status.data()[0], 0U);

  config.coordination_mode = controllers::BimanualCoordinationMode::kDecoupledCooperative;
  controllers::BimanualTaskKinematics decoupled_kinematics;
  ASSERT_TRUE(
      decoupled_kinematics
          .prepare(config,
                   {controllers::write_view(
                        task_error.data(), controllers::Device::kCuda, task_error_shape),
                    controllers::write_view(
                        task_jacobian.data(), controllers::Device::kCuda, task_jacobian_shape),
                    {status.data(), 1, controllers::Device::kCuda}})
          .ok());
  ASSERT_TRUE(decoupled_kinematics.reset({}, context).ok());
  const controllers::Result decoupled_result = decoupled_kinematics.step(
      {controllers::read_view(desired_absolute_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm1_tcp_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm2_tcp_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(desired_relative_pose.data(), controllers::Device::kCuda, pose_shape),
       controllers::read_view(arm1_jacobian.data(), controllers::Device::kCuda, arm_jacobian_shape),
       controllers::read_view(
           arm2_jacobian.data(), controllers::Device::kCuda, arm_jacobian_shape)},
      context);
  ASSERT_TRUE(decoupled_result.ok()) << decoupled_result.message;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_NEAR(task_jacobian.data()[6 * kJointDim], 0.0, 1.0e-12);
  EXPECT_NEAR(task_jacobian.data()[11 * kJointDim], -1.0, 1.0e-12);
  EXPECT_EQ(status.data()[0], 0U);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
#endif

TEST(BimanualOperationalMassTest, SupportsAFullTwelveByFourteenTask) {
  controllers::ControllerConfig config{controllers::Device::kCpu,
                                       controllers::DType::kFloat64,
                                       1,
                                       kJointDim,
                                       {},
                                       controllers::CartesianFrame::kMainFrameOrientation,
                                       kTaskDim};
  const std::array<double, 1> regularization{0.0};
  const std::array<double, 1> minimum_pivot{1.0e-12};
  std::array<double, kJointDim * kJointDim> joint_mass{};
  std::array<double, kTaskDim * kJointDim> jacobian{};
  for (std::size_t index = 0; index < kJointDim; ++index) {
    joint_mass[index * kJointDim + index] = static_cast<double>(index + 1);
  }
  for (std::size_t index = 0; index < kTaskDim; ++index) {
    jacobian[index * kJointDim + index] = 1.0;
  }
  std::array<double, kTaskDim * kTaskDim> task_mass{};
  std::array<double, kJointDim * kJointDim> joint_factor{};
  std::array<double, kJointDim * kTaskDim> solve_rhs{};
  std::array<double, kTaskDim * kTaskDim> task_inverse{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  controllers::OperationalSpaceMass mass;

  ASSERT_TRUE(mass.prepare(config,
                           {controllers::shared(read_cpu(regularization, 1)),
                            controllers::shared(read_cpu(minimum_pivot, 1))},
                           {write_cpu(task_mass, 1, kTaskDim, kTaskDim),
                            controllers::write_status(parameter_status, controllers::Device::kCpu),
                            controllers::write_status(status, controllers::Device::kCpu),
                            {write_cpu(joint_factor, 1, kJointDim, kJointDim),
                             write_cpu(solve_rhs, 1, kJointDim, kTaskDim),
                             write_cpu(task_inverse, 1, kTaskDim, kTaskDim)}})
                  .ok());
  ASSERT_TRUE(mass.step({read_cpu(joint_mass, 1, kJointDim, kJointDim),
                         read_cpu(jacobian, 1, kTaskDim, kJointDim)})
                  .ok());

  for (std::size_t row = 0; row < kTaskDim; ++row) {
    for (std::size_t col = 0; col < kTaskDim; ++col) {
      const double expected = row == col ? static_cast<double>(row + 1) : 0.0;
      EXPECT_NEAR(task_mass[row * kTaskDim + col], expected, 1.0e-9);
    }
  }
  EXPECT_EQ(status[0], 0U);
}

TEST(BimanualImpedanceTest, AppliesCoupledAbsoluteAndRelativeStiffness) {
  controllers::ControllerConfig config{controllers::Device::kCpu,
                                       controllers::DType::kFloat64,
                                       1,
                                       kJointDim,
                                       {},
                                       controllers::CartesianFrame::kMainFrameOrientation,
                                       kTaskDim};
  std::array<double, kTaskDim * kTaskDim> stiffness{};
  stiffness[0] = 100.0;
  stiffness[6 * kTaskDim + 6] = 80.0;
  stiffness[6] = 20.0;
  stiffness[6 * kTaskDim] = 20.0;
  for (std::size_t index = 1; index < kTaskDim; ++index) {
    if (index != 6) {
      stiffness[index * kTaskDim + index] = 10.0;
    }
  }
  std::array<double, kTaskDim> damping_factors{};
  std::array<double, kTaskDim> pose_error{};
  pose_error[5] = 0.2;
  pose_error[6] = 0.1;
  std::array<double, kTaskDim> velocity_error{};
  std::array<double, kTaskDim> desired_acceleration{};
  std::array<double, kJointDim> joint_velocity{};
  const std::array<double, kTaskDim * kTaskDim> task_mass = identity_matrix<kTaskDim>();
  std::array<double, kTaskDim * kJointDim> jacobian{};
  for (std::size_t index = 0; index < kTaskDim; ++index) {
    jacobian[index * kJointDim + index] = 1.0;
  }
  const std::array<double, 1> dt{0.001};
  std::array<double, kTaskDim> wrench{};
  std::array<double, kTaskDim * kTaskDim> previous_mass{};
  std::array<double, kTaskDim * kJointDim> previous_jacobian{};
  std::array<std::uint8_t, 1> initialized{};
  std::array<double, kTaskDim> last_valid{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  controllers::BimanualCartesianImpedanceController controller;

  ASSERT_TRUE(controller
                  .prepare(config,
                           {controllers::shared(read_cpu(stiffness, kTaskDim, kTaskDim)),
                            controllers::shared(read_cpu(damping_factors, kTaskDim))},
                           {{write_cpu(wrench, 1, kTaskDim),
                             controllers::write_status(parameter_status, controllers::Device::kCpu),
                             controllers::write_status(status, controllers::Device::kCpu),
                             {write_cpu(last_valid, 1, kTaskDim),
                              write_cpu(valid, 1),
                              write_cpu(hold_count, 1)}},
                            {write_cpu(previous_mass, 1, kTaskDim, kTaskDim),
                             write_cpu(previous_jacobian, 1, kTaskDim, kJointDim),
                             write_cpu(initialized, 1)}})
                  .ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(controller
                  .step({read_cpu(pose_error, 1, kTaskDim),
                         read_cpu(velocity_error, 1, kTaskDim),
                         read_cpu(desired_acceleration, 1, kTaskDim),
                         read_cpu(joint_velocity, 1, kJointDim),
                         read_cpu(task_mass, 1, kTaskDim, kTaskDim),
                         read_cpu(jacobian, 1, kTaskDim, kJointDim),
                         read_cpu(dt, 1)})
                  .ok());

  EXPECT_NEAR(wrench[0], 2.0, 1.0e-9);
  EXPECT_NEAR(wrench[5], 2.0, 1.0e-9);
  EXPECT_NEAR(wrench[6], 8.0, 1.0e-9);
  EXPECT_EQ(status[0], 0U);
}

TEST(BimanualImpedanceTest, MatchesTheSingleArmLawForBothSixDimensionalTasks) {
  constexpr std::size_t kSingleTaskDim = 6;
  const controllers::ControllerConfig single_config{controllers::Device::kCpu,
                                                    controllers::DType::kFloat64,
                                                    1,
                                                    kJointDim,
                                                    {},
                                                    controllers::CartesianFrame::kRobotBase,
                                                    kSingleTaskDim};
  const controllers::ControllerConfig bimanual_config{
      controllers::Device::kCpu,
      controllers::DType::kFloat64,
      1,
      kJointDim,
      {},
      controllers::CartesianFrame::kMainFrameOrientation,
      kTaskDim};

  std::array<double, kSingleTaskDim * kSingleTaskDim> absolute_stiffness{};
  std::array<double, kSingleTaskDim * kSingleTaskDim> relative_stiffness{};
  std::array<double, kSingleTaskDim> absolute_damping_factors{};
  std::array<double, kSingleTaskDim> relative_damping_factors{};
  std::array<double, kSingleTaskDim> absolute_pose_error{};
  std::array<double, kSingleTaskDim> relative_pose_error{};
  std::array<double, kSingleTaskDim> absolute_velocity_error{};
  std::array<double, kSingleTaskDim> relative_velocity_error{};
  std::array<double, kSingleTaskDim> absolute_desired_acceleration{};
  std::array<double, kSingleTaskDim> relative_desired_acceleration{};
  std::array<double, kSingleTaskDim * kSingleTaskDim> absolute_mass{};
  std::array<double, kSingleTaskDim * kSingleTaskDim> relative_mass{};
  std::array<double, kSingleTaskDim * kJointDim> absolute_jacobian{};
  std::array<double, kSingleTaskDim * kJointDim> relative_jacobian{};
  std::array<double, kJointDim> joint_velocity{};
  const std::array<double, 1> dt{0.01};

  for (std::size_t axis = 0; axis < kSingleTaskDim; ++axis) {
    absolute_stiffness[axis * kSingleTaskDim + axis] = 30.0 + 4.0 * axis;
    relative_stiffness[axis * kSingleTaskDim + axis] = 45.0 + 3.0 * axis;
    absolute_damping_factors[axis] = 0.55 + 0.03 * axis;
    relative_damping_factors[axis] = 0.85 - 0.04 * axis;
    absolute_pose_error[axis] = 0.01 * static_cast<double>(axis + 1);
    relative_pose_error[axis] = -0.015 * static_cast<double>(axis + 1);
    absolute_velocity_error[axis] = -0.02 * static_cast<double>(axis + 1);
    relative_velocity_error[axis] = 0.012 * static_cast<double>(axis + 1);
    absolute_desired_acceleration[axis] = 0.08 * static_cast<double>(axis + 1);
    relative_desired_acceleration[axis] = -0.05 * static_cast<double>(axis + 1);
    absolute_mass[axis * kSingleTaskDim + axis] = 1.0 + 0.2 * axis;
    relative_mass[axis * kSingleTaskDim + axis] = 2.0 + 0.15 * axis;
    absolute_jacobian[axis * kJointDim + axis] = 1.0 + 0.1 * axis;
    relative_jacobian[axis * kJointDim + 7 + axis] = 0.9 + 0.08 * axis;
    joint_velocity[axis] = 0.1 * static_cast<double>(axis + 1);
    joint_velocity[7 + axis] = -0.07 * static_cast<double>(axis + 1);
  }

  std::array<double, kTaskDim * kTaskDim> bimanual_stiffness{};
  std::array<double, kTaskDim> bimanual_damping_factors{};
  std::array<double, kTaskDim> bimanual_pose_error{};
  std::array<double, kTaskDim> bimanual_velocity_error{};
  std::array<double, kTaskDim> bimanual_desired_acceleration{};
  std::array<double, kTaskDim * kTaskDim> bimanual_mass{};
  std::array<double, kTaskDim * kJointDim> bimanual_jacobian{};
  const auto assemble_bimanual_inputs = [&]() {
    bimanual_stiffness.fill(0.0);
    bimanual_mass.fill(0.0);
    bimanual_jacobian.fill(0.0);
    for (std::size_t row = 0; row < kSingleTaskDim; ++row) {
      bimanual_damping_factors[row] = absolute_damping_factors[row];
      bimanual_damping_factors[kSingleTaskDim + row] = relative_damping_factors[row];
      bimanual_pose_error[row] = absolute_pose_error[row];
      bimanual_pose_error[kSingleTaskDim + row] = relative_pose_error[row];
      bimanual_velocity_error[row] = absolute_velocity_error[row];
      bimanual_velocity_error[kSingleTaskDim + row] = relative_velocity_error[row];
      bimanual_desired_acceleration[row] = absolute_desired_acceleration[row];
      bimanual_desired_acceleration[kSingleTaskDim + row] = relative_desired_acceleration[row];
      for (std::size_t col = 0; col < kSingleTaskDim; ++col) {
        bimanual_stiffness[row * kTaskDim + col] = absolute_stiffness[row * kSingleTaskDim + col];
        bimanual_stiffness[(kSingleTaskDim + row) * kTaskDim + kSingleTaskDim + col] =
            relative_stiffness[row * kSingleTaskDim + col];
        bimanual_mass[row * kTaskDim + col] = absolute_mass[row * kSingleTaskDim + col];
        bimanual_mass[(kSingleTaskDim + row) * kTaskDim + kSingleTaskDim + col] =
            relative_mass[row * kSingleTaskDim + col];
      }
      for (std::size_t joint = 0; joint < kJointDim; ++joint) {
        bimanual_jacobian[row * kJointDim + joint] = absolute_jacobian[row * kJointDim + joint];
        bimanual_jacobian[(kSingleTaskDim + row) * kJointDim + joint] =
            relative_jacobian[row * kJointDim + joint];
      }
    }
  };
  assemble_bimanual_inputs();

  ImpedanceStorageFixture<kSingleTaskDim, kJointDim> absolute_storage;
  ImpedanceStorageFixture<kSingleTaskDim, kJointDim> relative_storage;
  ImpedanceStorageFixture<kTaskDim, kJointDim> bimanual_storage;
  controllers::CartesianImpedanceController absolute_controller;
  controllers::CartesianImpedanceController relative_controller;
  controllers::BimanualCartesianImpedanceController bimanual_controller;

  ASSERT_TRUE(
      absolute_controller
          .prepare(
              single_config,
              {controllers::shared(read_cpu(absolute_stiffness, kSingleTaskDim, kSingleTaskDim)),
               controllers::shared(read_cpu(absolute_damping_factors, kSingleTaskDim))},
              absolute_storage.storage())
          .ok());
  ASSERT_TRUE(
      relative_controller
          .prepare(
              single_config,
              {controllers::shared(read_cpu(relative_stiffness, kSingleTaskDim, kSingleTaskDim)),
               controllers::shared(read_cpu(relative_damping_factors, kSingleTaskDim))},
              relative_storage.storage())
          .ok());
  ASSERT_TRUE(bimanual_controller
                  .prepare(bimanual_config,
                           {controllers::shared(read_cpu(bimanual_stiffness, kTaskDim, kTaskDim)),
                            controllers::shared(read_cpu(bimanual_damping_factors, kTaskDim))},
                           bimanual_storage.storage())
                  .ok());
  ASSERT_TRUE(absolute_controller.reset().ok());
  ASSERT_TRUE(relative_controller.reset().ok());
  ASSERT_TRUE(bimanual_controller.reset().ok());

  const auto step_all_controllers = [&]() {
    EXPECT_TRUE(absolute_controller
                    .step({read_cpu(absolute_pose_error, 1, kSingleTaskDim),
                           read_cpu(absolute_velocity_error, 1, kSingleTaskDim),
                           read_cpu(absolute_desired_acceleration, 1, kSingleTaskDim),
                           read_cpu(joint_velocity, 1, kJointDim),
                           read_cpu(absolute_mass, 1, kSingleTaskDim, kSingleTaskDim),
                           read_cpu(absolute_jacobian, 1, kSingleTaskDim, kJointDim),
                           read_cpu(dt, 1)})
                    .ok());
    EXPECT_TRUE(relative_controller
                    .step({read_cpu(relative_pose_error, 1, kSingleTaskDim),
                           read_cpu(relative_velocity_error, 1, kSingleTaskDim),
                           read_cpu(relative_desired_acceleration, 1, kSingleTaskDim),
                           read_cpu(joint_velocity, 1, kJointDim),
                           read_cpu(relative_mass, 1, kSingleTaskDim, kSingleTaskDim),
                           read_cpu(relative_jacobian, 1, kSingleTaskDim, kJointDim),
                           read_cpu(dt, 1)})
                    .ok());
    EXPECT_TRUE(bimanual_controller
                    .step({read_cpu(bimanual_pose_error, 1, kTaskDim),
                           read_cpu(bimanual_velocity_error, 1, kTaskDim),
                           read_cpu(bimanual_desired_acceleration, 1, kTaskDim),
                           read_cpu(joint_velocity, 1, kJointDim),
                           read_cpu(bimanual_mass, 1, kTaskDim, kTaskDim),
                           read_cpu(bimanual_jacobian, 1, kTaskDim, kJointDim),
                           read_cpu(dt, 1)})
                    .ok());
  };
  const auto expect_matching_wrenches = [&]() {
    for (std::size_t axis = 0; axis < kSingleTaskDim; ++axis) {
      EXPECT_NEAR(bimanual_storage.command[axis], absolute_storage.command[axis], 1.0e-10);
      EXPECT_NEAR(
          bimanual_storage.command[kSingleTaskDim + axis], relative_storage.command[axis], 1.0e-10);
    }
    EXPECT_EQ(absolute_storage.status[0], 0U);
    EXPECT_EQ(relative_storage.status[0], 0U);
    EXPECT_EQ(bimanual_storage.status[0], 0U);
  };

  step_all_controllers();
  expect_matching_wrenches();

  for (std::size_t axis = 0; axis < kSingleTaskDim; ++axis) {
    absolute_mass[axis * kSingleTaskDim + axis] += 0.02 * static_cast<double>(axis + 1);
    relative_mass[axis * kSingleTaskDim + axis] += 0.03 * static_cast<double>(axis + 1);
    absolute_jacobian[axis * kJointDim + axis] += 0.001 * static_cast<double>(axis + 1);
    relative_jacobian[axis * kJointDim + 7 + axis] -= 0.0015 * static_cast<double>(axis + 1);
  }
  assemble_bimanual_inputs();
  step_all_controllers();
  expect_matching_wrenches();
}

TEST(BimanualNullspaceTest, AcceptsTheFullTwelveRowTaskJacobian) {
  controllers::ControllerConfig config{controllers::Device::kCpu,
                                       controllers::DType::kFloat64,
                                       1,
                                       kJointDim,
                                       {},
                                       controllers::CartesianFrame::kMainFrameOrientation,
                                       kTaskDim};
  std::array<double, kJointDim * kJointDim> stiffness{};
  std::array<double, kJointDim * kJointDim> damping{};
  std::array<double, kJointDim * kJointDim> mass = identity_matrix<kJointDim>();
  std::array<double, kTaskDim * kJointDim> jacobian{};
  std::array<double, kJointDim> position_error{};
  std::array<double, kJointDim> velocity{};
  std::array<double, kJointDim> torque{};
  std::array<double, kJointDim> last_valid{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  for (std::size_t index = 0; index < kJointDim; ++index) {
    stiffness[index * kJointDim + index] = 5.0;
    damping[index * kJointDim + index] = 1.0;
  }
  for (std::size_t index = 0; index < kTaskDim; ++index) {
    jacobian[index * kJointDim + index] = 1.0;
  }
  position_error[kJointDim - 1] = 0.2;
  controllers::NullspaceController controller;

  ASSERT_TRUE(controller
                  .prepare(config,
                           {controllers::shared(read_cpu(stiffness, kJointDim, kJointDim)),
                            controllers::shared(read_cpu(damping, kJointDim, kJointDim))},
                           {write_cpu(torque, 1, kJointDim),
                            controllers::write_status(parameter_status, controllers::Device::kCpu),
                            controllers::write_status(status, controllers::Device::kCpu),
                            {write_cpu(last_valid, 1, kJointDim),
                             write_cpu(valid, 1),
                             write_cpu(hold_count, 1)}})
                  .ok());
  ASSERT_TRUE(controller.reset().ok());
  ASSERT_TRUE(controller
                  .step({read_cpu(position_error, 1, kJointDim),
                         read_cpu(velocity, 1, kJointDim),
                         read_cpu(mass, 1, kJointDim, kJointDim),
                         read_cpu(jacobian, 1, kTaskDim, kJointDim)})
                  .ok());

  EXPECT_NEAR(torque[kJointDim - 1], 1.0, 1.0e-9);
  EXPECT_EQ(status[0], 0U);
}

TEST(BimanualNullspaceTest, CompleteControllerProjectsEachArmIndependently) {
  controllers::BimanualTaskConfig config;
  config.device = controllers::Device::kCpu;
  config.dtype = controllers::DType::kFloat64;
  config.batch_size = 1;
  config.joint_dim = kJointDim;
  config.cartesian_frame = controllers::CartesianFrame::kMainFrameOrientation;
  config.task_dim = kTaskDim;
  const std::array<double, 1> regularization{1.0e-6};
  const std::array<double, 1> minimum_pivot{1.0e-12};
  std::array<double, kTaskDim * kTaskDim> task_stiffness{};
  std::array<double, kTaskDim> damping_factors{};
  std::array<double, kJointDim * kJointDim> nullspace_stiffness{};
  std::array<double, kJointDim * kJointDim> nullspace_damping{};
  std::array<double, kJointDim> lower_limit{};
  std::array<double, kJointDim> upper_limit{};
  std::array<double, kJointDim> maximum_rate{};
  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    task_stiffness[axis * kTaskDim + axis] = 1.0;
    damping_factors[axis] = 1.0;
  }
  for (std::size_t joint = 0; joint < kJointDim; ++joint) {
    nullspace_stiffness[joint * kJointDim + joint] = 5.0;
    nullspace_damping[joint * kJointDim + joint] = 1.0;
    lower_limit[joint] = -1000.0;
    upper_limit[joint] = 1000.0;
    maximum_rate[joint] = 1.0e9;
  }
  std::array<double, kTaskDim> task_error{};
  std::array<double, kTaskDim * kJointDim> task_jacobian{};
  std::array<std::uint64_t, 1> kinematics_status{};
  std::array<double, kTaskDim * kTaskDim> task_mass{};
  std::array<double, kJointDim * kJointDim> joint_factor{};
  std::array<double, kJointDim * kTaskDim> solve_rhs{};
  std::array<double, kTaskDim * kTaskDim> task_inverse{};
  std::array<std::uint64_t, 1> mass_parameter_status{};
  std::array<std::uint64_t, 1> mass_status{};
  ImpedanceStorageFixture<kTaskDim, kJointDim> task_storage;
  std::array<double, kJointDim> nullspace_torque{};
  std::array<double, kJointDim> last_valid_nullspace{};
  std::array<std::uint8_t, 1> nullspace_valid{};
  std::array<std::uint64_t, 1> nullspace_hold_count{};
  std::array<std::uint64_t, 1> nullspace_parameter_status{};
  std::array<std::uint64_t, 1> nullspace_status{};
  std::array<double, kJointDim> unfiltered_torque{};
  std::array<double, kJointDim> filtered_torque{};
  std::array<double, kJointDim> previous_torque{};
  std::array<double, kJointDim> last_valid_torque{};
  std::array<std::uint8_t, 1> filter_initialized{};
  std::array<std::uint8_t, 1> filter_valid{};
  std::array<std::uint64_t, 1> filter_hold_count{};
  std::array<std::uint64_t, 1> filter_parameter_status{};
  std::array<std::uint64_t, 1> filter_status{};
  std::array<double, kTaskDim> task_velocity_error{};
  controllers::BimanualCartesianImpedanceTorqueController controller;
  const controllers::Result prepare_result = controller.prepare(
      config,
      {{controllers::shared(read_cpu(regularization, 1)),
        controllers::shared(read_cpu(minimum_pivot, 1))},
       {controllers::shared(read_cpu(task_stiffness, kTaskDim, kTaskDim)),
        controllers::shared(read_cpu(damping_factors, kTaskDim))},
       {controllers::shared(read_cpu(nullspace_stiffness, kJointDim, kJointDim)),
        controllers::shared(read_cpu(nullspace_damping, kJointDim, kJointDim))},
       {controllers::shared(read_cpu(lower_limit, kJointDim)),
        controllers::shared(read_cpu(upper_limit, kJointDim)),
        controllers::shared(read_cpu(maximum_rate, kJointDim))}},
      {{write_cpu(task_error, 1, kTaskDim),
        write_cpu(task_jacobian, 1, kTaskDim, kJointDim),
        controllers::write_status(kinematics_status, controllers::Device::kCpu)},
       {write_cpu(task_mass, 1, kTaskDim, kTaskDim),
        controllers::write_status(mass_parameter_status, controllers::Device::kCpu),
        controllers::write_status(mass_status, controllers::Device::kCpu),
        {write_cpu(joint_factor, 1, kJointDim, kJointDim),
         write_cpu(solve_rhs, 1, kJointDim, kTaskDim),
         write_cpu(task_inverse, 1, kTaskDim, kTaskDim)}},
       task_storage.storage(),
       {write_cpu(nullspace_torque, 1, kJointDim),
        controllers::write_status(nullspace_parameter_status, controllers::Device::kCpu),
        controllers::write_status(nullspace_status, controllers::Device::kCpu),
        {write_cpu(last_valid_nullspace, 1, kJointDim),
         write_cpu(nullspace_valid, 1),
         write_cpu(nullspace_hold_count, 1)}},
       {write_cpu(unfiltered_torque, 1, kJointDim),
        {{write_cpu(filtered_torque, 1, kJointDim),
          controllers::write_status(filter_parameter_status, controllers::Device::kCpu),
          controllers::write_status(filter_status, controllers::Device::kCpu),
          {write_cpu(last_valid_torque, 1, kJointDim),
           write_cpu(filter_valid, 1),
           write_cpu(filter_hold_count, 1)}},
         {write_cpu(previous_torque, 1, kJointDim), write_cpu(filter_initialized, 1)}}},
       write_cpu(task_velocity_error, 1, kTaskDim)});
  ASSERT_TRUE(prepare_result.ok()) << prepare_result.message;
  ASSERT_TRUE(controller.reset().ok());

  const std::array<double, 16> identity = identity_pose();
  std::array<double, kSpatialDim * kArmJointDim> arm1_jacobian{};
  std::array<double, kSpatialDim * kArmJointDim> arm2_jacobian{};
  for (std::size_t axis = 0; axis < kSpatialDim; ++axis) {
    arm1_jacobian[axis * kArmJointDim + axis] = 1.0;
    arm2_jacobian[axis * kArmJointDim + axis] = 1.0;
  }
  std::array<double, kJointDim> position_error{};
  position_error[kArmJointDim - 1] = 0.2;
  position_error[kJointDim - 1] = -0.4;
  const std::array<double, kJointDim> zero_joint{};
  const std::array<double, kTaskDim> zero_task{};
  std::array<double, kTaskDim> desired_task_velocity{};
  desired_task_velocity[0] = 0.1;
  const std::array<double, kJointDim * kJointDim> joint_mass = identity_matrix<kJointDim>();
  const std::array<double, 1> dt{0.001};
  const controllers::Result step_result =
      controller.step({{read_cpu(identity, 1, 4, 4),
                        read_cpu(identity, 1, 4, 4),
                        read_cpu(identity, 1, 4, 4),
                        read_cpu(identity, 1, 4, 4),
                        read_cpu(arm1_jacobian, 1, kSpatialDim, kArmJointDim),
                        read_cpu(arm2_jacobian, 1, kSpatialDim, kArmJointDim)},
                       read_cpu(desired_task_velocity, 1, kTaskDim),
                       read_cpu(zero_task, 1, kTaskDim),
                       read_cpu(position_error, 1, kJointDim),
                       read_cpu(zero_joint, 1, kJointDim),
                       read_cpu(joint_mass, 1, kJointDim, kJointDim),
                       read_cpu(zero_joint, 1, kJointDim),
                       read_cpu(dt, 1)});
  ASSERT_TRUE(step_result.ok()) << step_result.message;
  EXPECT_NEAR(filtered_torque[kArmJointDim - 1], 1.0, 1.0e-9);
  EXPECT_NEAR(filtered_torque[kJointDim - 1], -2.0, 1.0e-9);
  EXPECT_NEAR(task_velocity_error[0], desired_task_velocity[0], 1.0e-12);
  for (std::size_t joint = 1; joint < kArmJointDim - 1; ++joint) {
    EXPECT_NEAR(filtered_torque[joint], 0.0, 1.0e-9);
    EXPECT_NEAR(filtered_torque[kArmJointDim + joint], 0.0, 1.0e-9);
  }

  nullspace_stiffness[kArmJointDim] = 1.0;
  ASSERT_TRUE(controller
                  .step({{read_cpu(identity, 1, 4, 4),
                          read_cpu(identity, 1, 4, 4),
                          read_cpu(identity, 1, 4, 4),
                          read_cpu(identity, 1, 4, 4),
                          read_cpu(arm1_jacobian, 1, kSpatialDim, kArmJointDim),
                          read_cpu(arm2_jacobian, 1, kSpatialDim, kArmJointDim)},
                         read_cpu(zero_task, 1, kTaskDim),
                         read_cpu(zero_task, 1, kTaskDim),
                         read_cpu(position_error, 1, kJointDim),
                         read_cpu(zero_joint, 1, kJointDim),
                         read_cpu(joint_mass, 1, kJointDim, kJointDim),
                         read_cpu(zero_joint, 1, kJointDim),
                         read_cpu(dt, 1)})
                  .ok());
  EXPECT_NE(
      nullspace_status[0] & controllers::status_bit(controllers::StatusFlag::kInvalidParameter),
      0U);
  EXPECT_NE(nullspace_status[0] & controllers::status_bit(controllers::StatusFlag::kSafeMode), 0U);
}

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
TEST(BimanualNullspaceTest, RunsTheFullTwelveRowTaskOnCuda) {
  std::array<double, kJointDim * kJointDim> host_stiffness{};
  std::array<double, kJointDim * kJointDim> host_damping{};
  const std::array<double, kJointDim * kJointDim> host_mass = identity_matrix<kJointDim>();
  std::array<double, kTaskDim * kJointDim> host_jacobian{};
  std::array<double, kJointDim> host_position_error{};
  const std::array<double, kJointDim> host_zero_vector{};
  const std::array<std::uint8_t, 1> host_zero_byte{};
  const std::array<std::uint64_t, 1> host_zero_status{};
  for (std::size_t index = 0; index < kJointDim; ++index) {
    host_stiffness[index * kJointDim + index] = 5.0;
    host_damping[index * kJointDim + index] = 1.0;
  }
  for (std::size_t index = 0; index < kTaskDim; ++index) {
    host_jacobian[index * kJointDim + index] = 1.0;
  }
  host_position_error[kJointDim - 1] = 0.2;

  CudaManagedBuffer<double> stiffness(host_stiffness.size());
  CudaManagedBuffer<double> damping(host_damping.size());
  CudaManagedBuffer<double> mass(host_mass.size());
  CudaManagedBuffer<double> jacobian(host_jacobian.size());
  CudaManagedBuffer<double> position_error(host_position_error.size());
  CudaManagedBuffer<double> velocity(host_zero_vector.size());
  CudaManagedBuffer<double> torque(host_zero_vector.size());
  CudaManagedBuffer<double> last_valid(host_zero_vector.size());
  CudaManagedBuffer<std::uint8_t> valid(1);
  CudaManagedBuffer<std::uint64_t> hold_count(1);
  CudaManagedBuffer<std::uint64_t> parameter_status(1);
  CudaManagedBuffer<std::uint64_t> status(1);
  stiffness.assign(host_stiffness);
  damping.assign(host_damping);
  mass.assign(host_mass);
  jacobian.assign(host_jacobian);
  position_error.assign(host_position_error);
  velocity.assign(host_zero_vector);
  torque.assign(host_zero_vector);
  last_valid.assign(host_zero_vector);
  valid.assign(host_zero_byte);
  hold_count.assign(host_zero_status);
  parameter_status.assign(host_zero_status);
  status.assign(host_zero_status);

  const std::array<std::int64_t, 2> joint_matrix_shape{kJointDim, kJointDim};
  const std::array<std::int64_t, 2> batch_joint_shape{1, kJointDim};
  const std::array<std::int64_t, 3> batch_joint_matrix_shape{1, kJointDim, kJointDim};
  const std::array<std::int64_t, 3> batch_jacobian_shape{1, kTaskDim, kJointDim};
  const std::array<std::int64_t, 1> batch_shape{1};
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  const controllers::ExecutionContext context{stream};
  const controllers::ControllerConfig config{controllers::Device::kCuda,
                                             controllers::DType::kFloat64,
                                             1,
                                             kJointDim,
                                             {},
                                             controllers::CartesianFrame::kMainFrameOrientation,
                                             kTaskDim};
  controllers::NullspaceController controller;
  const controllers::Result prepare_result = controller.prepare(
      config,
      {controllers::shared(controllers::read_view(
           stiffness.data(), controllers::Device::kCuda, joint_matrix_shape)),
       controllers::shared(
           controllers::read_view(damping.data(), controllers::Device::kCuda, joint_matrix_shape))},
      {controllers::write_view(torque.data(), controllers::Device::kCuda, batch_joint_shape),
       {parameter_status.data(), 1, controllers::Device::kCuda},
       {status.data(), 1, controllers::Device::kCuda},
       {controllers::write_view(last_valid.data(), controllers::Device::kCuda, batch_joint_shape),
        controllers::write_view(valid.data(), controllers::Device::kCuda, batch_shape),
        controllers::write_view(hold_count.data(), controllers::Device::kCuda, batch_shape)}},
      context);
  ASSERT_TRUE(prepare_result.ok()) << prepare_result.message;
  ASSERT_TRUE(controller.reset({}, context).ok());
  const controllers::Result step_result = controller.step(
      {controllers::read_view(position_error.data(), controllers::Device::kCuda, batch_joint_shape),
       controllers::read_view(velocity.data(), controllers::Device::kCuda, batch_joint_shape),
       controllers::read_view(mass.data(), controllers::Device::kCuda, batch_joint_matrix_shape),
       controllers::read_view(jacobian.data(), controllers::Device::kCuda, batch_jacobian_shape)},
      context);
  ASSERT_TRUE(step_result.ok()) << step_result.message;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_NEAR(torque.data()[kJointDim - 1], 1.0, 1.0e-9);
  EXPECT_EQ(status.data()[0], 0U);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(BimanualImpedanceTest, MatchesTheFullImpedanceLawOnCuda) {
  std::array<double, kTaskDim * kTaskDim> host_stiffness{};
  std::array<double, kTaskDim> host_damping{};
  std::array<double, kTaskDim> host_pose_error{};
  std::array<double, kTaskDim> host_velocity_error{};
  std::array<double, kTaskDim> host_desired_acceleration{};
  std::array<double, kJointDim> host_joint_velocity{};
  std::array<double, kTaskDim * kTaskDim> host_task_mass{};
  std::array<double, kTaskDim * kJointDim> host_jacobian{};
  const std::array<double, 1> host_dt{0.01};
  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    host_stiffness[axis * kTaskDim + axis] = 25.0 + 2.0 * axis;
    host_damping[axis] = 0.6 + 0.02 * axis;
    host_pose_error[axis] = 0.01 * static_cast<double>(axis + 1);
    host_velocity_error[axis] = -0.008 * static_cast<double>(axis + 1);
    host_desired_acceleration[axis] = 0.03 * static_cast<double>(axis + 1);
    host_task_mass[axis * kTaskDim + axis] = 1.2 + 0.1 * axis;
    const std::size_t joint = axis < kSpatialDim ? axis : 7 + axis - kSpatialDim;
    host_jacobian[axis * kJointDim + joint] = 0.8 + 0.04 * axis;
    host_joint_velocity[joint] = 0.05 * static_cast<double>(axis + 1);
  }
  const std::array<double, kTaskDim> host_zero_wrench{};
  const std::array<double, kTaskDim * kTaskDim> host_zero_task_matrix{};
  const std::array<double, kTaskDim * kJointDim> host_zero_jacobian{};
  const std::array<std::uint8_t, 1> host_zero_byte{};
  const std::array<std::uint64_t, 1> host_zero_status{};

  CudaManagedBuffer<double> stiffness(host_stiffness.size());
  CudaManagedBuffer<double> damping(host_damping.size());
  CudaManagedBuffer<double> pose_error(host_pose_error.size());
  CudaManagedBuffer<double> velocity_error(host_velocity_error.size());
  CudaManagedBuffer<double> desired_acceleration(host_desired_acceleration.size());
  CudaManagedBuffer<double> joint_velocity(host_joint_velocity.size());
  CudaManagedBuffer<double> task_mass(host_task_mass.size());
  CudaManagedBuffer<double> jacobian(host_jacobian.size());
  CudaManagedBuffer<double> dt(host_dt.size());
  CudaManagedBuffer<double> wrench(host_zero_wrench.size());
  CudaManagedBuffer<double> last_valid(host_zero_wrench.size());
  CudaManagedBuffer<double> previous_mass(host_zero_task_matrix.size());
  CudaManagedBuffer<double> previous_jacobian(host_zero_jacobian.size());
  CudaManagedBuffer<std::uint8_t> valid(1);
  CudaManagedBuffer<std::uint8_t> initialized(1);
  CudaManagedBuffer<std::uint64_t> hold_count(1);
  CudaManagedBuffer<std::uint64_t> parameter_status(1);
  CudaManagedBuffer<std::uint64_t> status(1);
  stiffness.assign(host_stiffness);
  damping.assign(host_damping);
  pose_error.assign(host_pose_error);
  velocity_error.assign(host_velocity_error);
  desired_acceleration.assign(host_desired_acceleration);
  joint_velocity.assign(host_joint_velocity);
  task_mass.assign(host_task_mass);
  jacobian.assign(host_jacobian);
  dt.assign(host_dt);
  wrench.assign(host_zero_wrench);
  last_valid.assign(host_zero_wrench);
  previous_mass.assign(host_zero_task_matrix);
  previous_jacobian.assign(host_zero_jacobian);
  valid.assign(host_zero_byte);
  initialized.assign(host_zero_byte);
  hold_count.assign(host_zero_status);
  parameter_status.assign(host_zero_status);
  status.assign(host_zero_status);

  const std::array<std::int64_t, 1> task_vector_shape{kTaskDim};
  const std::array<std::int64_t, 2> batch_task_shape{1, kTaskDim};
  const std::array<std::int64_t, 2> task_matrix_shape{kTaskDim, kTaskDim};
  const std::array<std::int64_t, 2> batch_joint_shape{1, kJointDim};
  const std::array<std::int64_t, 3> batch_task_matrix_shape{1, kTaskDim, kTaskDim};
  const std::array<std::int64_t, 3> batch_jacobian_shape{1, kTaskDim, kJointDim};
  const std::array<std::int64_t, 1> batch_shape{1};
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  controllers::BimanualCartesianImpedanceController controller;
  const controllers::ControllerConfig config{controllers::Device::kCuda,
                                             controllers::DType::kFloat64,
                                             1,
                                             kJointDim,
                                             {},
                                             controllers::CartesianFrame::kMainFrameOrientation,
                                             kTaskDim};
  const controllers::ExecutionContext context{stream};
  const controllers::Result prepare_result = controller.prepare(
      config,
      {controllers::shared(
           controllers::read_view(stiffness.data(), controllers::Device::kCuda, task_matrix_shape)),
       controllers::shared(
           controllers::read_view(damping.data(), controllers::Device::kCuda, task_vector_shape))},
      {{controllers::write_view(wrench.data(), controllers::Device::kCuda, batch_task_shape),
        {parameter_status.data(), 1, controllers::Device::kCuda},
        {status.data(), 1, controllers::Device::kCuda},
        {controllers::write_view(last_valid.data(), controllers::Device::kCuda, batch_task_shape),
         controllers::write_view(valid.data(), controllers::Device::kCuda, batch_shape),
         controllers::write_view(hold_count.data(), controllers::Device::kCuda, batch_shape)}},
       {controllers::write_view(
            previous_mass.data(), controllers::Device::kCuda, batch_task_matrix_shape),
        controllers::write_view(
            previous_jacobian.data(), controllers::Device::kCuda, batch_jacobian_shape),
        controllers::write_view(initialized.data(), controllers::Device::kCuda, batch_shape)}},
      context);
  ASSERT_TRUE(prepare_result.ok()) << prepare_result.message;
  ASSERT_TRUE(controller.reset({}, context).ok());
  const controllers::Result step_result = controller.step(
      {controllers::read_view(pose_error.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(velocity_error.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(
           desired_acceleration.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(joint_velocity.data(), controllers::Device::kCuda, batch_joint_shape),
       controllers::read_view(
           task_mass.data(), controllers::Device::kCuda, batch_task_matrix_shape),
       controllers::read_view(jacobian.data(), controllers::Device::kCuda, batch_jacobian_shape),
       controllers::read_view(dt.data(), controllers::Device::kCuda, batch_shape)},
      context);
  ASSERT_TRUE(step_result.ok()) << step_result.message;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    const double stiffness_value = host_stiffness[axis * kTaskDim + axis];
    const double mass_value = host_task_mass[axis * kTaskDim + axis];
    const double damping_value = 2.0 * host_damping[axis] * std::sqrt(mass_value * stiffness_value);
    const double expected = stiffness_value * host_pose_error[axis] +
                            damping_value * host_velocity_error[axis] +
                            mass_value * host_desired_acceleration[axis];
    EXPECT_NEAR(wrench.data()[axis], expected, 1.0e-9);
  }
  EXPECT_EQ(status.data()[0], 0U);

  const std::array<double, kTaskDim * kTaskDim> previous_host_mass = host_task_mass;
  const std::array<double, kTaskDim * kJointDim> previous_host_jacobian = host_jacobian;
  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    host_task_mass[axis * kTaskDim + axis] += 0.01 * static_cast<double>(axis + 1);
    const std::size_t joint = axis < kSpatialDim ? axis : 7 + axis - kSpatialDim;
    host_jacobian[axis * kJointDim + joint] += 0.0005 * static_cast<double>(axis + 1);
  }
  task_mass.assign(host_task_mass);
  jacobian.assign(host_jacobian);
  const controllers::Result second_step_result = controller.step(
      {controllers::read_view(pose_error.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(velocity_error.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(
           desired_acceleration.data(), controllers::Device::kCuda, batch_task_shape),
       controllers::read_view(joint_velocity.data(), controllers::Device::kCuda, batch_joint_shape),
       controllers::read_view(
           task_mass.data(), controllers::Device::kCuda, batch_task_matrix_shape),
       controllers::read_view(jacobian.data(), controllers::Device::kCuda, batch_jacobian_shape),
       controllers::read_view(dt.data(), controllers::Device::kCuda, batch_shape)},
      context);
  ASSERT_TRUE(second_step_result.ok()) << second_step_result.message;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  for (std::size_t axis = 0; axis < kTaskDim; ++axis) {
    const std::size_t joint = axis < kSpatialDim ? axis : 7 + axis - kSpatialDim;
    const double stiffness_value = host_stiffness[axis * kTaskDim + axis];
    const double mass_value = host_task_mass[axis * kTaskDim + axis];
    const double damping_value = 2.0 * host_damping[axis] * std::sqrt(mass_value * stiffness_value);
    const double mass_derivative =
        (mass_value - previous_host_mass[axis * kTaskDim + axis]) / host_dt[0];
    const double jacobian_derivative = (host_jacobian[axis * kJointDim + joint] -
                                        previous_host_jacobian[axis * kJointDim + joint]) /
                                       host_dt[0];
    const double expected = stiffness_value * host_pose_error[axis] +
                            damping_value * host_velocity_error[axis] +
                            mass_value * host_desired_acceleration[axis] +
                            0.5 * mass_derivative * host_velocity_error[axis] -
                            mass_value * jacobian_derivative * host_joint_velocity[joint];
    EXPECT_NEAR(wrench.data()[axis], expected, 1.0e-9);
  }
  EXPECT_EQ(status.data()[0], 0U);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
#endif

}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
