/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "./controller_test_support.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

namespace nvidia::srl::controllers {
namespace {

enum class Pipeline { kCartesian, kVariable, kUnified, kBimanual };
enum class Role { kState, kScratch, kOutput };

struct ExpectedTensor {
  Role role;
  DType dtype;
  std::vector<std::int64_t> shape;
  std::string units;
  std::string frame;
};

std::map<std::string, ExpectedTensor> expected_inventory(Pipeline pipeline,
                                                         DType dtype,
                                                         std::int64_t batch) {
  const std::int64_t joints = pipeline == Pipeline::kBimanual ? 14 : 7;
  const std::int64_t task = pipeline == Pipeline::kBimanual ? 12 : 6;
  const std::string frame =
      pipeline == Pipeline::kBimanual ? "main_frame_orientation" : "robot_base";
  std::map<std::string, ExpectedTensor> tensors;
  const auto add = [&](const std::string& name,
                       Role role,
                       DType type,
                       std::vector<std::int64_t> shape,
                       const std::string& units,
                       const std::string& coordinates) {
    tensors.emplace(name, ExpectedTensor{role, type, shape, units, coordinates});
  };
  const auto status = [&](const std::string& name, Role role) {
    add(name, role, DType::kUInt64, {batch}, "status_flags", "none");
  };
  const auto flag = [&](const std::string& name) {
    add(name, Role::kState, DType::kBool, {batch}, "unitless", "none");
  };
  const auto safety = [&](const std::string& prefix,
                          std::int64_t dimension,
                          const std::string& units,
                          const std::string& coordinates) {
    add(prefix + "last_valid_output", Role::kState, dtype, {batch, dimension}, units, coordinates);
    flag(prefix + "valid_mask");
    add(prefix + "hold_count", Role::kState, DType::kUInt64, {batch}, "control_steps", "none");
  };
  status("mass.parameter_status", Role::kState);
  status("task.parameter_status", Role::kState);
  status("nullspace.parameter_status", Role::kState);
  add("mass.joint_mass_factor",
      Role::kScratch,
      dtype,
      {batch, joints, joints},
      "mass",
      "joint_space");
  add("mass.mass_inverse_jacobian_transpose",
      Role::kScratch,
      dtype,
      {batch, joints, task},
      "inverse_mass_times_jacobian",
      "joint_to_task");
  add("mass.cartesian_inverse_mass_factor",
      Role::kScratch,
      dtype,
      {batch, task, task},
      "inverse_operational_space_mass",
      frame);
  add("mass.cartesian_mass_matrix",
      Role::kScratch,
      dtype,
      {batch, task, task},
      "operational_space_mass_units",
      frame);
  status("mass.status", Role::kScratch);
  add("task.previous_mass_matrix",
      Role::kState,
      dtype,
      {batch, task, task},
      "operational_space_mass_units",
      frame);
  add("task.previous_jacobian",
      Role::kState,
      dtype,
      {batch, task, joints},
      "unitless",
      "task_to_joint");
  flag(pipeline == Pipeline::kVariable ? "task.inertial_initialized" : "task.initialized");
  if (pipeline == Pipeline::kVariable) {
    for (const char* name : {"task.current_stiffness_matrix", "task.previous_stiffness_matrix"}) {
      add(name,
          Role::kState,
          dtype,
          {batch, task, task},
          "newton_per_meter_and_newton_meter_per_radian",
          frame);
    }
    flag("task.stiffness_initialized");
    for (const char* stage : {"stiffness", "damping", "inertial"}) {
      const std::string prefix = std::string("task.") + stage;
      add(prefix + "_wrench",
          Role::kScratch,
          dtype,
          {batch, task},
          "newton_and_newton_meter",
          frame);
      status(prefix + "_status", Role::kScratch);
    }
  }
  if (pipeline == Pipeline::kUnified) {
    for (const char* name : {"task.differentiator_previous_error",
                             "task.integrator_previous_error",
                             "task.integrated_error"}) {
      add(name, Role::kState, dtype, {batch, task}, "wrench_error", frame);
    }
    flag("task.differentiator_initialized");
    flag("task.integrator_initialized");
    for (const char* stage : {"impedance", "force"}) {
      const std::string prefix = std::string("task.") + stage;
      add(prefix + "_wrench",
          Role::kScratch,
          dtype,
          {batch, task},
          "newton_and_newton_meter",
          frame);
      status(prefix + "_status", Role::kScratch);
    }
  }
  safety("task.", task, "newton_and_newton_meter", frame);
  add("task.cartesian_wrench",
      Role::kScratch,
      dtype,
      {batch, task},
      "newton_and_newton_meter",
      frame);
  status("task.status", Role::kScratch);
  safety("nullspace.", joints, "newton_meter", "joint_space");
  add("nullspace.joint_torque",
      Role::kScratch,
      dtype,
      {batch, joints},
      "newton_meter",
      "joint_space");
  status("nullspace.status", Role::kScratch);
  add("previous_command", Role::kState, dtype, {batch, joints}, "command_units", "command_space");
  flag("initialized");
  safety("", joints, "command_units", "command_space");
  add("unfiltered_joint_torque",
      Role::kScratch,
      dtype,
      {batch, joints},
      "newton_meter",
      "joint_space");
  add("filtered_command", Role::kOutput, dtype, {batch, joints}, "command_units", "command_space");
  status("parameter_status", Role::kOutput);
  status("status", Role::kOutput);
  if (pipeline == Pipeline::kBimanual) {
    add("task_velocity_error",
        Role::kScratch,
        dtype,
        {batch, task},
        "meter_per_second_and_radian_per_second",
        "main_frame_orientation_at_task_origins");
    add("kinematics.task_pose_error",
        Role::kScratch,
        dtype,
        {batch, task},
        "meter_and_radian",
        "main_frame_orientation_at_task_origins");
    add("kinematics.task_jacobian",
        Role::kScratch,
        dtype,
        {batch, task, joints},
        "task_velocity_per_joint_velocity",
        "main_frame_orientation_at_task_origins");
    status("kinematics.status", Role::kScratch);
  }
  return tensors;
}

template <typename Controller>
void check_inventory(Pipeline pipeline, std::size_t state_count, std::size_t scratch_count) {
  for (DType dtype : {DType::kFloat32, DType::kFloat64}) {
    for (std::size_t batch : {1U, 3U}) {
      typename Controller::Config config;
      config.batch_size = batch;
      config.joint_dim = pipeline == Pipeline::kBimanual ? 14 : 7;
      config.task_dim = pipeline == Pipeline::kBimanual ? 12 : 6;
      config.cartesian_frame = pipeline == Pipeline::kBimanual
                                   ? CartesianFrame::kMainFrameOrientation
                                   : CartesianFrame::kRobotBase;
      config.dtype = dtype;
      Requirements requirements;
      ASSERT_TRUE(Controller::requirements(config, &requirements).ok());
      ASSERT_EQ(requirements.state_count, state_count);
      ASSERT_EQ(requirements.scratch_count, scratch_count);
      ASSERT_EQ(requirements.output_count, 3U);
      EXPECT_EQ(requirements.state.size(), 32U);
      EXPECT_EQ(requirements.scratch.size(), 32U);
      EXPECT_EQ(requirements.outputs.size(), 4U);
      EXPECT_EQ(requirements.batch_size, batch);
      EXPECT_EQ(requirements.joint_dim, config.joint_dim);
      auto expected = expected_inventory(pipeline, dtype, static_cast<std::int64_t>(batch));
      const auto check = [&](const TensorRequirement& actual, Role role) {
        ASSERT_NE(actual.name, nullptr);
        const auto found = expected.find(actual.name);
        ASSERT_NE(found, expected.end()) << actual.name;
        const ExpectedTensor& tensor = found->second;
        EXPECT_EQ(tensor.role, role) << actual.name;
        EXPECT_EQ(tensor.dtype, actual.dtype) << actual.name;
        EXPECT_EQ(actual.device, Device::kCpu);
        ASSERT_EQ(tensor.shape.size(), actual.rank) << actual.name;
        for (int dim = 0; dim < kMaxTensorRank; ++dim) {
          EXPECT_EQ(actual.shape[dim], dim < actual.rank ? tensor.shape[dim] : 0) << actual.name;
        }
        EXPECT_EQ(tensor.units, actual.units) << actual.name;
        EXPECT_EQ(tensor.frame, actual.frame) << actual.name;
        expected.erase(found);
      };
      for (std::size_t i = 0; i < requirements.state_count; ++i) {
        check(requirements.state[i], Role::kState);
      }
      for (std::size_t i = 0; i < requirements.scratch_count; ++i) {
        check(requirements.scratch[i], Role::kScratch);
      }
      for (std::size_t i = 0; i < requirements.output_count; ++i) {
        check(requirements.outputs[i], Role::kOutput);
      }
      EXPECT_TRUE(expected.empty());
      const std::array<const char*, 5> final_state = {
          "previous_command", "initialized", "last_valid_output", "valid_mask", "hold_count"};
      for (std::size_t i = 0; i < final_state.size(); ++i) {
        EXPECT_STREQ(requirements.state[state_count - final_state.size() + i].name, final_state[i]);
      }
    }
  }
}

TEST(CompositeRequirementsTest, IndependentCartesianInventory) {
  check_inventory<CartesianImpedanceTorqueController>(Pipeline::kCartesian, 17, 10);
}
TEST(CompositeRequirementsTest, IndependentVariableInventory) {
  check_inventory<VariableCartesianImpedanceTorqueController>(Pipeline::kVariable, 20, 16);
}
TEST(CompositeRequirementsTest, IndependentUnifiedInventory) {
  check_inventory<CartesianUnifiedForceImpedanceTorqueController>(Pipeline::kUnified, 22, 14);
}
TEST(CompositeRequirementsTest, IndependentBimanualInventory) {
  check_inventory<BimanualCartesianImpedanceTorqueController>(Pipeline::kBimanual, 17, 14);
}

template <typename T, typename Controller>
void check_lifecycle() {
  for (std::size_t batch : {1U, 3U}) {
    SCOPED_TRACE(batch);
    test_support::CompositeFixture<T, Controller> fixture(batch);
    test_support::CompositeFixture<T, Controller> reference(batch);
    ASSERT_TRUE(fixture.buffers.all_bound_once());
    ASSERT_TRUE(reference.buffers.all_bound_once());
    const Result prepared = fixture.prepare();
    ASSERT_TRUE(prepared.ok()) << prepared.message;
    ASSERT_TRUE(reference.prepare().ok());
    ASSERT_TRUE(fixture.reset().ok());
    ASSERT_TRUE(reference.reset().ok());
    for (int iteration = 0; iteration < 4; ++iteration) {
      SCOPED_TRACE(iteration);
      if (iteration != 0) fixture.buffers.poison_scratch();
      const Result stepped = fixture.step();
      ASSERT_TRUE(stepped.ok()) << stepped.message;
      ASSERT_TRUE(reference.step().ok());
      const auto& command = fixture.buffers.template values<T>("filtered_command");
      const auto& expected = reference.buffers.template values<T>("filtered_command");
      ASSERT_EQ(command.size(), batch * fixture.config.joint_dim);
      bool nonzero = false;
      for (std::size_t i = 0; i < command.size(); ++i) {
        EXPECT_TRUE(std::isfinite(command[i]));
        EXPECT_NEAR(command[i], expected[i], (std::is_same<T, float>::value ? 1.0e-5 : 1.0e-12));
        nonzero = nonzero || std::abs(command[i]) > T{0.01};
      }
      EXPECT_TRUE(nonzero);
      for (const char* name : {"parameter_status",
                               "mass.parameter_status",
                               "task.parameter_status",
                               "nullspace.parameter_status",
                               "status",
                               "mass.status",
                               "task.status",
                               "nullspace.status"}) {
        const auto& flags_by_row = fixture.buffers.template values<std::uint64_t>(name);
        EXPECT_EQ(flags_by_row, reference.buffers.template values<std::uint64_t>(name)) << name;
        for (std::uint64_t flags : flags_by_row) {
          EXPECT_EQ(flags & ~status_bit(StatusFlag::kSaturation), 0U) << name;
        }
      }
      EXPECT_EQ(fixture.buffers.template values<std::uint8_t>("valid_mask"),
                std::vector<std::uint8_t>(batch, 1));
      for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t index = row * fixture.config.joint_dim * fixture.config.joint_dim;
        fixture.joint_mass[index] += T{0.01};
        reference.joint_mass[index] += T{0.01};
      }
    }
    fixture.buffers.poison_scratch();
    ASSERT_TRUE(fixture.reset().ok());
    ASSERT_TRUE(reference.reset().ok());
    ASSERT_TRUE(fixture.step().ok());
    ASSERT_TRUE(reference.step().ok());
    EXPECT_EQ(fixture.buffers.template values<T>("filtered_command"),
              reference.buffers.template values<T>("filtered_command"));
  }
}

template <typename T>
class CompositeRequirementsTypedTest : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CompositeRequirementsTypedTest, FloatingTypes);

TYPED_TEST(CompositeRequirementsTypedTest, CartesianDescriptorStorageLifecycle) {
  check_lifecycle<TypeParam, CartesianImpedanceTorqueController>();
}
TYPED_TEST(CompositeRequirementsTypedTest, VariableDescriptorStorageLifecycle) {
  check_lifecycle<TypeParam, VariableCartesianImpedanceTorqueController>();
}
TYPED_TEST(CompositeRequirementsTypedTest, UnifiedDescriptorStorageLifecycle) {
  check_lifecycle<TypeParam, CartesianUnifiedForceImpedanceTorqueController>();
}
TYPED_TEST(CompositeRequirementsTypedTest, BimanualDescriptorStorageLifecycle) {
  check_lifecycle<TypeParam, BimanualCartesianImpedanceTorqueController>();
}

TYPED_TEST(CompositeRequirementsTypedTest, RejectsBimanualVelocityStorageOverlapBeforeWrites) {
  test_support::CompositeFixture<TypeParam, BimanualCartesianImpedanceTorqueController> fixture(2);
  auto& flags = fixture.buffers.template values<std::uint64_t>("mass.parameter_status");
  std::fill(flags.begin(), flags.end(), 123U);
  fixture.storage.task_velocity_error = fixture.storage.kinematics.task_pose_error;
  EXPECT_FALSE(fixture.prepare().ok());
  EXPECT_EQ(flags, std::vector<std::uint64_t>(2, 123U));
}

TYPED_TEST(CompositeRequirementsTypedTest, RejectsCrossStageStatusOverlapBeforeParameterWrites) {
  test_support::CompositeFixture<TypeParam, CartesianImpedanceTorqueController> fixture(2);
  auto& flags = fixture.buffers.template values<std::uint64_t>("mass.parameter_status");
  std::fill(flags.begin(), flags.end(), 123U);
  fixture.storage.task.parameter_status = fixture.storage.mass.parameter_status;
  EXPECT_EQ(fixture.prepare().code, ResultCode::kInvalidView);
  EXPECT_EQ(flags, (std::vector<std::uint64_t>{123U, 123U}));
  EXPECT_FALSE(fixture.step().ok());
}

TYPED_TEST(CompositeRequirementsTypedTest, AllowsScratchReuseAcrossDisjointStages) {
  using Controller = VariableCartesianImpedanceTorqueController;
  test_support::CompositeFixture<TypeParam, Controller> fixture(2);
  test_support::CompositeFixture<TypeParam, Controller> reference(2);
  fixture.storage.task.scratch.stiffness_wrench.data =
      fixture.storage.mass.scratch.joint_factor.data;
  ASSERT_TRUE(fixture.prepare().ok());
  ASSERT_TRUE(reference.prepare().ok());
  ASSERT_TRUE(fixture.reset().ok());
  ASSERT_TRUE(reference.reset().ok());
  for (int tick = 0; tick < 3; ++tick) {
    ASSERT_TRUE(fixture.step().ok());
    ASSERT_TRUE(reference.step().ok());
    EXPECT_EQ(fixture.buffers.template values<TypeParam>("filtered_command"),
              reference.buffers.template values<TypeParam>("filtered_command"));
    EXPECT_EQ(fixture.buffers.template values<std::uint64_t>("status"),
              reference.buffers.template values<std::uint64_t>("status"));
  }
}

TYPED_TEST(CompositeRequirementsTypedTest, AllowsExactFinalMappingFilterAlias) {
  test_support::CompositeFixture<TypeParam, CartesianImpedanceTorqueController> fixture(2);
  fixture.storage.output.unfiltered_joint_torque = fixture.storage.output.filter.command;
  ASSERT_TRUE(fixture.prepare().ok());
  ASSERT_TRUE(fixture.reset().ok());
  ASSERT_TRUE(fixture.step().ok());
  for (const auto flags : fixture.buffers.template values<std::uint64_t>("status")) {
    EXPECT_FALSE(has_status_flag(flags, StatusFlag::kSafeMode));
  }
}

template <typename Controller>
void check_rejected_query(typename Controller::Config config) {
  Requirements output;
  output.state_count = 1;
  output.scratch_count = 1;
  output.output_count = 1;
  output.state[0].name = "stale";
  EXPECT_EQ(Controller::requirements(config, &output).code, ResultCode::kInvalidArgument);
  EXPECT_EQ(output.state_count, 0U);
  EXPECT_EQ(output.scratch_count, 0U);
  EXPECT_EQ(output.output_count, 0U);
  EXPECT_EQ(output.batch_size, 0U);
  for (const TensorRequirement& tensor : output.state) EXPECT_EQ(tensor.name, nullptr);
  for (const TensorRequirement& tensor : output.scratch) EXPECT_EQ(tensor.name, nullptr);
  for (const TensorRequirement& tensor : output.outputs) EXPECT_EQ(tensor.name, nullptr);
  typename Controller::ComponentRequirements complete;
  complete.output.filter.outputs[0].name = "stale";
  complete.output.filter.output_count = 1;
  EXPECT_EQ(Controller::component_requirements(config, &complete).code,
            ResultCode::kInvalidArgument);
  EXPECT_EQ(complete.output.filter.output_count, 0U);
  EXPECT_EQ(complete.output.filter.outputs[0].name, nullptr);
  Controller controller;
  EXPECT_EQ(controller.prepare(config, {}, {}).code, ResultCode::kInvalidArgument);
  EXPECT_FALSE(controller.reset().ok());
  EXPECT_FALSE(controller.step({}).ok());
  EXPECT_EQ(Controller::requirements(config, nullptr).code, ResultCode::kInvalidArgument);
}

TEST(CompositeRequirementsTest, RejectsLeafDimensionAndFrameMismatchesWithoutPartialResults) {
  ControllerConfig config;
  config.batch_size = 1;
  config.joint_dim = 33;
  config.cartesian_frame = CartesianFrame::kRobotBase;
  check_rejected_query<CartesianImpedanceTorqueController>(config);
  check_rejected_query<VariableCartesianImpedanceTorqueController>(config);
  CompositeControllerConfig unified;
  static_cast<ControllerConfig&>(unified) = config;
  check_rejected_query<CartesianUnifiedForceImpedanceTorqueController>(unified);
  config.joint_dim = 7;
  config.task_dim = 12;
  check_rejected_query<VariableCartesianImpedanceTorqueController>(config);
  static_cast<ControllerConfig&>(unified) = config;
  check_rejected_query<CartesianUnifiedForceImpedanceTorqueController>(unified);
  config.task_dim = 6;
  config.cartesian_frame = CartesianFrame::kUnspecified;
  check_rejected_query<CartesianImpedanceTorqueController>(config);
  check_rejected_query<VariableCartesianImpedanceTorqueController>(config);
  static_cast<ControllerConfig&>(unified) = config;
  check_rejected_query<CartesianUnifiedForceImpedanceTorqueController>(unified);
  BimanualTaskConfig bimanual;
  bimanual.batch_size = 1;
  bimanual.joint_dim = 14;
  bimanual.task_dim = 12;
  bimanual.cartesian_frame = CartesianFrame::kRobotBase;
  check_rejected_query<BimanualCartesianImpedanceTorqueController>(bimanual);
  bimanual.cartesian_frame = CartesianFrame::kMainFrameOrientation;
  bimanual.joint_dim = 7;
  check_rejected_query<BimanualCartesianImpedanceTorqueController>(bimanual);
  bimanual.joint_dim = 14;
  bimanual.task_dim = 6;
  check_rejected_query<BimanualCartesianImpedanceTorqueController>(bimanual);
  bimanual.task_dim = 12;
  bimanual.frames.arm1_T_arm2[0] = 2.0;
  check_rejected_query<BimanualCartesianImpedanceTorqueController>(bimanual);
  config.cartesian_frame = CartesianFrame::kRobotBase;
  config.batch_size = std::numeric_limits<std::size_t>::max();
  check_rejected_query<CartesianImpedanceTorqueController>(config);
}

TEST(CompositeRequirementsTest, KeepsJointBoundsLocalToJointAndNullspaceControllers) {
  ControllerConfig config;
  config.batch_size = 1;
  config.cartesian_frame = CartesianFrame::kRobotBase;
  Requirements output;
  for (std::size_t dimension : {32U, 33U, 64U}) {
    config.joint_dim = dimension;
    EXPECT_EQ(JointImpedanceController::requirements(config, &output).ok(), dimension <= 32);
    EXPECT_EQ(NullspaceController::requirements(config, &output).ok(), dimension <= 32);
    EXPECT_EQ(CartesianImpedanceTorqueController::requirements(config, &output).ok(),
              dimension <= 32);
    EXPECT_TRUE(TorqueFilter::requirements(config, &output).ok());
    EXPECT_TRUE(OperationalSpaceMass::requirements(config, &output).ok());
    EXPECT_TRUE(TaskWrenchToJointTorqueController::requirements(config, &output).ok());
    if (dimension > 32) {
      JointImpedanceController joint;
      NullspaceController nullspace;
      EXPECT_EQ(joint.prepare(config, {}, {}).code, ResultCode::kInvalidArgument);
      EXPECT_EQ(nullspace.prepare(config, {}, {}).code, ResultCode::kInvalidArgument);
    }
  }
}

template <typename Controller>
bool check_name_lifetime() {
  typename Controller::Config config;
  constexpr bool bimanual =
      std::is_same<Controller, BimanualCartesianImpedanceTorqueController>::value;
  config.batch_size = 1;
  config.joint_dim = bimanual ? 14 : 7;
  config.task_dim = bimanual ? 12 : 6;
  config.cartesian_frame =
      bimanual ? CartesianFrame::kMainFrameOrientation : CartesianFrame::kRobotBase;
  Requirements original;
  if (!Controller::requirements(config, &original).ok()) return false;
  const Requirements copied = original;
  original = Requirements{};
  std::vector<std::pair<const char*, std::string>> retained;
  for (std::size_t i = 0; i < copied.state_count; ++i) {
    retained.emplace_back(copied.state[i].name, copied.state[i].name);
  }
  for (std::size_t i = 0; i < copied.scratch_count; ++i) {
    retained.emplace_back(copied.scratch[i].name, copied.scratch[i].name);
  }
  for (std::size_t i = 0; i < copied.output_count; ++i) {
    retained.emplace_back(copied.outputs[i].name, copied.outputs[i].name);
  }
  for (std::size_t iteration = 0; iteration < 100; ++iteration) {
    config.batch_size = iteration + 1;
    Requirements overwritten;
    if (!Controller::requirements(config, &overwritten).ok()) return false;
    for (const auto& name : retained) {
      if (std::strcmp(name.first, name.second.c_str()) != 0) return false;
    }
  }
  return true;
}

TEST(CompositeRequirementsTest, QualifiedNamesSurviveCopiesRepeatedAndConcurrentQueries) {
  std::array<std::future<bool>, 8> workers;
  for (auto& worker : workers) {
    worker = std::async(std::launch::async, [] {
      return check_name_lifetime<CartesianImpedanceTorqueController>() &&
             check_name_lifetime<VariableCartesianImpedanceTorqueController>() &&
             check_name_lifetime<CartesianUnifiedForceImpedanceTorqueController>() &&
             check_name_lifetime<BimanualCartesianImpedanceTorqueController>();
    });
  }
  for (auto& worker : workers) EXPECT_TRUE(worker.get());
}

TEST(CompositeRequirementsTest,
     ContiguousConstructorsRejectArithmeticOverflowBeforeMultiplication) {
  double scalar = 0.0;
  const double* constant = &scalar;
  constexpr std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
  const std::array<std::array<std::int64_t, 3>, 4> invalid = {
      std::array<std::int64_t, 3>{0, maximum, 2}, {maximum, 2, 1}, {-1, 2, 3}, {1, 1, maximum}};
  for (const auto& extents : invalid) {
    EXPECT_EQ(make_contiguous_tensor_view(&scalar, Device::kCpu, extents).dtype, DType::kUnknown);
    EXPECT_EQ(make_contiguous_tensor_view(constant, Device::kCpu, extents).dtype, DType::kUnknown);
    EXPECT_EQ(write_view(&scalar, Device::kCpu, extents).dtype, DType::kUnknown);
    EXPECT_EQ(read_view(constant, Device::kCpu, extents).dtype, DType::kUnknown);
  }
  EXPECT_EQ(read_view(constant, Device::kCpu, std::numeric_limits<std::uint64_t>::max()).dtype,
            DType::kUnknown);
  EXPECT_EQ(write_view(&scalar, Device::kCpu, std::numeric_limits<std::uint64_t>::max()).dtype,
            DType::kUnknown);
  for (const auto& extents : {std::array<std::int64_t, 3>{0, 2, 3}, {2, 0, 3}, {2, 3, 0}}) {
    const TensorView view = make_contiguous_tensor_view(constant, Device::kCpu, extents);
    EXPECT_EQ(view.rank, 3);
    EXPECT_EQ(view.dtype, DType::kFloat64);
    EXPECT_EQ(view.strides[2], 1);
    std::vector<double> empty;
    EXPECT_EQ(read_view(empty, Device::kCpu, extents).dtype, DType::kFloat64);
    EXPECT_EQ(write_view(empty, Device::kCpu, extents).dtype, DType::kFloat64);
  }
}

}  // namespace
}  // namespace nvidia::srl::controllers
