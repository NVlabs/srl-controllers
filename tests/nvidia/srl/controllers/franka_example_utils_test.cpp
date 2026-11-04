/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include <gtest/gtest.h>

#include <franka/rate_limiting.h>
#include <eigen3/Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "examples/cartesian_math.hpp"
#include "examples/franka/franka_admittance.hpp"
#include "examples/franka/franka_cartesian_field.hpp"
#include "examples/franka/franka_force_tank.hpp"
#include "examples/franka/franka_utils.hpp"

namespace {

#if defined(NVIDIA_SRL_FRANKA_FER)
constexpr double kExpectedAdmittanceTranslationMass = 29.25;
constexpr double kExpectedAdmittanceRotationMass = 3.9;
#else
constexpr double kExpectedAdmittanceTranslationMass = 1.95;
constexpr double kExpectedAdmittanceRotationMass = 1.95;
#endif

class StreamRedirect {
 public:
  explicit StreamRedirect(const std::string& input)
      : input_(input), old_input_(std::cin.rdbuf(input_.rdbuf())), old_output_(std::cout.rdbuf()) {
    std::cout.rdbuf(output_.rdbuf());
  }

  StreamRedirect(const StreamRedirect&) = delete;
  StreamRedirect& operator=(const StreamRedirect&) = delete;

  ~StreamRedirect() {
    std::cin.rdbuf(old_input_);
    std::cout.rdbuf(old_output_);
  }

 private:
  std::istringstream input_;
  std::ostringstream output_;
  std::streambuf* old_input_;
  std::streambuf* old_output_;
};

TEST(FrankaExampleUtilsTest, CollisionThresholdsMatchTheDocumentedMaxima) {
  const std::array<double, 7> expected_torque{60.0, 60.0, 60.0, 60.0, 10.0, 10.0, 10.0};
  const std::array<double, 6> expected_wrench{40.0, 40.0, 40.0, 10.0, 10.0, 10.0};
  EXPECT_EQ(franka_example::kCollisionTorqueThresholdNm, expected_torque);
  EXPECT_EQ(franka_example::kCollisionWrenchThreshold, expected_wrench);
}

TEST(FrankaExampleUtilsTest, ConfirmationRequiresAnEmptyLine) {
  {
    StreamRedirect streams("\n");
    EXPECT_TRUE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
  }
  {
    StreamRedirect streams("execute\n");
    EXPECT_FALSE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
  }
  {
    StreamRedirect streams(" \n");
    EXPECT_FALSE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
  }
  {
    StreamRedirect streams("");
    EXPECT_FALSE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
  }
}

TEST(FrankaExampleUtilsTest, ControllerStartRequiresASeparateEmptyLine) {
  for (const std::string second_response : {"\n", "cancel\n", " \n", ""}) {
    StreamRedirect streams("\n" + second_response);
    ASSERT_TRUE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
    EXPECT_EQ(franka_example::confirm_controller_start(), second_response == "\n");
  }
}

TEST(FrankaExampleUtilsTest, ControllerConfirmationRequiresAnEmptyLine) {
  for (const std::string& input : {"\n", "start\n", " \n", "\t\n", ""}) {
    SCOPED_TRACE(input);
    StreamRedirect streams(input);
    EXPECT_EQ(franka_example::confirm_controller_start(), input == "\n");
  }
}

TEST(FrankaExampleUtilsTest, ControllerActivationRequiresASeparateConfirmationLine) {
  for (const std::string& input : {"\n\n", "\ncancel\n", "\n"}) {
    SCOPED_TRACE(input);
    StreamRedirect streams(input);
    ASSERT_TRUE(franka_example::confirm_execution(
        "test", "configuration", franka_example::kInitialJointConfiguration, 10.0));
    EXPECT_EQ(franka_example::confirm_controller_start(), input == "\n\n");
  }
}

TEST(FrankaExampleUtilsTest, ParsesOptionalDurationAndNonRealtimeModeAfterRobotHost) {
  char executable[] = "example";
  char robot_host[] = "robot.example";
  char non_rt[] = "--non-rt";
  char duration[] = "--duration-seconds=60.5";
  char unknown[] = "--unknown";
  franka_example::ProgramArguments arguments;

  char* default_argv[] = {executable, robot_host};
  ASSERT_TRUE(franka_example::parse_arguments(2, default_argv, &arguments));
  EXPECT_EQ(arguments.robot_host, robot_host);
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 10.0);
  EXPECT_FALSE(arguments.non_rt_requested);
  EXPECT_EQ(franka_example::realtime_config(arguments.non_rt_requested),
            franka::RealtimeConfig::kEnforce);

  char* non_rt_argv[] = {executable, robot_host, non_rt};
  ASSERT_TRUE(franka_example::parse_arguments(3, non_rt_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 10.0);
  EXPECT_TRUE(arguments.non_rt_requested);
  EXPECT_EQ(franka_example::realtime_config(arguments.non_rt_requested),
            franka::RealtimeConfig::kIgnore);

  char* duration_argv[] = {executable, robot_host, duration};
  ASSERT_TRUE(franka_example::parse_arguments(3, duration_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 60.5);
  EXPECT_FALSE(arguments.non_rt_requested);

  char* options_argv[] = {executable, robot_host, duration, non_rt};
  ASSERT_TRUE(franka_example::parse_arguments(4, options_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 60.5);
  EXPECT_TRUE(arguments.non_rt_requested);

  char* reversed_argv[] = {executable, robot_host, non_rt, duration};
  ASSERT_TRUE(franka_example::parse_arguments(4, reversed_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 60.5);
  EXPECT_TRUE(arguments.non_rt_requested);

  ASSERT_TRUE(franka_example::parse_arguments(2, default_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 10.0);
  EXPECT_FALSE(arguments.non_rt_requested);

  char* unknown_argv[] = {executable, robot_host, unknown};
  EXPECT_FALSE(franka_example::parse_arguments(3, unknown_argv, &arguments));
  char* duplicate_argv[] = {executable, robot_host, non_rt, non_rt};
  EXPECT_FALSE(franka_example::parse_arguments(4, duplicate_argv, &arguments));
  char* duplicate_duration_argv[] = {executable, robot_host, duration, duration};
  EXPECT_FALSE(franka_example::parse_arguments(4, duplicate_duration_argv, &arguments));
  char* misplaced_argv[] = {executable, non_rt, robot_host};
  EXPECT_FALSE(franka_example::parse_arguments(3, misplaced_argv, &arguments));
}

TEST(FrankaExampleUtilsTest, RejectsInvalidDurations) {
  char executable[] = "example";
  char robot_host[] = "robot.example";
  for (const char* value :
       {"", "0", "-1", "nan", "inf", "-inf", "1e999", "1e-999", "abc", "60s", "1.5.0"}) {
    SCOPED_TRACE(value);
    std::string duration = std::string("--duration-seconds=") + value;
    char* argv[] = {executable, robot_host, duration.data()};
    franka_example::ProgramArguments arguments;
    EXPECT_FALSE(franka_example::parse_arguments(3, argv, &arguments));
  }
}

TEST(FrankaExampleUtilsTest, WrenchFilterUsesMeasuredPeriodAndResets) {
  franka_example::WrenchLowPassFilter filter(7.0);
  const Eigen::Vector<double, 6> input = Eigen::Vector<double, 6>::Constant(2.0);
  const double time_constant = 1.0 / (2.0 * franka_example::kPi * 7.0);
  const double first = 2.0 * 0.001 / (time_constant + 0.001);
  EXPECT_TRUE(
      filter.update(input, 0.001).isApprox(Eigen::Vector<double, 6>::Constant(first), 1.0e-12));
  const double second = first + (2.0 - first) * 0.002 / (time_constant + 0.002);
  EXPECT_TRUE(
      filter.update(input, 0.002).isApprox(Eigen::Vector<double, 6>::Constant(second), 1.0e-12));
  filter.reset();
  EXPECT_TRUE(
      filter.update(input, 0.001).isApprox(Eigen::Vector<double, 6>::Constant(first), 1.0e-12));
}

TEST(FrankaExampleUtilsTest, AdmittanceFiltersWrenchWithoutDeadbandAndRejectsInvalidInputs) {
  franka_example::AdmittanceWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  Eigen::Vector<double, 6> wrench = Eigen::Vector<double, 6>::Zero();
  ASSERT_TRUE(workspace.step_admittance(wrench, 0.001));
  wrench[0] = 2.0;
  ASSERT_TRUE(workspace.step_admittance(wrench, 0.001));
  const double alpha = 0.001 / (1.0 / (2.0 * franka_example::kPi * 7.0) + 0.001);
  EXPECT_NEAR(
      workspace.acceleration()[0], -2.0 * alpha / kExpectedAdmittanceTranslationMass, 1.0e-12);
  for (int step = 0; step < 20; ++step) {
    ASSERT_TRUE(workspace.step_admittance(wrench, 0.001));
  }
  EXPECT_LT(workspace.pose_offset()[0], 0.0);
  EXPECT_DOUBLE_EQ(workspace.pose_offset()[1], 0.0);
  const auto previous_offset = workspace.pose_offset();
  wrench[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(workspace.step_admittance(wrench, 0.001));
  EXPECT_EQ(workspace.pose_offset(), previous_offset);
  wrench.setZero();
  EXPECT_FALSE(workspace.step_admittance(wrench, 0.0));
  EXPECT_FALSE(workspace.step_admittance(wrench, std::numeric_limits<double>::infinity()));
  EXPECT_EQ(workspace.pose_offset(), previous_offset);
  EXPECT_TRUE(workspace.step_admittance(wrench, 0.001));
  EXPECT_TRUE(franka_example::all_finite(workspace.pose_offset()));
}

TEST(FrankaExampleUtilsTest, AdmittanceDoesNotClampAccelerationVelocityOrPose) {
  franka_example::AdmittanceWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  const Eigen::Vector<double, 6> wrench = Eigen::Vector<double, 6>::Constant(100.0);
  ASSERT_TRUE(workspace.step_admittance(wrench, 0.001));
  const double alpha = 0.001 / (1.0 / (2.0 * franka_example::kPi * 7.0) + 0.001);
  for (std::size_t axis = 0; axis < 6; ++axis) {
    EXPECT_NEAR(
        workspace.acceleration()[axis],
        -100.0 * alpha /
            (axis < 3 ? kExpectedAdmittanceTranslationMass : kExpectedAdmittanceRotationMass),
        1.0e-12);
  }
  std::array<double, 6> maximum_speed{};
  for (int step = 0; step < 500; ++step) {
    ASSERT_TRUE(workspace.step_admittance(wrench, 0.001));
    for (std::size_t axis = 0; axis < 6; ++axis) {
      maximum_speed[axis] = std::max(maximum_speed[axis], std::abs(workspace.twist()[axis]));
    }
  }
  for (std::size_t axis = 0; axis < 6; ++axis) {
    EXPECT_GT(maximum_speed[axis], 0.1);
    EXPECT_LT(workspace.pose_offset()[axis], axis < 3 ? -0.03 : -0.2);
  }
  ASSERT_TRUE(workspace.prepare().ok());
  EXPECT_TRUE(workspace.step_admittance(Eigen::Vector<double, 6>::Zero(), 0.001));
  for (const double value : workspace.pose_offset()) {
    EXPECT_DOUBLE_EQ(value, 0.0);
  }
}

TEST(FrankaExampleUtilsTest, AdmittanceCartesianImpedanceUsesPoseTwistAndAcceleration) {
  franka_example::CartesianImpedanceWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  Eigen::Matrix<double, 7, 7> mass = Eigen::Matrix<double, 7, 7>::Identity();
  Eigen::Matrix<double, 6, 7> jacobian = Eigen::Matrix<double, 6, 7>::Zero();
  jacobian.leftCols<6>().setIdentity();
  Eigen::Vector<double, 6> pose_error = Eigen::Vector<double, 6>::Zero();
  Eigen::Vector<double, 6> velocity_error = Eigen::Vector<double, 6>::Zero();
  Eigen::Vector<double, 6> desired_acceleration = Eigen::Vector<double, 6>::Zero();
  const Eigen::Vector<double, 7> joint_velocity = Eigen::Vector<double, 7>::Zero();
  pose_error[0] = 0.01;
  velocity_error[1] = 0.1;
  desired_acceleration[2] = 0.5;

  ASSERT_TRUE(workspace.step(
      pose_error, velocity_error, desired_acceleration, joint_velocity, mass, jacobian, 0.001));
  ASSERT_TRUE(workspace.last_step_invoked());
  EXPECT_NEAR(workspace.wrench()[0], 20.0, 1.0e-9);
  EXPECT_GT(workspace.wrench()[1], 0.0);
  EXPECT_NEAR(workspace.wrench()[2], 0.5, 1.0e-6);
  const auto trace = workspace.last_step_trace();
  EXPECT_DOUBLE_EQ(trace.pose_error[0], 0.01);
  EXPECT_DOUBLE_EQ(trace.velocity_error[1], 0.1);
  EXPECT_DOUBLE_EQ(trace.desired_acceleration[2], 0.5);
  EXPECT_EQ(trace.impedance_status, 0u);
}

TEST(FrankaExampleUtilsTest, FullCartesianImpedanceKeepsRotationalStiffnessAsInertiaChanges) {
  Eigen::Matrix<double, 6, 7> jacobian = Eigen::Matrix<double, 6, 7>::Zero();
  jacobian.leftCols<6>().setIdentity();
  Eigen::Vector<double, 6> pose_error = Eigen::Vector<double, 6>::Zero();
  pose_error[4] = 0.1;
  const Eigen::Vector<double, 6> zero_cartesian = Eigen::Vector<double, 6>::Zero();
  const Eigen::Vector<double, 7> zero_joint_velocity = Eigen::Vector<double, 7>::Zero();

  franka_example::CartesianImpedanceWorkspace nominal;
  ASSERT_TRUE(nominal.prepare().ok());
  Eigen::Matrix<double, 7, 7> mass = Eigen::Matrix<double, 7, 7>::Identity();
  ASSERT_TRUE(nominal.step(
      pose_error, zero_cartesian, zero_cartesian, zero_joint_velocity, mass, jacobian, 0.001));

  franka_example::CartesianImpedanceWorkspace light_rotation;
  ASSERT_TRUE(light_rotation.prepare().ok());
  mass.diagonal().segment<3>(3).setConstant(0.02);
  ASSERT_TRUE(light_rotation.step(
      pose_error, zero_cartesian, zero_cartesian, zero_joint_velocity, mass, jacobian, 0.001));

  EXPECT_NEAR(nominal.wrench()[4], 10.0, 1.0e-9);
  EXPECT_NEAR(light_rotation.wrench()[4], 10.0, 1.0e-9);
}

TEST(FrankaExampleUtilsTest, InitialMotionStartsAtMeasuredStateAndReachesTarget) {
  franka::RobotState state;
  state.q = {0.1, -0.2, 0.3, -0.4, 0.2, 0.5, -0.1};
  franka_example::InitialMotionGenerator motion_generator(
      franka_example::kInitialMotionSpeedFactor, franka_example::kInitialJointConfiguration);

  franka::JointPositions command = motion_generator(state, franka::Duration(0));
  EXPECT_EQ(command.q, state.q);
  EXPECT_FALSE(command.motion_finished);

  for (std::size_t step = 0; step < 10000 && !command.motion_finished; ++step) {
    command = motion_generator(state, franka::Duration(1));
    EXPECT_TRUE(franka_example::all_finite(command.q));
  }

  EXPECT_TRUE(command.motion_finished);
  for (std::size_t joint = 0; joint < command.q.size(); ++joint) {
    EXPECT_NEAR(command.q[joint], franka_example::kInitialJointConfiguration[joint], 1e-12);
  }
}

TEST(FrankaExampleUtilsTest, FiniteChecksRejectInvalidStateAndTorqueShape) {
  std::array<double, 7> values{};
  EXPECT_TRUE(franka_example::all_finite(values));
  values[3] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(franka_example::all_finite(values));

  Eigen::VectorXd torques = Eigen::VectorXd::Zero(7);
  EXPECT_TRUE(franka_example::valid_joint_torques(torques));
  torques(2) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(franka_example::valid_joint_torques(torques));
  EXPECT_FALSE(franka_example::valid_joint_torques(Eigen::VectorXd::Zero(6)));
}

TEST(FrankaExampleUtilsTest, TorqueRateLimitMatchesLibfranka) {
  franka::RobotState state;
  state.tau_J_d = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0};
  Eigen::VectorXd desired(7);
  desired << 10.0, -10.0, 8.0, -8.0, 6.0, -6.0, 4.0;

  std::array<double, 7> desired_array{};
  std::copy_n(desired.data(), desired_array.size(), desired_array.begin());
  const std::array<double, 7> expected =
      franka::limitRate(franka::kMaxTorqueRate, desired_array, state.tau_J_d);
  const franka::Torques actual = franka_example::filtered_torques(desired, state);

  EXPECT_EQ(actual.tau_J, expected);
}

TEST(FrankaExampleUtilsTest, CartesianPoseErrorUsesBaseFrameSo3LogCoordinates) {
  Eigen::Matrix4d desired = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d current = Eigen::Matrix4d::Identity();
  desired.topLeftCorner<3, 3>() =
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()).toRotationMatrix();
  desired(0, 3) = 0.2;

  const Eigen::Vector<double, 6> error =
      nvidia::srl::controllers::examples::cartesian_pose_error(desired, current);
  EXPECT_NEAR(error(0), 0.2, 1.0e-12);
  EXPECT_NEAR(error(4), 0.4, 1.0e-12);
  EXPECT_TRUE(error(Eigen::seqN(1, 2)).isZero(1.0e-12));
  EXPECT_TRUE(error(Eigen::seqN(3, 1)).isZero(1.0e-12));
  EXPECT_TRUE(error(Eigen::seqN(5, 1)).isZero(1.0e-12));
}

TEST(FrankaExampleUtilsTest, AdmittancePoseOffsetIsRelativeToFixedReferenceInBaseFrame) {
  Eigen::Matrix4d reference = Eigen::Matrix4d::Identity();
  reference.topRightCorner<3, 1>() = Eigen::Vector3d(0.4, -0.2, 0.6);
  reference.topLeftCorner<3, 3>() =
      Eigen::AngleAxisd(franka_example::kPi / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector<double, 6> offset;
  offset << 0.02, -0.01, 0.03, 0.2, 0.0, 0.0;

  const Eigen::Matrix4d desired =
      nvidia::srl::controllers::examples::apply_cartesian_pose_offset(reference, offset);
  EXPECT_TRUE((desired.topRightCorner<3, 1>().isApprox(Eigen::Vector3d(0.42, -0.21, 0.63))));
  // A base-x rotation leaves the reference's local y axis (-base x) unchanged.
  EXPECT_TRUE((desired.topLeftCorner<3, 3>().col(1).isApprox(-Eigen::Vector3d::UnitX())));
  EXPECT_NEAR(desired(2, 0), std::sin(0.2), 1.0e-12);
  EXPECT_NEAR(desired(1, 2), -std::sin(0.2), 1.0e-12);

  // Reusing or resetting the state must not accumulate a second pose trajectory.
  EXPECT_TRUE(nvidia::srl::controllers::examples::apply_cartesian_pose_offset(reference, offset)
                  .isApprox(desired));
  offset.setZero();
  EXPECT_TRUE(nvidia::srl::controllers::examples::apply_cartesian_pose_offset(reference, offset)
                  .isApprox(reference));
}

}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(FrankaForceTank, PowerSignDepletionAndUpstreamFailure) {
  franka_example::force_tank::ForceTankWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  const Eigen::Matrix<double, 7, 7> mass = Eigen::Matrix<double, 7, 7>::Identity();
  const Eigen::Matrix<double, 6, 7> jacobian = Eigen::Matrix<double, 6, 7>::Identity();
  Eigen::Vector<double, 7> velocity = Eigen::Vector<double, 7>::Zero();
  velocity[2] = -0.1;
  Eigen::Vector<double, 6> zero = Eigen::Vector<double, 6>::Zero();
  Eigen::Vector<double, 6> desired = zero;
  desired[2] = -10;
  ASSERT_TRUE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 1.0));
  EXPECT_LT(workspace.power_watts(), 0);
  EXPECT_DOUBLE_EQ(workspace.raw_force_wrench()[2], -20.0);
  for (std::size_t axis = 0; axis < 6; ++axis) {
    EXPECT_DOUBLE_EQ(workspace.wrench()[axis],
                     workspace.impedance_wrench()[axis] +
                         workspace.scale() * workspace.raw_force_wrench()[axis]);
  }
  EXPECT_LT(workspace.scale(), 1);
  const double expected_recharge = 0.5 * 1.4 * std::sqrt(50.0 / 1.000001) * 0.01;
  EXPECT_NEAR(workspace.damping_recharge_watts(), expected_recharge, 1e-12);
  EXPECT_NEAR(workspace.energy_joules(), 0.01 + expected_recharge, 1e-12);
  const double before_recharge = workspace.energy_joules();
  velocity[2] = 0.1;
  ASSERT_TRUE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 0.001));
  EXPECT_GT(workspace.power_watts(), 0);
  EXPECT_LT(workspace.scale(), 1);
  EXPECT_NEAR(
      workspace.energy_joules(),
      before_recharge + 0.001 * (workspace.scale() * workspace.power_watts() + expected_recharge),
      1e-12);
  velocity.setZero();
  const double before_stationary = workspace.energy_joules();
  ASSERT_TRUE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 0.01));
  EXPECT_EQ(workspace.power_watts(), 0);
  EXPECT_NEAR(workspace.energy_joules(), before_stationary, 1e-12);
  const double held = workspace.energy_joules();
  Eigen::Matrix<double, 7, 7> invalid = mass;
  invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(workspace.step(zero, velocity, invalid, jacobian, zero, desired, 0.01));
  EXPECT_EQ(workspace.energy_joules(), held);
  for (double axis : workspace.wrench()) EXPECT_EQ(axis, 0);
  EXPECT_FALSE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 0));
  EXPECT_EQ(workspace.energy_joules(), held);
  ASSERT_TRUE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 0.01));
  EXPECT_NEAR(workspace.energy_joules(), held, 1e-12);
}

TEST(FrankaForceTank, RechargeUsesOnlyImpedanceDampingAndPreservesSampledBalance) {
  franka_example::force_tank::ForceTankWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  Eigen::Matrix<double, 7, 7> mass = Eigen::Matrix<double, 7, 7>::Identity();
  mass(0, 1) = mass(1, 0) = 0.2;
  const Eigen::Matrix<double, 6, 7> jacobian = Eigen::Matrix<double, 6, 7>::Identity();
  Eigen::Vector<double, 7> velocity = Eigen::Vector<double, 7>::Zero();
  velocity[0] = 0.01;
  velocity[1] = -0.02;
  Eigen::Vector<double, 6> zero = Eigen::Vector<double, 6>::Zero();
  Eigen::Vector<double, 6> desired = zero;
  desired[2] = -10;
  // With zero error/acceleration and constant mass/J, impedance contains only damping.
  ASSERT_TRUE(workspace.step(zero, velocity, mass, jacobian, zero, desired, 0.001));
  const Eigen::Vector<double, 6> twist = jacobian * velocity;
  const double damping_power =
      -twist.dot(Eigen::Map<const Eigen::Vector<double, 6>>(workspace.impedance_wrench().data()));
  EXPECT_GT(damping_power, 0);
  EXPECT_NEAR(workspace.damping_recharge_watts(), 0.5 * damping_power, 1e-12);
  Eigen::Vector<double, 6> error = zero;
  error[0] = 0.03;
  const double recharge_without_spring = workspace.damping_recharge_watts();
  ASSERT_TRUE(workspace.step(error, velocity, mass, jacobian, zero, desired, 0.001));
  EXPECT_NEAR(workspace.damping_recharge_watts(), recharge_without_spring, 1e-12);
  for (int step = 0; step < 3000; ++step) {
    velocity[2] = step < 1000 ? -0.1 : 0.001 * std::sin(0.15 * step);
    const double before = workspace.energy_joules();
    ASSERT_TRUE(workspace.step(error, velocity, mass, jacobian, zero, desired, 0.001));
    const double spent = std::max(-workspace.power_watts(), 0.0) * workspace.scale() * 0.001;
    EXPECT_LE(spent, before - 0.01 + 1e-12);
    EXPECT_NEAR(workspace.energy_joules(),
                std::min(1.0,
                         before + 0.001 * (workspace.scale() * workspace.power_watts() +
                                           workspace.damping_recharge_watts())),
                1e-12);
    EXPECT_GE(workspace.energy_joules(), 0.01 - 1e-12);
  }
}

TEST(FrankaCartesianField, PreparedWorkspaceHasUncappedMomentAndRejectsInvalidModel) {
  namespace field = franka_example::cartesian_field;
  field::FieldWorkspace workspace;
  ASSERT_TRUE(workspace.prepare().ok());
  field::Jacobian jacobian = field::Jacobian::Zero();
  jacobian.leftCols<6>().setIdentity();
  field::Matrix7 mass = field::Matrix7::Identity();
  field::Vector6 error = field::Vector6::Zero();
  error[3] = 2.0;
  nvidia::srl::controllers::examples::CartesianFieldSample sample;
  sample.force_n = {1.0, 2.0, 3.0};
  ASSERT_TRUE(workspace.step(error,
                             field::Vector6::Zero(),
                             field::Vector7::Zero(),
                             field::Vector7::Zero(),
                             mass,
                             jacobian,
                             sample));
  EXPECT_DOUBLE_EQ(workspace.wrench()[0], 1.0);
  EXPECT_DOUBLE_EQ(workspace.wrench()[3], 20.0);
  mass(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(workspace.step(error,
                              field::Vector6::Zero(),
                              field::Vector7::Zero(),
                              field::Vector7::Zero(),
                              mass,
                              jacobian,
                              sample));
}

TEST(FrankaCartesianField, WallsCommandOnlyUncappedTranslationTorque) {
  namespace field = franka_example::cartesian_field;
  namespace examples = nvidia::srl::controllers::examples;
  field::FieldWorkspace workspace;
  ASSERT_TRUE(workspace.prepare(examples::CartesianFieldKind::kVirtualWalls).ok());
  field::Jacobian jacobian = field::Jacobian::Zero();
  jacobian.leftCols<6>().setIdentity();
  examples::CartesianFieldSample sample;
  for (const auto force : {examples::FieldVector3{}, examples::FieldVector3{120, -100, 40}}) {
    sample.force_n = force;
    ASSERT_TRUE(workspace.step(field::Vector6::Ones(),
                               field::Vector6::Ones(),
                               field::Vector7::Ones(),
                               field::Vector7::Ones(),
                               field::Matrix7::Identity(),
                               jacobian,
                               sample));
    field::Vector7 expected = field::Vector7::Zero();
    for (std::size_t axis = 0; axis < 3; ++axis) expected[axis] = force[axis];
    EXPECT_TRUE(workspace.joint_torques(jacobian, field::Vector7::Constant(2.0)) == expected);
    for (const double value : workspace.nullspace_torque()) EXPECT_DOUBLE_EQ(value, 0.0);
  }
}
