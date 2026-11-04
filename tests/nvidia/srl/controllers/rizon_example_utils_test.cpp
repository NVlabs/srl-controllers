/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>

#include <array>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "examples/rizon/rizon_workspaces.hpp"

namespace {

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

TEST(RizonExampleUtilsTest, CartesianPoseOffsetKeepsTranslationAndRotationIndependent) {
  Eigen::Matrix4d reference_pose = Eigen::Matrix4d::Identity();
  reference_pose.topRightCorner<3, 1>() = Eigen::Vector3d(0.3, -0.2, 0.7);
  Eigen::Vector<double, 6> pose_offset = Eigen::Vector<double, 6>::Zero();
  pose_offset(1) = 0.2;
  pose_offset(3) = 0.4;

  const Eigen::Matrix4d desired_pose =
      nvidia::srl::controllers::examples::apply_cartesian_pose_offset(reference_pose, pose_offset);
  const Eigen::Vector<double, 6> pose_error =
      nvidia::srl::controllers::examples::cartesian_pose_error(desired_pose, reference_pose);

  EXPECT_NEAR(pose_error(0), 0.0, 1.0e-12);
  EXPECT_NEAR(pose_error(1), 0.2, 1.0e-12);
  EXPECT_NEAR(pose_error(2), 0.0, 1.0e-12);
  EXPECT_NEAR(pose_error(3), 0.4, 1.0e-12);
  EXPECT_NEAR(pose_error(4), 0.0, 1.0e-12);
  EXPECT_NEAR(pose_error(5), 0.0, 1.0e-12);
}

TEST(RizonExampleUtilsTest, ConfirmationRequiresAnEmptyLine) {
  const rizon_example::ExampleDescription description{"test", "configuration"};
  EXPECT_DOUBLE_EQ(description.torque_magnitude_limit_factor, 1.0);
  EXPECT_DOUBLE_EQ(description.friction_compensation_percent, 100.0);
  {
    StreamRedirect streams("\n");
    EXPECT_TRUE(rizon_example::confirm_execution(description, 10.0, false));
  }
  {
    StreamRedirect streams("execute\n");
    EXPECT_FALSE(rizon_example::confirm_execution(description, 10.0, false));
  }
  {
    StreamRedirect streams("");
    EXPECT_FALSE(rizon_example::confirm_execution(description, 10.0, false));
  }
  {
    StreamRedirect streams("\n");
    EXPECT_TRUE(rizon_example::confirm_controller_start());
  }
  {
    StreamRedirect streams("start\n");
    EXPECT_FALSE(rizon_example::confirm_controller_start());
  }
}

TEST(RizonExampleUtilsTest, ParsesOptionalDurationAndNonRealtimeMode) {
  char executable[] = "example";
  char serial[] = "Rizon4s-123456";
  char non_rt[] = "--non-rt";
  char duration[] = "--duration-seconds=60.5";
  char invalid_duration[] = "--duration-seconds=0";
  char unknown[] = "--unknown";
  rizon_example::ProgramArguments arguments;

  char* default_argv[] = {executable, serial};
  ASSERT_TRUE(rizon_example::parse_arguments(2, default_argv, &arguments));
  EXPECT_EQ(arguments.robot_serial_number, serial);
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 10.0);
  EXPECT_FALSE(arguments.non_rt_requested);

  char* options_argv[] = {executable, serial, duration, non_rt};
  ASSERT_TRUE(rizon_example::parse_arguments(4, options_argv, &arguments));
  EXPECT_DOUBLE_EQ(arguments.run_duration_seconds, 60.5);
  EXPECT_TRUE(arguments.non_rt_requested);

  char* unknown_argv[] = {executable, serial, unknown};
  EXPECT_FALSE(rizon_example::parse_arguments(3, unknown_argv, &arguments));
  char* duplicate_argv[] = {executable, serial, non_rt, non_rt};
  EXPECT_FALSE(rizon_example::parse_arguments(4, duplicate_argv, &arguments));
  char* duplicate_duration_argv[] = {executable, serial, duration, duration};
  EXPECT_FALSE(rizon_example::parse_arguments(4, duplicate_duration_argv, &arguments));
  char* invalid_duration_argv[] = {executable, serial, invalid_duration};
  EXPECT_FALSE(rizon_example::parse_arguments(3, invalid_duration_argv, &arguments));
  char* misplaced_argv[] = {executable, non_rt, serial};
  EXPECT_FALSE(rizon_example::parse_arguments(3, misplaced_argv, &arguments));
}

TEST(RizonExampleUtilsTest, StartupOnlyEnablesHealthyDisabledRobot) {
  using flexiv::rdk::OperationalStatus;
  using rizon_example::StartupAction;
  EXPECT_EQ(rizon_example::startup_action(OperationalStatus::READY, false, false, false, true),
            StartupAction::kReady);
  EXPECT_EQ(
      rizon_example::startup_action(OperationalStatus::NOT_ENABLED, false, false, false, true),
      StartupAction::kEnable);
  EXPECT_EQ(rizon_example::startup_action(OperationalStatus::BOOTING, false, false, false, true),
            StartupAction::kWait);
  EXPECT_EQ(
      rizon_example::startup_action(OperationalStatus::RELEASING_BRAKE, false, false, false, true),
      StartupAction::kWait);
  EXPECT_EQ(rizon_example::startup_action(OperationalStatus::MINOR_FAULT, true, false, false, true),
            StartupAction::kReject);
  EXPECT_EQ(
      rizon_example::startup_action(OperationalStatus::IN_RECOVERY_STATE, false, false, true, true),
      StartupAction::kReject);
  EXPECT_EQ(
      rizon_example::startup_action(OperationalStatus::IN_REDUCED_STATE, false, true, false, true),
      StartupAction::kReject);
  EXPECT_EQ(rizon_example::startup_action(
                OperationalStatus::ESTOP_NOT_RELEASED, false, false, false, false),
            StartupAction::kReject);
}

TEST(RizonExampleUtilsTest, RobotInfoRequiresSevenJointsAndNoExternalAxes) {
  flexiv::rdk::RobotInfo info;
  info.DoF_m = 7;
  info.DoF = 7;
  info.q_min.assign(7, -1.0);
  info.q_max.assign(7, 1.0);
  info.dq_max.assign(7, 2.0);
  info.tau_max.assign(7, 10.0);
  EXPECT_TRUE(rizon_example::valid_robot_info(info));
  info.DoF_e = 1;
  info.DoF = 8;
  EXPECT_FALSE(rizon_example::valid_robot_info(info));
  info.DoF_e = 0;
  info.DoF = 7;
  info.tau_max[3] = 0.0;
  EXPECT_FALSE(rizon_example::valid_robot_info(info));
}

TEST(RizonExampleUtilsTest, InitialPositionUsesDegreesAndRequiresConvergence) {
  flexiv::rdk::RobotInfo info;
  info.DoF_m = 7;
  info.DoF = 7;
  info.q_min.assign(7, -2.0);
  info.q_max.assign(7, 2.0);
  info.dq_max.assign(7, 2.0);
  info.tau_max.assign(7, 10.0);
  ASSERT_TRUE(rizon_example::is_initial_position_valid(info));
  EXPECT_NEAR(rizon_example::kInitialJointPosition[1], -0.2617993877991494, 1e-12);
  EXPECT_NEAR(rizon_example::kInitialJointPosition[3], 1.3089969389957472, 1e-12);

  flexiv::rdk::RobotStates state;
  state.q.assign(rizon_example::kInitialJointPosition.begin(),
                 rizon_example::kInitialJointPosition.end());
  state.dq.assign(7, 0.0);
  EXPECT_TRUE(rizon_example::has_reached_initial_position(state));
  state.q[3] += 2.0 * rizon_example::kInitialPositionTolerance;
  EXPECT_FALSE(rizon_example::has_reached_initial_position(state));

  info.q_max[3] = 1.0;
  EXPECT_FALSE(rizon_example::is_initial_position_valid(info));
}

TEST(RizonExampleUtilsTest, CustomInitialPositionIsValidatedFormattedAndRequiresConvergence) {
  const std::array<double, rizon_example::kJointCount> custom_position{
      0.0,
      -15.0 * rizon_example::kDegreesToRadians,
      0.0,
      75.0 * rizon_example::kDegreesToRadians,
      45.0 * rizon_example::kDegreesToRadians,
      0.0,
      0.0};
  flexiv::rdk::RobotInfo info;
  info.DoF_m = 7;
  info.DoF = 7;
  info.q_min.assign(7, -2.0);
  info.q_max.assign(7, 2.0);
  info.dq_max.assign(7, 2.0);
  info.tau_max.assign(7, 10.0);
  EXPECT_TRUE(rizon_example::is_initial_position_valid(info, custom_position));
  EXPECT_EQ(rizon_example::format_joint_position_degrees(custom_position),
            "[0, -15, 0, 75, 45, 0, 0]");
  flexiv::rdk::RobotStates state;
  state.q.assign(custom_position.begin(), custom_position.end());
  state.dq.assign(7, 0.0);
  EXPECT_TRUE(rizon_example::has_reached_initial_position(state, custom_position));
  state.q[4] += 2.0 * rizon_example::kInitialPositionTolerance;
  EXPECT_FALSE(rizon_example::has_reached_initial_position(state, custom_position));
  info.q_max[4] = 0.5;
  EXPECT_FALSE(rizon_example::is_initial_position_valid(info, custom_position));
}

TEST(RizonExampleUtilsTest, PoseConversionUsesWxyzQuaternionOrdering) {
  constexpr double kSqrtHalf = 0.7071067811865475244;
  const std::array<double, flexiv::rdk::kPoseSize> pose{
      1.0, 2.0, 3.0, kSqrtHalf, 0.0, 0.0, kSqrtHalf};
  Eigen::Matrix4d transform;
  ASSERT_TRUE(rizon_example::pose_from_rdk(pose, transform));
  EXPECT_TRUE((transform.topRightCorner<3, 1>().isApprox(Eigen::Vector3d(1.0, 2.0, 3.0))));
  EXPECT_TRUE((transform.topLeftCorner<3, 3>() * Eigen::Vector3d::UnitX())
                  .isApprox(Eigen::Vector3d::UnitY(), 1e-12));
}

TEST(RizonExampleUtilsTest, TcpJacobianIncludesToolOffsetAndBodyRotation) {
  rizon_example::Jacobian flange = rizon_example::Jacobian::Zero();
  flange.bottomRows<3>().leftCols<3>().setIdentity();
  Eigen::Matrix4d world_flange = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d world_tcp = Eigen::Matrix4d::Identity();
  world_tcp(0, 3) = 0.2;
  world_tcp.topLeftCorner<3, 3>() =
      Eigen::AngleAxisd(0.5 * 3.14159265358979323846, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  rizon_example::Jacobian world_jacobian;
  rizon_example::Jacobian tcp_jacobian;
  ASSERT_TRUE(rizon_example::derive_tcp_jacobians(
      flange, world_flange, world_tcp, world_jacobian, tcp_jacobian));
  EXPECT_NEAR(world_jacobian(1, 2), 0.2, 1e-12);
  EXPECT_TRUE(tcp_jacobian.topRows<3>().isApprox(
      world_tcp.topLeftCorner<3, 3>().transpose() * world_jacobian.topRows<3>(), 1e-12));
  EXPECT_TRUE(tcp_jacobian.bottomRows<3>().isApprox(
      world_tcp.topLeftCorner<3, 3>().transpose() * world_jacobian.bottomRows<3>(), 1e-12));
}

TEST(RizonExampleUtilsTest, ConvertsAndValidatesRobotTimestamps) {
  std::int64_t timestamp_nanoseconds = 0;
  EXPECT_TRUE(rizon_example::convert_timestamp_to_nanoseconds({12, 345}, timestamp_nanoseconds));
  EXPECT_EQ(timestamp_nanoseconds, 12000000345);
  EXPECT_FALSE(
      rizon_example::convert_timestamp_to_nanoseconds({12, 1000000000}, timestamp_nanoseconds));
  EXPECT_FALSE(rizon_example::convert_timestamp_to_nanoseconds({-1, 0}, timestamp_nanoseconds));
}

TEST(RizonExampleUtilsTest, TorqueFilterAppliesMagnitudeMarginAndSlewLimit) {
  rizon_example::JointVector desired = rizon_example::JointVector::Constant(10.0);
  rizon_example::JointVector previous = rizon_example::JointVector::Zero();
  std::vector<double> limits(7, 20.0);
  std::vector<double> streamed(7, 0.0);
  std::array<bool, 7> magnitude_limited{};
  std::array<bool, 7> slew_limited{};
  ASSERT_TRUE(rizon_example::filter_torques(
      desired, limits, 0.001, 0.75, previous, streamed, magnitude_limited, slew_limited));
  for (const double torque : streamed) {
    EXPECT_NEAR(torque, 1.0, 1e-12);
  }
  EXPECT_FALSE(magnitude_limited[0]);
  EXPECT_TRUE(slew_limited[0]);
  desired[2] = 21.0;
  ASSERT_TRUE(rizon_example::filter_torques(
      desired, limits, 1.0, 0.75, previous, streamed, magnitude_limited, slew_limited));
  EXPECT_NEAR(streamed[2], 15.0, 1e-12);
  EXPECT_TRUE(magnitude_limited[2]);
  EXPECT_FALSE(slew_limited[2]);
  previous[2] = 0.0;
  ASSERT_TRUE(rizon_example::filter_torques(
      desired, limits, 1.0, 1.0, previous, streamed, magnitude_limited, slew_limited));
  EXPECT_NEAR(streamed[2], 20.0, 1e-12);
  EXPECT_TRUE(magnitude_limited[2]);
  EXPECT_FALSE(slew_limited[2]);
  desired[2] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(rizon_example::filter_torques(
      desired, limits, 0.001, 0.75, previous, streamed, magnitude_limited, slew_limited));
}

TEST(RizonExampleUtilsTest, EveryControllerWorkspacePreparesWithoutRobotHardware) {
  rizon_example::JointImpedanceExample joint_impedance;
  const auto joint_impedance_result = joint_impedance.prepare();
  ASSERT_TRUE(joint_impedance_result.ok()) << joint_impedance_result.message;

  rizon_example::CartesianImpedanceExample cartesian_impedance;
  const auto cartesian_impedance_result = cartesian_impedance.prepare();
  ASSERT_TRUE(cartesian_impedance_result.ok()) << cartesian_impedance_result.message;
  rizon_example::CycleData cartesian_cycle;
  cartesian_cycle.joint_position.setZero();
  cartesian_cycle.joint_velocity.setZero();
  cartesian_cycle.mass.setIdentity();
  cartesian_cycle.world_jacobian.leftCols<rizon_example::kCartesianDim>().setIdentity();
  ASSERT_TRUE(cartesian_impedance.set_initial_state(cartesian_cycle));
  cartesian_cycle.world_tcp_pose(0, 3) = 0.01;
  cartesian_cycle.joint_velocity[0] = 0.1;
  cartesian_cycle.joint_position[6] = 0.1;
  rizon_example::JointVector cartesian_torque;
  ASSERT_TRUE(cartesian_impedance.step(
      cartesian_cycle, 0.0, rizon_example::kNominalPeriodSeconds, cartesian_torque));
  constexpr double kExpectedTranslationTorque =
      -2000.0 * 0.01 - 2.0 * 0.7 * std::sqrt(2000.0) * 0.1;
  EXPECT_NEAR(cartesian_torque[0], kExpectedTranslationTorque, 4e-6);
  EXPECT_TRUE(cartesian_torque.segment<5>(1).isZero());
  EXPECT_NEAR(cartesian_torque[6], -1.0, 1e-12);

  rizon_example::VariableCartesianImpedanceExample variable_cartesian_impedance;
  const auto variable_cartesian_impedance_result = variable_cartesian_impedance.prepare();
  ASSERT_TRUE(variable_cartesian_impedance_result.ok())
      << variable_cartesian_impedance_result.message;

  rizon_example::CartesianAdmittanceExample cartesian_admittance;
  const auto cartesian_admittance_result = cartesian_admittance.prepare();
  ASSERT_TRUE(cartesian_admittance_result.ok()) << cartesian_admittance_result.message;
  rizon_example::CycleData admittance_cycle;
  admittance_cycle.mass.setIdentity();
  admittance_cycle.world_jacobian.leftCols<rizon_example::kCartesianDim>().setIdentity();
  ASSERT_TRUE(cartesian_admittance.set_initial_state(admittance_cycle));
  admittance_cycle.external_wrench_world[0] = 10.0;
  rizon_example::CartesianMotionCommand admittance_command;
  for (std::size_t sample = 0; sample < 3; ++sample) {
    ASSERT_TRUE(cartesian_admittance.step(admittance_cycle,
                                          sample * rizon_example::kNominalPeriodSeconds,
                                          rizon_example::kNominalPeriodSeconds,
                                          admittance_command));
  }
  EXPECT_GT(admittance_command.pose[0], 0.0);

  rizon_example::ForceImpedanceExample force_impedance_world(
      nvidia::srl::controllers::CartesianFrame::kRobotBase, -15.0);
  const auto force_impedance_world_result = force_impedance_world.prepare();
  ASSERT_TRUE(force_impedance_world_result.ok()) << force_impedance_world_result.message;

  rizon_example::ForceImpedanceExample force_impedance_ee(
      nvidia::srl::controllers::CartesianFrame::kEndEffector, 10.0);
  const auto force_impedance_ee_result = force_impedance_ee.prepare();
  ASSERT_TRUE(force_impedance_ee_result.ok()) << force_impedance_ee_result.message;
}

}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
