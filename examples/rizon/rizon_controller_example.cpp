/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file rizon_controller_example.cpp
 * @brief Entry point shared by the seven controller Rizon executables.
 */

#include <cstdlib>
#include <iostream>
#include <string_view>

#include "./rizon_utils.hpp"
#include "./rizon_workspaces.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

#ifndef RIZON_EXAMPLE_NAME
#error "RIZON_EXAMPLE_NAME must identify the configured Rizon example"
#endif

int main(int argc, char** argv) {
  constexpr std::string_view kExampleName{RIZON_EXAMPLE_NAME};

  if (kExampleName == "rizon_joint_impedance_controller") {
    rizon_example::JointImpedanceExample workspace;
    return rizon_example::run_example(
        argc,
        argv,
        {"Rizon joint impedance hold",
         "Command: hold the measured start angles for all seven joints.\n"
         "Feedback: link-side q (rad) and dq (rad/s); the model term is the mass matrix.",
         1.0,
         100.0},
        workspace);
  }
  if (kExampleName == "rizon_cartesian_impedance_controller") {
    rizon_example::CartesianImpedanceExample workspace;
    return rizon_example::run_example(
        argc,
        argv,
        {"Rizon Cartesian impedance hold",
         "Command: hold the measured WORLD_T_TCP pose and measured post-motion joint posture.\n"
         "Control: full CartesianImpedanceController with K=[2000,2000,2000,200,200,200], "
         "zeta=[0.7,0.7,0.7,0.7,0.7,0.7], inertial coupling, model Coriolis feedforward, and "
         "nullspace K=[120,80,60,40,30,20,10], D=[12,8,6,4,3,2,1].\n"
         "Frame: WORLD is fixed at the robot base; TCP is the configured tool center. "
         "Translation is m and quaternion input order is [qw, qx, qy, qz]."},
        workspace);
  }
  if (kExampleName == "rizon_variable_cartesian_impedance_controller") {
    rizon_example::VariableCartesianImpedanceExample workspace;
    return rizon_example::run_example(
        argc,
        argv,
        {"Rizon variable Cartesian impedance hold",
         "Command: hold measured WORLD_T_TCP while WORLD-z stiffness varies sinusoidally from "
         "1 to 2000 N/m at 0.1 Hz.\nFrame: WORLD is fixed at the robot base; TCP "
         "is the configured tool center."},
        workspace);
  }
  if (kExampleName == "rizon_cartesian_admittance_controller") {
    rizon_example::CartesianAdmittanceExample workspace;
    return rizon_example::run_cartesian_motion_example(
        argc,
        argv,
        {"Rizon Cartesian admittance",
         "Command: estimated WORLD-frame external wrench displaces a virtual 6-DoF system from "
         "the measured start pose without acceleration, twist, or pose limiting. The resulting "
         "pose is streamed through RDK RT Cartesian position control.\n"
         "Wrench is ext_wrench_in_world ordered force (N), moment (Nm); a physical F/T sensor is "
         "not required."},
        workspace);
  }
  if (kExampleName == "rizon_cartesian_force_impedance_controller") {
    rizon_example::ForceImpedanceExample workspace(
        nvidia::srl::controllers::CartesianFrame::kRobotBase, -15.0);
    return rizon_example::run_example(
        argc,
        argv,
        {"Rizon WORLD-frame force/impedance",
         "Setup: place the tool in unloaded compliant contact whose normal is WORLD -z; verify "
         "the configured TCP and payload first.\nCommand: hold measured WORLD_T_TCP and ramp "
         "WORLD-z force from 0 to -15 N over 2 s. The shaping envelope is 0.06 m/rad to "
         "0.10 m/rad."},
        workspace);
  }
  if (kExampleName == "rizon_cartesian_force_impedance_controller_in_ee") {
    constexpr std::array<double, rizon_example::kJointCount> kInitialJointPosition{
        0.0,
        -15.0 * rizon_example::kDegreesToRadians,
        0.0,
        75.0 * rizon_example::kDegreesToRadians,
        45.0 * rizon_example::kDegreesToRadians,
        0.0,
        0.0};
    rizon_example::ForceImpedanceExample workspace(
        nvidia::srl::controllers::CartesianFrame::kEndEffector, 10.0);
    return rizon_example::run_example(
        argc,
        argv,
        {"Rizon TCP-frame force/impedance",
         "Setup: place the tool in unloaded compliant contact whose normal is TCP -z; verify the "
         "configured TCP and payload first.\nCommand: hold measured WORLD_T_TCP and ramp TCP-z "
         "force from 0 to +10 N over 2 s. The shaping envelope is 0.06 m/rad to 0.10 m/rad.",
         1.0,
         100.0,
         kInitialJointPosition},
        workspace);
  }

  std::cerr << "Unsupported configured Rizon example: " << kExampleName << '\n';
  return EXIT_FAILURE;
}
