// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#pragma once
#include "nvidia/srl/controllers/composition.hpp"
#include "nvidia/srl/controllers/controllers.hpp"
#include "nvidia/srl/controllers/equations.hpp"

// This consumer is compiled against a relocated installed prefix only.
inline bool public_surface_is_available() {
  namespace c = nvidia::srl::controllers;
  c::Requirements requirements;
  {
    c::TorqueFilter::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::TorqueFilter::requirements(config, &requirements).ok()) return false;
  }
  {
    c::JointImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::JointImpedanceController::requirements(config, &requirements).ok()) return false;
  }
  {
    c::NullspaceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::NullspaceController::requirements(config, &requirements).ok()) return false;
  }
  {
    c::OperationalSpaceMass::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::OperationalSpaceMass::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CartesianImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianImpedanceController::requirements(config, &requirements).ok()) return false;
  }
  {
    c::BimanualCartesianImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 14;
    config.cartesian_frame = c::CartesianFrame::kMainFrameOrientation;
    config.task_dim = 12;
    if (!c::BimanualCartesianImpedanceController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::BimanualTaskKinematics::Config config;
    config.batch_size = 1;
    config.joint_dim = 14;
    config.cartesian_frame = c::CartesianFrame::kMainFrameOrientation;
    config.task_dim = 12;
    if (!c::BimanualTaskKinematics::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CartesianAdmittancePoseController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianAdmittancePoseController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::OperationalSpaceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::OperationalSpaceController::requirements(config, &requirements).ok()) return false;
    if (!c::OperationalSpaceInverseDynamicsController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianForceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianForceController::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CartesianUnifiedForceImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianUnifiedForceImpedanceController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::StaticJointImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    if (!c::StaticJointImpedanceController::requirements(config, &requirements).ok()) return false;
  }
  {
    c::StaticCartesianImpedanceController::Config config;
    config.batch_size = 1;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::StaticCartesianImpedanceController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianUnifiedForceAdmittanceController::Config config;
    config.batch_size = 1;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianUnifiedForceAdmittanceController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::VariableCartesianImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::VariableCartesianImpedanceController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::TaskWrenchToJointTorqueController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::TaskWrenchToJointTorqueController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianImpedanceTorqueController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianImpedanceTorqueController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::VariableCartesianImpedanceTorqueController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::VariableCartesianImpedanceTorqueController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianUnifiedForceImpedanceTorqueController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianUnifiedForceImpedanceTorqueController::requirements(config, &requirements)
             .ok())
      return false;
  }
  {
    c::BimanualCartesianImpedanceTorqueController::Config config;
    config.batch_size = 1;
    config.joint_dim = 14;
    config.cartesian_frame = c::CartesianFrame::kMainFrameOrientation;
    config.task_dim = 12;
    if (!c::BimanualCartesianImpedanceTorqueController::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::ProportionalBlock::Config config;
    config.batch_size = 1;
    config.signal_dim = 97;
    if (!c::ProportionalBlock::requirements(config, &requirements).ok()) return false;
  }
  {
    c::IntegratorBlock::Config config;
    config.batch_size = 1;
    config.signal_dim = 97;
    if (!c::IntegratorBlock::requirements(config, &requirements).ok()) return false;
  }
  {
    c::DifferentiatorBlock::Config config;
    config.batch_size = 1;
    config.signal_dim = 97;
    if (!c::DifferentiatorBlock::requirements(config, &requirements).ok()) return false;
  }
  {
    c::JointStiffness::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::JointStiffness::requirements(config, &requirements).ok()) return false;
  }
  {
    c::JointDamping::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::JointDamping::requirements(config, &requirements).ok()) return false;
  }
  {
    c::JointDampingDoubleDiagonalization::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::JointDampingDoubleDiagonalization::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianStiffness::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianStiffness::requirements(config, &requirements).ok()) return false;
  }
  {
    c::VariableCartesianStiffness::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::VariableCartesianStiffness::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CartesianDamping::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianDamping::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CartesianDampingDoubleDiagonalization::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianDampingDoubleDiagonalization::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianDampingFactorizationDesign::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianDampingFactorizationDesign::requirements(config, &requirements).ok())
      return false;
  }
  {
    c::CartesianVirtualMass::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::CartesianVirtualMass::requirements(config, &requirements).ok()) return false;
  }
  {
    c::ShapingFunction::Config config;
    config.batch_size = 1;
    config.signal_dim = 97;
    if (!c::ShapingFunction::requirements(config, &requirements).ok()) return false;
  }
  {
    c::CommandObserver::Config config;
    config.batch_size = 1;
    config.signal_dim = 97;
    if (!c::CommandObserver::requirements(config, &requirements).ok()) return false;
  }
  {
    c::FlowFunnel::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::FlowFunnel::requirements(config, &requirements).ok()) return false;
  }
  {
    c::InverseAPF::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::InverseAPF::requirements(config, &requirements).ok()) return false;
  }
  {
    c::InertialCoupling::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::InertialCoupling::requirements(config, &requirements).ok()) return false;
  }
  {
    c::EnergyTank::Config config;
    config.batch_size = 1;
    config.signal_dim = 1;
    if (!c::EnergyTank::requirements(config, &requirements).ok()) return false;
  }
  {
    c::VariableCartesianUnifiedForceImpedanceController::Config config;
    config.batch_size = 1;
    config.joint_dim = 7;
    config.cartesian_frame = c::CartesianFrame::kRobotBase;
    if (!c::VariableCartesianUnifiedForceImpedanceController::requirements(config, &requirements)
             .ok())
      return false;
  }
  c::CartesianImpedanceTorqueController::Config config;
  config.batch_size = 1;
  config.joint_dim = 7;
  config.cartesian_frame = c::CartesianFrame::kRobotBase;
  c::CartesianTorqueRequirements complete;
  if (!c::CartesianImpedanceTorqueController::component_requirements(config, &complete).ok() ||
      complete.mass.output_count == 0 || complete.task.state_count == 0)
    return false;
  if (c::equations::proportional_value(2., 3.) != 6.) return false;
  std::uint64_t fault = static_cast<std::uint64_t>(c::StatusFlag::kInvalidInput), flags = 0;
  const c::StatusView input{&fault, 1, c::Device::kCpu};
  return c::merge_status(&input, 1, {&flags, 1, c::Device::kCpu}).ok() &&
         (flags & static_cast<std::uint64_t>(c::StatusFlag::kSafeMode));
}
