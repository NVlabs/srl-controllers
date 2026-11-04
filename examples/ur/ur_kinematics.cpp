/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file ur_kinematics.cpp
 * @brief Fixed-size Pinocchio kinematics for UR impedance examples.
 */

#include "ur_kinematics.hpp"  // NOLINT(build/include_subdir)

#include <eigen3/Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace ur_impedance_example {

namespace {

constexpr const char* kFlangeFrame = "flange";
constexpr const char* kBaseFrame = "base";
constexpr const char* kTcpFrame = "configured_tcp";
constexpr std::array<const char*, kUrJointCount> kJointNames{"shoulder_pan_joint",
                                                             "shoulder_lift_joint",
                                                             "elbow_joint",
                                                             "wrist_1_joint",
                                                             "wrist_2_joint",
                                                             "wrist_3_joint"};

bool finite_array(const std::array<double, 6>& values) noexcept {
  return std::all_of(
      values.begin(), values.end(), [](const double value) { return std::isfinite(value); });
}

Eigen::Matrix3d rotation_vector_to_matrix(const Eigen::Vector3d& rotation_vector) noexcept {
  const double angle = rotation_vector.norm();
  return angle > 1.0e-12 ? Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix()
                         : Eigen::Matrix3d::Identity();
}

}  // namespace

UrKinematics::UrKinematics(const std::string& urdf_path) {
  pinocchio::urdf::buildModel(urdf_path, model_);
  data_ = pinocchio::Data(model_);
  base_frame_ = model_.getFrameId(kBaseFrame);
  flange_frame_ = model_.getFrameId(kFlangeFrame);
  if (!dimensions_are_supported()) {
    throw std::runtime_error(
        "URDF must contain the expected six UR joints plus base and flange frames");
  }
}

bool UrKinematics::dimensions_are_supported() const noexcept {
  if (model_.nq != static_cast<int>(kUrJointCount) ||
      model_.nv != static_cast<int>(kUrJointCount) ||
      base_frame_ >= static_cast<pinocchio::FrameIndex>(model_.nframes) ||
      flange_frame_ >= static_cast<pinocchio::FrameIndex>(model_.nframes)) {
    return false;
  }
  for (const char* joint_name : kJointNames) {
    if (!model_.existJointName(joint_name)) {
      return false;
    }
  }
  return true;
}

bool UrKinematics::configure_tcp(const std::array<double, 6>& tcp_offset) {
  if (tcp_configured_ || !finite_array(tcp_offset) || !dimensions_are_supported()) {
    return false;
  }
  const Eigen::Vector3d translation(tcp_offset[0], tcp_offset[1], tcp_offset[2]);
  const Eigen::Vector3d rotation_vector(tcp_offset[3], tcp_offset[4], tcp_offset[5]);
  const pinocchio::SE3 flange_to_tcp(rotation_vector_to_matrix(rotation_vector), translation);
  const pinocchio::Frame& flange = model_.frames[flange_frame_];
  tcp_frame_ = model_.addFrame(pinocchio::Frame(kTcpFrame,
                                                flange.parentJoint,
                                                flange_frame_,
                                                flange.placement * flange_to_tcp,
                                                pinocchio::OP_FRAME));
  data_ = pinocchio::Data(model_);
  tcp_configured_ = tcp_frame_ < static_cast<pinocchio::FrameIndex>(model_.nframes);
  return tcp_configured_;
}

bool UrKinematics::compute(const std::array<double, 6>& joint_position,
                           Eigen::Matrix4d* base_to_tcp,
                           Jacobian* base_aligned_tcp_jacobian) noexcept {
  if (!tcp_configured_ || base_to_tcp == nullptr || base_aligned_tcp_jacobian == nullptr ||
      !finite_array(joint_position)) {
    return false;
  }
  for (std::size_t joint = 0; joint < kUrJointCount; ++joint) {
    configuration_[static_cast<Eigen::Index>(joint)] = joint_position[joint];
  }
  pinocchio::forwardKinematics(model_, data_, configuration_);
  pinocchio::computeJointJacobians(model_, data_, configuration_);
  pinocchio::updateFramePlacements(model_, data_);
  frame_jacobian_.setZero();
  pinocchio::getFrameJacobian(
      model_, data_, tcp_frame_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, frame_jacobian_);
  const pinocchio::SE3& world_from_base = data_.oMf[base_frame_];
  const pinocchio::SE3& world_from_tcp = data_.oMf[tcp_frame_];
  const Eigen::Matrix3d base_from_world = world_from_base.rotation().transpose();
  base_to_tcp->setIdentity();
  base_to_tcp->topLeftCorner<3, 3>() = base_from_world * world_from_tcp.rotation();
  base_to_tcp->topRightCorner<3, 1>() =
      base_from_world * (world_from_tcp.translation() - world_from_base.translation());
  base_aligned_tcp_jacobian->topRows<3>() = base_from_world * frame_jacobian_.topRows<3>();
  base_aligned_tcp_jacobian->bottomRows<3>() = base_from_world * frame_jacobian_.bottomRows<3>();
  return base_to_tcp->allFinite() && base_aligned_tcp_jacobian->allFinite();
}

Eigen::Matrix4d pose_vector_to_transform(const std::array<double, 6>& pose) noexcept {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  if (!finite_array(pose)) {
    transform.setConstant(std::numeric_limits<double>::quiet_NaN());
    return transform;
  }
  transform.topRightCorner<3, 1>() = Eigen::Vector3d(pose[0], pose[1], pose[2]);
  transform.topLeftCorner<3, 3>() =
      rotation_vector_to_matrix(Eigen::Vector3d(pose[3], pose[4], pose[5]));
  return transform;
}

}  // namespace ur_impedance_example
