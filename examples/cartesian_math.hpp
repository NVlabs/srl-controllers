/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_math.hpp
 * @brief Eigen-based pose helpers used only by the hardware examples.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace nvidia {
namespace srl {
namespace controllers {
namespace examples {

inline Eigen::Vector3d calculate_cartesian_translation_error(const Eigen::Matrix4d& desired_pose,
                                                             const Eigen::Matrix4d& current_pose) {
  return desired_pose.topRightCorner<3, 1>() - current_pose.topRightCorner<3, 1>();
}

inline Eigen::Vector3d calculate_cartesian_rotation_error(const Eigen::Matrix4d& desired_pose,
                                                          const Eigen::Matrix4d& current_pose) {
  const Eigen::AngleAxisd orientation_error(desired_pose.topLeftCorner<3, 3>() *
                                            current_pose.topLeftCorner<3, 3>().transpose());
  return orientation_error.angle() * orientation_error.axis();
}

inline Eigen::Vector<double, 6> cartesian_pose_error(const Eigen::Matrix4d& desired_pose,
                                                     const Eigen::Matrix4d& current_pose) {
  Eigen::Vector<double, 6> error;
  error.head<3>() = calculate_cartesian_translation_error(desired_pose, current_pose);
  error.tail<3>() = calculate_cartesian_rotation_error(desired_pose, current_pose);
  return error;
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

inline Eigen::Matrix3d left_jacobian_so3(const Eigen::Vector3d& rotation) {
  const double angle = rotation.norm();
  const Eigen::Matrix3d rotation_skew = skew(rotation);
  if (angle < 1.0e-12) {
    return Eigen::Matrix3d::Identity() + 0.5 * rotation_skew;
  }
  const double angle_squared = angle * angle;
  return Eigen::Matrix3d::Identity() + (1.0 - std::cos(angle)) / angle_squared * rotation_skew +
         (angle - std::sin(angle)) / (angle_squared * angle) * rotation_skew * rotation_skew;
}

inline Eigen::Matrix3d exponential_so3(const Eigen::Vector3d& rotation) {
  const double angle = rotation.norm();
  const Eigen::Matrix3d rotation_skew = skew(rotation);
  if (angle < 1.0e-12) {
    return Eigen::Matrix3d::Identity() + rotation_skew;
  }
  const double angle_squared = angle * angle;
  return Eigen::Matrix3d::Identity() + std::sin(angle) / angle * rotation_skew +
         (1.0 - std::cos(angle)) / angle_squared * rotation_skew * rotation_skew;
}

inline Eigen::Matrix4d apply_cartesian_pose_offset(const Eigen::Matrix4d& reference_pose,
                                                   const Eigen::Vector<double, 6>& pose_offset) {
  Eigen::Matrix4d desired_pose = reference_pose;
  desired_pose.topRightCorner<3, 1>() += pose_offset.head<3>();
  desired_pose.topLeftCorner<3, 3>() =
      exponential_so3(pose_offset.tail<3>()) * reference_pose.topLeftCorner<3, 3>();
  return desired_pose;
}

inline Eigen::Matrix4d exponential_se3(const Eigen::Vector<double, 6>& twist) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<3, 3>() = exponential_so3(twist.tail<3>());
  transform.topRightCorner<3, 1>() = left_jacobian_so3(twist.tail<3>()) * twist.head<3>();
  return transform;
}

inline Eigen::Matrix4d integrated_twist(const Eigen::Vector<double, 6>& twist,
                                        const Eigen::Matrix4d& current_pose,
                                        const double dt,
                                        const bool straight_line = false) {
  Eigen::Vector3d position = current_pose.topRightCorner<3, 1>();
  if (straight_line) {
    position += 0.5 * twist.head<3>() * dt;
  }
  Eigen::Vector<double, 6> corrected_twist;
  corrected_twist.head<3>() = twist.head<3>() - twist.tail<3>().cross(position);
  corrected_twist.tail<3>() = twist.tail<3>();

  Eigen::Matrix4d integrated_pose = exponential_se3(corrected_twist * dt) * current_pose;
  const Eigen::JacobiSVD<Eigen::Matrix3d> decomposition(integrated_pose.topLeftCorner<3, 3>(),
                                                        Eigen::ComputeFullU | Eigen::ComputeFullV);
  integrated_pose.topLeftCorner<3, 3>() =
      decomposition.matrixU() * decomposition.matrixV().transpose();
  return integrated_pose;
}

inline Eigen::VectorXd joint_torques_from_wrench(const Eigen::Vector<double, 6>& wrench,
                                                 const Eigen::MatrixXd& jacobian) {
  if (jacobian.rows() != wrench.rows()) {
    throw std::invalid_argument("Jacobian and wrench have incompatible dimensions.");
  }
  return jacobian.transpose() * wrench;
}

}  // namespace examples
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
