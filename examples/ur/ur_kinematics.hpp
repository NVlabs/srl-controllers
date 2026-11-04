/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file ur_kinematics.hpp
 * @brief Fixed-size Pinocchio kinematics for UR5e and UR10e example URDFs.
 */

#pragma once

#include <eigen3/Eigen/Dense>

#include <array>
#include <cstddef>
#include <string>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ur_impedance_example {

constexpr std::size_t kUrJointCount = 6;
using JointVector = Eigen::Matrix<double, 6, 1>;
using CartesianVector = Eigen::Matrix<double, 6, 1>;
using Jacobian = Eigen::Matrix<double, 6, 6>;

class UrKinematics {
 public:
  explicit UrKinematics(const std::string& urdf_path);

  UrKinematics(const UrKinematics&) = delete;
  UrKinematics& operator=(const UrKinematics&) = delete;

  bool configure_tcp(const std::array<double, 6>& tcp_offset);
  bool compute(const std::array<double, 6>& joint_position,
               Eigen::Matrix4d* base_to_tcp,
               Jacobian* base_aligned_tcp_jacobian) noexcept;

 private:
  bool dimensions_are_supported() const noexcept;

  pinocchio::Model model_{};
  pinocchio::Data data_{model_};
  pinocchio::FrameIndex base_frame_{0U};
  pinocchio::FrameIndex flange_frame_{0U};
  pinocchio::FrameIndex tcp_frame_{0U};
  JointVector configuration_{JointVector::Zero()};
  Jacobian frame_jacobian_{Jacobian::Zero()};
  bool tcp_configured_{false};
};

Eigen::Matrix4d pose_vector_to_transform(const std::array<double, 6>& pose) noexcept;

}  // namespace ur_impedance_example
