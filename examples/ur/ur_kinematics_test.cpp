/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file ur_kinematics_test.cpp
 * @brief Focused finite-difference checks for the UR example Jacobian.
 */

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "./ur_kinematics.hpp"

namespace {

constexpr double kDifferenceStep = 1.0e-7;
constexpr double kTolerance = 2.0e-5;

Eigen::Vector3d rotation_difference(const Eigen::Matrix3d& plus, const Eigen::Matrix3d& minus) {
  const Eigen::AngleAxisd difference(plus * minus.transpose());
  return difference.axis() * difference.angle() / (2.0 * kDifferenceStep);
}

bool check_model(const std::string& urdf_path) {
  ur_impedance_example::UrKinematics model(urdf_path);
  const std::array<double, 6> tcp_offset{0.02, -0.01, 0.15, 0.1, -0.05, 0.02};
  if (!model.configure_tcp(tcp_offset)) {
    std::cerr << urdf_path << ": failed to configure TCP\n";
    return false;
  }
  const std::array<double, 6> zero_position{};
  Eigen::Matrix4d zero_pose = Eigen::Matrix4d::Identity();
  ur_impedance_example::Jacobian zero_jacobian = ur_impedance_example::Jacobian::Zero();
  if (!model.compute(zero_position, &zero_pose, &zero_jacobian) || zero_pose(0, 3) >= -0.5) {
    std::cerr << urdf_path << ": kinematics are not expressed in the UR controller base frame\n";
    return false;
  }
  const std::array<double, 6> position{0.2, -1.0, 1.1, -0.7, 0.5, 0.3};
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  ur_impedance_example::Jacobian jacobian = ur_impedance_example::Jacobian::Zero();
  if (!model.compute(position, &pose, &jacobian) || !pose.allFinite() || !jacobian.allFinite()) {
    std::cerr << urdf_path << ": non-finite pose or Jacobian\n";
    return false;
  }

  ur_impedance_example::Jacobian finite_difference = ur_impedance_example::Jacobian::Zero();
  for (std::size_t joint = 0; joint < position.size(); ++joint) {
    std::array<double, 6> plus_position = position;
    std::array<double, 6> minus_position = position;
    plus_position[joint] += kDifferenceStep;
    minus_position[joint] -= kDifferenceStep;
    Eigen::Matrix4d plus_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d minus_pose = Eigen::Matrix4d::Identity();
    ur_impedance_example::Jacobian unused = ur_impedance_example::Jacobian::Zero();
    if (!model.compute(plus_position, &plus_pose, &unused) ||
        !model.compute(minus_position, &minus_pose, &unused)) {
      std::cerr << urdf_path << ": perturbed kinematics failed\n";
      return false;
    }
    finite_difference.template topRows<3>().col(static_cast<Eigen::Index>(joint)) =
        (plus_pose.topRightCorner<3, 1>() - minus_pose.topRightCorner<3, 1>()) /
        (2.0 * kDifferenceStep);
    finite_difference.template bottomRows<3>().col(static_cast<Eigen::Index>(joint)) =
        rotation_difference(plus_pose.topLeftCorner<3, 3>(), minus_pose.topLeftCorner<3, 3>());
  }

  const double maximum_error = (jacobian - finite_difference).cwiseAbs().maxCoeff();
  if (!std::isfinite(maximum_error) || maximum_error > kTolerance) {
    std::cerr << urdf_path << ": Jacobian finite-difference error=" << maximum_error
              << ", tolerance=" << kTolerance << '\n';
    return false;
  }
  std::cout << urdf_path << ": maximum Jacobian error=" << maximum_error << '\n';
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " UR5E_URDF UR10E_URDF\n";
    return EXIT_FAILURE;
  }
  try {
    return check_model(argv[1]) && check_model(argv[2]) ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "UR kinematics test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
