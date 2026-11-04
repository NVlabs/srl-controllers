/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file component_core_test.cpp
 * @brief Coverage for the allocation-free controller equations.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

#include "nvidia/srl/controllers/detail/component_core.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace {

TEST(ComponentCoreTest, StridedViewsUseCanonicalDenseFeedbackEquation) {
  const double matrix_storage[]{2.0, -99.0, 1.0, -99.0, -3.0, -99.0, 4.0};
  const double vector_storage[]{5.0, -99.0, 7.0};
  const detail::ConstMatrixView<double> matrix{matrix_storage, 2, 2, 4, 2};
  const detail::ConstVectorView<double> vector{vector_storage, 2, 2};

  EXPECT_DOUBLE_EQ(detail::cartesian_stiffness_value(matrix, vector, 0), 17.0);
  EXPECT_DOUBLE_EQ(detail::joint_damping_value(matrix, vector, 1), 13.0);
}

TEST(ComponentCoreTest, StatefulAndCompositeEquationsAreCanonical) {
  const auto integral = detail::integrator_trapezoid_value(0.5, 2.0, 1.0, 0.1, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(integral.value, 0.65);
  EXPECT_FALSE(integral.saturated);

  bool saturated = false;
  EXPECT_DOUBLE_EQ(detail::torque_filter_value(4.0, 0.5, -2.0, 2.0, 1.0, 0.1, true, &saturated),
                   0.6);
  EXPECT_TRUE(saturated);

  EXPECT_DOUBLE_EQ(detail::cartesian_impedance_value(1.0, 2.0, 3.0, 4.0), 10.0);
  EXPECT_DOUBLE_EQ(detail::admittance_force_balance_value(4.0, 3.0, 2.0, 1.0), 4.0);
  EXPECT_DOUBLE_EQ(detail::unified_force_impedance_value(5.0, -2.0), 3.0);
  EXPECT_DOUBLE_EQ(detail::safe_output_value(true, true, true, 7.0, 4.0), 4.0);
  EXPECT_DOUBLE_EQ(detail::safe_output_value(true, false, true, 7.0, 4.0), 0.0);
}

TEST(ComponentCoreTest, CartesianGeometryUsesCanonicalEquations) {
  const std::array<double, 3> start{0.0, 0.0, 0.0};
  const std::array<double, 3> end{2.0, 0.0, 0.0};
  const std::array<double, 3> position{1.0, 3.0, 0.0};
  std::array<double, 3> direction{};
  std::array<double, 3> direction_to_projection{};
  const auto geometry = detail::flow_funnel_geometry(
      detail::ConstVectorView<double>{start.data(), start.size(), 1},
      detail::ConstVectorView<double>{end.data(), end.size(), 1},
      detail::ConstVectorView<double>{position.data(), position.size(), 1},
      detail::MutableVectorView<double>{direction.data(), direction.size(), 1},
      detail::MutableVectorView<double>{
          direction_to_projection.data(), direction_to_projection.size(), 1});

  EXPECT_TRUE(geometry.valid);
  EXPECT_DOUBLE_EQ(geometry.length, 2.0);
  EXPECT_DOUBLE_EQ(geometry.projection_distance, 1.0);
  EXPECT_DOUBLE_EQ(geometry.radius, 3.0);
  EXPECT_EQ(direction, (std::array<double, 3>{1.0, 0.0, 0.0}));
  EXPECT_EQ(direction_to_projection, (std::array<double, 3>{0.0, -1.0, 0.0}));

  const std::array<double, 16> identity{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 16> quarter_turn_z{
      0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> rotation_error{};
  EXPECT_TRUE(detail::so3_log_error(
      detail::ConstMatrixView<double>{identity.data(), 4, 4, 4, 1},
      detail::ConstMatrixView<double>{quarter_turn_z.data(), 4, 4, 4, 1},
      detail::MutableVectorView<double>{rotation_error.data(), rotation_error.size(), 1}));
  EXPECT_NEAR(rotation_error[0], 0.0, 1.0e-12);
  EXPECT_NEAR(rotation_error[1], 0.0, 1.0e-12);
  EXPECT_NEAR(rotation_error[2], detail::kPi / 2.0, 1.0e-12);

  EXPECT_DOUBLE_EQ(detail::nullspace_projector_value(1U, 1U, 0.25), 0.75);
  EXPECT_DOUBLE_EQ(detail::nullspace_projector_value(1U, 0U, 0.25), -0.25);
  EXPECT_DOUBLE_EQ(detail::nullspace_projected_torque_value(4.0, 1.5), 2.5);
}

}  // namespace
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
