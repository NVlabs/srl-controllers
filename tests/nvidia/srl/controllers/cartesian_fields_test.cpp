// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "examples/cartesian_fields.hpp"

namespace {
namespace field = nvidia::srl::controllers::examples;
namespace equations = nvidia::srl::controllers::equations;

TEST(CartesianFields, AllSixFacesPushInwardAndCornersCompose) {
  const field::CartesianFieldParameters parameters;
  ASSERT_TRUE(field::validate_cartesian_field_parameters(parameters));
  const field::FieldVector3 origin{};
  field::CartesianFieldSample sample;
  ASSERT_TRUE(field::evaluate_cartesian_field(
      field::CartesianFieldKind::kVirtualWalls, parameters, origin, origin, origin, 1.0, &sample));
  EXPECT_EQ(sample.force_n, origin);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      field::FieldVector3 position{};
      position[axis] = sign * 0.11;
      ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                                  parameters,
                                                  origin,
                                                  position,
                                                  origin,
                                                  1,
                                                  &sample));
      EXPECT_NEAR(sample.force_n[axis], -sign * 10.0, 1e-12);
      for (std::size_t other = 0; other < 3; ++other)
        if (other != axis) {
          EXPECT_DOUBLE_EQ(sample.force_n[other], 0.0);
        }
    }
  }
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                              parameters,
                                              origin,
                                              {0.11, -0.11, 0.11},
                                              origin,
                                              1,
                                              &sample));
  EXPECT_NEAR(sample.force_n[0], -10.0, 1e-12);
  EXPECT_NEAR(sample.force_n[1], 10.0, 1e-12);
  EXPECT_NEAR(sample.force_n[2], -10.0, 1e-12);
}

TEST(CartesianFields, MovingInsideBoxHasZeroForceAndPenetrationForceIsUncapped) {
  const field::CartesianFieldParameters parameters;
  const field::FieldVector3 origin{};
  field::CartesianFieldSample sample;
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                              parameters,
                                              origin,
                                              {0.05, -0.1, 0.1},
                                              {1.0, -2.0, 3.0},
                                              1,
                                              &sample));
  EXPECT_EQ(sample.force_n, origin);
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                              parameters,
                                              origin,
                                              {0.2, -0.2, 0.2},
                                              {0.5, -0.5, 0.0},
                                              1,
                                              &sample));
  EXPECT_NEAR(sample.force_n[0], -120.0, 1e-12);
  EXPECT_NEAR(sample.force_n[1], 120.0, 1e-12);
  EXPECT_NEAR(sample.force_n[2], -100.0, 1e-12);
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                              parameters,
                                              origin,
                                              {0.11, 0, 0},
                                              {-1.0, 2.0, 3.0},
                                              1,
                                              &sample));
  EXPECT_EQ(sample.force_n, origin);  // inward motion cannot make a wall pull outward
}

TEST(CartesianFields, UnilateralWallIsContinuousAndDissipatesNormalMotion) {
  EXPECT_DOUBLE_EQ(equations::virtual_wall_force_value(0.0, -1.0, 500.0, 20.0, 0.005), 0.0);
  EXPECT_DOUBLE_EQ(equations::virtual_wall_force_value(0.01, -1.0, 500.0, 20.0, 0.005), 0.0);
  EXPECT_LT(equations::virtual_wall_force_value(-1e-10, -1.0, 500.0, 20.0, 0.005), 1e-6);
  EXPECT_DOUBLE_EQ(equations::virtual_wall_force_value(-0.01, 1.0, 500.0, 20.0, 0.005), 0.0);
  EXPECT_DOUBLE_EQ(equations::virtual_wall_force_value(-0.01, -1.0, 500.0, 20.0, 0.005), 25.0);
}

TEST(CartesianFields, SphereForceIsNegativePotentialGradientAndVanishesAtInfluenceBoundary) {
  constexpr double gain = 0.00005, influence = 0.035, clearance = 0.02, epsilon = 1e-7;
  const auto potential = [](const double distance) {
    const double inverse_difference = 1.0 / distance - 1.0 / influence;
    return 0.5 * gain * inverse_difference * inverse_difference;
  };
  const double negative_gradient =
      -(potential(clearance + epsilon) - potential(clearance - epsilon)) / (2.0 * epsilon);
  EXPECT_NEAR(equations::spherical_apf_repulsion_value(clearance, influence, gain),
              negative_gradient,
              1e-8);
  EXPECT_DOUBLE_EQ(equations::spherical_apf_repulsion_value(influence, influence, gain), 0.0);
  EXPECT_DOUBLE_EQ(equations::spherical_apf_repulsion_value(0.1, influence, gain), 0.0);
}

TEST(CartesianFields, InvalidEquationInputsProduceZeroForce) {
  const field::CartesianFieldParameters parameters;
  const field::FieldVector3 origin{};
  field::CartesianFieldSample sample;
  for (const field::FieldVector3 position :
       {parameters.obstacle_offset_m, field::FieldVector3{-0.025, 0.07, 0.0}}) {
    EXPECT_FALSE(field::evaluate_cartesian_field(field::CartesianFieldKind::kPotentialField,
                                                 parameters,
                                                 origin,
                                                 position,
                                                 origin,
                                                 1,
                                                 &sample));
    EXPECT_EQ(sample.force_n, origin);
  }
  EXPECT_FALSE(field::evaluate_cartesian_field(field::CartesianFieldKind::kVirtualWalls,
                                               parameters,
                                               origin,
                                               {std::numeric_limits<double>::quiet_NaN(), 0, 0},
                                               origin,
                                               1,
                                               &sample));
  EXPECT_EQ(sample.force_n, origin);
  auto invalid = parameters;
  invalid.damping_activation_distance_m = 0;
  EXPECT_FALSE(field::validate_cartesian_field_parameters(invalid));
}

TEST(CartesianFields, UncappedForceAndRampAreAppliedAfterSuperposition) {
  const field::CartesianFieldParameters parameters;
  const field::FieldVector3 origin{};
  field::CartesianFieldSample sample;
  ASSERT_TRUE(field::evaluate_cartesian_field(
      field::CartesianFieldKind::kPotentialField, parameters, origin, origin, origin, 0, &sample));
  EXPECT_EQ(sample.force_n, origin);
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kPotentialField,
                                              parameters,
                                              origin,
                                              origin,
                                              origin,
                                              0.5,
                                              &sample));
  EXPECT_DOUBLE_EQ(sample.force_n[0], 0.0);
  EXPECT_DOUBLE_EQ(sample.force_n[1], 3.0);
  ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kPotentialField,
                                              parameters,
                                              origin,
                                              origin,
                                              {2.0, -2.0, 2.0},
                                              1,
                                              &sample));
  EXPECT_DOUBLE_EQ(sample.force_n[0], -40.0);
  EXPECT_DOUBLE_EQ(sample.force_n[1], 46.0);
  EXPECT_DOUBLE_EQ(sample.force_n[2], -40.0);
}

TEST(CartesianFields, PositiveClearanceBelowFiveMillimetersHasUncappedRepulsion) {
  const field::CartesianFieldParameters parameters;
  const field::FieldVector3 origin{};
  field::CartesianFieldSample sample;
  for (const double clearance : {0.004, 0.001}) {
    auto position = parameters.obstacle_offset_m;
    position[1] += parameters.obstacle_radius_m + clearance;
    ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kPotentialField,
                                                parameters,
                                                origin,
                                                position,
                                                origin,
                                                1,
                                                &sample));
    const double distance = sample.clearance_m;
    const double repulsion = 0.00005 * (1.0 / distance - 1.0 / 0.035) / (distance * distance);
    EXPECT_NEAR(sample.force_n[1], 50.0 * (0.12 - position[1]) + repulsion, 1e-6);
    EXPECT_TRUE(sample.repulsion_active);
    EXPECT_GT(field::field_norm(sample.force_n), 10.0);
  }
}

TEST(CartesianFields, DampedPointMassReachesGoalAroundObstacle) {
  const field::CartesianFieldParameters parameters;
  field::FieldVector3 position{}, velocity{}, origin{};
  double minimum_clearance = 1.0, settled_seconds = 0;
  std::size_t repulsion_samples = 0;
  field::CartesianFieldSample sample;
  constexpr double dt = 0.001;
  for (std::size_t tick = 0; tick < 10000; ++tick) {
    ASSERT_TRUE(field::evaluate_cartesian_field(field::CartesianFieldKind::kPotentialField,
                                                parameters,
                                                origin,
                                                position,
                                                velocity,
                                                tick * dt,
                                                &sample));
    minimum_clearance = std::min(minimum_clearance, sample.clearance_m);
    repulsion_samples += sample.repulsion_active ? 1 : 0;
    settled_seconds = sample.goal_error_m <= 0.005 && field::field_norm(velocity) <= 0.01
                          ? settled_seconds + dt
                          : 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      velocity[axis] += sample.force_n[axis] * dt;  // independent 1 kg plant
      position[axis] += velocity[axis] * dt;
    }
  }
  EXPECT_GT(repulsion_samples, 0U);
  EXPECT_GT(minimum_clearance, 0.0);
  EXPECT_LE(sample.goal_error_m, 0.005);
  EXPECT_GE(settled_seconds, 0.5);
}
}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
