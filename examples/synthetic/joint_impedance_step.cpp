/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_impedance_step.cpp
 * @brief Executes one synthetic CPU joint-impedance step.
 */

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include "nvidia/srl/controllers/controllers.hpp"

namespace {
namespace controllers = nvidia::srl::controllers;

constexpr std::size_t kJointCount = 7;

}  // namespace

int main() {
  std::array<double, kJointCount> position_error{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07};
  std::array<double, kJointCount> velocity_error{};
  std::array<double, kJointCount> desired_acceleration{};
  std::array<double, kJointCount * kJointCount> mass{};
  std::array<double, kJointCount * kJointCount> stiffness{};
  std::array<double, kJointCount> damping_factors{};
  std::array<double, kJointCount> torque{};
  std::array<double, kJointCount> last_valid_torque{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    mass[joint * kJointCount + joint] = 1.0 + 0.1 * static_cast<double>(joint);
    stiffness[joint * kJointCount + joint] = 100.0;
    damping_factors[joint] = 1.0;
  }

  constexpr controllers::Device kDevice = controllers::Device::kCpu;
  const controllers::JointImpedanceController::Inputs inputs{
      controllers::read_view(position_error, kDevice, 1, kJointCount),
      controllers::read_view(velocity_error, kDevice, 1, kJointCount),
      controllers::read_view(desired_acceleration, kDevice, 1, kJointCount),
      controllers::read_view(mass, kDevice, 1, kJointCount, kJointCount)};

  controllers::JointImpedanceController controller;
  const controllers::ControllerConfig config{
      kDevice, controllers::DType::kFloat64, 1, kJointCount, {}};
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::per_batch(
          controllers::read_view(stiffness, kDevice, 1, kJointCount, kJointCount)),
      controllers::shared(controllers::read_view(damping_factors, kDevice, kJointCount))};
  const controllers::JointImpedanceController::Storage storage{
      controllers::write_view(torque, kDevice, 1, kJointCount),
      controllers::write_status(parameter_status, kDevice),
      controllers::write_status(status, kDevice),
      controllers::make_safety_state(
          controllers::write_view(last_valid_torque, kDevice, 1, kJointCount),
          controllers::write_view(valid, kDevice, 1),
          controllers::write_view(hold_count, kDevice, 1))};
  const controllers::Result prepare_result = controller.prepare(config, parameters, storage);
  if (!prepare_result.ok()) {
    std::fprintf(stderr, "Controller preparation failed: %s\n", prepare_result.message);
    return 1;
  }
  const controllers::Result reset_result = controller.reset();
  if (!reset_result.ok()) {
    std::fprintf(stderr, "Controller reset failed: %s\n", reset_result.message);
    return 1;
  }

  const controllers::Result result = controller.step(inputs);
  if (!result.ok() || controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode)) {
    std::fprintf(stderr, "Controller step failed: status=0x%" PRIx64 "\n", status[0]);
    return 2;
  }

  std::printf("classification=synthetic backend=cpu torque_Nm=");
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    std::printf("%s%.6f", joint == 0 ? "[" : ",", torque[joint]);
  }
  std::printf("] status=0x%" PRIx64 "\n", status[0]);
  return 0;
}
