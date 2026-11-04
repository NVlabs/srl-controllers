/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file initial_motion.hpp
 * @brief Shared initial joint motion for Universal Robots hardware examples.
 */

#ifndef EXAMPLES_UR_INITIAL_MOTION_HPP_
#define EXAMPLES_UR_INITIAL_MOTION_HPP_

#include <ur_client_library/types.h>
#include <ur_client_library/ur/instruction_executor.h>
#include <ur_client_library/ur/ur_driver.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

namespace nvidia::srl::controllers::examples::ur {

constexpr double kDegreesToRadians = 0.01745329251994329577;
constexpr double kInitialMotionAcceleration = 0.5;
constexpr double kInitialMotionVelocity = 0.5;
inline constexpr urcl::vector6d_t kInitialJointPosition{0.0,
                                                        -90.0 * kDegreesToRadians,
                                                        -90.0 * kDegreesToRadians,
                                                        -90.0 * kDegreesToRadians,
                                                        90.0 * kDegreesToRadians,
                                                        0.0};
inline constexpr urcl::vector6d_t kEndEffectorForceInitialJointPosition{0.0,
                                                                        -90.0 * kDegreesToRadians,
                                                                        -90.0 * kDegreesToRadians,
                                                                        -90.0 * kDegreesToRadians,
                                                                        135.0 * kDegreesToRadians,
                                                                        0.0};

inline bool confirm_controller_start() {
  std::cout << "Initial joint motion completed.\n"
            << "Press Enter to start the example controller (type anything to cancel): "
            << std::flush;
  std::string response;
  if (!std::getline(std::cin, response) || !response.empty()) {
    std::cout << "Execution cancelled before starting the controller.\n";
    return false;
  }
  return true;
}

inline bool move_to_initial_joint_position(const std::shared_ptr<urcl::UrDriver>& driver,
                                           const urcl::vector6d_t& target = kInitialJointPosition) {
  if (!driver) {
    return false;
  }
  std::cout << "Moving to initial joint position [";
  for (std::size_t joint = 0; joint < target.size(); ++joint) {
    std::cout << target[joint] / kDegreesToRadians;
    std::cout << (joint + 1 == target.size() ? "]" : ", ");
  }
  std::cout << " deg at " << kInitialMotionVelocity << " rad/s and " << kInitialMotionAcceleration
            << " rad/s^2.\n";
  urcl::InstructionExecutor executor(driver);
  return executor.moveJ(target, kInitialMotionAcceleration, kInitialMotionVelocity, 0.0, 0.0);
}

}  // namespace nvidia::srl::controllers::examples::ur

#endif  // EXAMPLES_UR_INITIAL_MOTION_HPP_
