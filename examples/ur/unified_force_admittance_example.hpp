/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file unified_force_admittance_example.hpp
 * @brief Shared runner for the two UR force/admittance examples.
 */

#ifndef EXAMPLES_UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_HPP_
#define EXAMPLES_UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_HPP_

namespace nvidia::srl::controllers::examples::ur {

enum class ForceTargetFrame {
  kRobotBase,
  kEndEffector,
};

int run_unified_force_admittance_example(int argc,
                                         char* argv[],
                                         ForceTargetFrame force_target_frame);

}  // namespace nvidia::srl::controllers::examples::ur

#endif  // EXAMPLES_UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_HPP_
