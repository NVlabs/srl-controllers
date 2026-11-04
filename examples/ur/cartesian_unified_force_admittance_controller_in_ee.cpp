/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_unified_force_admittance_controller_in_ee.cpp
 * @brief Runs unified force/admittance control with a UR end-effector-frame force target.
 */

#include "examples/ur/unified_force_admittance_example.hpp"

int main(int argc, char* argv[]) {
  return nvidia::srl::controllers::examples::ur::run_unified_force_admittance_example(
      argc, argv, nvidia::srl::controllers::examples::ur::ForceTargetFrame::kEndEffector);
}
