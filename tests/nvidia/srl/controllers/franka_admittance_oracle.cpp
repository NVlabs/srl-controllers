/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @brief Hardware-free trace oracle using the unchanged physical admittance helper.
 */
#include <iomanip>
#include <iostream>

#include "examples/franka/franka_admittance.hpp"

int main() {
  franka_example::AdmittanceWorkspace workspace;
  if (!workspace.prepare().ok()) return 1;
  std::cout << std::setprecision(17);
  for (int tick = 0; tick < 8; ++tick) {
    Eigen::Vector<double, 6> wrench;
    for (int axis = 0; axis < 6; ++axis) wrench[axis] = (axis + 1) * (tick + 1) * 0.125;
    const double dt = 0.001 + tick * 0.00001;
    if (!workspace.step_admittance(wrench, dt)) return 2;
    const auto trace = workspace.last_step_trace();
    for (const auto* vector : {&trace.external_wrench,
                               &trace.acceleration,
                               &trace.twist,
                               &trace.pose_offset,
                               &trace.pose_previous_signal,
                               &trace.twist_previous_signal,
                               &trace.last_valid_acceleration}) {
      for (const double scalar : *vector) std::cout << scalar << ' ';
    }
    std::cout << trace.status << ' ' << trace.parameter_status << ' '
              << static_cast<int>(trace.pose_initialized) << ' '
              << static_cast<int>(trace.twist_initialized) << ' '
              << static_cast<int>(trace.safety_valid) << ' ' << trace.hold_count << '\n';
  }
  return 0;
}
