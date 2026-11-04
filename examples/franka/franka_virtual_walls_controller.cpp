// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include "./franka_cartesian_field.hpp"

int main(int argc, char** argv) {
  return franka_example::cartesian_field::run_cartesian_field_example(
      argc, argv, nvidia::srl::controllers::examples::CartesianFieldKind::kVirtualWalls);
}
