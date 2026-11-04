// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

// Standard Library

#include <iostream>

#include "nvidia/srl/controllers/version.hpp"

int main() {
  std::cout << NVIDIA_SRL_CONTROLLERS_VERSION << '\n';
  return 0;
}
