/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file controllers_python.cpp
 * @brief Private Python bridge for the controller API.
 */

#include <pybind11/pybind11.h>

#include "src/nvidia/srl/controllers/controllers_python.hpp"

namespace nvidia {
namespace srl {
namespace controllers {

PYBIND11_MODULE(nvidia_srl_controllers_cpp, module) {
  controllers_python::bind_controllers(module);
}

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
