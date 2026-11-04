// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

// Standard Library
#include <string>

// NVIDIA
#include "nvidia/srl/controllers/controllers.hpp"

namespace nvidia {
namespace srl {
namespace controllers {

//! Return the current full version as a string.
std::string get_full_version() { return std::string(NVIDIA_SRL_CONTROLLERS_VERSION); }

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
