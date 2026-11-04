// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

// Standard Library
#include <string>

// Third Party
#include "gtest/gtest.h"

// NVIDIA
#include "nvidia/srl/controllers/controllers.hpp"

namespace nvidia {
namespace srl {
namespace controllers {

TEST(Version, FullVersion) {
  EXPECT_EQ(std::string(NVIDIA_SRL_CONTROLLERS_VERSION), get_full_version());
}

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
