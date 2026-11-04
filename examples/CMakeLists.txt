# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

add_subdirectory(synthetic)

if(ENABLE_FRANKA_EXAMPLES)
  add_subdirectory(franka)
endif()

if(ENABLE_RIZON_EXAMPLES)
  add_subdirectory(rizon)
endif()

if(ENABLE_FLEXIV_DUAL_ARM_RDK_EXAMPLE)
  add_subdirectory(flexiv_dual)
endif()

if(ENABLE_UR_EXAMPLES)
  add_subdirectory(ur)
endif()

if(NV_EXAMPLE_TARGETS)
  set(NV_EXAMPLE_TARGETS ${NV_EXAMPLE_TARGETS} PARENT_SCOPE)
endif()
