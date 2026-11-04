# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

nv_add_example_target(
  rt_reference_loop.cpp
  "nvidia::srl::controllers"
  RT_REFERENCE_LOOP_TARGET
)
list(APPEND NV_EXAMPLE_TARGETS ${RT_REFERENCE_LOOP_TARGET})
if(RT_REFERENCE_LOOP_TARGET AND TARGET ${RT_REFERENCE_LOOP_TARGET})
  target_link_options(${RT_REFERENCE_LOOP_TARGET} PRIVATE "-Wl,-z,now")
endif()

nv_add_example(joint_impedance_step.cpp nvidia::srl::controllers)

if(NV_EXAMPLE_TARGETS)
  set(NV_EXAMPLE_TARGETS ${NV_EXAMPLE_TARGETS} PARENT_SCOPE)
endif()
