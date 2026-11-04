# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22


# For each tool add it with a separate call to `nv_add_tool`
# nv_add_tool(tool_source.cpp "list of linked libs")

nv_add_tool(controllers_version.cpp nvidia::srl::controllers)

if(ENABLE_CUDA AND ENABLE_TENSOR_JOINT_IMPEDANCE_CUDA_BENCHMARK)
  nv_add_tool(tensor_joint_impedance_cuda_benchmark.cpp
    nvidia_srl_controllers_cuda_tools_backend CUDA::cudart)
  nv_add_tool(tensor_joint_impedance_cuda_qualification.cpp
    nvidia_srl_controllers_cuda_tools_backend CUDA::cudart)
  target_include_directories(nvidia_srl_tensor_joint_impedance_cuda_benchmark
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/nvidia/srl/controllers
  )
  target_include_directories(nvidia_srl_tensor_joint_impedance_cuda_qualification
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/nvidia/srl/controllers
  )
endif()

if(ENABLE_TESTS)
  add_test(NAME nvidia_srl_controllers_version_tool COMMAND nvidia_srl_controllers_version)
  set_tests_properties(nvidia_srl_controllers_version_tool PROPERTIES
    PASS_REGULAR_EXPRESSION "${nvidia_srl_controllers_VERSION}"
  )
endif()


# ==============================================================================
# DO NOT EDIT THE CONTENT BELOW THIS LINE
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# Update parent scope variables
if(NV_TOOL_TARGETS)
  set(NV_TOOL_TARGETS ${NV_TOOL_TARGETS} PARENT_SCOPE)
endif()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# DO NOT EDIT THIS SECTION
# ==============================================================================
