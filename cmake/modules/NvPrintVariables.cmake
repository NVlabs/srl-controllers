# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

macro(nv_print_variables)
  message(STATUS "==================================================")
  message(STATUS "CMake configuration:")
  cmake_print_variables(
    PROJECT_NAME
    PROJECT_VERSION
    CMAKE_BUILD_TYPE
    CMAKE_SYSTEM_PROCESSOR
    CMAKE_INSTALL_PREFIX
    DEVEL
    VERBOSITY
    ENABLE_BENCHMARK
    ENABLE_COVERAGE
    ENABLE_CUDA
    ENABLE_DOCS
    ENABLE_EXAMPLES
    ENABLE_FRANKA_EXAMPLES
    ENABLE_FLEXIV_DUAL_ARM_RDK_EXAMPLE
    ENABLE_RIZON_EXAMPLES
    ENABLE_UR_EXAMPLES
    ENABLE_PYBIND
    ENABLE_TESTS
    ENABLE_TOOLS
    WERROR
  )

  if(VERBOSITY STRGREATER_EQUAL 1)
    message(STATUS "Project configuration:")
    cmake_print_variables(
      NV_PROJECT_DASH
      NV_PROJECT_COLON
      NV_VERSION_H_DIR
      NV_TARGETS_EXPORT_SET
      NV_CPACK_DEPENDS
      ${PROJECT_NAME}_VERSION_FULL
      PROJECT_SOURCE_DIR
      CMAKE_BINARY_DIR
      CMAKE_PREFIX_PATH
      CMAKE_INSTALL_RPATH
      CMAKE_CXX_COMPILER
      PYBIND11_PYTHON_VERSION
    )
  endif()
  message(STATUS "==================================================")
endmacro()

macro(nv_print_targets)
  if(VERBOSITY STRGREATER_EQUAL 1)
    message(STATUS "CMake targets:")
    cmake_print_variables(
      NV_EXAMPLE_TARGETS
      NV_LIBRARY_TARGETS
      NV_PYBIND_TARGETS
      NV_TEST_TARGETS
      NV_TOOL_TARGETS
      NV_CUSTOM_TARGETS
    )
  endif()
endmacro()
