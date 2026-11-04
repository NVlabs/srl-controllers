# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Configure coverage with default target name of "coverage", and add to list of
# custom targets.
#
# Usage:
#   cmake -DENABLE_COVERAGE=ON ..
#   make
#   make test
#   make coverage
#   xdg-open coverage/cpp/html/index.html
macro(nv_configure_coverage)
  set(TARGET_NAME coverage)
  nv_configure_coverage_target(${TARGET_NAME})
  list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})
endmacro()

# ==============================================================================
# Configure coverage
#
# Args:
#   TARGET_NAME (str) - Name of coverage target.
function(nv_configure_coverage_target TARGET_NAME)
  # Check if gcovr is installed
  find_program(GCOVR_PRG gcovr)
  if(GCOVR_PRG STREQUAL "GCOVR_PRG-NOTFOUND")
    message(FATAL_ERROR "'gcovr' not found, which is required to make coverage report. For more info see: https://gcovr.com")
  endif()

  # Set the correct extensions for gcovr to find data files
  # https://texus.me/2015/09/06/cmake-and-gcov/
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

  # Create a "Coverage" build type that sets the correct flags for coverage
  # https://stackoverflow.com/questions/11437692/how-to-add-a-custom-build-type-to-cmake-targeting-make
  set(COVERAGE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Concurrent tests update shared-library counters; single updates corrupt branch counts.
    string(APPEND COVERAGE_CXX_FLAGS " -fprofile-update=atomic")
  endif()
  SET(CMAKE_CXX_FLAGS_COVERAGE
      "${COVERAGE_CXX_FLAGS}"
      CACHE STRING "Flags used by the C++ compiler during coverage builds."
      FORCE )
  MARK_AS_ADVANCED(
      CMAKE_CXX_FLAGS_COVERAGE
      CMAKE_C_FLAGS_COVERAGE)

  # Create custom target to run coverage command and store output in build/coverage/cpp
  message(STATUS "${GCOVR_CMD}")
  add_custom_target(${TARGET_NAME}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    COMMAND mkdir -p coverage/cpp/html
    COMMAND gcovr -r "${PROJECT_SOURCE_DIR}" -s --xml coverage/cpp/coverage.xml --xml-pretty --html coverage/cpp/html/index.html --html-details -j8  --exclude=${PROJECT_SOURCE_DIR}/examples --exclude=${PROJECT_SOURCE_DIR}/tools --exclude=${PROJECT_SOURCE_DIR}/tests --exclude=${PROJECT_SOURCE_DIR}/.*/.*_test.cpp
  )

endfunction()
