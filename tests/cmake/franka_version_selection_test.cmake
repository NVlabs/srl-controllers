# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

if(NOT DEFINED CONTROLLERS_SOURCE_DIR)
  message(FATAL_ERROR "CONTROLLERS_SOURCE_DIR is required")
endif()
include("${CONTROLLERS_SOURCE_DIR}/examples/franka/LibfrankaVersions.cmake")

if(DEFINED PROBE_VERSION)
  nvidia_srl_controllers_select_libfranka_version("${PROBE_VERSION}")
  return()
endif()

function(expect_version VERSION FAMILY REVISION)
  nvidia_srl_controllers_select_libfranka_version("${VERSION}")
  if(NOT FRANKA_ROBOT_FAMILY STREQUAL FAMILY OR
     NOT FRANKA_GIT_REVISION STREQUAL REVISION)
    message(FATAL_ERROR
      "${VERSION}: expected ${FAMILY}/${REVISION}, found "
      "${FRANKA_ROBOT_FAMILY}/${FRANKA_GIT_REVISION}")
  endif()
endfunction()

expect_version(0.9.2 "FER/Panda" "f3b8d775a9c847cab32684c8a316f67867761674")
expect_version(0.10.0 FR3 "4f9e3cc666e42d267f1ab566869c4f4c552e5b57")
expect_version(0.15.3 FR3 "09385a5a619d5445bda461b3a89a62d6fa40a59a")
expect_version(0.21.3 FR3 "85912fe02258d8cb811d3eff1f11e52ce89e3217")

foreach(INVALID_VERSION latest 0.9.1 0.9.3)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DCONTROLLERS_SOURCE_DIR=${CONTROLLERS_SOURCE_DIR}"
      "-DPROBE_VERSION=${INVALID_VERSION}"
      -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE OUTPUT
    ERROR_VARIABLE ERROR)
  if(RESULT EQUAL 0)
    message(FATAL_ERROR "${INVALID_VERSION} must be rejected")
  endif()
endforeach()

# A configure with Franka enabled must not discover the host SDK. The examples
# and their package resolve only after the private ExternalProject install.
file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -G "${TEST_CMAKE_GENERATOR}"
    -S "${CONTROLLERS_SOURCE_DIR}"
    -B "${TEST_BINARY_DIR}"
    "-DCMAKE_MAKE_PROGRAM=${TEST_CMAKE_MAKE_PROGRAM}"
    -DDEVEL=OFF
    -DENABLE_COVERAGE=OFF
    -DENABLE_DOCS=OFF
    -DENABLE_EXAMPLES=ON
    -DENABLE_FRANKA_EXAMPLES=ON
    -DENABLE_PYBIND=OFF
    -DENABLE_TESTS=OFF
    -DENABLE_TOOLS=OFF
    -DLIBFRANKA_VERSION=0.9.2
    "-DFranka_DIR=${TEST_BINARY_DIR}/intentionally-nonexistent"
  RESULT_VARIABLE RESULT
  OUTPUT_VARIABLE OUTPUT
  ERROR_VARIABLE ERROR)
if(NOT RESULT EQUAL 0 OR
   NOT OUTPUT MATCHES "libfranka 0.9.2 .* locally built")
  message(FATAL_ERROR
    "The FER configure must select its local SDK without a host package.\n${OUTPUT}\n${ERROR}")
endif()
