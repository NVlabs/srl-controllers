# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.18)

foreach(REQUIRED_VARIABLE
    CONTROLLERS_SOURCE_DIR
    TEST_BINARY_DIR
    TEST_CMAKE_GENERATOR
    TEST_CMAKE_MAKE_PROGRAM)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required for the Rizon selector test")
  endif()
endforeach()

file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")
set(VERSION_MODULE
    "${CONTROLLERS_SOURCE_DIR}/examples/rizon/FlexivRdkVersions.cmake")
set(SELECTION_FIXTURE
    "${CONTROLLERS_SOURCE_DIR}/tests/cmake/rizon_version_selection_fixture.cmake")

function(run_selector SELECTED_VERSION RESULT_OUT LOG_OUT)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVERSION_MODULE=${VERSION_MODULE}"
      "-DSELECTED_VERSION=${SELECTED_VERSION}"
      -P "${SELECTION_FIXTURE}"
    RESULT_VARIABLE SELECTOR_RESULT
    OUTPUT_VARIABLE SELECTOR_OUTPUT
    ERROR_VARIABLE SELECTOR_ERROR)
  set(${RESULT_OUT} "${SELECTOR_RESULT}" PARENT_SCOPE)
  set(${LOG_OUT} "${SELECTOR_OUTPUT}\n${SELECTOR_ERROR}" PARENT_SCOPE)
endfunction()

function(expect_selector_success SELECTED_VERSION EXPECTED_METADATA)
  string(JOIN "" EXPECTED_METADATA ${EXPECTED_METADATA} ${ARGN})
  run_selector("${SELECTED_VERSION}" SELECTOR_RESULT SELECTOR_LOG)
  if(NOT SELECTOR_RESULT EQUAL 0 OR NOT SELECTOR_LOG MATCHES "${EXPECTED_METADATA}")
    message(FATAL_ERROR
      "Selector metadata mismatch for ${SELECTED_VERSION}:\n${SELECTOR_LOG}")
  endif()
endfunction()

function(expect_selector_failure SELECTED_VERSION EXPECTED_MESSAGE)
  run_selector("${SELECTED_VERSION}" SELECTOR_RESULT SELECTOR_LOG)
  if(SELECTOR_RESULT EQUAL 0 OR NOT SELECTOR_LOG MATCHES "${EXPECTED_MESSAGE}")
    message(FATAL_ERROR
      "Selector should reject '${SELECTED_VERSION}' with '${EXPECTED_MESSAGE}':\n"
      "${SELECTOR_LOG}")
  endif()
endfunction()

expect_selector_success(
  1.9.1
  "1.9.1\\|5707121ffadc5a1577c5e0eabf2afa20b9842acb\\|"
  "41875278fd89af2db0834458d72f518e8916c26b34fdadb5fa85244caf2cd370\\|"
  "libflexiv_rdk.x86_64-linux-gnu.a\\|libflexiv_rdk.a\\|static\\|TRUE")
expect_selector_success(
  1.9.2
  "1.9.2\\|20e2edf4f5776f8e97380a1fb14b709dffbf67d9\\|"
  "3b1c923b4bed6c1cc94ce774d57071906c5808a30302a5d7d7be40b349c69b58\\|"
  "libflexiv_rdk.x86_64-linux-gnu.a\\|libflexiv_rdk.a\\|static\\|TRUE")
expect_selector_success(
  1.9.3
  "1.9.3\\|bbb7ffc1feb125f6078eb99d2048264ceaab65e8\\|"
  "b3314d3095dd4615391448ea5e471ee2414696fea92e1232accafa54f2a39cfc\\|"
  "libflexiv_rdk.x86_64-linux-gnu.a\\|libflexiv_rdk.a\\|static\\|TRUE")
expect_selector_success(
  1.9.4
  "1.9.4\\|9439b408d3815b93843de5bceea2bf5b951164b6\\|"
  "1b2f4b6155c913b2955fd8110bf4064ab74dcab700dfb16fd13b0ad3c113d1c8\\|"
  "libflexiv_rdk.x86_64-linux-gnu.so\\|libflexiv_rdk.so\\|shared\\|FALSE")

expect_selector_failure(latest "must use major.minor.patch syntax")
expect_selector_failure(2.0.0 "Unsupported FLEXIV_RDK_VERSION")

set(DEFAULT_BINARY_DIR "${TEST_BINARY_DIR}/default-project")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -G "${TEST_CMAKE_GENERATOR}"
    -S "${CONTROLLERS_SOURCE_DIR}"
    -B "${DEFAULT_BINARY_DIR}"
    "-DCMAKE_MAKE_PROGRAM=${TEST_CMAKE_MAKE_PROGRAM}"
    -DDEVEL=OFF
    -DENABLE_COVERAGE=OFF
    -DENABLE_DOCS=OFF
    -DENABLE_EXAMPLES=OFF
    -DENABLE_PYBIND=OFF
    -DENABLE_TESTS=OFF
    -DENABLE_TOOLS=OFF
  RESULT_VARIABLE DEFAULT_CONFIGURE_RESULT
  OUTPUT_VARIABLE DEFAULT_CONFIGURE_OUTPUT
  ERROR_VARIABLE DEFAULT_CONFIGURE_ERROR)
if(NOT DEFAULT_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Default project configuration failed:\n"
    "${DEFAULT_CONFIGURE_OUTPUT}\n${DEFAULT_CONFIGURE_ERROR}")
endif()
file(STRINGS "${DEFAULT_BINARY_DIR}/CMakeCache.txt" DEFAULT_VERSION_LINE
  REGEX "^FLEXIV_RDK_VERSION:STRING=")
if(NOT DEFAULT_VERSION_LINE STREQUAL "FLEXIV_RDK_VERSION:STRING=1.9.1")
  message(FATAL_ERROR
    "Expected default FLEXIV_RDK_VERSION:STRING=1.9.1, found '${DEFAULT_VERSION_LINE}'")
endif()
