# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

if(NOT DEFINED EXAMPLE_BINARY OR "${EXAMPLE_BINARY}" STREQUAL "")
  message(FATAL_ERROR "EXAMPLE_BINARY must be set")
endif()
if(NOT EXISTS "${EXAMPLE_BINARY}")
  message(FATAL_ERROR "Example binary does not exist: ${EXAMPLE_BINARY}")
endif()
if(NOT DEFINED EXAMPLE_ARGUMENTS OR "${EXAMPLE_ARGUMENTS}" STREQUAL "")
  message(FATAL_ERROR "EXAMPLE_ARGUMENTS must be a pipe-separated argument list")
endif()

string(REPLACE "|" ";" EXAMPLE_ARGUMENT_LIST "${EXAMPLE_ARGUMENTS}")
set(EXAMPLE_COMMAND "${EXAMPLE_BINARY}")
if(DEFINED EXAMPLE_LIBRARY_PATH AND NOT "${EXAMPLE_LIBRARY_PATH}" STREQUAL "")
  set(EXAMPLE_COMMAND
    ${CMAKE_COMMAND} -E env
    "LD_LIBRARY_PATH=${EXAMPLE_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
    "${EXAMPLE_BINARY}")
endif()
execute_process(
  COMMAND ${EXAMPLE_COMMAND} ${EXAMPLE_ARGUMENT_LIST}
  INPUT_FILE "/dev/null"
  RESULT_VARIABLE EXAMPLE_RESULT
  OUTPUT_VARIABLE EXAMPLE_OUTPUT
  ERROR_VARIABLE EXAMPLE_ERROR
  TIMEOUT 10)

set(EXAMPLE_LOG "${EXAMPLE_OUTPUT}\n${EXAMPLE_ERROR}")
if(NOT EXAMPLE_RESULT EQUAL 1)
  message(FATAL_ERROR
    "${EXAMPLE_BINARY} did not cancel cleanly after confirmation input reached EOF "
    "(result=${EXAMPLE_RESULT}):\n"
    "${EXAMPLE_LOG}")
endif()
if(NOT EXAMPLE_LOG MATCHES "non_rt_requested=true")
  message(FATAL_ERROR
    "${EXAMPLE_BINARY} did not accept --non-rt and reach its pre-connection confirmation:\n"
    "${EXAMPLE_LOG}")
endif()
if(NOT EXAMPLE_LOG MATCHES
   "Press Enter to connect|Press Enter to initialize DDS")
  message(FATAL_ERROR
    "${EXAMPLE_BINARY} accepted --non-rt but did not reach its connection prompt:\n"
    "${EXAMPLE_LOG}")
endif()

if(DEFINED EXPECTED_OUTPUT)
  string(FIND "${EXAMPLE_OUTPUT}" "${EXPECTED_OUTPUT}" EXPECTED_OUTPUT_INDEX)
  if(EXPECTED_OUTPUT_INDEX EQUAL -1)
    message(FATAL_ERROR
      "${EXAMPLE_BINARY} did not print '${EXPECTED_OUTPUT}' before connection:\n${EXAMPLE_LOG}")
  endif()
endif()

function(assert_trailing_argument_rejected REJECTED_ARGUMENT DESCRIPTION)
  execute_process(
    COMMAND ${EXAMPLE_COMMAND} ${EXAMPLE_ARGUMENT_LIST} "${REJECTED_ARGUMENT}"
    INPUT_FILE "/dev/null"
    RESULT_VARIABLE REJECTED_RESULT
    OUTPUT_VARIABLE REJECTED_OUTPUT
    ERROR_VARIABLE REJECTED_ERROR
    TIMEOUT 10)
  if(NOT REJECTED_RESULT EQUAL 1 OR NOT REJECTED_ERROR MATCHES "Usage:" OR
     REJECTED_OUTPUT MATCHES "non_rt_requested=true")
    message(FATAL_ERROR
      "${EXAMPLE_BINARY} did not reject ${DESCRIPTION} during argument validation "
      "(result=${REJECTED_RESULT}):\n${REJECTED_OUTPUT}\n${REJECTED_ERROR}")
  endif()
endfunction()

assert_trailing_argument_rejected("--non-rt" "duplicate --non-rt")
if(EXAMPLE_ARGUMENTS MATCHES "--duration-seconds=")
  assert_trailing_argument_rejected(
    "--duration-seconds=1" "duplicate --duration-seconds")
endif()
assert_trailing_argument_rejected("--unknown" "an unknown option")
