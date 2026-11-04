# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Add test targets based on given source files and linked libraries.
#
# NOTE: The GTest::gtest target and the library target automatically get added
# as linked libraries.
#
# See `nv_add_test_targets` for more details.
#
# Args:
#   LIB_NAME (str) - The name of the library that these tests run against.
#   TEST_FILES (list) - List of test file paths.
#   TEST_LINK_LIBS (list) - List of libraries to privately link to the
#       test target.
macro(nv_add_tests LIB_NAME TEST_FILES TEST_LINK_LIBS)
  nv_add_test_targets(${LIB_NAME} "${TEST_FILES}" "${TEST_LINK_LIBS}" TEST_TARGETS)
  list(APPEND NV_TEST_TARGETS ${TEST_TARGETS})
endmacro()


# ==============================================================================
# Create a test executable for every test file that is given and link the
# given libraries to every test target.
#
# NOTE: The GTest::gtest target and the library target automatically get added
# as linked libraries.
#
# See `nv_add_test_target` for more details.
#
# Args:
#   LIB_NAME (str) - The name of the library that these tests run against.
#   TEST_FILES (list) - List of test file paths.
#   TEST_LINK_LIBS (list) - List of libraries to privately link to the
#       test target.
function(nv_add_test_targets LIB_NAME TEST_FILES TEST_LINK_LIBS)

  # Set the library target name
  string(JOIN "_" LIB_TARGET_NAME ${NV_NAMESPACE} ${LIB_NAME})

  # Append additional link libs
  list(APPEND TEST_LINK_LIBS "GTest::gtest" ${LIB_TARGET_NAME})

  if(VERBOSITY STRGREATER_EQUAL 2)
    message(STATUS "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    message(STATUS "From nv_add_tests()")
    message(STATUS "--------------------------------------------------")
    cmake_print_variables(LIB_NAME)
    cmake_print_variables(TEST_FILES)
    cmake_print_variables(TEST_LINK_LIBS)
    message(STATUS "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
  endif()

  set(TEST_TARGETS)

  foreach(TEST_FILE ${TEST_FILES})
    nv_add_test_target(${LIB_NAME} ${TEST_FILE} "${TEST_LINK_LIBS}" TARGET_NAME)
    list(APPEND TEST_TARGETS ${TARGET_NAME})

    # cmake_print_variables(TARGET_NAME)
  endforeach()

  # Throw an error if more than one output variable is provided
  list(LENGTH ARGN N_OUPUTS)
  if(N_OUPUTS GREATER 1)
    message(FATAL_ERROR "Function invoked with incorrect number of arguments")
  endif()
  # Return target name, if output variable is provided
  if(ARGN)
    LIST(GET ARGN 0 OUTPUT)
    set(${OUTPUT} ${TEST_TARGETS} PARENT_SCOPE)
  endif()

endfunction()


# ==============================================================================
# Add a test executable target based on test file. The name of the target is the
# test file name without the extension. And link the libraries listed in
# LINK_LIBS to the target. In addition, adds the test target as a dependency to
# the "coverage" target if it exist.
#
# NOTE: The GTest::gtest target and the library target automatically get added
# as linked libraries.
#
# Args:
#   LIB_NAME (str) - The name of the library that this test runs against.
#   TEST_FILE (str) - Test file path.
#   TEST_LINK_LIBS (list) - List of libraries to privately link to the
#       test target.
function(nv_add_test_target LIB_NAME TEST_FILE TEST_LINK_LIBS)
  # Set the library target name
  string(JOIN "_" LIB_TARGET_NAME ${NV_NAMESPACE} ${LIB_NAME})

  # Set test target name
  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  string(JOIN "_" TEST_TARGET_NAME ${LIB_TARGET_NAME} ${TEST_NAME})

  # Add test target
  add_executable(${TEST_TARGET_NAME} "${TEST_FILE}")

  # Link dependency libraries to the target
  # NOTE: Best practice is to default to PRIVATE, only use PUBLIC or INTERFACE if the
  # relationship genuinely requires it.
  # From C. Scott, https://www.mail-archive.com/cmake@cmake.org/msg61604.html
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${TEST_LINK_LIBS}")

  # Set the output file name for the executable
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES OUTPUT_NAME ${TEST_NAME})

  # Set target compile options
  nv_set_target_compile_options(${TEST_TARGET_NAME})

  # Make tests discoverable to used with ctest
  gtest_discover_tests(${TEST_TARGET_NAME} DISCOVERY_TIMEOUT 20 DISCOVERY_MODE PRE_TEST)

  # Add coverage
  if(TARGET coverage)
    add_dependencies(coverage ${TEST_TARGET_NAME})
  endif()

  list(LENGTH ARGN N_OUTPUTS)
  if(N_OUTPUTS GREATER 1)
    message(FATAL_ERROR "Function invoked with incorrect number of arguments")
  endif()
  # Return test target name, if output variable is provided
  if(ARGN)
    LIST(GET ARGN 0 OUTPUT)
    set(${OUTPUT} ${TEST_TARGET_NAME} PARENT_SCOPE)
  endif()

endfunction()
