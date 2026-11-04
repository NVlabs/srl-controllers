# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)
project(rizon_examples_consumer LANGUAGES CXX)

include(GNUInstallDirs)

foreach(REQUIRED_VARIABLE
    CONTROLLERS_INCLUDE_DIR
    CONTROLLERS_LIBRARY
    RIZON_SOURCE_DIR
    FLEXIV_RDK_EXPECTED_VERSION)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()
if(NOT EXISTS "${CONTROLLERS_INCLUDE_DIR}/nvidia/srl/controllers/controllers.hpp" OR
   NOT EXISTS "${CONTROLLERS_LIBRARY}" OR
   NOT EXISTS "${RIZON_SOURCE_DIR}/rizon_controller_example.cpp")
  message(FATAL_ERROR "Rizon consumer inputs must name built controller and example artifacts")
endif()

find_package(flexiv_rdk ${FLEXIV_RDK_EXPECTED_VERSION} EXACT REQUIRED CONFIG)

add_library(controllers_local SHARED IMPORTED)
set_target_properties(controllers_local PROPERTIES
  IMPORTED_LOCATION "${CONTROLLERS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CONTROLLERS_INCLUDE_DIR}"
)

get_filename_component(CONTROLLERS_LIBRARY_DIR "${CONTROLLERS_LIBRARY}" DIRECTORY)
set(RIZON_BUILD_RPATH
    "${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR};${CONTROLLERS_LIBRARY_DIR}")
set(RIZON_EXAMPLE_NAMES
  rizon_cartesian_admittance_controller
  rizon_cartesian_force_impedance_controller
  rizon_cartesian_force_impedance_controller_in_ee
  rizon_cartesian_impedance_controller
  rizon_joint_impedance_controller
  rizon_variable_cartesian_impedance_controller
)

foreach(RIZON_EXAMPLE IN LISTS RIZON_EXAMPLE_NAMES)
  add_executable(${RIZON_EXAMPLE} "${RIZON_SOURCE_DIR}/rizon_controller_example.cpp")
  target_compile_features(${RIZON_EXAMPLE} PRIVATE cxx_std_17)
  target_compile_options(${RIZON_EXAMPLE} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(${RIZON_EXAMPLE} PRIVATE
    "RIZON_EXAMPLE_NAME=\"${RIZON_EXAMPLE}\"")
  target_include_directories(${RIZON_EXAMPLE} PRIVATE "${RIZON_SOURCE_DIR}")
  target_link_options(${RIZON_EXAMPLE} PRIVATE "LINKER:--disable-new-dtags")
  target_link_libraries(${RIZON_EXAMPLE} PRIVATE controllers_local flexiv::flexiv_rdk)
  set_target_properties(${RIZON_EXAMPLE} PROPERTIES
    BUILD_RPATH "${RIZON_BUILD_RPATH}"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )
  install(TARGETS ${RIZON_EXAMPLE} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

if(BUILD_RIZON_TESTS)
  if(NOT DEFINED RIZON_TEST_SOURCE OR NOT EXISTS "${RIZON_TEST_SOURCE}")
    message(FATAL_ERROR "RIZON_TEST_SOURCE must name the Rizon utility test")
  endif()
  find_package(GTest REQUIRED)
  add_executable(rizon_example_utils_test "${RIZON_TEST_SOURCE}")
  target_compile_features(rizon_example_utils_test PRIVATE cxx_std_17)
  target_compile_options(rizon_example_utils_test PRIVATE -Wall -Wextra -Wpedantic)
  target_include_directories(rizon_example_utils_test PRIVATE "${RIZON_SOURCE_DIR}/../..")
  target_link_options(rizon_example_utils_test PRIVATE "LINKER:--disable-new-dtags")
  target_link_libraries(rizon_example_utils_test PRIVATE
    controllers_local flexiv::flexiv_rdk GTest::gtest)
  set_target_properties(rizon_example_utils_test PROPERTIES
    BUILD_RPATH "${RIZON_BUILD_RPATH}"
  )
endif()
