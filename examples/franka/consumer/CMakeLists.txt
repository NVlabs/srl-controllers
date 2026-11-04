# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)
project(franka_examples_consumer LANGUAGES CXX)
include(GNUInstallDirs)

foreach(REQUIRED_VARIABLE
    CONTROLLERS_INCLUDE_DIR CONTROLLERS_GENERATED_INCLUDE_DIR
    CONTROLLERS_LIBRARY FRANKA_SOURCE_DIR
    FRANKA_LOCAL_PREFIX FRANKA_EXPECTED_VERSION)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()
if(NOT EXISTS "${CONTROLLERS_INCLUDE_DIR}/nvidia/srl/controllers/controllers.hpp" OR
   NOT EXISTS "${CONTROLLERS_GENERATED_INCLUDE_DIR}/nvidia/srl/controllers/version.hpp" OR
   NOT EXISTS "${CONTROLLERS_LIBRARY}")
  message(FATAL_ERROR "Franka consumer requires the built controller library and headers")
endif()

if(FRANKA_PINOCCHIO_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${FRANKA_PINOCCHIO_PREFIX}")
  find_package(pinocchio REQUIRED CONFIG
    PATHS "${FRANKA_PINOCCHIO_PREFIX}" NO_DEFAULT_PATH)
endif()
find_package(Eigen3 REQUIRED CONFIG)
find_package(Franka ${FRANKA_EXPECTED_VERSION} EXACT REQUIRED CONFIG
  PATHS "${FRANKA_LOCAL_PREFIX}" NO_DEFAULT_PATH)
if(NOT Franka_DIR MATCHES "^${FRANKA_LOCAL_PREFIX}/")
  message(FATAL_ERROR "Franka must resolve from the private build prefix")
endif()

add_library(controllers_local SHARED IMPORTED)
set_target_properties(controllers_local PROPERTIES
  IMPORTED_LOCATION "${CONTROLLERS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${CONTROLLERS_INCLUDE_DIR};${CONTROLLERS_GENERATED_INCLUDE_DIR}")
get_filename_component(CONTROLLERS_LIBRARY_DIR "${CONTROLLERS_LIBRARY}" DIRECTORY)
set(FRANKA_BUILD_RPATH
    "${FRANKA_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR};${CONTROLLERS_LIBRARY_DIR}")
if(FRANKA_PINOCCHIO_PREFIX)
  list(APPEND FRANKA_BUILD_RPATH "${FRANKA_PINOCCHIO_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

set(FRANKA_EXAMPLE_NAMES
  franka_virtual_walls_controller
  franka_artificial_potential_field_controller
  franka_cartesian_admittance_controller
  franka_cartesian_force_impedance_controller
  franka_cartesian_force_impedance_controller_with_tank
  franka_cartesian_force_impedance_controller_in_ee
  franka_cartesian_impedance_controller
  franka_joint_impedance_controller
  franka_variable_cartesian_impedance_controller)
foreach(FRANKA_EXAMPLE IN LISTS FRANKA_EXAMPLE_NAMES)
  add_executable(${FRANKA_EXAMPLE} "${FRANKA_SOURCE_DIR}/${FRANKA_EXAMPLE}.cpp")
  target_compile_features(${FRANKA_EXAMPLE} PRIVATE cxx_std_17)
  target_include_directories(${FRANKA_EXAMPLE} PRIVATE "${FRANKA_SOURCE_DIR}/../..")
  target_link_libraries(${FRANKA_EXAMPLE} PRIVATE
    controllers_local Franka::Franka Eigen3::Eigen)
  target_link_options(${FRANKA_EXAMPLE} PRIVATE "LINKER:--disable-new-dtags")
  set_target_properties(${FRANKA_EXAMPLE} PROPERTIES
    BUILD_RPATH "${FRANKA_BUILD_RPATH}"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  if(FRANKA_EXPECTED_VERSION STREQUAL "0.9.2" AND
     FRANKA_EXAMPLE STREQUAL "franka_cartesian_admittance_controller")
    target_compile_definitions(${FRANKA_EXAMPLE} PRIVATE NVIDIA_SRL_FRANKA_FER=1)
  endif()
  install(TARGETS ${FRANKA_EXAMPLE} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

# Make the build-tree example installation directly runnable and self-contained
# with respect to the locally built controller and SDK libraries.
install(FILES "${CONTROLLERS_LIBRARY}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${FRANKA_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/"
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(FRANKA_PINOCCHIO_PREFIX)
  install(DIRECTORY "${FRANKA_PINOCCHIO_PREFIX}/${CMAKE_INSTALL_LIBDIR}/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(BUILD_FRANKA_TESTS)
  find_package(GTest REQUIRED)
  add_executable(franka_example_utils_test "${FRANKA_TEST_SOURCE}")
  target_compile_features(franka_example_utils_test PRIVATE cxx_std_17)
  target_include_directories(franka_example_utils_test PRIVATE "${FRANKA_SOURCE_DIR}/../..")
  target_link_libraries(franka_example_utils_test PRIVATE
    controllers_local Franka::Franka Eigen3::Eigen GTest::gtest_main)
  target_link_options(franka_example_utils_test PRIVATE "LINKER:--disable-new-dtags")
  set_target_properties(franka_example_utils_test PROPERTIES
    BUILD_RPATH "${FRANKA_BUILD_RPATH}")
  if(FRANKA_EXPECTED_VERSION STREQUAL "0.9.2")
    target_compile_definitions(franka_example_utils_test PRIVATE NVIDIA_SRL_FRANKA_FER=1)
  endif()
endif()
