# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)
project(ur_example_consumer LANGUAGES CXX)

include(GNUInstallDirs)

foreach(REQUIRED_VARIABLE
    CONTROLLERS_INCLUDE_DIR
    CONTROLLERS_GENERATED_INCLUDE_DIR
    CONTROLLERS_LIBRARY
    PINOCCHIO_EXPECTED_VERSION
    UR_EXAMPLE_SOURCE_DIR
    UR_EXAMPLE_PROJECT_SOURCE_DIR
    URCL_RESOURCE_DIR)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()
if(NOT EXISTS "${CONTROLLERS_INCLUDE_DIR}/nvidia/srl/controllers/controllers.hpp" OR
   NOT EXISTS
     "${CONTROLLERS_GENERATED_INCLUDE_DIR}/nvidia/srl/controllers/version.hpp" OR
   NOT EXISTS "${CONTROLLERS_LIBRARY}" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/cartesian_admittance_controller.cpp" OR
   NOT EXISTS
     "${UR_EXAMPLE_SOURCE_DIR}/cartesian_unified_force_admittance_controller.cpp" OR
   NOT EXISTS
     "${UR_EXAMPLE_SOURCE_DIR}/cartesian_unified_force_admittance_controller_in_ee.cpp" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/unified_force_admittance_example.hpp" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/joint_impedance_controller.cpp" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/cartesian_impedance_controller.cpp" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/ur_kinematics.cpp" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/models/ur5e.urdf" OR
   NOT EXISTS "${UR_EXAMPLE_SOURCE_DIR}/models/ur10e.urdf" OR
   NOT EXISTS "${URCL_RESOURCE_DIR}/external_control.urscript")
  message(FATAL_ERROR
    "Built controller artifacts, UR example source, and the pinned URCL script are required")
endif()

find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(pinocchio ${PINOCCHIO_EXPECTED_VERSION} EXACT REQUIRED CONFIG)
find_package(ur_client_library REQUIRED CONFIG)

add_library(controllers_local SHARED IMPORTED)
set_target_properties(controllers_local PROPERTIES
  IMPORTED_LOCATION "${CONTROLLERS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${CONTROLLERS_INCLUDE_DIR};${CONTROLLERS_GENERATED_INCLUDE_DIR}")

add_library(ur_kinematics STATIC "${UR_EXAMPLE_SOURCE_DIR}/ur_kinematics.cpp")
target_compile_features(ur_kinematics PUBLIC cxx_std_17)
target_compile_options(ur_kinematics PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ur_kinematics PUBLIC "${UR_EXAMPLE_SOURCE_DIR}")
target_link_libraries(ur_kinematics PUBLIC pinocchio::pinocchio Eigen3::Eigen)

add_executable(
  cartesian_admittance_controller
  "${UR_EXAMPLE_SOURCE_DIR}/cartesian_admittance_controller.cpp")
target_compile_features(cartesian_admittance_controller PRIVATE cxx_std_17)
target_compile_options(
  cartesian_admittance_controller PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(cartesian_admittance_controller PRIVATE
  "UR_EXAMPLE_RESOURCE_RELATIVE_DIR=\"../${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur\"")
target_include_directories(
  cartesian_admittance_controller PRIVATE "${UR_EXAMPLE_PROJECT_SOURCE_DIR}")
target_link_options(
  cartesian_admittance_controller PRIVATE "LINKER:--disable-new-dtags")
target_link_libraries(
  cartesian_admittance_controller PRIVATE
  controllers_local ur_client_library::urcl Eigen3::Eigen)

add_executable(
  joint_impedance_controller
  "${UR_EXAMPLE_SOURCE_DIR}/joint_impedance_controller.cpp")
target_compile_features(joint_impedance_controller PRIVATE cxx_std_17)
target_compile_options(joint_impedance_controller PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(joint_impedance_controller PRIVATE
  "UR_EXAMPLE_RESOURCE_RELATIVE_DIR=\"../${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur\"")
target_include_directories(
  joint_impedance_controller PRIVATE "${UR_EXAMPLE_PROJECT_SOURCE_DIR}")
target_link_options(joint_impedance_controller PRIVATE "LINKER:--disable-new-dtags")
target_link_libraries(
  joint_impedance_controller PRIVATE
  controllers_local ur_client_library::urcl Eigen3::Eigen)

add_executable(
  cartesian_impedance_controller
  "${UR_EXAMPLE_SOURCE_DIR}/cartesian_impedance_controller.cpp")
target_compile_features(cartesian_impedance_controller PRIVATE cxx_std_17)
target_compile_options(cartesian_impedance_controller PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(cartesian_impedance_controller PRIVATE
  "UR_EXAMPLE_RESOURCE_RELATIVE_DIR=\"../${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur\"")
target_include_directories(
  cartesian_impedance_controller PRIVATE "${UR_EXAMPLE_PROJECT_SOURCE_DIR}")
target_link_options(cartesian_impedance_controller PRIVATE "LINKER:--disable-new-dtags")
target_link_libraries(
  cartesian_impedance_controller PRIVATE
  ur_kinematics controllers_local ur_client_library::urcl Eigen3::Eigen)

add_executable(
  cartesian_unified_force_admittance_controller
  "${UR_EXAMPLE_SOURCE_DIR}/cartesian_unified_force_admittance_controller.cpp")
target_compile_features(cartesian_unified_force_admittance_controller PRIVATE cxx_std_17)
target_compile_options(
  cartesian_unified_force_admittance_controller PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(cartesian_unified_force_admittance_controller PRIVATE
  "UR_EXAMPLE_RESOURCE_RELATIVE_DIR=\"../${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur\"")
target_include_directories(
  cartesian_unified_force_admittance_controller PRIVATE "${UR_EXAMPLE_PROJECT_SOURCE_DIR}")
target_link_options(
  cartesian_unified_force_admittance_controller PRIVATE "LINKER:--disable-new-dtags")
target_link_libraries(
  cartesian_unified_force_admittance_controller PRIVATE
  controllers_local ur_client_library::urcl Eigen3::Eigen)

add_executable(
  cartesian_unified_force_admittance_controller_in_ee
  "${UR_EXAMPLE_SOURCE_DIR}/cartesian_unified_force_admittance_controller.cpp"
  "${UR_EXAMPLE_SOURCE_DIR}/cartesian_unified_force_admittance_controller_in_ee.cpp")
target_compile_features(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE cxx_std_17)
target_compile_options(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE
  -Wall -Wextra -Wpedantic)
target_compile_definitions(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE
  "UR_EXAMPLE_RESOURCE_RELATIVE_DIR=\"../${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur\""
  UR_UNIFIED_FORCE_ADMITTANCE_NO_MAIN=1)
target_include_directories(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE
  "${UR_EXAMPLE_PROJECT_SOURCE_DIR}")
target_link_options(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE
  "LINKER:--disable-new-dtags")
target_link_libraries(
  cartesian_unified_force_admittance_controller_in_ee PRIVATE
  controllers_local ur_client_library::urcl Eigen3::Eigen)

get_filename_component(CONTROLLERS_LIBRARY_DIR "${CONTROLLERS_LIBRARY}" DIRECTORY)
set_target_properties(
  cartesian_admittance_controller cartesian_unified_force_admittance_controller
  cartesian_unified_force_admittance_controller_in_ee
  joint_impedance_controller cartesian_impedance_controller
  PROPERTIES
  BUILD_RPATH "${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR};${CONTROLLERS_LIBRARY_DIR}"
  INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(
  TARGETS
    cartesian_admittance_controller
    cartesian_unified_force_admittance_controller
    cartesian_unified_force_admittance_controller_in_ee
    joint_impedance_controller
    cartesian_impedance_controller
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(
  FILES
    "${URCL_RESOURCE_DIR}/external_control.urscript"
    "${UR_EXAMPLE_SOURCE_DIR}/rtde_input_recipe.txt"
    "${UR_EXAMPLE_SOURCE_DIR}/rtde_output_recipe.txt"
    "${UR_EXAMPLE_SOURCE_DIR}/rtde_torque_output_recipe.txt"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur")
install(
  FILES
    "${UR_EXAMPLE_SOURCE_DIR}/models/ur5e.urdf"
    "${UR_EXAMPLE_SOURCE_DIR}/models/ur10e.urdf"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/nvidia_srl_controllers/examples/ur/models")

if(ENABLE_TESTS)
  add_executable(
    ur_kinematics_test
    "${UR_EXAMPLE_SOURCE_DIR}/ur_kinematics_test.cpp")
  target_compile_features(ur_kinematics_test PRIVATE cxx_std_17)
  target_compile_options(ur_kinematics_test PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(ur_kinematics_test PRIVATE ur_kinematics Eigen3::Eigen)
endif()
