# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

include(ExternalProject)
include(GNUInstallDirs)
include("${CMAKE_CURRENT_SOURCE_DIR}/LibfrankaVersions.cmake")
nvidia_srl_controllers_select_libfranka_version("${LIBFRANKA_VERSION}")

set(FRANKA_EXAMPLE_ROOT "${CMAKE_BINARY_DIR}/_deps/libfranka-${LIBFRANKA_VERSION}")
set(FRANKA_SOURCE_DIR "${FRANKA_EXAMPLE_ROOT}/src/libfranka")
set(FRANKA_LOCAL_PREFIX "${FRANKA_EXAMPLE_ROOT}/install")
set(FRANKA_EXAMPLE_PREFIX "${FRANKA_EXAMPLE_ROOT}/example-install")
set(FRANKA_EXAMPLE_BINARY_DIR "${FRANKA_EXAMPLE_ROOT}/example-build")
set(FRANKA_LOCAL_LIBRARY
    "${FRANKA_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}franka${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(FRANKA_EXAMPLE_NAMES
  franka_virtual_walls_controller
  franka_artificial_potential_field_controller
  franka_cartesian_admittance_controller
  franka_cartesian_force_impedance_controller
  franka_cartesian_force_impedance_controller_with_tank
  franka_cartesian_force_impedance_controller_in_ee
  franka_cartesian_impedance_controller
  franka_joint_impedance_controller
  franka_variable_cartesian_impedance_controller
)
set(FRANKA_EXAMPLE_BINARIES)
foreach(FRANKA_EXAMPLE IN LISTS FRANKA_EXAMPLE_NAMES)
  list(APPEND FRANKA_EXAMPLE_BINARIES
    "${FRANKA_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/${FRANKA_EXAMPLE}")
endforeach()
set(FRANKA_EXAMPLE_UTILS_TEST_BINARY
    "${FRANKA_EXAMPLE_BINARY_DIR}/franka_example_utils_test")

set(FRANKA_CMAKE_PREFIX_PATH "")
set(FRANKA_DEPENDENCIES)
if(NOT LIBFRANKA_VERSION STREQUAL "0.9.2")
  # Recent libfranka versions use Pinocchio to construct robot models.
  set(FRANKA_PINOCCHIO_REVISION "ed3bb75ce96cf26e84aebd8b73785407950a0f1f")
  set(FRANKA_PINOCCHIO_PREFIX "${FRANKA_EXAMPLE_ROOT}/pinocchio-install")
  ExternalProject_Add(franka_pinocchio
    PREFIX "${FRANKA_EXAMPLE_ROOT}/pinocchio"
    SOURCE_DIR "${FRANKA_EXAMPLE_ROOT}/pinocchio/src/pinocchio"
    GIT_REPOSITORY "https://github.com/stack-of-tasks/pinocchio.git"
    GIT_TAG "${FRANKA_PINOCCHIO_REVISION}"
    GIT_SHALLOW FALSE
    GIT_SUBMODULES cmake
    GIT_SUBMODULES_RECURSE TRUE
    UPDATE_DISCONNECTED TRUE
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_INSTALL_PREFIX=${FRANKA_PINOCCHIO_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
      "-DBUILD_PYTHON_INTERFACE=OFF"
      "-DBUILD_TESTING=OFF"
      "-DBUILD_EXAMPLES=OFF"
      "-DBUILD_BENCHMARK=OFF"
      "-DBUILD_UTILS=OFF"
      "-DINSTALL_DOCUMENTATION=OFF"
      "-DBUILD_WITH_URDF_SUPPORT=ON"
      "-DBUILD_WITH_COLLISION_SUPPORT=OFF"
      "-DBUILD_WITH_SDF_SUPPORT=OFF"
      "-DBUILD_WITH_AUTODIFF_SUPPORT=OFF"
      "-DBUILD_WITH_CASADI_SUPPORT=OFF"
      "-DBUILD_WITH_OPENMP_SUPPORT=OFF"
    BUILD_BYPRODUCTS
      "${FRANKA_PINOCCHIO_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libpinocchio_default.so"
      "${FRANKA_PINOCCHIO_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libpinocchio_parsers.so"
  )
  set(FRANKA_CMAKE_PREFIX_PATH "${FRANKA_PINOCCHIO_PREFIX}")
  list(APPEND FRANKA_DEPENDENCIES franka_pinocchio)
endif()

message(STATUS
  "Franka examples: ${FRANKA_ROBOT_FAMILY}, libfranka ${LIBFRANKA_VERSION} "
  "(${FRANKA_GIT_REVISION}, locally built)")

ExternalProject_Add(libfranka_external
  PREFIX "${FRANKA_EXAMPLE_ROOT}/libfranka"
  SOURCE_DIR "${FRANKA_SOURCE_DIR}"
  GIT_REPOSITORY "https://github.com/frankarobotics/libfranka.git"
  GIT_TAG "${FRANKA_GIT_REVISION}"
  GIT_SHALLOW FALSE
  GIT_SUBMODULES common
  GIT_SUBMODULES_RECURSE TRUE
  UPDATE_DISCONNECTED TRUE
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${FRANKA_CMAKE_PREFIX_PATH}"
    "-DCMAKE_INSTALL_PREFIX=${FRANKA_LOCAL_PREFIX}"
    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    "-DBUILD_TESTS=OFF"
    "-DBUILD_EXAMPLES=OFF"
    "-DBUILD_DOCUMENTATION=OFF"
    "-DGENERATE_PYLIBFRANKA=OFF"
  BUILD_BYPRODUCTS "${FRANKA_LOCAL_LIBRARY}"
  DEPENDS ${FRANKA_DEPENDENCIES}
)

# Configure examples only after the private SDK package exists. The consumer
# restricts package discovery to this prefix, so /usr/local cannot be selected.
ExternalProject_Add(franka_examples_project
  PREFIX "${FRANKA_EXAMPLE_ROOT}/example"
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
  BINARY_DIR "${FRANKA_EXAMPLE_BINARY_DIR}"
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${FRANKA_EXAMPLE_PREFIX}"
    "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    "-DCONTROLLERS_INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include"
    "-DCONTROLLERS_GENERATED_INCLUDE_DIR=${CMAKE_BINARY_DIR}/include"
    "-DCONTROLLERS_LIBRARY=$<TARGET_FILE:nvidia_srl_controllers>"
    "-DFRANKA_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DFRANKA_TEST_SOURCE=${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/franka_example_utils_test.cpp"
    "-DFRANKA_LOCAL_PREFIX=${FRANKA_LOCAL_PREFIX}"
    "-DFRANKA_PINOCCHIO_PREFIX=${FRANKA_PINOCCHIO_PREFIX}"
    "-DFRANKA_EXPECTED_VERSION=${LIBFRANKA_VERSION}"
    "-DBUILD_FRANKA_TESTS=${ENABLE_TESTS}"
  BUILD_BYPRODUCTS ${FRANKA_EXAMPLE_BINARIES}
  BUILD_ALWAYS TRUE
  DEPENDS libfranka_external nvidia_srl_controllers
)

foreach(FRANKA_EXAMPLE_BINARY IN LISTS FRANKA_EXAMPLE_BINARIES)
  install(CODE
    "if(NOT EXISTS \"${FRANKA_EXAMPLE_BINARY}\")\n  message(FATAL_ERROR \"Missing built Franka example: ${FRANKA_EXAMPLE_BINARY}\")\nendif()")
endforeach()
install(PROGRAMS ${FRANKA_EXAMPLE_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${FRANKA_LOCAL_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(FILES "${FRANKA_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/libfranka")
# Preserve upstream attribution and subcomponent license files, including NOTICE.
install(DIRECTORY "${FRANKA_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/libfranka"
  FILES_MATCHING
    PATTERN "LICENSE*"
    PATTERN "COPYING*"
    PATTERN "NOTICE*"
    PATTERN ".git" EXCLUDE)
if(FRANKA_PINOCCHIO_PREFIX)
  install(DIRECTORY "${FRANKA_PINOCCHIO_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
  install(FILES "${FRANKA_EXAMPLE_ROOT}/pinocchio/src/pinocchio/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/pinocchio")
endif()

if(ENABLE_TESTS)
  foreach(FRANKA_EXAMPLE IN LISTS FRANKA_EXAMPLE_NAMES)
    add_test(
      NAME "nvidia_srl_controllers_${FRANKA_EXAMPLE}_non_rt_preconnection"
      COMMAND ${CMAKE_COMMAND}
        "-DEXAMPLE_BINARY=${FRANKA_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/${FRANKA_EXAMPLE}"
        "-DEXAMPLE_ARGUMENTS=robot.invalid|--duration-seconds=60.5|--non-rt"
        "-DEXPECTED_OUTPUT=runs the controller for 60.5 s."
        -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  endforeach()
  add_test(
    NAME nvidia_srl_controllers_franka_example_utils
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${FRANKA_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${FRANKA_PINOCCHIO_PREFIX}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
      "${FRANKA_EXAMPLE_UTILS_TEST_BINARY}")
endif()
