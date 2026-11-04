# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

include(ExternalProject)
include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.4 REQUIRED CONFIG)
find_package(Boost 1.71 REQUIRED COMPONENTS filesystem serialization)
pkg_check_modules(UR_URDFDOM REQUIRED IMPORTED_TARGET "urdfdom>=1.0")

set(URCL_VERSION "2.14.1")
set(URCL_GIT_REVISION "a854de920d602300ce7c0fe3bd79655cb85bd678")
set(UR_PINOCCHIO_VERSION "3.9.0")
set(UR_PINOCCHIO_GIT_REVISION "ed3bb75ce96cf26e84aebd8b73785407950a0f1f")
set(UR_DESCRIPTION_VERSION "3.5.1")
set(UR_DESCRIPTION_GIT_REVISION "710505166d0399b506838d55663f634880d235fe")
set(UR_EXAMPLE_ROOT "${CMAKE_BINARY_DIR}/_deps/ur")
set(URCL_SOURCE_DIR "${UR_EXAMPLE_ROOT}/urcl/src/ur_client_library")
set(URCL_LOCAL_PREFIX "${UR_EXAMPLE_ROOT}/install")
set(UR_PINOCCHIO_SOURCE_DIR "${UR_EXAMPLE_ROOT}/pinocchio/src/pinocchio_ur")
set(UR_PINOCCHIO_LOCAL_PREFIX "${UR_EXAMPLE_ROOT}/pinocchio-install")
set(URCL_LOCAL_LIBRARY
    "${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}urcl${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(UR_PINOCCHIO_LOCAL_LIBRARIES
    "${UR_PINOCCHIO_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libpinocchio_default.so"
    "${UR_PINOCCHIO_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libpinocchio_parsers.so")
set(UR_EXAMPLE_PREFIX "${UR_EXAMPLE_ROOT}/example-install")
set(UR_ADMITTANCE_EXAMPLE_BINARY
    "${UR_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/cartesian_admittance_controller")
set(UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_BINARY
    "${UR_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/cartesian_unified_force_admittance_controller")
set(UR_UNIFIED_FORCE_ADMITTANCE_EE_EXAMPLE_BINARY
    "${UR_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/cartesian_unified_force_admittance_controller_in_ee")
set(UR_JOINT_IMPEDANCE_EXAMPLE_BINARY
    "${UR_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/joint_impedance_controller")
set(UR_CARTESIAN_IMPEDANCE_EXAMPLE_BINARY
    "${UR_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/cartesian_impedance_controller")
set(UR_KINEMATICS_TEST_BINARY
    "${UR_EXAMPLE_ROOT}/example/src/ur_example_project-build/ur_kinematics_test")

message(STATUS
  "UR example: ur_client_library ${URCL_VERSION} "
  "(${URCL_GIT_REVISION}, exact bundled revision)")
message(STATUS
  "UR Cartesian impedance: Pinocchio ${UR_PINOCCHIO_VERSION} "
  "(${UR_PINOCCHIO_GIT_REVISION}, exact bundled revision); generated URDFs from "
  "ur_description ${UR_DESCRIPTION_VERSION} (${UR_DESCRIPTION_GIT_REVISION})")

ExternalProject_Add(ur_client_library_external
  PREFIX "${UR_EXAMPLE_ROOT}/urcl"
  SOURCE_DIR "${URCL_SOURCE_DIR}"
  GIT_REPOSITORY
    "https://github.com/UniversalRobots/Universal_Robots_Client_Library.git"
  GIT_TAG "${URCL_GIT_REVISION}"
  GIT_SHALLOW FALSE
  UPDATE_DISCONNECTED TRUE
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${URCL_LOCAL_PREFIX}"
    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    "-DBUILDING_TESTS=OFF"
    "-DBUILD_SHARED_LIBS=ON"
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target urcl --parallel
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR>
  BUILD_BYPRODUCTS "${URCL_LOCAL_LIBRARY}"
)

# Keep the Cartesian model dependency private to the opt-in UR examples. Eigen,
# Boost, urdfdom, console_bridge, and tinyxml remain version-checked host prerequisites.
ExternalProject_Add(pinocchio_ur
  PREFIX "${UR_EXAMPLE_ROOT}/pinocchio"
  SOURCE_DIR "${UR_PINOCCHIO_SOURCE_DIR}"
  GIT_REPOSITORY "https://github.com/stack-of-tasks/pinocchio.git"
  GIT_TAG "${UR_PINOCCHIO_GIT_REVISION}"
  GIT_SHALLOW FALSE
  GIT_SUBMODULES cmake
  GIT_SUBMODULES_RECURSE TRUE
  UPDATE_DISCONNECTED TRUE
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=${UR_PINOCCHIO_LOCAL_PREFIX}"
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
  BUILD_BYPRODUCTS ${UR_PINOCCHIO_LOCAL_LIBRARIES}
)

ExternalProject_Add(ur_example_project
  PREFIX "${UR_EXAMPLE_ROOT}/example"
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${URCL_LOCAL_PREFIX}"
    "-DCMAKE_INSTALL_PREFIX=${UR_EXAMPLE_PREFIX}"
    "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
    "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    "-DCONTROLLERS_INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include"
    "-DCONTROLLERS_GENERATED_INCLUDE_DIR=${CMAKE_BINARY_DIR}/include"
    "-DCONTROLLERS_LIBRARY=$<TARGET_FILE:nvidia_srl_controllers>"
    "-DEigen3_DIR=${Eigen3_DIR}"
    "-Dpinocchio_DIR=${UR_PINOCCHIO_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/pinocchio"
    "-DPINOCCHIO_EXPECTED_VERSION=${UR_PINOCCHIO_VERSION}"
    "-DENABLE_TESTS=${ENABLE_TESTS}"
    "-DUR_EXAMPLE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DUR_EXAMPLE_PROJECT_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DURCL_RESOURCE_DIR=${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_DATADIR}/ur_client_library/resources"
  BUILD_BYPRODUCTS
    "${UR_ADMITTANCE_EXAMPLE_BINARY}"
    "${UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_BINARY}"
    "${UR_UNIFIED_FORCE_ADMITTANCE_EE_EXAMPLE_BINARY}"
    "${UR_JOINT_IMPEDANCE_EXAMPLE_BINARY}"
    "${UR_CARTESIAN_IMPEDANCE_EXAMPLE_BINARY}"
  BUILD_ALWAYS TRUE
  DEPENDS ur_client_library_external pinocchio_ur nvidia_srl_controllers
)

install(CODE
  "if(NOT EXISTS \"${UR_ADMITTANCE_EXAMPLE_BINARY}\" OR\n   NOT EXISTS \"${UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_BINARY}\" OR\n   NOT EXISTS \"${UR_UNIFIED_FORCE_ADMITTANCE_EE_EXAMPLE_BINARY}\" OR\n   NOT EXISTS \"${UR_JOINT_IMPEDANCE_EXAMPLE_BINARY}\" OR\n   NOT EXISTS \"${UR_CARTESIAN_IMPEDANCE_EXAMPLE_BINARY}\")\n  message(FATAL_ERROR \"The UR examples have not been built; run cmake --build before cmake --install.\")\nendif()")
install(DIRECTORY "${UR_EXAMPLE_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(DIRECTORY "${URCL_LOCAL_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(DIRECTORY "${UR_PINOCCHIO_LOCAL_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(FILES "${URCL_SOURCE_DIR}/LICENSE_apache_2_0"
  DESTINATION
    "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/ur_client_library")
# Preserve upstream attribution and subcomponent license files, including NOTICE.
install(DIRECTORY "${URCL_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/ur_client_library"
  FILES_MATCHING
    PATTERN "LICENSE*"
    PATTERN "COPYING*"
    PATTERN "NOTICE*"
    PATTERN "Apache-*.txt"
    PATTERN "MIT.txt"
    PATTERN ".git" EXCLUDE)
install(FILES "${UR_PINOCCHIO_SOURCE_DIR}/LICENSE"
  DESTINATION
    "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/pinocchio")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ur_description_LICENSE"
  DESTINATION
    "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/ur_description")

if(ENABLE_TESTS)
  add_test(
    NAME nvidia_srl_controllers_ur_cartesian_admittance_non_rt_preconnection
    COMMAND ${CMAKE_COMMAND}
      "-DEXAMPLE_BINARY=${UR_ADMITTANCE_EXAMPLE_BINARY}"
      "-DEXAMPLE_ARGUMENTS=robot.invalid|--duration-seconds=60|--non-rt"
      "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  add_test(
    NAME nvidia_srl_controllers_ur_unified_force_admittance_non_rt_preconnection
    COMMAND ${CMAKE_COMMAND}
      "-DEXAMPLE_BINARY=${UR_UNIFIED_FORCE_ADMITTANCE_EXAMPLE_BINARY}"
      "-DEXAMPLE_ARGUMENTS=robot.invalid|--duration-seconds=60|--non-rt"
      "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  add_test(
    NAME nvidia_srl_controllers_ur_unified_force_admittance_ee_non_rt_preconnection
    COMMAND ${CMAKE_COMMAND}
      "-DEXAMPLE_BINARY=${UR_UNIFIED_FORCE_ADMITTANCE_EE_EXAMPLE_BINARY}"
      "-DEXAMPLE_ARGUMENTS=robot.invalid|--duration-seconds=60|--non-rt"
      "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  add_test(
    NAME nvidia_srl_controllers_ur_joint_impedance_non_rt_preconnection
    COMMAND ${CMAKE_COMMAND}
      "-DEXAMPLE_BINARY=${UR_JOINT_IMPEDANCE_EXAMPLE_BINARY}"
      "-DEXAMPLE_ARGUMENTS=robot.invalid|--robot-model=ur5e|--duration-seconds=60|--non-rt"
      "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  add_test(
    NAME nvidia_srl_controllers_ur_cartesian_impedance_non_rt_preconnection
    COMMAND ${CMAKE_COMMAND}
      "-DEXAMPLE_BINARY=${UR_CARTESIAN_IMPEDANCE_EXAMPLE_BINARY}"
      "-DEXAMPLE_ARGUMENTS=robot.invalid|--robot-model=ur10e|--duration-seconds=60|--non-rt"
      "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${URCL_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${UR_PINOCCHIO_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  add_test(
    NAME nvidia_srl_controllers_ur_kinematics
    COMMAND ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${UR_PINOCCHIO_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
      "${UR_KINEMATICS_TEST_BINARY}"
      "${CMAKE_CURRENT_SOURCE_DIR}/models/ur5e.urdf"
      "${CMAKE_CURRENT_SOURCE_DIR}/models/ur10e.urdf")
endif()

list(APPEND NV_EXAMPLE_TARGETS ur_example_project)
set(NV_EXAMPLE_TARGETS ${NV_EXAMPLE_TARGETS} PARENT_SCOPE)
