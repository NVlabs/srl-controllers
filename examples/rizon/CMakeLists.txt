# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

include(ExternalProject)
include(GNUInstallDirs)

include("${CMAKE_CURRENT_SOURCE_DIR}/FlexivRdkVersions.cmake")
nvidia_srl_controllers_select_flexiv_rdk_version("${FLEXIV_RDK_VERSION}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  message(FATAL_ERROR
    "Rizon examples are qualified only for Linux x86_64; received "
    "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}.")
endif()
if(FLEXIV_RDK_REQUIRES_DEPENDENCIES AND NOT CMAKE_INSTALL_LIBDIR STREQUAL "lib")
  message(FATAL_ERROR
    "Flexiv RDK ${FLEXIV_RDK_VERSION} uses upstream static-dependency installers "
    "that require CMAKE_INSTALL_LIBDIR=lib.")
endif()

message(STATUS
  "Rizon examples: Flexiv RDK ${FLEXIV_RDK_VERSION} "
  "(${FLEXIV_RDK_INSTALL_KIND}, exact bundled revision)")

set(FLEXIV_RDK_ROOT "${CMAKE_BINARY_DIR}/_deps/flexiv_rdk-${FLEXIV_RDK_VERSION}")
set(FLEXIV_RDK_SOURCE_DIR "${FLEXIV_RDK_ROOT}/src/flexiv_rdk")
set(FLEXIV_RDK_LOCAL_PREFIX "${FLEXIV_RDK_ROOT}/install")
set(FLEXIV_RDK_DEPENDENCY_SOURCE_DIR "${FLEXIV_RDK_ROOT}/dependencies-build/cloned")
set(FLEXIV_RDK_LOCAL_LIBRARY
    "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${FLEXIV_RDK_INSTALLED_LIBRARY}")
set(FLEXIV_RDK_DEPENDENCY_PINS_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/FlexivRdkDependencyPins.cmake")

set(FLEXIV_RDK_CMAKE_ARGS
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_PREFIX_PATH=${FLEXIV_RDK_LOCAL_PREFIX}"
  "-DCMAKE_INSTALL_PREFIX=${FLEXIV_RDK_LOCAL_PREFIX}"
  "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
)

if(FLEXIV_RDK_REQUIRES_DEPENDENCIES)
  # RDK 1.9.1-1.9.3 ship a static archive whose public target depends on the
  # release's Fast DDS, RBDyn, and support-library stack. Build the exact pinned
  # dependency sources first and keep the entire installation in one private prefix.
  ExternalProject_Add(flexiv_rdk_dependencies
    PREFIX "${FLEXIV_RDK_ROOT}/dependencies"
    SOURCE_DIR "${FLEXIV_RDK_SOURCE_DIR}"
    BINARY_DIR "${FLEXIV_RDK_ROOT}/dependencies-build"
    GIT_REPOSITORY "https://github.com/flexivrobotics/flexiv_rdk.git"
    GIT_TAG "${FLEXIV_RDK_GIT_REVISION}"
    GIT_SHALLOW FALSE
    UPDATE_DISCONNECTED TRUE
    PATCH_COMMAND
      ${CMAKE_COMMAND}
        "-DRDK_SOURCE_DIR=<SOURCE_DIR>"
        "-DDEPENDENCY_PINS_FILE=${FLEXIV_RDK_DEPENDENCY_PINS_FILE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/PatchFlexivRdkDependencies.cmake"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      bash "<SOURCE_DIR>/thirdparty/build_and_install_dependencies.sh"
        "${FLEXIV_RDK_LOCAL_PREFIX}" 4
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libfastrtps.so"
      "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libRBDyn.so"
  )

  ExternalProject_Add(flexiv_rdk
    PREFIX "${FLEXIV_RDK_ROOT}/rdk"
    SOURCE_DIR "${FLEXIV_RDK_SOURCE_DIR}"
    BINARY_DIR "${FLEXIV_RDK_ROOT}/rdk-build"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND
      ${CMAKE_COMMAND}
        "-DRDK_HASH_FILE=<SOURCE_DIR>/lib/${FLEXIV_RDK_ASSET_NAME}.sha256"
        "-DRDK_HASH=${FLEXIV_RDK_LINUX_X86_64_SHA256}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/PatchFlexivRdkChecksum.cmake"
    CMAKE_ARGS ${FLEXIV_RDK_CMAKE_ARGS}
    BUILD_BYPRODUCTS "${FLEXIV_RDK_LOCAL_LIBRARY}"
    DEPENDS flexiv_rdk_dependencies
  )
else()
  list(APPEND FLEXIV_RDK_CMAKE_ARGS "-DEigen3_DIR=${Eigen3_DIR}")
  ExternalProject_Add(flexiv_rdk
    PREFIX "${FLEXIV_RDK_ROOT}/rdk"
    SOURCE_DIR "${FLEXIV_RDK_SOURCE_DIR}"
    BINARY_DIR "${FLEXIV_RDK_ROOT}/rdk-build"
    GIT_REPOSITORY "https://github.com/flexivrobotics/flexiv_rdk.git"
    GIT_TAG "${FLEXIV_RDK_GIT_REVISION}"
    GIT_SHALLOW FALSE
    UPDATE_DISCONNECTED TRUE
    PATCH_COMMAND
      ${CMAKE_COMMAND}
        "-DRDK_HASH_FILE=<SOURCE_DIR>/lib/${FLEXIV_RDK_ASSET_NAME}.sha256"
        "-DRDK_HASH=${FLEXIV_RDK_LINUX_X86_64_SHA256}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/PatchFlexivRdkChecksum.cmake"
    CMAKE_ARGS ${FLEXIV_RDK_CMAKE_ARGS}
    BUILD_BYPRODUCTS "${FLEXIV_RDK_LOCAL_LIBRARY}"
  )
endif()

set(RIZON_EXAMPLE_NAMES
  rizon_cartesian_admittance_controller
  rizon_cartesian_force_impedance_controller
  rizon_cartesian_force_impedance_controller_in_ee
  rizon_cartesian_impedance_controller
  rizon_joint_impedance_controller
  rizon_variable_cartesian_impedance_controller
)
set(RIZON_EXAMPLE_PREFIX "${FLEXIV_RDK_ROOT}/example-install")
set(RIZON_EXAMPLE_BINARY_DIR "${FLEXIV_RDK_ROOT}/example-build")
set(RIZON_EXAMPLE_BINARIES)
foreach(RIZON_EXAMPLE IN LISTS RIZON_EXAMPLE_NAMES)
  list(APPEND RIZON_EXAMPLE_BINARIES
    "${RIZON_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/${RIZON_EXAMPLE}")
endforeach()
set(RIZON_EXAMPLE_UTILS_TEST_BINARY
    "${RIZON_EXAMPLE_BINARY_DIR}/rizon_example_utils_test")

set(RIZON_CONSUMER_CMAKE_ARGS
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_PREFIX_PATH=${FLEXIV_RDK_LOCAL_PREFIX}"
  "-DCMAKE_INSTALL_PREFIX=${RIZON_EXAMPLE_PREFIX}"
  "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
  "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
  "-DCONTROLLERS_INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include"
  "-DCONTROLLERS_LIBRARY=$<TARGET_FILE:nvidia_srl_controllers>"
  "-DRIZON_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
  "-DRIZON_TEST_SOURCE=${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/rizon_example_utils_test.cpp"
  "-DFLEXIV_RDK_EXPECTED_VERSION=${FLEXIV_RDK_VERSION}"
  "-DBUILD_RIZON_TESTS=${ENABLE_TESTS}"
)

# Configure the examples only after the complete RDK package exists. This lets
# every selected version provide its own official transitive link interface.
ExternalProject_Add(rizon_examples_project
  PREFIX "${FLEXIV_RDK_ROOT}/example"
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
  BINARY_DIR "${RIZON_EXAMPLE_BINARY_DIR}"
  CMAKE_ARGS ${RIZON_CONSUMER_CMAKE_ARGS}
  BUILD_BYPRODUCTS ${RIZON_EXAMPLE_BINARIES}
  BUILD_ALWAYS TRUE
  DEPENDS flexiv_rdk nvidia_srl_controllers
)

foreach(RIZON_EXAMPLE_BINARY IN LISTS RIZON_EXAMPLE_BINARIES)
  install(CODE
    "if(NOT EXISTS \"${RIZON_EXAMPLE_BINARY}\")\n  message(FATAL_ERROR \"Missing built Rizon example: ${RIZON_EXAMPLE_BINARY}\")\nendif()")
endforeach()
install(PROGRAMS ${RIZON_EXAMPLE_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${FLEXIV_RDK_LOCAL_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(FILES "${FLEXIV_RDK_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/flexiv-rdk")
if(FLEXIV_RDK_REQUIRES_DEPENDENCIES)
  install(DIRECTORY "${FLEXIV_RDK_DEPENDENCY_SOURCE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/flexiv-rdk-dependencies"
    FILES_MATCHING
    PATTERN "LICENSE*"
    PATTERN "COPYING*"
    PATTERN "NOTICE*")
endif()

if(ENABLE_TESTS)
  foreach(RIZON_EXAMPLE IN LISTS RIZON_EXAMPLE_NAMES)
    add_test(
      NAME "nvidia_srl_controllers_${RIZON_EXAMPLE}_non_rt_preconnection"
      COMMAND ${CMAKE_COMMAND}
        "-DEXAMPLE_BINARY=${RIZON_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/${RIZON_EXAMPLE}"
        "-DEXAMPLE_ARGUMENTS=robot.invalid|--duration-seconds=60|--non-rt"
        "-DEXAMPLE_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        -P "${CMAKE_SOURCE_DIR}/tests/cmake/non_rt_preconnection_smoke_test.cmake")
  endforeach()
  add_test(
    NAME nvidia_srl_controllers_rizon_example_utils
    COMMAND "${RIZON_EXAMPLE_UTILS_TEST_BINARY}"
  )
  add_test(
    NAME nvidia_srl_controllers_rizon_local_install
    COMMAND ${CMAKE_COMMAND}
      "-DPRODUCER_BINARY_DIR=${CMAKE_BINARY_DIR}"
      "-DSTAGE_DIR=${CMAKE_BINARY_DIR}/rizon-install-test"
      "-DRIZON_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
      "-DRIZON_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DDEPENDENCY_PINS_FILE=${FLEXIV_RDK_DEPENDENCY_PINS_FILE}"
      "-DEXPECTED_RDK_VERSION=${FLEXIV_RDK_VERSION}"
      "-DEXPECTED_RDK_LIBRARY=${FLEXIV_RDK_INSTALLED_LIBRARY}"
      "-DEXPECT_BUNDLED_DEPENDENCIES=${FLEXIV_RDK_REQUIRES_DEPENDENCIES}"
      "-DINSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
      "-DINSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/rizon_local_install_test.cmake"
  )
endif()

list(APPEND NV_EXAMPLE_TARGETS rizon_examples_project)
set(NV_EXAMPLE_TARGETS ${NV_EXAMPLE_TARGETS} PARENT_SCOPE)
