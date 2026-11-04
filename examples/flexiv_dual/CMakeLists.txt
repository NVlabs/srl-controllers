# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

include(ExternalProject)
include(GNUInstallDirs)

include("${CMAKE_CURRENT_SOURCE_DIR}/../rizon/FlexivRdkVersions.cmake")
nvidia_srl_controllers_select_flexiv_rdk_version("${FLEXIV_RDK_VERSION}")
if(NOT FLEXIV_RDK_VERSION STREQUAL "1.9.1")
  message(FATAL_ERROR
    "The two-Rizon dual-arm example requires the qualified Flexiv RDK 1.9.1 release; "
    "received FLEXIV_RDK_VERSION=${FLEXIV_RDK_VERSION}.")
endif()

set(FLEXIV_DUAL_ROOT "${CMAKE_BINARY_DIR}/_deps/flexiv_rdk-${FLEXIV_RDK_VERSION}")
set(FLEXIV_RDK_SOURCE_DIR "${FLEXIV_DUAL_ROOT}/src/flexiv_rdk")
set(FLEXIV_RDK_LOCAL_PREFIX "${FLEXIV_DUAL_ROOT}/install")
set(FLEXIV_RDK_DEPENDENCY_SOURCE_DIR "${FLEXIV_DUAL_ROOT}/dependencies-build/cloned")
set(FLEXIV_DUAL_EXAMPLE_PREFIX "${FLEXIV_DUAL_ROOT}/example-install")
set(FLEXIV_RDK_LOCAL_LIBRARY
    "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${FLEXIV_RDK_INSTALLED_LIBRARY}")
set(FLEXIV_RDK_DEPENDENCY_PINS_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/../rizon/FlexivRdkDependencyPins.cmake")
set(FLEXIV_DUAL_EXAMPLE_NAMES
  flexiv_dual_bimanual_cartesian_impedance
  flexiv_dual_bimanual_compliant_hold)
set(FLEXIV_DUAL_EXAMPLE_BINARIES)
foreach(EXAMPLE_NAME IN LISTS FLEXIV_DUAL_EXAMPLE_NAMES)
  list(APPEND FLEXIV_DUAL_EXAMPLE_BINARIES
    "${FLEXIV_DUAL_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/${EXAMPLE_NAME}")
endforeach()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  message(FATAL_ERROR
    "The Flexiv dual-arm example is qualified only for Linux x86_64; received "
    "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}.")
endif()

# Reuse the exact RDK 1.9.1 build when the single-arm Rizon examples are also
# selected. Otherwise create the same checksum-verified, dependency-pinned
# private package so the dual-arm option remains independently buildable.
if(TARGET flexiv_rdk)
  set(FLEXIV_DUAL_RDK_TARGET flexiv_rdk)
else()
  ExternalProject_Add(flexiv_dual_rdk_dependencies
    PREFIX "${FLEXIV_DUAL_ROOT}/dependencies"
    SOURCE_DIR "${FLEXIV_RDK_SOURCE_DIR}"
    BINARY_DIR "${FLEXIV_DUAL_ROOT}/dependencies-build"
    GIT_REPOSITORY "https://github.com/flexivrobotics/flexiv_rdk.git"
    GIT_TAG "${FLEXIV_RDK_GIT_REVISION}"
    GIT_SHALLOW FALSE
    UPDATE_DISCONNECTED TRUE
    PATCH_COMMAND
      ${CMAKE_COMMAND}
        "-DRDK_SOURCE_DIR=<SOURCE_DIR>"
        "-DDEPENDENCY_PINS_FILE=${FLEXIV_RDK_DEPENDENCY_PINS_FILE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../rizon/PatchFlexivRdkDependencies.cmake"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      bash "<SOURCE_DIR>/thirdparty/build_and_install_dependencies.sh"
        "${FLEXIV_RDK_LOCAL_PREFIX}" 4
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libfastrtps.so"
      "${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libRBDyn.so"
  )

  ExternalProject_Add(flexiv_dual_rdk
    PREFIX "${FLEXIV_DUAL_ROOT}/rdk"
    SOURCE_DIR "${FLEXIV_RDK_SOURCE_DIR}"
    BINARY_DIR "${FLEXIV_DUAL_ROOT}/rdk-build"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND
      ${CMAKE_COMMAND}
        "-DRDK_HASH_FILE=<SOURCE_DIR>/lib/${FLEXIV_RDK_ASSET_NAME}.sha256"
        "-DRDK_HASH=${FLEXIV_RDK_LINUX_X86_64_SHA256}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../rizon/PatchFlexivRdkChecksum.cmake"
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_PREFIX_PATH=${FLEXIV_RDK_LOCAL_PREFIX}"
      "-DCMAKE_INSTALL_PREFIX=${FLEXIV_RDK_LOCAL_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    BUILD_BYPRODUCTS "${FLEXIV_RDK_LOCAL_LIBRARY}"
    DEPENDS flexiv_dual_rdk_dependencies
  )
  set(FLEXIV_DUAL_RDK_TARGET flexiv_dual_rdk)
endif()

# This staged consumer is configured only after RDK and all of its package
# metadata are installed. It therefore links the official flexiv::flexiv_rdk
# target instead of duplicating the vendor's transitive link interface here.
ExternalProject_Add(flexiv_dual_controller_examples_project
  PREFIX "${FLEXIV_DUAL_ROOT}/example"
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
  CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_PREFIX_PATH=${FLEXIV_RDK_LOCAL_PREFIX}"
    "-DCMAKE_INSTALL_PREFIX=${FLEXIV_DUAL_EXAMPLE_PREFIX}"
    "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
    "-DCONTROLLERS_INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include"
    "-DCONTROLLERS_LIBRARY=$<TARGET_FILE:nvidia_srl_controllers>"
    "-DFLEXIV_RDK_EXPECTED_VERSION=${FLEXIV_RDK_VERSION}"
  BUILD_BYPRODUCTS ${FLEXIV_DUAL_EXAMPLE_BINARIES}
  BUILD_ALWAYS TRUE
  DEPENDS ${FLEXIV_DUAL_RDK_TARGET} nvidia_srl_controllers
)

foreach(EXAMPLE_BINARY IN LISTS FLEXIV_DUAL_EXAMPLE_BINARIES)
  install(CODE "if(NOT EXISTS \"${EXAMPLE_BINARY}\")
    message(FATAL_ERROR \"The Flexiv dual-arm examples have not been built; run cmake --build before cmake --install.\")
  endif()")
endforeach()
install(PROGRAMS ${FLEXIV_DUAL_EXAMPLE_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${FLEXIV_RDK_LOCAL_PREFIX}/" DESTINATION "." USE_SOURCE_PERMISSIONS)
install(FILES "${FLEXIV_RDK_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/flexiv-rdk")
install(DIRECTORY "${FLEXIV_RDK_DEPENDENCY_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/third-party/flexiv-rdk-dependencies"
  FILES_MATCHING
  PATTERN "LICENSE*"
  PATTERN "COPYING*"
  PATTERN "NOTICE*")

if(ENABLE_TESTS)
  add_test(
    NAME nvidia_srl_controllers_flexiv_dual_bimanual_kinematics
    COMMAND
      ${CMAKE_COMMAND} -E env
      "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/nvidia/srl/controllers:${FLEXIV_RDK_LOCAL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
      "${FLEXIV_DUAL_EXAMPLE_PREFIX}/${CMAKE_INSTALL_BINDIR}/flexiv_dual_bimanual_compliant_hold"
      --self-check)
  add_test(
    NAME nvidia_srl_controllers_flexiv_dual_local_install
    COMMAND ${CMAKE_COMMAND}
      "-DPRODUCER_BINARY_DIR=${CMAKE_BINARY_DIR}"
      "-DSTAGE_DIR=${CMAKE_BINARY_DIR}/flexiv-dual-install-test"
      -P "${CMAKE_SOURCE_DIR}/tests/cmake/flexiv_dual_local_install_test.cmake"
  )
endif()

list(APPEND NV_EXAMPLE_TARGETS flexiv_dual_controller_examples_project)
set(NV_EXAMPLE_TARGETS ${NV_EXAMPLE_TARGETS} PARENT_SCOPE)
