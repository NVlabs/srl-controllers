# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

if(NOT DEFINED PRODUCER_BINARY_DIR OR "${PRODUCER_BINARY_DIR}" STREQUAL "" OR
   NOT DEFINED STAGE_DIR OR "${STAGE_DIR}" STREQUAL "")
  message(FATAL_ERROR "PRODUCER_BINARY_DIR and STAGE_DIR are required")
endif()

file(REMOVE_RECURSE "${STAGE_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${PRODUCER_BINARY_DIR}" --prefix "${STAGE_DIR}"
  RESULT_VARIABLE INSTALL_RESULT
  OUTPUT_VARIABLE INSTALL_OUTPUT
  ERROR_VARIABLE INSTALL_ERROR)
if(NOT INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "Flexiv dual-arm staged install failed:\n${INSTALL_OUTPUT}\n${INSTALL_ERROR}")
endif()

set(REQUIRED_FILES
  "${STAGE_DIR}/bin/flexiv_dual_bimanual_cartesian_impedance"
  "${STAGE_DIR}/bin/flexiv_dual_bimanual_compliant_hold"
  "${STAGE_DIR}/include/flexiv/rdk/robot.hpp"
  "${STAGE_DIR}/include/flexiv/rdk/model.hpp"
  "${STAGE_DIR}/lib/libflexiv_rdk.a"
  "${STAGE_DIR}/lib/cmake/flexiv_rdk/flexiv_rdk-config.cmake"
  "${STAGE_DIR}/share/fastrtps/cmake/fastrtps-config.cmake"
  "${STAGE_DIR}/lib/cmake/RBDyn/RBDynConfig.cmake"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/third-party/flexiv-rdk/LICENSE"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/third-party/flexiv-rdk-dependencies/Fast-DDS/LICENSE")
foreach(REQUIRED_FILE IN LISTS REQUIRED_FILES)
  if(NOT EXISTS "${REQUIRED_FILE}")
    message(FATAL_ERROR "Missing locally installed Flexiv dependency: ${REQUIRED_FILE}")
  endif()
endforeach()

set(INSTALLED_CONSUMER_BINARY_DIR "${STAGE_DIR}/consumer-build")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_LIST_DIR}/../../examples/flexiv_dual/consumer"
    -B "${INSTALLED_CONSUMER_BINARY_DIR}"
    "-DCMAKE_PREFIX_PATH=${STAGE_DIR}"
    "-DCONTROLLERS_INCLUDE_DIR=${STAGE_DIR}/include"
    "-DCONTROLLERS_LIBRARY=${STAGE_DIR}/lib/libnvidia_srl_controllers.so"
    -DFLEXIV_RDK_EXPECTED_VERSION=1.9.1
  RESULT_VARIABLE CONSUMER_CONFIGURE_RESULT
  OUTPUT_VARIABLE CONSUMER_CONFIGURE_OUTPUT
  ERROR_VARIABLE CONSUMER_CONFIGURE_ERROR)
if(NOT CONSUMER_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Installed Flexiv packages are not independently consumable:\n"
    "${CONSUMER_CONFIGURE_OUTPUT}\n${CONSUMER_CONFIGURE_ERROR}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${INSTALLED_CONSUMER_BINARY_DIR}" --parallel 2
  RESULT_VARIABLE CONSUMER_BUILD_RESULT
  OUTPUT_VARIABLE CONSUMER_BUILD_OUTPUT
  ERROR_VARIABLE CONSUMER_BUILD_ERROR)
if(NOT CONSUMER_BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Installed Flexiv consumer build failed:\n"
    "${CONSUMER_BUILD_OUTPUT}\n${CONSUMER_BUILD_ERROR}")
endif()

file(READ "${STAGE_DIR}/lib/cmake/flexiv_rdk/flexiv_rdk-config-version.cmake" VERSION_TEXT)
if(NOT VERSION_TEXT MATCHES "set\\(PACKAGE_VERSION \"1\\.9\\.1\"\\)")
  message(FATAL_ERROR "Installed Flexiv RDK package does not report exact version 1.9.1")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${STAGE_DIR}/bin/flexiv_dual_bimanual_cartesian_impedance"
  RESULT_VARIABLE READELF_RESULT
  OUTPUT_VARIABLE READELF_OUTPUT
  ERROR_VARIABLE READELF_ERROR)
if(NOT READELF_RESULT EQUAL 0 OR
   NOT READELF_OUTPUT MATCHES "\\(RPATH\\).*\\[\\$ORIGIN/\\.\\./lib\\]")
  message(FATAL_ERROR "Installed dual-arm example lacks its transitive relative RPATH: ${READELF_ERROR}")
endif()

find_program(LDD_EXECUTABLE ldd REQUIRED)
execute_process(
  COMMAND "${LDD_EXECUTABLE}" "${STAGE_DIR}/bin/flexiv_dual_bimanual_cartesian_impedance"
  RESULT_VARIABLE LDD_RESULT
  OUTPUT_VARIABLE LDD_OUTPUT
  ERROR_VARIABLE LDD_ERROR)
if(NOT LDD_RESULT EQUAL 0 OR LDD_OUTPUT MATCHES "not found")
  message(FATAL_ERROR "Installed dual-arm example has unresolved libraries:\n${LDD_OUTPUT}\n${LDD_ERROR}")
endif()

execute_process(
  COMMAND "${STAGE_DIR}/bin/flexiv_dual_bimanual_cartesian_impedance"
  RESULT_VARIABLE EXECUTE_RESULT
  OUTPUT_VARIABLE EXECUTE_OUTPUT
  ERROR_VARIABLE EXECUTE_ERROR)
if(NOT EXECUTE_RESULT EQUAL 1 OR NOT EXECUTE_ERROR MATCHES "Usage:")
  message(FATAL_ERROR
    "Installed dual-arm example did not reach argument validation:\n"
    "${EXECUTE_OUTPUT}\n${EXECUTE_ERROR}")
endif()
