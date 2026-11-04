# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)
project(flexiv_dual_controller_examples LANGUAGES CXX)

if(NOT DEFINED CONTROLLERS_INCLUDE_DIR OR NOT DEFINED CONTROLLERS_LIBRARY OR
   NOT EXISTS "${CONTROLLERS_INCLUDE_DIR}/nvidia/srl/controllers/controllers.hpp" OR
   NOT EXISTS "${CONTROLLERS_LIBRARY}")
  message(FATAL_ERROR "CONTROLLERS_INCLUDE_DIR and CONTROLLERS_LIBRARY must name built artifacts")
endif()

find_package(flexiv_rdk ${FLEXIV_RDK_EXPECTED_VERSION} EXACT REQUIRED CONFIG)

add_library(controllers_local SHARED IMPORTED)
set_target_properties(controllers_local PROPERTIES
  IMPORTED_LOCATION "${CONTROLLERS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CONTROLLERS_INCLUDE_DIR}"
)

set(FLEXIV_DUAL_EXAMPLES
  flexiv_dual_bimanual_cartesian_impedance
  flexiv_dual_bimanual_compliant_hold)
set(FLEXIV_DUAL_PROFILES 0 1)
list(LENGTH FLEXIV_DUAL_EXAMPLES FLEXIV_DUAL_EXAMPLE_COUNT)
math(EXPR FLEXIV_DUAL_LAST_INDEX "${FLEXIV_DUAL_EXAMPLE_COUNT} - 1")
foreach(INDEX RANGE ${FLEXIV_DUAL_LAST_INDEX})
  list(GET FLEXIV_DUAL_EXAMPLES ${INDEX} EXAMPLE_NAME)
  list(GET FLEXIV_DUAL_PROFILES ${INDEX} EXAMPLE_PROFILE)
  add_executable(${EXAMPLE_NAME} flexiv_dual_controller_example.cpp)
  target_compile_features(${EXAMPLE_NAME} PRIVATE cxx_std_17)
  target_compile_definitions(${EXAMPLE_NAME} PRIVATE
    FLEXIV_DUAL_EXAMPLE_PROFILE=${EXAMPLE_PROFILE})
  target_compile_options(${EXAMPLE_NAME} PRIVATE -Wall -Wextra -Wpedantic)
  target_link_options(${EXAMPLE_NAME} PRIVATE "LINKER:--disable-new-dtags")
  target_link_libraries(${EXAMPLE_NAME} PRIVATE controllers_local flexiv::flexiv_rdk)
  set_target_properties(${EXAMPLE_NAME} PROPERTIES
    BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    INSTALL_RPATH "\$ORIGIN/../lib")
endforeach()
install(TARGETS ${FLEXIV_DUAL_EXAMPLES} RUNTIME DESTINATION bin)
