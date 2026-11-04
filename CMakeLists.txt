# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)

set(NV_ORG "nvidia")
set(NV_GRP "srl")

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# DO NOT EDIT THIS SECTION
# ==============================================================================



# Manually set project variables
# ------------------------------
# (Note: is is assumed the source code will live at
# `src/${NV_ORG}/${NV_GRP}/${NV_PKG}`
set(NV_PKG "controllers")

option(ENABLE_PYBIND "Build pybindings for public interfaces" ON)
option(ENABLE_TENSOR_JOINT_IMPEDANCE_CUDA_BENCHMARK
       "Build the tensor joint-impedance CUDA benchmark tool" OFF)
set(LIBFRANKA_VERSION "0.21.3" CACHE STRING
    "libfranka source version built locally for the Franka examples")
set(FLEXIV_RDK_VERSION "1.9.1" CACHE STRING
    "Exact bundled Flexiv RDK version used to build the Flexiv/Rizon examples")
set_property(CACHE FLEXIV_RDK_VERSION PROPERTY STRINGS 1.9.1 1.9.2 1.9.3 1.9.4)



# ==============================================================================
# DO NOT EDIT THIS SECTION
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# Create project
project("${NV_ORG}_${NV_GRP}_${NV_PKG}")

# Include CMake modules
include(${PROJECT_SOURCE_DIR}/cmake/NvCMakeModules.cmake)
# Set Cmake command line options
nv_set_cmake_options()

if((ENABLE_FRANKA_EXAMPLES OR ENABLE_FLEXIV_DUAL_ARM_RDK_EXAMPLE OR
    ENABLE_RIZON_EXAMPLES OR
    ENABLE_UR_EXAMPLES) AND
   NOT ENABLE_EXAMPLES)
  message(FATAL_ERROR
    "Vendor examples require ENABLE_EXAMPLES=ON in addition to the selected vendor option.")
endif()

if(ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    message(FATAL_ERROR
      "ENABLE_CUDA=ON requires -DCMAKE_CUDA_ARCHITECTURES=<SM>-real, for example 89-real. "
      "Controllers ships native SASS only and does not rely on PTX JIT for deployment.")
  endif()
  list(LENGTH CMAKE_CUDA_ARCHITECTURES NVIDIASRLCONTROLLERS_CUDA_ARCHITECTURE_COUNT)
  if(NOT NVIDIASRLCONTROLLERS_CUDA_ARCHITECTURE_COUNT EQUAL 1 OR
     NOT "${CMAKE_CUDA_ARCHITECTURES}" MATCHES "^[0-9]+-real$")
    message(FATAL_ERROR
      "CMAKE_CUDA_ARCHITECTURES must contain exactly one native architecture in <SM>-real "
      "form (for example 89-real); received '${CMAKE_CUDA_ARCHITECTURES}'.")
  endif()
  string(REGEX REPLACE "-real$" "" NVIDIASRLCONTROLLERS_CUDA_BUILD_SM
    "${CMAKE_CUDA_ARCHITECTURES}")
  set(NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia/srl/controllers/tensor/joint_impedance_cuda_tuning_profiles.inc"
      CACHE FILEPATH
      "CUDA joint-impedance tuning profile table generated on a deployment GPU")
  if(NOT EXISTS "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE}")
    message(FATAL_ERROR
      "NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE does not exist: "
      "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE}")
  endif()
  set(NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia/srl/controllers/tensor/cuda_component_tuning_profiles.inc"
      CACHE FILEPATH
      "CUDA tensor-component launch profile table generated on a deployment GPU")
  if(NOT EXISTS "${NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE}")
    message(FATAL_ERROR
      "NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE does not exist: "
      "${NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE}")
  endif()
  set(NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE ON CACHE BOOL
      "Require committed CUDA launch profiles for the selected SM")
  if(NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE)
    file(STRINGS "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE}"
      NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FOR_BUILD
      REGEX "^NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE\\(${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM},[ ]*kFloat(32|64),")
    if(NOT NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FOR_BUILD)
      message(FATAL_ERROR
        "No committed CUDA joint-impedance tuning profile exists for SM "
        "${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM}. Run the qualification tool on that GPU "
        "and commit its generated dispatch table. Candidate-only builds may set "
        "NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF.")
    endif()
    file(STRINGS "${NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE}"
      NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FOR_BUILD
      REGEX "^NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE(_SCALARS)?\\(${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM},")
    if(NOT NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FOR_BUILD)
      message(FATAL_ERROR
        "No committed CUDA tensor-component tuning profile exists for SM "
        "${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM}. Run qualification on that GPU and "
        "commit its generated dispatch tables. Candidate-only builds may set "
        "NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF.")
    endif()
    set(NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE_VALUE 1)
  else()
    set(NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE_VALUE 0)
  endif()
  set(NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_INCLUDE_ROOT
      "${CMAKE_BINARY_DIR}/generated/nvidia/srl/controllers")
  file(MAKE_DIRECTORY "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_INCLUDE_ROOT}/tensor")
  configure_file("${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE}"
                 "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_INCLUDE_ROOT}/tensor/joint_impedance_cuda_tuning_profiles.inc"
                 COPYONLY)
  configure_file("${NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE}"
                 "${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_INCLUDE_ROOT}/tensor/cuda_component_tuning_profiles.inc"
                 COPYONLY)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# DO NOT EDIT THIS SECTION
# ==============================================================================

# set(ENABLE_EXAMPLES OFF)

set(NV_APT_PKG_LIB_DEPENDS "")


# Find Packages and set apt dependencies
# --------------------------------------
# Update list of apt package dependencies necessary for the library
# (e.g. "libeigen3-dev (>= 3.2.0), libeigen3-dev (<< 4.0)")
if(ENABLE_FRANKA_EXAMPLES OR ENABLE_RIZON_EXAMPLES OR ENABLE_UR_EXAMPLES)
  nv_find_package(Eigen3 3.3 REQUIRED CONFIG)
endif()

# Set library folders
# --------------------------------------
option(NVIDIASRLCONTROLLERS_PYTHON_WHEEL "Build native artifacts for a relocatable wheel" OFF)
option(NVIDIASRLCONTROLLERS_PYTHON_SOURCE_SYMLINK
  "Link the development extension into the source Python package" ON)

set(NV_LIBS controllers)
set(NV_PUBLIC_HEADER_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nvidia/srl/controllers/composition.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nvidia/srl/controllers/equations.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nvidia/srl/controllers/controllers.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nvidia/srl/controllers/example_specs.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/nvidia/srl/controllers/types.hpp"
)


# ==============================================================================
# DO NOT EDIT THIS SECTION
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
nv_configure_project()

if(ENABLE_TESTS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(
      NAME nvidia_srl_controllers_install_prefix
      COMMAND ${CMAKE_COMMAND}
        "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-prefix"
        "-DTEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
        "-DTEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/install_prefix_test.cmake"
    )
  endif()
  add_test(
    NAME nvidia_srl_controllers_example_contract
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/example_contract_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_legacy_surface_contract
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/legacy_surface_contract_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_exported_surface_contract
    COMMAND ${CMAKE_COMMAND}
      "-DLIBRARY_FILE=$<TARGET_FILE:nvidia_srl_controllers>"
      "-DNM_TOOL=${CMAKE_NM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/exported_surface_contract_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_franka_version_selection
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/franka-version-selection"
      "-DTEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
      "-DTEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/franka_version_selection_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_rizon_option_off
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/rizon-option-off"
      "-DTEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
      "-DTEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/rizon_option_off_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_flexiv_dual_option_off
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/flexiv-dual-option-off"
      "-DTEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
      "-DTEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/flexiv_dual_option_off_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_rizon_version_selection
    COMMAND ${CMAKE_COMMAND}
      "-DCONTROLLERS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/rizon-version-selection"
      "-DTEST_CMAKE_GENERATOR=${CMAKE_GENERATOR}"
      "-DTEST_CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/rizon_version_selection_test.cmake"
  )
  add_test(
    NAME nvidia_srl_controllers_package_consumer
    COMMAND ${CMAKE_COMMAND}
      "-DPRODUCER_BINARY_DIR=${CMAKE_BINARY_DIR}"
      "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/package-consumer"
      "-DCONSUMER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer-build"
      "-DSTAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer-stage"
      "-DEXPECTED_VERSION=${nvidia_srl_controllers_VERSION}"
      "-DCHECK_VERSION_TOOL=${ENABLE_TOOLS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/package_consumer_test.cmake"
  )
  if(ENABLE_CUDA)
    add_test(
      NAME nvidia_srl_controllers_cuda_tensor_execution_ownership
      COMMAND ${CMAKE_COMMAND}
        "-DTENSOR_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia/srl/controllers/tensor"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/cuda_tensor_execution_ownership_test.cmake"
    )
    add_test(
      NAME nvidia_srl_controllers_cuda_package_consumer
      COMMAND ${CMAKE_COMMAND}
        "-DPRODUCER_BINARY_DIR=${CMAKE_BINARY_DIR}"
        "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/package-consumer"
        "-DCONSUMER_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cuda-package-consumer-build"
        "-DSTAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/cuda-package-consumer-stage"
        "-DEXPECTED_VERSION=${nvidia_srl_controllers_VERSION}"
        "-DCHECK_VERSION_TOOL=${ENABLE_TOOLS}"
        "-DENABLE_CUDA_CONSUMER=ON"
        "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
        "-DEXPECTED_CUDA_SM=${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/package_consumer_test.cmake"
    )
  endif()
endif()
