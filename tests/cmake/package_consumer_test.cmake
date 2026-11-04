# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.18)

foreach(REQUIRED_VARIABLE
    PRODUCER_BINARY_DIR
    CONSUMER_SOURCE_DIR
    CONSUMER_BINARY_DIR
    STAGE_DIR
    EXPECTED_VERSION
    CHECK_VERSION_TOOL)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} must be set for the package-consumer test")
  endif()
endforeach()
if(ENABLE_CUDA_CONSUMER)
  foreach(REQUIRED_CUDA_VARIABLE CMAKE_CUDA_ARCHITECTURES EXPECTED_CUDA_SM)
    if(NOT DEFINED ${REQUIRED_CUDA_VARIABLE} OR "${${REQUIRED_CUDA_VARIABLE}}" STREQUAL "")
      message(FATAL_ERROR "${REQUIRED_CUDA_VARIABLE} must be set for the CUDA package-consumer test")
    endif()
  endforeach()
endif()

function(run_or_fail DESCRIPTION)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE OUTPUT
    ERROR_VARIABLE ERROR
  )
  if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR
      "${DESCRIPTION} failed with exit code ${RESULT}.\n"
      "stdout:\n${OUTPUT}\n"
      "stderr:\n${ERROR}"
    )
  endif()
endfunction()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef RUN_ID)
set(CONSUMER_BINARY_DIR "${CONSUMER_BINARY_DIR}-${RUN_ID}")
if(EXISTS "${CONSUMER_BINARY_DIR}" OR IS_SYMLINK "${CONSUMER_BINARY_DIR}")
  message(FATAL_ERROR "Refusing to reuse consumer build ${CONSUMER_BINARY_DIR}")
endif()
if(NOT PACKAGE_ALREADY_INSTALLED)
  set(STAGE_DIR "${STAGE_DIR}-${RUN_ID}")
  if(EXISTS "${STAGE_DIR}" OR IS_SYMLINK "${STAGE_DIR}")
    message(FATAL_ERROR "Refusing to reuse package stage ${STAGE_DIR}")
  endif()
  run_or_fail(
    "Installing the Controllers package"
    "${CMAKE_COMMAND}" -E env --unset=DESTDIR
      "${CMAKE_COMMAND}" --install "${PRODUCER_BINARY_DIR}" --prefix "${STAGE_DIR}"
  )
endif()
foreach(INSTALL_DIRECTORY LIBDIR BINDIR)
  file(STRINGS "${PRODUCER_BINARY_DIR}/CMakeCache.txt" DIRECTORY_LINE
    REGEX "^CMAKE_INSTALL_${INSTALL_DIRECTORY}:[^=]*=")
  if(DIRECTORY_LINE STREQUAL "")
    message(FATAL_ERROR "The producer cache is missing CMAKE_INSTALL_${INSTALL_DIRECTORY}")
  endif()
  string(REGEX REPLACE "^[^=]*=" "" CMAKE_INSTALL_${INSTALL_DIRECTORY} "${DIRECTORY_LINE}")
endforeach()

set(RELOCATED_STAGE_DIR "${STAGE_DIR}-relocated")
file(REMOVE_RECURSE "${RELOCATED_STAGE_DIR}")
file(RENAME "${STAGE_DIR}" "${RELOCATED_STAGE_DIR}")
set(STAGE_DIR "${RELOCATED_STAGE_DIR}")

set(REQUIRED_STAGE_FILES
  "${STAGE_DIR}/include/nvidia/srl/controllers/controllers.hpp"
  "${STAGE_DIR}/include/nvidia/srl/controllers/version.hpp"
  "${STAGE_DIR}/cmake/nvidia-srl-controllers-config.cmake"
  "${STAGE_DIR}/cmake/nvidia-srl-controllers-config-version.cmake"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/LICENSE"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/CONTRIBUTING.md"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/THIRD_PARTY_NOTICES.md"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/LICENSES/Apache-2.0.txt"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/examples/ur/ur_description_LICENSE"
)
if(CHECK_VERSION_TOOL)
  list(APPEND REQUIRED_STAGE_FILES
    "${STAGE_DIR}/${CMAKE_INSTALL_BINDIR}/controllers_version${CMAKE_EXECUTABLE_SUFFIX}"
  )
endif()
foreach(REQUIRED_STAGE_FILE IN LISTS REQUIRED_STAGE_FILES)
  if(NOT EXISTS "${REQUIRED_STAGE_FILE}")
    message(FATAL_ERROR "The staged package is missing ${REQUIRED_STAGE_FILE}")
  endif()
endforeach()

file(SHA256 "${STAGE_DIR}/share/doc/nvidia-srl-controllers/LICENSE" PRIMARY_LICENSE_SHA256)
if(NOT PRIMARY_LICENSE_SHA256 STREQUAL
    "0799fbb323e03399571dcc8279d7cd7433f2374e33502b2c723961635ea9fea2")
  message(FATAL_ERROR "The installed primary license differs from NVIDIA License dated 22 March 2022")
endif()

file(GLOB_RECURSE STAGED_CONTROLLER_HEADERS RELATIVE
  "${STAGE_DIR}/include/nvidia/srl/controllers"
  "${STAGE_DIR}/include/nvidia/srl/controllers/*.hpp"
)
set(EXPECTED_CONTROLLER_HEADERS
  controllers.hpp
  equations.hpp
  composition.hpp
  example_specs.hpp
  types.hpp
  version.hpp
)
list(SORT STAGED_CONTROLLER_HEADERS)
list(SORT EXPECTED_CONTROLLER_HEADERS)
if(NOT STAGED_CONTROLLER_HEADERS STREQUAL EXPECTED_CONTROLLER_HEADERS)
  message(FATAL_ERROR
    "The staged public headers are '${STAGED_CONTROLLER_HEADERS}', expected exactly "
    "'${EXPECTED_CONTROLLER_HEADERS}'."
  )
endif()

file(GLOB STAGED_LIBRARIES "${STAGE_DIR}/${CMAKE_INSTALL_LIBDIR}/*nvidia_srl_controllers*")
if(STAGED_LIBRARIES STREQUAL "")
  message(FATAL_ERROR "The staged package does not contain the Controllers shared library")
endif()

if(WIN32)
  set(RUNTIME_ENVIRONMENT "PATH=${STAGE_DIR}/${CMAKE_INSTALL_BINDIR};$ENV{PATH}")
  string(REPLACE ";" "\\;" RUNTIME_ENVIRONMENT "${RUNTIME_ENVIRONMENT}")
elseif(APPLE)
  set(RUNTIME_ENVIRONMENT "DYLD_LIBRARY_PATH=${STAGE_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{DYLD_LIBRARY_PATH}")
else()
  set(RUNTIME_ENVIRONMENT --unset=LD_LIBRARY_PATH --unset=LD_PRELOAD --unset=LD_AUDIT)
endif()
if(CHECK_VERSION_TOOL)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${RUNTIME_ENVIRONMENT}
      "${STAGE_DIR}/${CMAKE_INSTALL_BINDIR}/controllers_version${CMAKE_EXECUTABLE_SUFFIX}"
    RESULT_VARIABLE VERSION_TOOL_RESULT
    OUTPUT_VARIABLE VERSION_TOOL_OUTPUT
    ERROR_VARIABLE VERSION_TOOL_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT VERSION_TOOL_RESULT EQUAL 0)
    message(FATAL_ERROR
      "The staged controllers_version tool failed with exit code ${VERSION_TOOL_RESULT}.\n"
      "stdout:\n${VERSION_TOOL_OUTPUT}\n"
      "stderr:\n${VERSION_TOOL_ERROR}"
    )
  endif()
  if(NOT "${VERSION_TOOL_OUTPUT}" STREQUAL "${EXPECTED_VERSION}")
    message(FATAL_ERROR
      "The staged controllers_version output '${VERSION_TOOL_OUTPUT}' does not match "
      "the configured version '${EXPECTED_VERSION}'."
    )
  endif()
endif()

set(CONSUMER_CONFIGURE_ARGUMENTS
  -S "${CONSUMER_SOURCE_DIR}"
  -B "${CONSUMER_BINARY_DIR}"
  "-Dnvidia-srl-controllers_DIR=${STAGE_DIR}/cmake"
  "-DCMAKE_INSTALL_PREFIX=${STAGE_DIR}"
  "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
  "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE
  -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=TRUE
)
if(ENABLE_CUDA_CONSUMER)
  list(APPEND CONSUMER_CONFIGURE_ARGUMENTS
    -DENABLE_CUDA_CONSUMER=ON
    "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    "-DEXPECTED_CUDA_SM=${EXPECTED_CUDA_SM}"
  )
  if(DEFINED CMAKE_CUDA_COMPILER AND NOT "${CMAKE_CUDA_COMPILER}" STREQUAL "")
    list(APPEND CONSUMER_CONFIGURE_ARGUMENTS
      "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
    )
  endif()
  if(DEFINED CUDAToolkit_ROOT AND NOT "${CUDAToolkit_ROOT}" STREQUAL "")
    list(APPEND CONSUMER_CONFIGURE_ARGUMENTS "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  endif()
endif()
run_or_fail(
  "Configuring the external package consumer"
  "${CMAKE_COMMAND}"
  ${CONSUMER_CONFIGURE_ARGUMENTS}
)

file(STRINGS "${CONSUMER_BINARY_DIR}/CMakeCache.txt" PACKAGE_DIR_LINE
  REGEX "^nvidia-srl-controllers_DIR:[^=]*="
)
if(PACKAGE_DIR_LINE STREQUAL "")
  message(FATAL_ERROR "The consumer cache does not record nvidia-srl-controllers_DIR")
endif()
string(REGEX REPLACE "^[^=]*=" "" RESOLVED_PACKAGE_DIR "${PACKAGE_DIR_LINE}")
file(REAL_PATH "${STAGE_DIR}/cmake" EXPECTED_PACKAGE_DIR)
file(REAL_PATH "${RESOLVED_PACKAGE_DIR}" RESOLVED_PACKAGE_DIR)
if(NOT RESOLVED_PACKAGE_DIR STREQUAL EXPECTED_PACKAGE_DIR)
  message(FATAL_ERROR
    "The consumer resolved Controllers from '${RESOLVED_PACKAGE_DIR}', expected "
    "'${EXPECTED_PACKAGE_DIR}'."
  )
endif()

run_or_fail(
  "Building the external package consumer"
  "${CMAKE_COMMAND}" --build "${CONSUMER_BINARY_DIR}"
)

run_or_fail(
  "Installing the external package consumer"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR
    "${CMAKE_COMMAND}" --install "${CONSUMER_BINARY_DIR}"
)

set(ORIGINAL_STAGE_DIR "${STAGE_DIR}")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(STAGE_DIR "${ORIGINAL_STAGE_DIR}-relocated")
  if(EXISTS "${STAGE_DIR}" OR IS_SYMLINK "${STAGE_DIR}")
    message(FATAL_ERROR "Refusing to replace relocation destination ${STAGE_DIR}")
  endif()
  file(RENAME "${ORIGINAL_STAGE_DIR}" "${STAGE_DIR}")
  if(EXISTS "${ORIGINAL_STAGE_DIR}")
    message(FATAL_ERROR "The original installation remains available after relocation")
  endif()
endif()
set(CONSUMER_EXECUTABLE
  "${STAGE_DIR}/${CMAKE_INSTALL_BINDIR}/controllers_package_consumer${CMAKE_EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${CONSUMER_EXECUTABLE}")
  message(FATAL_ERROR "The external consumer executable was not installed: ${CONSUMER_EXECUTABLE}")
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  find_program(READELF_TOOL readelf REQUIRED)
  find_program(LDD_TOOL ldd REQUIRED)
  file(REAL_PATH "${STAGE_DIR}/${CMAKE_INSTALL_LIBDIR}/libnvidia_srl_controllers.so"
    EXPECTED_LIBRARY)
  file(STRINGS "${PRODUCER_BINARY_DIR}/CMakeCache.txt" SOURCE_DIRECTORY_LINE
    REGEX "^CMAKE_HOME_DIRECTORY:[^=]*=")
  string(REGEX REPLACE "^[^=]*=" "" PRODUCER_SOURCE_DIR "${SOURCE_DIRECTORY_LINE}")
  set(INSTALLED_ELF_FILES "${EXPECTED_LIBRARY}" "${CONSUMER_EXECUTABLE}")
  set(INSTALLED_EXECUTABLES "${CONSUMER_EXECUTABLE}")
  if(CHECK_VERSION_TOOL)
    list(APPEND INSTALLED_ELF_FILES "${STAGE_DIR}/${CMAKE_INSTALL_BINDIR}/controllers_version")
  endif()
  foreach(INSTALLED_ELF IN LISTS INSTALLED_ELF_FILES)
    execute_process(COMMAND "${READELF_TOOL}" -d "${INSTALLED_ELF}"
      RESULT_VARIABLE ELF_RESULT OUTPUT_VARIABLE ELF_DYNAMIC ERROR_VARIABLE ELF_ERROR)
    if(NOT ELF_RESULT EQUAL 0 OR NOT ELF_DYNAMIC MATCHES "(RUNPATH|RPATH).*\\$ORIGIN")
      message(FATAL_ERROR "Missing relocatable RPATH in ${INSTALLED_ELF}: ${ELF_DYNAMIC}${ELF_ERROR}")
    endif()
    foreach(FORBIDDEN_PATH IN ITEMS "${PRODUCER_BINARY_DIR}" "${PRODUCER_SOURCE_DIR}"
        "${CONSUMER_BINARY_DIR}" "${ORIGINAL_STAGE_DIR}")
      string(FIND "${ELF_DYNAMIC}" "${FORBIDDEN_PATH}" FORBIDDEN_INDEX)
      if(NOT FORBIDDEN_INDEX EQUAL -1)
        message(FATAL_ERROR "Installed ELF ${INSTALLED_ELF} contains ${FORBIDDEN_PATH}")
      endif()
    endforeach()
  endforeach()
  foreach(INSTALLED_EXECUTABLE IN LISTS INSTALLED_EXECUTABLES)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env ${RUNTIME_ENVIRONMENT}
        "${LDD_TOOL}" "${INSTALLED_EXECUTABLE}"
      RESULT_VARIABLE LDD_RESULT OUTPUT_VARIABLE LDD_OUTPUT ERROR_VARIABLE LDD_ERROR)
    if(NOT LDD_RESULT EQUAL 0 OR
        NOT LDD_OUTPUT MATCHES "libnvidia_srl_controllers[^ ]* => ([^ \n]+)")
      message(FATAL_ERROR "Cannot resolve installed Controllers library: ${LDD_OUTPUT}${LDD_ERROR}")
    endif()
    file(REAL_PATH "${CMAKE_MATCH_1}" LOADED_LIBRARY)
    if(NOT LOADED_LIBRARY STREQUAL EXPECTED_LIBRARY)
      message(FATAL_ERROR "${INSTALLED_EXECUTABLE} loaded ${LOADED_LIBRARY}, expected ${EXPECTED_LIBRARY}")
    endif()
  endforeach()
endif()
run_or_fail(
  "Running the external package consumer against the relocated library"
  "${CMAKE_COMMAND}" -E env ${RUNTIME_ENVIRONMENT} "${CONSUMER_EXECUTABLE}" "${EXPECTED_LIBRARY}"
)
if(CHECK_VERSION_TOOL)
  run_or_fail(
    "Running the relocated version tool"
    "${CMAKE_COMMAND}" -E env ${RUNTIME_ENVIRONMENT}
      "${STAGE_DIR}/${CMAKE_INSTALL_BINDIR}/controllers_version${CMAKE_EXECUTABLE_SUFFIX}"
  )
endif()
message(STATUS "Verified installed consumer and relocated library at ${STAGE_DIR}")
