# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.27)

foreach(REQUIRED_VARIABLE CONTROLLERS_SOURCE_DIR TEST_BINARY_DIR)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} must be set for the install-prefix test")
  endif()
endforeach()
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The install-prefix regression requires Linux")
endif()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef RUN_ID)
set(TEST_ROOT "${TEST_BINARY_DIR}-${RUN_ID}")
if(EXISTS "${TEST_ROOT}" OR IS_SYMLINK "${TEST_ROOT}")
  message(FATAL_ERROR "Refusing to reuse test tree ${TEST_ROOT}")
endif()
file(MAKE_DIRECTORY "${TEST_ROOT}")

function(run_or_fail DESCRIPTION)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE RESULT OUTPUT_VARIABLE OUTPUT ERROR_VARIABLE ERROR)
  if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "${DESCRIPTION} failed (${RESULT}).\n${OUTPUT}\n${ERROR}")
  endif()
  set(LAST_COMMAND_OUTPUT "${OUTPUT}${ERROR}" PARENT_SCOPE)
endfunction()

function(configure_producer BUILD_DIR)
  set(GENERATOR_ARGUMENTS)
  if(TEST_CMAKE_GENERATOR)
    list(APPEND GENERATOR_ARGUMENTS -G "${TEST_CMAKE_GENERATOR}")
  endif()
  if(TEST_CMAKE_MAKE_PROGRAM)
    list(APPEND GENERATOR_ARGUMENTS "-DCMAKE_MAKE_PROGRAM=${TEST_CMAKE_MAKE_PROGRAM}")
  endif()
  run_or_fail("Configuring fresh producer ${BUILD_DIR}"
    "${CMAKE_COMMAND}" -E env --unset=CMAKE_INSTALL_PREFIX --unset=DESTDIR
    "${CMAKE_COMMAND}" -S "${CONTROLLERS_SOURCE_DIR}" -B "${BUILD_DIR}"
    ${GENERATOR_ARGUMENTS}
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_PYBIND=OFF
    -DENABLE_TESTS=OFF -DENABLE_COVERAGE=OFF -DENABLE_DOCS=OFF -DENABLE_EXAMPLES=OFF
    -DENABLE_TOOLS=ON ${ARGN})
endfunction()

function(assert_prefix BUILD_DIR EXPECTED_PREFIX)
  file(STRINGS "${BUILD_DIR}/CMakeCache.txt" CACHE_PREFIX REGEX "^CMAKE_INSTALL_PREFIX:[^=]*=")
  string(REGEX REPLACE "^[^=]*=" "" CACHE_PREFIX "${CACHE_PREFIX}")
  if(NOT CACHE_PREFIX STREQUAL EXPECTED_PREFIX)
    message(FATAL_ERROR "Cached prefix '${CACHE_PREFIX}', expected '${EXPECTED_PREFIX}'")
  endif()
  file(READ "${BUILD_DIR}/cmake_install.cmake" INSTALL_SCRIPT)
  string(FIND "${INSTALL_SCRIPT}" "set(CMAKE_INSTALL_PREFIX \"${EXPECTED_PREFIX}\")" PREFIX_INDEX)
  if(PREFIX_INDEX EQUAL -1)
    message(FATAL_ERROR "Generated install prefix does not honor '${EXPECTED_PREFIX}' in ${BUILD_DIR}")
  endif()
endfunction()

function(build_producer BUILD_DIR)
  run_or_fail("Building producer ${BUILD_DIR}" "${CMAKE_COMMAND}" --build "${BUILD_DIR}"
    --parallel 2 --target nvidia_srl_controllers nvidia_srl_controllers_version)
endfunction()

function(check_package BUILD_DIR INSTALL_DIR)
  file(STRINGS "${BUILD_DIR}/include/nvidia/srl/controllers/version.hpp" VERSION_LINE
    REGEX "^#define NVIDIA_SRL_CONTROLLERS_VERSION ")
  string(REGEX REPLACE "^[^\"]*\"([^\"]*)\".*" "\\1" EXPECTED_VERSION "${VERSION_LINE}")
  run_or_fail("Checking installed and relocated consumer in ${INSTALL_DIR}"
    "${CMAKE_COMMAND}" "-DPRODUCER_BINARY_DIR=${BUILD_DIR}"
    "-DCONSUMER_SOURCE_DIR=${CONTROLLERS_SOURCE_DIR}/tests/cmake/package-consumer"
    "-DCONSUMER_BINARY_DIR=${TEST_ROOT}/consumer" "-DSTAGE_DIR=${INSTALL_DIR}"
    "-DEXPECTED_VERSION=${EXPECTED_VERSION}" -DCHECK_VERSION_TOOL=ON -DPACKAGE_ALREADY_INSTALLED=ON
    -P "${CONTROLLERS_SOURCE_DIR}/tests/cmake/package_consumer_test.cmake")
endfunction()

function(assert_no_devel_link BUILD_DIR)
  if(EXISTS "${BUILD_DIR}/devel/controllers" OR IS_SYMLINK "${BUILD_DIR}/devel/controllers")
    message(FATAL_ERROR "Installation escaped staging via ${BUILD_DIR}/devel/controllers")
  endif()
endfunction()

set(EXPLICIT_BUILD "${TEST_ROOT}/explicit-build")
set(EXPLICIT_PREFIX "${TEST_ROOT}/explicit-install")
configure_producer("${EXPLICIT_BUILD}" -DDEVEL=ON "-DCMAKE_INSTALL_PREFIX=${EXPLICIT_PREFIX}"
  -DCMAKE_INSTALL_LIBDIR=lib64/controllers -DCMAKE_INSTALL_BINDIR=tools/bin
  "-DCMAKE_INSTALL_RPATH=$ORIGIN/vendor-runtime")
assert_prefix("${EXPLICIT_BUILD}" "${EXPLICIT_PREFIX}")
configure_producer("${EXPLICIT_BUILD}")
assert_prefix("${EXPLICIT_BUILD}" "${EXPLICIT_PREFIX}")
build_producer("${EXPLICIT_BUILD}")
run_or_fail("Installing with the configured explicit prefix"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${EXPLICIT_BUILD}")
assert_no_devel_link("${EXPLICIT_BUILD}")
find_program(READELF_TOOL readelf REQUIRED)
foreach(INSTALLED_ELF IN ITEMS
    "${EXPLICIT_PREFIX}/lib64/controllers/libnvidia_srl_controllers.so"
    "${EXPLICIT_PREFIX}/tools/bin/controllers_version")
  run_or_fail("Inspecting the explicit install RPATH" "${READELF_TOOL}" -d "${INSTALLED_ELF}")
  string(FIND "${LAST_COMMAND_OUTPUT}" "$ORIGIN/vendor-runtime" CUSTOM_RPATH_INDEX)
  if(CUSTOM_RPATH_INDEX EQUAL -1)
    message(FATAL_ERROR "The target-specific RPATH discarded an explicit custom path: ${INSTALLED_ELF}")
  endif()
endforeach()
string(FIND "${LAST_COMMAND_OUTPUT}" "$ORIGIN/../../lib64/controllers" TOOL_RPATH_INDEX)
if(TOOL_RPATH_INDEX EQUAL -1)
  message(FATAL_ERROR "The installed tool RPATH does not account for custom binary/library directories")
endif()
check_package("${EXPLICIT_BUILD}" "${EXPLICIT_PREFIX}")

set(OVERRIDE_PREFIX "${TEST_ROOT}/runtime-override")
run_or_fail("Installing with --prefix"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${EXPLICIT_BUILD}"
  --prefix "${OVERRIDE_PREFIX}")
assert_no_devel_link("${EXPLICIT_BUILD}")
check_package("${EXPLICIT_BUILD}" "${OVERRIDE_PREFIX}")

set(DESTDIR_ROOT "${TEST_ROOT}/explicit-destdir")
run_or_fail("Staging an explicit prefix with DESTDIR"
  "${CMAKE_COMMAND}" -E env "DESTDIR=${DESTDIR_ROOT}"
  "${CMAKE_COMMAND}" --install "${EXPLICIT_BUILD}")
if(EXISTS "${EXPLICIT_PREFIX}")
  message(FATAL_ERROR "DESTDIR installation recreated the unstaged explicit prefix")
endif()
assert_no_devel_link("${EXPLICIT_BUILD}")
check_package("${EXPLICIT_BUILD}" "${DESTDIR_ROOT}${EXPLICIT_PREFIX}")

set(DEFAULT_BUILD "${TEST_ROOT}/default-build")
configure_producer("${DEFAULT_BUILD}" -DDEVEL=ON)
file(STRINGS "${DEFAULT_BUILD}/nvidia-srl-controllers-config-version.cmake" VERSION_LINE
  REGEX "^set\\(PACKAGE_VERSION ")
string(REGEX REPLACE "^[^\"]*\"([^\"]*)\".*" "\\1" FULL_VERSION "${VERSION_LINE}")
set(DEFAULT_PREFIX "${DEFAULT_BUILD}/devel/controllers-${FULL_VERSION}")
assert_prefix("${DEFAULT_BUILD}" "${DEFAULT_PREFIX}")
configure_producer("${DEFAULT_BUILD}")
assert_prefix("${DEFAULT_BUILD}" "${DEFAULT_PREFIX}")
configure_producer("${DEFAULT_BUILD}" -DDEVEL=OFF)
assert_prefix("${DEFAULT_BUILD}" "/opt/nvidia/srl/controllers-${FULL_VERSION}")
configure_producer("${DEFAULT_BUILD}" -DDEVEL=ON)
assert_prefix("${DEFAULT_BUILD}" "${DEFAULT_PREFIX}")
build_producer("${DEFAULT_BUILD}")

find_program(CPACK_TOOL cpack REQUIRED)
set(PACKAGE_DIR "${DEFAULT_BUILD}/packages")
run_or_fail("Creating a fresh CPack stage"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CPACK_TOOL}"
  --config "${DEFAULT_BUILD}/CPackConfig.cmake" -G TGZ -B "${PACKAGE_DIR}")
assert_no_devel_link("${DEFAULT_BUILD}")
if(EXISTS "${DEFAULT_PREFIX}")
  message(FATAL_ERROR "CPack wrote to the configured prefix outside its stage")
endif()
file(GLOB PACKAGES "${PACKAGE_DIR}/*.tar.gz")
list(LENGTH PACKAGES PACKAGE_COUNT)
if(NOT PACKAGE_COUNT EQUAL 1)
  message(FATAL_ERROR "Expected one CPack archive in ${PACKAGE_DIR}, got '${PACKAGES}': ${LAST_COMMAND_OUTPUT}")
endif()
list(GET PACKAGES 0 PACKAGE_ARCHIVE)
set(EXTRACT_DIR "${TEST_ROOT}/archive")
file(ARCHIVE_EXTRACT INPUT "${PACKAGE_ARCHIVE}" DESTINATION "${EXTRACT_DIR}")
file(GLOB_RECURSE ARCHIVE_FILES LIST_DIRECTORIES TRUE "${EXTRACT_DIR}/*")
foreach(ARCHIVE_FILE IN LISTS ARCHIVE_FILES)
  if(IS_SYMLINK "${ARCHIVE_FILE}")
    file(REAL_PATH "${ARCHIVE_FILE}" LINK_DESTINATION)
    cmake_path(IS_PREFIX EXTRACT_DIR "${LINK_DESTINATION}" NORMALIZE LINK_IN_STAGE)
    if(NOT LINK_IN_STAGE)
      message(FATAL_ERROR "CPack archive link escapes its stage: ${ARCHIVE_FILE}")
    endif()
  endif()
endforeach()
file(GLOB_RECURSE ARCHIVE_LIBRARIES "${EXTRACT_DIR}/*libnvidia_srl_controllers.so")
if(NOT ARCHIVE_LIBRARIES)
  message(FATAL_ERROR "CPack archive contains no Controllers library")
endif()

set(DEFAULT_OVERRIDE "${TEST_ROOT}/default-override")
run_or_fail("Overriding a project-managed default at install time"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}"
  --prefix "${DEFAULT_OVERRIDE}")
assert_no_devel_link("${DEFAULT_BUILD}")
check_package("${DEFAULT_BUILD}" "${DEFAULT_OVERRIDE}")

set(DEFAULT_DESTDIR "${TEST_ROOT}/default-destdir")
run_or_fail("Staging a project-managed default with DESTDIR"
  "${CMAKE_COMMAND}" -E env "DESTDIR=${DEFAULT_DESTDIR}"
  "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}")
assert_no_devel_link("${DEFAULT_BUILD}")
if(EXISTS "${DEFAULT_PREFIX}")
  message(FATAL_ERROR "DESTDIR installation wrote outside its stage")
endif()
check_package("${DEFAULT_BUILD}" "${DEFAULT_DESTDIR}${DEFAULT_PREFIX}")

configure_producer("${DEFAULT_BUILD}" -DDEVEL=OFF)
set(PRODUCTION_PREFIX "/opt/nvidia/srl/controllers-${FULL_VERSION}")
assert_prefix("${DEFAULT_BUILD}" "${PRODUCTION_PREFIX}")
run_or_fail("Staging the production default with DESTDIR"
  "${CMAKE_COMMAND}" -E env "DESTDIR=${TEST_ROOT}/production-destdir"
  "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}")
assert_no_devel_link("${DEFAULT_BUILD}")
check_package("${DEFAULT_BUILD}" "${TEST_ROOT}/production-destdir${PRODUCTION_PREFIX}")
configure_producer("${DEFAULT_BUILD}" -DDEVEL=ON)
assert_prefix("${DEFAULT_BUILD}" "${DEFAULT_PREFIX}")

file(MAKE_DIRECTORY "${DEFAULT_BUILD}/devel")
file(WRITE "${DEFAULT_BUILD}/devel/controllers" "preserve unrelated file\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}"
  RESULT_VARIABLE COLLISION_RESULT OUTPUT_VARIABLE COLLISION_OUTPUT ERROR_VARIABLE COLLISION_ERROR)
if(COLLISION_RESULT EQUAL 0 OR NOT COLLISION_ERROR MATCHES "Refusing to replace")
  message(FATAL_ERROR "The development-link collision was not reported: ${COLLISION_OUTPUT}${COLLISION_ERROR}")
endif()
file(READ "${DEFAULT_BUILD}/devel/controllers" PRESERVED_FILE)
if(NOT PRESERVED_FILE STREQUAL "preserve unrelated file\n")
  message(FATAL_ERROR "The development link replaced an unrelated file")
endif()
file(RENAME "${DEFAULT_BUILD}/devel/controllers" "${DEFAULT_BUILD}/devel/preserved-file")
run_or_fail("Installing to the development default"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}")
run_or_fail("Updating the development convenience link"
  "${CMAKE_COMMAND}" -E env --unset=DESTDIR "${CMAKE_COMMAND}" --install "${DEFAULT_BUILD}")
if(NOT IS_SYMLINK "${DEFAULT_BUILD}/devel/controllers")
  message(FATAL_ERROR "The managed development install did not create its convenience link")
endif()
file(READ_SYMLINK "${DEFAULT_BUILD}/devel/controllers" DEFAULT_LINK)
if(NOT DEFAULT_LINK STREQUAL "controllers-${FULL_VERSION}" AND
    NOT DEFAULT_LINK STREQUAL "./controllers-${FULL_VERSION}")
  message(FATAL_ERROR "Unexpected development convenience link: ${DEFAULT_LINK}")
endif()
check_package("${DEFAULT_BUILD}" "${DEFAULT_PREFIX}")

set(RECONFIGURED_PREFIX "${TEST_ROOT}/reconfigured-install")
configure_producer("${DEFAULT_BUILD}" "-DCMAKE_INSTALL_PREFIX=${RECONFIGURED_PREFIX}")
assert_prefix("${DEFAULT_BUILD}" "${RECONFIGURED_PREFIX}")
configure_producer("${DEFAULT_BUILD}" -DDEVEL=OFF)
assert_prefix("${DEFAULT_BUILD}" "${RECONFIGURED_PREFIX}")
message(STATUS "Verified explicit/default prefixes, reconfiguration, DESTDIR, --prefix, custom directories, CPack and relocation in ${TEST_ROOT}")
