# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

foreach(REQUIRED_VARIABLE
    PRODUCER_BINARY_DIR
    STAGE_DIR
    RIZON_CONSUMER_SOURCE_DIR
    RIZON_SOURCE_DIR
    DEPENDENCY_PINS_FILE
    EXPECTED_RDK_VERSION
    EXPECTED_RDK_LIBRARY
    EXPECT_BUNDLED_DEPENDENCIES
    INSTALL_BINDIR
    INSTALL_LIBDIR)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${STAGE_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${PRODUCER_BINARY_DIR}" --prefix "${STAGE_DIR}"
  RESULT_VARIABLE INSTALL_RESULT
  OUTPUT_VARIABLE INSTALL_OUTPUT
  ERROR_VARIABLE INSTALL_ERROR)
if(NOT INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "Rizon staged install failed:\n${INSTALL_OUTPUT}\n${INSTALL_ERROR}")
endif()

file(GLOB RIZON_BINARIES "${STAGE_DIR}/${INSTALL_BINDIR}/rizon_*")
list(LENGTH RIZON_BINARIES RIZON_BINARY_COUNT)
if(NOT RIZON_BINARY_COUNT EQUAL 6)
  message(FATAL_ERROR "Expected six installed Rizon executables, found ${RIZON_BINARY_COUNT}")
endif()

set(REQUIRED_RDK_FILES
  "${STAGE_DIR}/include/flexiv/rdk/robot.hpp"
  "${STAGE_DIR}/include/flexiv/rdk/model.hpp"
  "${STAGE_DIR}/${INSTALL_LIBDIR}/${EXPECTED_RDK_LIBRARY}"
  "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/flexiv_rdk/flexiv_rdk-config.cmake"
  "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/flexiv_rdk/flexiv_rdk-config-version.cmake"
  "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/flexiv_rdk/flexiv_rdk-targets.cmake"
  "${STAGE_DIR}/share/doc/nvidia-srl-controllers/third-party/flexiv-rdk/LICENSE")
if(EXPECT_BUNDLED_DEPENDENCIES)
  include("${DEPENDENCY_PINS_FILE}")
  list(APPEND REQUIRED_RDK_FILES
    "${STAGE_DIR}/${INSTALL_LIBDIR}/libfastrtps.so"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/libfastcdr.so"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/libRBDyn.so"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/libfoonathan_memory-0.7.3.so"
    "${STAGE_DIR}/share/fastrtps/cmake/fastrtps-config.cmake"
    "${STAGE_DIR}/share/foonathan_memory_vendor/cmake/foonathan_memory_vendorConfig.cmake"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/foonathan_memory/cmake/foonathan_memory-config.cmake"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/SpaceVecAlg/SpaceVecAlgConfig.cmake"
    "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/RBDyn/RBDynConfig.cmake"
    "${STAGE_DIR}/share/doc/nvidia-srl-controllers/third-party/flexiv-rdk-dependencies/Fast-DDS/LICENSE")
endif()
foreach(REQUIRED_FILE IN LISTS REQUIRED_RDK_FILES)
  if(NOT EXISTS "${REQUIRED_FILE}")
    message(FATAL_ERROR "Missing installed Flexiv RDK file: ${REQUIRED_FILE}")
  endif()
endforeach()

if(EXPECT_BUNDLED_DEPENDENCIES)
  find_program(GIT_EXECUTABLE git REQUIRED)
  set(DEPENDENCY_SOURCE_ROOT
      "${PRODUCER_BINARY_DIR}/_deps/flexiv_rdk-${EXPECTED_RDK_VERSION}/dependencies-build/cloned")

  function(assert_git_revision SOURCE_DIR EXPECTED_REVISION)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse HEAD
      RESULT_VARIABLE REVISION_RESULT
      OUTPUT_VARIABLE ACTUAL_REVISION
      ERROR_VARIABLE REVISION_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT REVISION_RESULT EQUAL 0 OR
       NOT ACTUAL_REVISION STREQUAL EXPECTED_REVISION)
      message(FATAL_ERROR
        "Dependency revision mismatch in ${SOURCE_DIR}: expected ${EXPECTED_REVISION}, "
        "found '${ACTUAL_REVISION}' (${REVISION_ERROR})")
    endif()
  endfunction()

  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/eigen" "${FLEXIV_RDK_EIGEN_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/spdlog" "${FLEXIV_RDK_SPDLOG_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/tinyxml2" "${FLEXIV_RDK_TINYXML2_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/yaml-cpp" "${FLEXIV_RDK_YAML_CPP_COMMIT}")
  assert_git_revision(
    "${DEPENDENCY_SOURCE_ROOT}/foonathan_memory_vendor"
    "${FLEXIV_RDK_FOONATHAN_VENDOR_COMMIT}")
  assert_git_revision(
    "${DEPENDENCY_SOURCE_ROOT}/foonathan_memory_vendor/build/foo_mem-ext-prefix/src/foo_mem-ext"
    "${FLEXIV_RDK_FOONATHAN_MEMORY_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/Fast-CDR" "${FLEXIV_RDK_FAST_CDR_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/Fast-DDS" "${FLEXIV_RDK_FAST_DDS_COMMIT}")
  assert_git_revision(
    "${DEPENDENCY_SOURCE_ROOT}/SpaceVecAlg" "${FLEXIV_RDK_SPACE_VEC_ALG_COMMIT}")
  assert_git_revision("${DEPENDENCY_SOURCE_ROOT}/RBDyn" "${FLEXIV_RDK_RBDYN_COMMIT}")

  foreach(SUBMODULE_REPOSITORY Fast-DDS SpaceVecAlg RBDyn)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C
        "${DEPENDENCY_SOURCE_ROOT}/${SUBMODULE_REPOSITORY}"
        submodule status --recursive
      RESULT_VARIABLE SUBMODULE_RESULT
      OUTPUT_VARIABLE SUBMODULE_OUTPUT
      ERROR_VARIABLE SUBMODULE_ERROR)
    if(NOT SUBMODULE_RESULT EQUAL 0 OR SUBMODULE_OUTPUT MATCHES "(^|\n)[-+U]")
      message(FATAL_ERROR
        "${SUBMODULE_REPOSITORY} submodules do not match the pinned gitlinks:\n"
        "${SUBMODULE_OUTPUT}\n${SUBMODULE_ERROR}")
    endif()
  endforeach()

  set(BOOST_ARCHIVE "${DEPENDENCY_SOURCE_ROOT}/boost_1_83_0.tar.bz2")
  if(NOT EXISTS "${BOOST_ARCHIVE}")
    message(FATAL_ERROR "Pinned Boost source archive is missing: ${BOOST_ARCHIVE}")
  endif()
  file(SHA256 "${BOOST_ARCHIVE}" BOOST_ARCHIVE_SHA256)
  if(NOT BOOST_ARCHIVE_SHA256 STREQUAL FLEXIV_RDK_BOOST_SHA256)
    message(FATAL_ERROR
      "Boost source hash mismatch: expected ${FLEXIV_RDK_BOOST_SHA256}, "
      "found ${BOOST_ARCHIVE_SHA256}")
  endif()

  set(FAST_DDS_CACHE "${DEPENDENCY_SOURCE_ROOT}/Fast-DDS/build/CMakeCache.txt")
  file(READ "${FAST_DDS_CACHE}" FAST_DDS_CACHE_TEXT)
  if(NOT FAST_DDS_CACHE_TEXT MATCHES "THIRDPARTY_Asio:STRING=FORCE" OR
     NOT FAST_DDS_CACHE_TEXT MATCHES
         "Asio_INCLUDE_DIR:PATH=.*/Fast-DDS/thirdparty/asio/asio/include" OR
     NOT FAST_DDS_CACHE_TEXT MATCHES
         "foonathan_memory_DIR:PATH=.*/flexiv_rdk-${EXPECTED_RDK_VERSION}/install/")
    message(FATAL_ERROR
      "Fast DDS did not use its pinned Asio and foonathan-memory dependencies")
  endif()
endif()

file(READ
  "${STAGE_DIR}/${INSTALL_LIBDIR}/cmake/flexiv_rdk/flexiv_rdk-config-version.cmake"
  VERSION_TEXT)
if(NOT VERSION_TEXT MATCHES
    "set\\(PACKAGE_VERSION \"${EXPECTED_RDK_VERSION}\"\\)")
  message(FATAL_ERROR
    "Installed Flexiv RDK package does not report version ${EXPECTED_RDK_VERSION}")
endif()

# Reconfigure and relink the consumer against only the staged installation.
set(INSTALLED_CONSUMER_BINARY_DIR
    "${PRODUCER_BINARY_DIR}/rizon-installed-consumer-${EXPECTED_RDK_VERSION}")
file(REMOVE_RECURSE "${INSTALLED_CONSUMER_BINARY_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${RIZON_CONSUMER_SOURCE_DIR}"
    -B "${INSTALLED_CONSUMER_BINARY_DIR}"
    "-DCMAKE_PREFIX_PATH=${STAGE_DIR}"
    "-DCONTROLLERS_INCLUDE_DIR=${STAGE_DIR}/include"
    "-DCONTROLLERS_LIBRARY=${STAGE_DIR}/${INSTALL_LIBDIR}/libnvidia_srl_controllers.so"
    "-DCMAKE_INSTALL_LIBDIR=${INSTALL_LIBDIR}"
    "-DRIZON_SOURCE_DIR=${RIZON_SOURCE_DIR}"
    "-DFLEXIV_RDK_EXPECTED_VERSION=${EXPECTED_RDK_VERSION}"
    -DBUILD_RIZON_TESTS=OFF
  RESULT_VARIABLE CONSUMER_CONFIGURE_RESULT
  OUTPUT_VARIABLE CONSUMER_CONFIGURE_OUTPUT
  ERROR_VARIABLE CONSUMER_CONFIGURE_ERROR)
if(NOT CONSUMER_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Installed Rizon packages are not independently consumable:\n"
    "${CONSUMER_CONFIGURE_OUTPUT}\n${CONSUMER_CONFIGURE_ERROR}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${INSTALLED_CONSUMER_BINARY_DIR}" --parallel 2
  RESULT_VARIABLE CONSUMER_BUILD_RESULT
  OUTPUT_VARIABLE CONSUMER_BUILD_OUTPUT
  ERROR_VARIABLE CONSUMER_BUILD_ERROR)
if(NOT CONSUMER_BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Installed Rizon consumer build failed:\n"
    "${CONSUMER_BUILD_OUTPUT}\n${CONSUMER_BUILD_ERROR}")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
find_program(LDD_EXECUTABLE ldd REQUIRED)
foreach(RIZON_BINARY IN LISTS RIZON_BINARIES)
  execute_process(
    COMMAND "${READELF_EXECUTABLE}" -d "${RIZON_BINARY}"
    RESULT_VARIABLE READELF_RESULT
    OUTPUT_VARIABLE READELF_OUTPUT
    ERROR_VARIABLE READELF_ERROR)
  string(FIND
    "${READELF_OUTPUT}" "[$ORIGIN/../${INSTALL_LIBDIR}]" RPATH_INDEX)
  if(NOT READELF_RESULT EQUAL 0 OR RPATH_INDEX EQUAL -1)
    message(FATAL_ERROR
      "Missing relative Rizon RPATH in ${RIZON_BINARY}: ${READELF_ERROR}")
  endif()

  execute_process(
    COMMAND "${LDD_EXECUTABLE}" "${RIZON_BINARY}"
    RESULT_VARIABLE LDD_RESULT
    OUTPUT_VARIABLE LDD_OUTPUT
    ERROR_VARIABLE LDD_ERROR)
  if(NOT LDD_RESULT EQUAL 0 OR LDD_OUTPUT MATCHES "not found")
    message(FATAL_ERROR
      "Installed Rizon binary has unresolved libraries:\n${LDD_OUTPUT}\n${LDD_ERROR}")
  endif()

  execute_process(
    COMMAND "${RIZON_BINARY}"
    RESULT_VARIABLE EXECUTE_RESULT
    OUTPUT_VARIABLE EXECUTE_OUTPUT
    ERROR_VARIABLE EXECUTE_ERROR)
  if(NOT EXECUTE_RESULT EQUAL 1 OR NOT EXECUTE_ERROR MATCHES "Usage:")
    message(FATAL_ERROR
      "Installed Rizon binary did not stop at argument validation:\n"
      "${EXECUTE_OUTPUT}\n${EXECUTE_ERROR}")
  endif()
endforeach()

file(GLOB_RECURSE BUNDLED_CMAKE_FILES "${STAGE_DIR}/*.cmake")
foreach(BUNDLED_CMAKE_FILE IN LISTS BUNDLED_CMAKE_FILES)
  file(READ "${BUNDLED_CMAKE_FILE}" BUNDLED_CMAKE_TEXT)
  string(FIND "${BUNDLED_CMAKE_TEXT}" "${PRODUCER_BINARY_DIR}" PRODUCER_PATH_INDEX)
  if(NOT PRODUCER_PATH_INDEX EQUAL -1 OR
     BUNDLED_CMAKE_TEXT MATCHES "/home/[^/]+/repos/flexiv_rdk" OR
     BUNDLED_CMAKE_TEXT MATCHES "/usr/local")
    message(FATAL_ERROR
      "Installed dependency metadata contains a non-relocatable path: ${BUNDLED_CMAKE_FILE}")
  endif()
endforeach()
