# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

foreach(REQUIRED_VARIABLE RDK_SOURCE_DIR DEPENDENCY_PINS_FILE)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()

if(NOT IS_DIRECTORY "${RDK_SOURCE_DIR}/thirdparty/scripts")
  message(FATAL_ERROR "Flexiv RDK dependency scripts are missing from ${RDK_SOURCE_DIR}")
endif()
if(NOT EXISTS "${DEPENDENCY_PINS_FILE}")
  message(FATAL_ERROR "Dependency pin file does not exist: ${DEPENDENCY_PINS_FILE}")
endif()

include("${DEPENDENCY_PINS_FILE}")

function(patch_dependency_script SCRIPT_NAME EXPECTED_TAG PINNED_COMMIT)
  set(SCRIPT_PATH "${RDK_SOURCE_DIR}/thirdparty/scripts/${SCRIPT_NAME}")
  file(READ "${SCRIPT_PATH}" SCRIPT_TEXT)

  set(TAG_LINE "VER_TAG=${EXPECTED_TAG}")
  string(FIND "${SCRIPT_TEXT}" "${TAG_LINE}" TAG_INDEX)
  if(TAG_INDEX EQUAL -1)
    message(FATAL_ERROR "${SCRIPT_NAME} no longer contains expected pin '${TAG_LINE}'")
  endif()
  string(REPLACE "${TAG_LINE}" "VER_TAG=${PINNED_COMMIT}" SCRIPT_TEXT "${SCRIPT_TEXT}")

  string(FIND "${SCRIPT_TEXT}" " --branch $VER_TAG" BRANCH_INDEX)
  if(BRANCH_INDEX EQUAL -1)
    message(FATAL_ERROR "${SCRIPT_NAME} no longer clones the expected version tag")
  endif()
  string(REPLACE " --branch $VER_TAG" "" SCRIPT_TEXT "${SCRIPT_TEXT}")

  string(FIND "${SCRIPT_TEXT}" "  git checkout $VER_TAG\nfi" CHECKOUT_INDEX)
  if(CHECKOUT_INDEX EQUAL -1)
    message(FATAL_ERROR "${SCRIPT_NAME} no longer checks out the expected version tag")
  endif()
  string(REPLACE
    "  git checkout $VER_TAG\nfi"
    "fi\ngit checkout --detach \"$VER_TAG\"\ngit reset --hard \"$VER_TAG\""
    SCRIPT_TEXT
    "${SCRIPT_TEXT}")

  file(WRITE "${SCRIPT_PATH}" "${SCRIPT_TEXT}")
endfunction()

patch_dependency_script(install_eigen.sh 3.4.0 "${FLEXIV_RDK_EIGEN_COMMIT}")
patch_dependency_script(install_spdlog.sh v1.9.2 "${FLEXIV_RDK_SPDLOG_COMMIT}")
patch_dependency_script(install_tinyxml2.sh 9.0.0 "${FLEXIV_RDK_TINYXML2_COMMIT}")
patch_dependency_script(install_yaml-cpp.sh yaml-cpp-0.7.0 "${FLEXIV_RDK_YAML_CPP_COMMIT}")
patch_dependency_script(
  install_foonathan_memory.sh v1.3.1 "${FLEXIV_RDK_FOONATHAN_VENDOR_COMMIT}")
patch_dependency_script(install_Fast-CDR.sh v1.0.28 "${FLEXIV_RDK_FAST_CDR_COMMIT}")
patch_dependency_script(install_Fast-DDS.sh v2.6.10 "${FLEXIV_RDK_FAST_DDS_COMMIT}")
patch_dependency_script(
  install_SpaceVecAlg.sh v1.2.4 "${FLEXIV_RDK_SPACE_VEC_ALG_COMMIT}")
patch_dependency_script(install_RBDyn.sh v1.9.0 "${FLEXIV_RDK_RBDYN_COMMIT}")

foreach(SUBMODULE_SCRIPT install_Fast-DDS.sh install_SpaceVecAlg.sh install_RBDyn.sh)
  set(SCRIPT_PATH "${RDK_SOURCE_DIR}/thirdparty/scripts/${SUBMODULE_SCRIPT}")
  file(READ "${SCRIPT_PATH}" SCRIPT_TEXT)
  set(RESET_LINE "git reset --hard \"$VER_TAG\"")
  string(FIND "${SCRIPT_TEXT}" "${RESET_LINE}" RESET_INDEX)
  if(RESET_INDEX EQUAL -1)
    message(FATAL_ERROR "${SUBMODULE_SCRIPT} is missing the pinned reset command")
  endif()
  string(REPLACE
    "${RESET_LINE}"
    "${RESET_LINE}\ngit submodule update --init --recursive"
    SCRIPT_TEXT
    "${SCRIPT_TEXT}")
  file(WRITE "${SCRIPT_PATH}" "${SCRIPT_TEXT}")
endforeach()

set(FOONATHAN_SCRIPT
    "${RDK_SOURCE_DIR}/thirdparty/scripts/install_foonathan_memory.sh")
file(READ "${FOONATHAN_SCRIPT}" FOONATHAN_TEXT)
set(FOONATHAN_RESET "git reset --hard \"$VER_TAG\"")
string(FIND "${FOONATHAN_TEXT}" "${FOONATHAN_RESET}" FOONATHAN_RESET_INDEX)
if(FOONATHAN_RESET_INDEX EQUAL -1)
  message(FATAL_ERROR "install_foonathan_memory.sh is missing the pinned reset command")
endif()
string(REPLACE
  "${FOONATHAN_RESET}"
  "${FOONATHAN_RESET}\nsed -i 's/GIT_TAG v0.7-3/GIT_TAG ${FLEXIV_RDK_FOONATHAN_MEMORY_COMMIT}/' CMakeLists.txt"
  FOONATHAN_TEXT
  "${FOONATHAN_TEXT}")
set(FOONATHAN_CONFIGURE "cmake .. $SHARED_CMAKE_ARGS")
string(FIND "${FOONATHAN_TEXT}" "${FOONATHAN_CONFIGURE}" FOONATHAN_CONFIGURE_INDEX)
if(FOONATHAN_CONFIGURE_INDEX EQUAL -1)
  message(FATAL_ERROR "install_foonathan_memory.sh has an unexpected configure command")
endif()
string(REPLACE
  "${FOONATHAN_CONFIGURE}"
  "${FOONATHAN_CONFIGURE} -DCMAKE_DISABLE_FIND_PACKAGE_foonathan_memory=ON"
  FOONATHAN_TEXT
  "${FOONATHAN_TEXT}")
file(WRITE "${FOONATHAN_SCRIPT}" "${FOONATHAN_TEXT}")

set(FAST_DDS_SCRIPT "${RDK_SOURCE_DIR}/thirdparty/scripts/install_Fast-DDS.sh")
file(READ "${FAST_DDS_SCRIPT}" FAST_DDS_TEXT)
string(FIND "${FAST_DDS_TEXT}" "-DTHIRDPARTY_Asio=ON" FAST_DDS_ASIO_INDEX)
if(FAST_DDS_ASIO_INDEX EQUAL -1)
  message(FATAL_ERROR "install_Fast-DDS.sh no longer selects its pinned Asio submodule")
endif()
string(REPLACE
  "-DTHIRDPARTY_Asio=ON"
  "-DTHIRDPARTY_Asio=FORCE"
  FAST_DDS_TEXT
  "${FAST_DDS_TEXT}")
file(WRITE "${FAST_DDS_SCRIPT}" "${FAST_DDS_TEXT}")

set(BOOST_SCRIPT "${RDK_SOURCE_DIR}/thirdparty/scripts/install_boost.sh")
file(READ "${BOOST_SCRIPT}" BOOST_TEXT)
set(BOOST_EXTRACT_LINE "echo \"-- Extracting: boost_$VER_STR.tar.bz2\"")
string(FIND "${BOOST_TEXT}" "${BOOST_EXTRACT_LINE}" BOOST_EXTRACT_INDEX)
if(BOOST_EXTRACT_INDEX EQUAL -1)
  message(FATAL_ERROR "install_boost.sh no longer has the expected extraction step")
endif()
string(REPLACE
  "${BOOST_EXTRACT_LINE}"
  "echo '${FLEXIV_RDK_BOOST_SHA256}  boost_'$VER_STR'.tar.bz2' | sha256sum --check -\n  ${BOOST_EXTRACT_LINE}"
  BOOST_TEXT
  "${BOOST_TEXT}")

set(BOOST_INSTALL_END "fi\n\necho \"Installed boost\"")
string(FIND "${BOOST_TEXT}" "${BOOST_INSTALL_END}" BOOST_INSTALL_END_INDEX)
if(BOOST_INSTALL_END_INDEX EQUAL -1)
  message(FATAL_ERROR "install_boost.sh no longer has the expected install epilogue")
endif()
string(REPLACE
  "${BOOST_INSTALL_END}"
  [=[fi

# Boost records its original prefix as an optional symlink optimization. Remove
# that block so the installed CMake packages contain no producer-build paths.
find "$INSTALL_DIR" -path '*/boost_*-config.cmake' -type f -print0 |
  while IFS= read -r -d '' config_file; do
    sed -i '\|^# If the computed and the original directories are symlink-equivalent|,/^$/d' "$config_file"
  done

echo "Installed boost"]=]
  BOOST_TEXT
  "${BOOST_TEXT}")
file(WRITE "${BOOST_SCRIPT}" "${BOOST_TEXT}")
