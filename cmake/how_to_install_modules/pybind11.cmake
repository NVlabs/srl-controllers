# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
# Extract out package name from this file name

# ==============================================================================
# Print out how to install.
function(nv_how_to_install_pybind11)
	set(PKG_NAME pybind11)
	set(GIT_TAG v3.1.0)
	set(GIT_URL https://github.com/pybind/pybind11.git)
	set(CMAKE_ARGS "")

	set(TMP_DIR /tmp/${PKG_NAME})
	set(PKG_REPO_DIR ${TMP_DIR}/${PKG_NAME})
	set(PKG_BUILD_DIR ${PKG_REPO_DIR}/build)

  message(STATUS "mkdir -p ${TMP_DIR}")
  message(STATUS "rm -rf ${PKG_REPO_DIR}")
  message(STATUS "git clone --branch ${GIT_TAG} ${GIT_URL} ${PKG_REPO_DIR}")
  message(STATUS "mkdir ${PKG_BUILD_DIR}")
  message(STATUS "cmake ${CMAKE_ARGS} -H${PKG_REPO_DIR} -B${PKG_BUILD_DIR}")
  message(STATUS "make -j -C ${PKG_BUILD_DIR}")
  message(STATUS "sudo make -C ${PKG_BUILD_DIR} install")

endfunction()

# ==============================================================================
# Print install instructions in the terminal when running with
# cmake -P <this file>
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
  nv_how_to_install_pybind11()
endif()
