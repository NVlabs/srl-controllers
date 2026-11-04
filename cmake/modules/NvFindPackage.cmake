# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Find a package, showing repository-local installation guidance where available.
function(nv_find_package PKG_NAME)
  if(VERBOSITY STRGREATER_EQUAL 1)
    message(STATUS "Finding package: ${PKG_NAME}")
  endif()

  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
  string(TOLOWER "${PKG_NAME}" PKG_NAME_LOWER)
  set(INSTALL_HELP_FILE
    "${PROJECT_SOURCE_DIR}/cmake/how_to_install_modules/${PKG_NAME_LOWER}.cmake"
  )

  if(NOT EXISTS "${INSTALL_HELP_FILE}")
    find_package(${PKG_NAME} ${ARGN})
    return()
  endif()

  include("${INSTALL_HELP_FILE}")
  set(FIND_PACKAGE_ARGS ${ARGN})
  list(FIND FIND_PACKAGE_ARGS REQUIRED REQUIRED_INDEX)
  if(REQUIRED_INDEX EQUAL -1)
    set(PACKAGE_REQUIRED FALSE)
  else()
    set(PACKAGE_REQUIRED TRUE)
    list(REMOVE_ITEM FIND_PACKAGE_ARGS REQUIRED)
  endif()
  list(APPEND FIND_PACKAGE_ARGS QUIET)
  list(REMOVE_DUPLICATES FIND_PACKAGE_ARGS)
  find_package(${PKG_NAME} ${FIND_PACKAGE_ARGS})

  if(NOT ${PKG_NAME}_FOUND)
    string(REPLACE "-" "_" PKG_NAME_UNDER "${PKG_NAME_LOWER}")
    set(INSTALL_HELP_FUNCTION "nv_how_to_install_${PKG_NAME_UNDER}")
    message(STATUS "${PKG_NAME} was not found. To install it, run:")
    message(STATUS "```")
    cmake_language(CALL ${INSTALL_HELP_FUNCTION})
    message(STATUS "```")
    if(PACKAGE_REQUIRED)
      message(FATAL_ERROR "Required package ${PKG_NAME} was not found")
    endif()
  elseif(VERBOSITY STRGREATER_EQUAL 1)
    if(DEFINED ${PKG_NAME}_VERSION)
      message(STATUS "Found ${PKG_NAME} ${${PKG_NAME}_VERSION}")
    else()
      message(STATUS "Found ${PKG_NAME}")
    endif()
  endif()
endfunction()
