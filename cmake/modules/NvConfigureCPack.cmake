# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22


# ==============================================================================
# Configure the package for packager (i.e. cpack).
macro(nv_configure_cpack)
  # Set basic defaults to configure CPack before including it.
  # NOTE: From C. Scott, "Professional CMake 3rd Edition", p.343
  set(CPACK_PACKAGE_NAME ${NV_PROJECT_DASH})
  set(CPACK_PACKAGE_VENDOR "${NV_ORG} ${NV_GRP}")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${NV_ORG_UPPER} ${NV_GRP_UPPER} <${NV_GRP}@${NV_ORG}.com>")

  set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
  set(CPACK_PACKAGING_INSTALL_PREFIX ${NV_CPACK_INSTALL_PREFIX})

  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")


  # TODO (roflaherty): Configure these variables
  # set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack example project")
  # set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_LIST_DIR}/Description.txt)
  # set(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_CURRENT_LIST_DIR}/Welcome.txt)

  set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${PROJECT_VERSION})
  set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/packages")

  set(CPACK_VERBATIM_VARIABLES YES)
  set(CPACK_STRIP_FILES YES)

  set(
    CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
  )

  # Need the `dpkg-shlibdeps` tool. Install with `sudo apt install dpkg-dev`.
  # NOTE: From C. Scott, "Professional CMake 3rd Edition", p.370
  # TODO (roflaherty): Configure these variables
  # set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS YES)

  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://${NV_GRP}.gitlab-master-pages.${NV_ORG}.com/${NV_PKG}")

  # The `libdevel` section is used for packages that contain files used for
  # development purposes, such as libraries, header files, and development
  # tools.
  set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")

  # Update the debian package depenceny list based on what is set in top level
  # CMakeLists.txt
  string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${NV_CPACK_DEPENDS}")

  # Set CPack generators to facilitate running cpack with no arguments.
  # NOTE: From C. Scott, "Professional CMake 3rd Edition", p.346
  set(CPACK_GENERATOR DEB)
  if(WIN32)
    set(CPACK_GENERATOR ZIP WIX)
  elseif(APPLE)
    set(CPACK_GENERATOR TGZ productbuild)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR DEB)
  else()
    set(CPACK_GENERATOR TGZ)
  endif()

  include(CPack)
endmacro()
