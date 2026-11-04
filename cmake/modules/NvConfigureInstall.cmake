# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22


# ==============================================================================
# Configure the package for installation.
function(nv_configure_install)
  # if(VERBOSITY STRGREATER_EQUAL 2)
  #   message(STATUS "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
  #   message(STATUS "From nv_configure_install")
  #   message(STATUS "--------------------------------------------------")
  #   message(STATUS "--------------------------------------------------")
  # endif()

  # Install CMake targets (libraries, executables, and CMake config files) from export set.

  # NOTE: Refer to C. Scott, "Professional CMake 3rd Edition", p.322 for install(EXPORT ...) setup
  # NOTE: Refer to C. Scott, "Professional CMake 3rd Edition", p.331 and p.332 for the config file setup
  # NOTE: Relative paths to DESTINATION are relative to the CMAKE_INSTALL_PREFIX
  # variable, and relative paths are prefered (see DESTINATION documentation).
  if(NV_LIBRARY_TARGETS OR NV_TOOL_TARGETS)
    install(EXPORT ${NV_TARGETS_EXPORT_SET}
      DESTINATION "cmake"
      NAMESPACE ${NV_NAMESPACE_COLON}::
      FILE "${NV_PROJECT_DASH}-config.cmake"
    )
  endif()

  if(NV_PUBLIC_HEADER_FILES)
    install(FILES ${NV_PUBLIC_HEADER_FILES}
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${NV_ORG}/${NV_GRP}/${NV_PKG}"
    )
  else()
    install(DIRECTORY "${NV_INCLUDE_DIR}/${NV_ORG}"
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING PATTERN *.hpp PATTERN *.tpp
    )
  endif()

  # NOTE: Refer to C. Scott, "Professional CMake 3rd Edition", p.335 for configuring the version file
  write_basic_package_version_file("${NV_PROJECT_DASH}-config-version.cmake"
    VERSION ${${PROJECT_NAME}_VERSION_FULL}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${NV_PROJECT_DASH}-config-version.cmake
    DESTINATION "cmake"
  )

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${NV_VERSION_H_DIR}/version.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${NV_VERSION_H_DIR}"
  )

  install(FILES
    "${CMAKE_SOURCE_DIR}/README.md"
    "${CMAKE_SOURCE_DIR}/CHANGELOG.md"
    "${CMAKE_SOURCE_DIR}/CLAUDE.md"
    "${CMAKE_SOURCE_DIR}/LICENSE"
    "${CMAKE_SOURCE_DIR}/CONTRIBUTING.md"
    "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}"
  )

  install(DIRECTORY "${CMAKE_SOURCE_DIR}/LICENSES/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/LICENSES"
    FILES_MATCHING PATTERN "*.txt")
  install(FILES "${CMAKE_SOURCE_DIR}/examples/ur/ur_description_LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/examples/ur")

  install(DIRECTORY "${CMAKE_SOURCE_DIR}/docs/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/docs"
    FILES_MATCHING PATTERN "*.md" PATTERN "*.json")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/examples/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/${NV_PROJECT_DASH}/examples"
    FILES_MATCHING PATTERN "*.md" PATTERN "*.cpp" PATTERN "*.cu" PATTERN "*.hpp" PATTERN "*.py"
      PATTERN "*.json" PATTERN "*.urdf" PATTERN "*.cmake" PATTERN "CMakeLists.txt"
      PATTERN "__pycache__" EXCLUDE)

  if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND DEVEL AND NV_USE_VERSIONED_INSTALL_PREFIX)
    # Link the unspecified version to the current version
    set(NAME_VERSION ${NV_PKG}-${${PROJECT_NAME}_VERSION_FULL})
    string(CONFIGURE [=[
      if(CMAKE_INSTALL_PREFIX STREQUAL "@NV_INSTALL_PREFIX_TO_VERSION@" AND
          "$ENV{DESTDIR}" STREQUAL "")
        if(EXISTS "@NV_INSTALL_PREFIX_TO_NAME@" AND NOT IS_SYMLINK "@NV_INSTALL_PREFIX_TO_NAME@")
          message(FATAL_ERROR "Refusing to replace @NV_INSTALL_PREFIX_TO_NAME@ with a development link")
        endif()
        file(CREATE_LINK "./@NAME_VERSION@" "@NV_INSTALL_PREFIX_TO_NAME@"
          SYMBOLIC RESULT NV_LINK_RESULT)
        if(NOT NV_LINK_RESULT STREQUAL "0")
          message(FATAL_ERROR "Cannot create development link @NV_INSTALL_PREFIX_TO_NAME@: ${NV_LINK_RESULT}")
        endif()
        message(STATUS "Link @NV_INSTALL_PREFIX_TO_NAME@ -> ./@NAME_VERSION@")
      endif()
    ]=] NV_INSTALL_LINK_CODE @ONLY)
    install(CODE "${NV_INSTALL_LINK_CODE}")
  endif()

endfunction()
