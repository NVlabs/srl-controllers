# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22


# ==============================================================================
# Get the current project version based on the latest git tag using git
# describe.
function(nv_get_git_version)
  find_package(Git)
  if(GIT_EXECUTABLE)
    # Generate a git-describe version string from Git repository tags
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --dirty --match "v*"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
      RESULT_VARIABLE GIT_DESCRIBE_ERROR_CODE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    if(NOT GIT_DESCRIBE_ERROR_CODE)
      string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR ${GIT_DESCRIBE_VERSION})
      string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR ${GIT_DESCRIBE_VERSION})
      string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH ${GIT_DESCRIBE_VERSION})
      # string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+-(\\d-g\\w{7}(-dirty)?)$" "\\1" VERSION_SHA ${GIT_DESCRIBE_VERSION})
      string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+-(.*)" "\\1" VERSION_SHA ${GIT_DESCRIBE_VERSION})

      # Hack because I can't figure out the proper regex expression to have the VERSION_SHA be empty if it doesn't exist.
      if(${VERSION_SHA} STREQUAL ${GIT_DESCRIBE_VERSION})
        set(VERSION_SHA "")
      endif()

    else()
      message(WARNING "No git tags were found. Package version is being set to v0.0.0")
      set(GIT_DESCRIBE_VERSION "v0.0.0")
    endif()
    set(${PROJECT_NAME}_VERSION_MAJOR ${VERSION_MAJOR} PARENT_SCOPE)
    set(${PROJECT_NAME}_VERSION_MINOR ${VERSION_MINOR} PARENT_SCOPE)
    set(${PROJECT_NAME}_VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE)
    set(${PROJECT_NAME}_VERSION_SHA ${VERSION_SHA} PARENT_SCOPE)
    set(${PROJECT_NAME}_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" PARENT_SCOPE)
    if("${VERSION_SHA}" STREQUAL "")
      set(${PROJECT_NAME}_VERSION_FULL "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" PARENT_SCOPE)
    else()
      set(${PROJECT_NAME}_VERSION_FULL "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}-${VERSION_SHA}" PARENT_SCOPE)
    endif()

    set(PROJECT_VERSION_MAJOR ${VERSION_MAJOR} PARENT_SCOPE)
    set(PROJECT_VERSION_MINOR ${VERSION_MINOR} PARENT_SCOPE)
    set(PROJECT_VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE)
    set(PROJECT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}" PARENT_SCOPE)
  endif()
endfunction()
