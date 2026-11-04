# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

macro(nv_configure_project)
  # Set the default build type.
  if(ENABLE_TESTS AND ENABLE_COVERAGE)
    set(DEFAULT_BUILD_TYPE Coverage)
  else()
    set(DEFAULT_BUILD_TYPE Release)
  endif()

  # Set C++ language required standard and disable extensions for all targets
  # Recommended practice from C. Scott "Professional CMake 3rd Edition", p. 130
  # is to always set all three of these variables
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Set default build type and update the help string
  if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE}
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Coverage."
       FORCE)
  endif()

  # Set additional global variables
  # NOTE: These variables are global and used in functions that are in the
  # modules in this same directory.
  set(NV_NAMESPACE ${NV_ORG} ${NV_GRP})

  string(TOUPPER ${NV_ORG} NV_ORG_UPPER)
  string(TOUPPER ${NV_GRP} NV_GRP_UPPER)
  string(TOUPPER ${NV_PKG} NV_PKG_UPPER)

  string(REPLACE "_" "-" NV_PKG_DASH ${NV_PKG})

  list(JOIN NV_NAMESPACE "_" NV_NAMESPACE_UNDER)
  list(JOIN NV_NAMESPACE "-" NV_NAMESPACE_DASH)
  list(JOIN NV_NAMESPACE "::" NV_NAMESPACE_COLON)
  list(JOIN NV_NAMESPACE "/" NV_NAMESPACE_SLASH)

  set(NV_PROJECT_UNDER ${PROJECT_NAME})
  set(NV_PROJECT_DASH ${NV_NAMESPACE_DASH}-${NV_PKG_DASH})
  set(NV_PROJECT_SLASH ${NV_NAMESPACE_SLASH}/${NV_PKG})
  set(NV_PROJECT_COLON ${NV_NAMESPACE_COLON}::${NV_PKG})
  string(TOUPPER ${PROJECT_NAME} NV_PROJECT_UPPER)

  set(NV_VERSION_H_DIR ${NV_PROJECT_SLASH})

  set(NV_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
  set(NV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
  set(NV_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
  set(NV_SRC_CODE_DIR ${PROJECT_SOURCE_DIR}/src/${NV_NAMESPACE_SLASH})
  set(NV_TEST_DIR ${PROJECT_SOURCE_DIR}/tests)
  set(NV_TEST_CODE_DIR ${PROJECT_SOURCE_DIR}/tests/${NV_NAMESPACE_SLASH})
  set(NV_TARGETS_EXPORT_SET "${NV_PROJECT_UNDER}Targets")
  set(NV_DEVEL_DIR ${CMAKE_BINARY_DIR}/devel)
  set(NV_OPT_DIR /opt)
  set(NV_CPACK_INSTALL_PREFIX /usr)
  set(NV_PYTHON_ROOT_DIR ${PROJECT_SOURCE_DIR}/python)
  set(NV_PYTHON_SRC_DIR ${NV_PYTHON_ROOT_DIR}/src)
  set(NV_PYTHON_TEST_DIR ${NV_PYTHON_ROOT_DIR}/tests)
  set(NV_PYTHON_NAMESPACE_DIR ${NV_PYTHON_SRC_DIR}/${NV_NAMESPACE_SLASH})
  set(NV_PYTHON_PKG_DIR ${NV_PYTHON_NAMESPACE_DIR}/${NV_PKG})
  set(NV_PYTHON_PRIVATE_PKG_DIR ${NV_PYTHON_NAMESPACE_DIR}/_${NV_PKG})


  # Set variables to keep track of all the targets that are created
  set(NV_EXAMPLE_TARGETS)
  set(NV_LIBRARY_TARGETS)
  set(NV_PYBIND_TARGETS)
  set(NV_TEST_TARGETS)
  set(NV_TOOL_TARGETS)
  set(NV_CUSTOM_TARGETS)

  # Set package version
  nv_get_git_version()

  # Set variables used to set default install prefix

  # Set default install prefix if not on windows and the current project is the top level project and not a
  # subproject.
  # NOTE: From C. Scott, "Professional CMake 3rd Edition", p.307
  set(NV_USE_VERSIONED_INSTALL_PREFIX OFF)
  if((NOT WIN32) AND (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR))
    if(DEVEL)
      set(NV_INSTALL_PREFIX_TO_NAME ${NV_DEVEL_DIR}/${NV_PKG})
      set(NV_INSTALL_PREFIX_TO_VERSION ${NV_DEVEL_DIR}/${NV_PKG}-${${PROJECT_NAME}_VERSION_FULL})
    else()
      set(NV_INSTALL_PREFIX_TO_NAME ${NV_OPT_DIR}/${NV_PROJECT_SLASH})
      set(NV_INSTALL_PREFIX_TO_VERSION ${NV_OPT_DIR}/${NV_PROJECT_SLASH}-${${PROJECT_NAME}_VERSION_FULL})

      # NOTE (roflaherty): This was error prone because if the version value was
      # updated then the cached variables would need to be flushed for the
      # CMAKE_INSTALL_PREFIX to get the new version value.

      # # Allow the user to override the default with a command line option
      # if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      #   set(CMAKE_INSTALL_PREFIX ${NV_INSTALL_PREFIX_TO_VERSION} CACHE PATH "Set package install directory" FORCE)
      # endif()
    endif()
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR
        (DEFINED NV_MANAGED_INSTALL_PREFIX AND
         CMAKE_INSTALL_PREFIX STREQUAL NV_MANAGED_INSTALL_PREFIX))
      set(CMAKE_INSTALL_PREFIX "${NV_INSTALL_PREFIX_TO_VERSION}"
        CACHE PATH "Set package install directory" FORCE)
      set(NV_MANAGED_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
        CACHE INTERNAL "Last project-managed versioned install prefix" FORCE)
      set(NV_USE_VERSIONED_INSTALL_PREFIX ON)
    else()
      unset(NV_MANAGED_INSTALL_PREFIX CACHE)
    endif()
  endif()

  # Set preferred Python version to use for Pybind11 (defaults to system `python3 --version`)
  if(ENABLE_PYBIND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Get cmake directory for the pybind11 module that is installed with the
    # given python version
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Set pybind11_DIR to find the above pybind11 module
    set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

    # Find pybind11
    nv_find_package(pybind11 3.1...<4.0 REQUIRED CONFIG)

    # Set python version to use in pybind11
    if (NOT PYBIND11_PYTHON_VERSION)
      set(PYBIND11_PYTHON_VERSION  ${Python_VERSION})
    endif()
    set(PYBIND11_FINDPYTHON TRUE)
  endif()

  # Set compile definitions for compiler
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # using Clang
    add_compile_definitions(CXX_COMPILER_ID_APPLE_CLANG)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    add_compile_definitions(CXX_COMPILER_ID_CLANG)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    add_compile_definitions(CXX_COMPILER_ID_GNU)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
    add_compile_definitions(CXX_COMPILER_ID_INTEL)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_compile_definitions(CXX_COMPILER_ID_MSVC)
  endif()

  # Set compile definitions for platform type
  # NOTE: APPLE needs to come before UNIX because APPLE is a subset of UNIX
  # systems
  if(APPLE)
    add_compile_definitions(PLATFORM_MACOS)
  elseif(UNIX)
    add_compile_definitions(PLATFORM_LINUX)
  elseif(WIN32)
    add_compile_definitions(PLATFORM_WINDOWS)
  endif()

  # Include additional helper modules
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Set library rpath
  # See: https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()

  # Create additional helper targets (e.g. clean and uninstall),
  # but only if the project is the top level project.
  # NOTE: From C. Scott, "Professional CMake 3rd Edition", <TODO: find the page number>
  if(CMAKE_SOURCE_DIR STREQUAL ${PROJECT_NAME}_SOURCE_DIR)
    # Create "check" target to run linter
    set(TARGET_NAME check)
    add_custom_target(${TARGET_NAME}
      COMMENT "Running license and cpplint checks"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND python3 tools/check_license_headers.py --self-test
      COMMAND python3 tools/check_license_headers.py
      COMMAND cpplint --recursive --quiet examples include src tests tools
    )
    list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

    # Create "format" target to code formatter
    set(TARGET_NAME format)
    add_custom_target(${TARGET_NAME}
      COMMENT "Running clang-format to format files"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND find ./include ./examples ./src ./tests ./tools -name "*.hpp" -o -name "*.cpp" -o -name "*.tpp" | xargs clang-format -i
    )
    list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

    # Create "uninstall" target to remove installed files
    # NOTE: proper way to do this is shown here:
    # https://gitlab.kitware.com/cmake/community/wikis/FAQ#can-i-do-make-uninstall-with-cmake
    set(TARGET_NAME uninstall)
    add_custom_target(${TARGET_NAME}
      COMMENT "Removing all CMake installed files"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMAND cat install_manifest.txt && echo "" && xargs rm -rf < install_manifest.txt 2>&1 || true
    )
    list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

  endif()

  # Generate version.h header
  # "version.hpp" is generated from "version.hpp.in" after replacing embedded CMake
  # variables. The generated file will be placed under "include/nvidia/srl/" in both
  # the build and installation directories.
  configure_file(${NV_CMAKE_DIR}/templates/version.hpp.in include/${NV_VERSION_H_DIR}/version.hpp)

  # Set cpack dependency list
  set(NV_CPACK_DEPENDS "${NV_APT_PKG_LIB_DEPENDS}")
  if(NOT "${NV_APT_PKG_TOOLS_DEPENDS}" STREQUAL "")
    list(APPEND NV_CPACK_DEPENDS "${NV_APT_PKG_TOOLS_DEPENDS}")
  endif()

  # Print variables
  nv_print_variables()

  # Enable testing
  if(ENABLE_TESTS)
    enable_testing()
    nv_find_package(GTest REQUIRED)

    # Including GoogleTest and using `gtest_discover_tests` provides a lot more
    # functionality than `add_test` and it replaces `gtest_add_tests`, the older
    # version.
    # NOTE: From C. Scott, "Professional CMake 3rd Edition", p.298
    # See https://cmake.org/cmake/help/git-stage/module/GoogleTest.html
    # NOTE: `gtest_discovered_tests` is used in NvAddTest.cmake.
    include(GoogleTest)
  endif()

  # Add source directory
  foreach(LIB_NAME ${NV_LIBS})
    set(LIB_DIR ${NV_SRC_CODE_DIR}/${LIB_NAME})
    if(EXISTS ${LIB_DIR})
      add_subdirectory(${LIB_DIR})

      if (ENABLE_TESTS)
        set(LIB_TEST_DIR ${NV_TEST_CODE_DIR}/${LIB_NAME})
          if(EXISTS ${LIB_TEST_DIR})
            add_subdirectory(${LIB_TEST_DIR})
          endif()
      endif()

    endif()
  endforeach()

  # Add custom test targets
  if(ENABLE_TESTS)
    # Create a custom target (test_cpp) to run C++ tests
    set(TARGET_NAME test_cpp)
    add_custom_target(${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo "Running C++ tests..."
        COMMAND ctest .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

    # Create a custom target (test_all) to run C++ tests and Python tests (if necessary)
    set(TARGET_NAME test_all)
    add_custom_target(${TARGET_NAME})
    list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

    if(ENABLE_PYBIND)
      # Create a custom target (test_python) to run Python tests (if necessary)
      set(TARGET_NAME test_python)
      add_custom_target(${TARGET_NAME}
          COMMAND ${CMAKE_COMMAND} -E echo "Running Python tests..."
          COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${NV_PYTHON_ROOT_DIR}/src"
            pytest ${NV_PYTHON_TEST_DIR}
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      )
      list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})
      add_dependencies(test_all test_cpp test_python)
    else()
      add_dependencies(test_all test_cpp)
    endif()
  endif()


  if(ENABLE_TOOLS)
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools)
      add_subdirectory(tools)
    endif()
  endif()

  if(ENABLE_EXAMPLES)
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/examples)
      add_subdirectory(examples)
    endif()
  endif()

  # Set up documentation generation
  if(ENABLE_DOCS)
    nv_configure_docs()
  endif()

  # Set up coverage report generation
  if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    nv_configure_coverage()
  endif()

  # Configure package install
  nv_configure_install()

  # Configure cpack
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    nv_configure_cpack()
  endif()

  # Print all targets
  nv_print_targets()

endmacro()
