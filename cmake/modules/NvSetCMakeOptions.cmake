# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Set any option at the command line with `-D <option name>=<option value>`.
# NOTE: Option values are cached in the `CMakeCache.txt` between executations of
# the `cmake` command.
macro(nv_set_cmake_options)
  # Set developer mode
  # Developer mode builds are installed into the `<project root>/devel`
  # directory instead of `/opt`.  See `NvConfigurePackage.cmake` for details
  # about the installation path.  Additionally, the defaults for other options
  # will change based on this value.
  option(DEVEL "Enable developer mode" ON)

  # Set the level of the verbose output for the CMake build configuration.
  # option(VERBOSITY "Set the level of verbose output" 0)
  if(NOT DEVEL)
    set(VERBOSITY 0 CACHE STRING "Set the level of verbose output")
  else()
    set(VERBOSITY 2 CACHE STRING "Set the level of verbose output")
  endif()

  # Build targets that depend on google/benchmark
  option(ENABLE_BENCHMARK "Build targets that depend on google/benchmark" OFF)

  # Enable coverage report generation.
  option(ENABLE_COVERAGE "Enable coverage report generation" ${DEVEL})

  # Build targets that depend on CUDA
  option(ENABLE_CUDA "Build targets that depend on CUDA" OFF)

  # Set up documentation generation targets
  option(ENABLE_DOCS "Enable documentation generation" ${DEVEL})

  # Build examples. Vendor integrations remain explicitly opt-in because their
  # SDKs are not dependencies of the controllers library itself.
  option(ENABLE_EXAMPLES "Allow building examples" OFF)
  option(ENABLE_FRANKA_EXAMPLES "Build examples that depend on libfranka" OFF)
  option(ENABLE_RIZON_EXAMPLES
         "Build Rizon examples with a pinned, locally installed Flexiv RDK" OFF)
  option(ENABLE_FLEXIV_DUAL_ARM_RDK_EXAMPLE
         "Build the dual-arm hold example with a pinned, locally installed Flexiv RDK v2" OFF)
  option(ENABLE_UR_EXAMPLES "Build examples that depend on ur_client_library" OFF)

  # Build pybindings for public interfaces
  option(ENABLE_PYBIND "Build pybindings for public interfaces" OFF)

  # Build tests (including unit and integration tests).
  option(ENABLE_TESTS "Build tests" ${ENABLE_COVERAGE})

  # Build tools.
  option(ENABLE_TOOLS "Build tools" ON)

  # This option allows including or not including the `-Werror` compiler option
  # to the "Release" and "Coverage" builds. Note, `-Werror` is never included in
  # the "Debug" build. See `cmake/NvSetTargetCompileOptions.cmake` file for the
  # current settings.
  option(WERROR "Use -Werror compiler option" "NOT ${DEVEL}")
endmacro()
