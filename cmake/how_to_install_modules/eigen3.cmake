# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
# Extract out package name from this file name

# ==============================================================================
# Print out how to install.
function(nv_how_to_install_eigen3)
  if(APPLE)
    message(STATUS "brew install eigen")
  else()
    message(STATUS "sudo apt-get install libeigen3-dev")
  endif()
endfunction()

# ==============================================================================
# Print install instructions in the terminal when running with
# cmake -P <this file>
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
  nv_how_to_install_eigen3()
endif()
