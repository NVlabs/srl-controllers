# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

function(nvidia_srl_controllers_select_libfranka_version VERSION)
  if(NOT "${VERSION}" MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
      "LIBFRANKA_VERSION must use major.minor.patch syntax; received '${VERSION}'.")
  endif()
  if(VERSION STREQUAL "0.9.2")
    set(FRANKA_ROBOT_FAMILY "FER/Panda")
    set(FRANKA_GIT_REVISION "f3b8d775a9c847cab32684c8a316f67867761674")
  elseif(VERSION VERSION_GREATER_EQUAL "0.10.0")
    set(FRANKA_ROBOT_FAMILY "FR3")
    if(VERSION STREQUAL "0.10.0")
      set(FRANKA_GIT_REVISION "4f9e3cc666e42d267f1ab566869c4f4c552e5b57")
    elseif(VERSION STREQUAL "0.15.3")
      set(FRANKA_GIT_REVISION "09385a5a619d5445bda461b3a89a62d6fa40a59a")
    elseif(VERSION STREQUAL "0.21.3")
      set(FRANKA_GIT_REVISION "85912fe02258d8cb811d3eff1f11e52ce89e3217")
    else()
      set(FRANKA_GIT_REVISION "${VERSION}")
    endif()
  else()
    message(FATAL_ERROR
      "LIBFRANKA_VERSION must be 0.9.2 for FER/Panda or at least 0.10.0 for FR3; "
      "received '${VERSION}'.")
  endif()
  set(FRANKA_ROBOT_FAMILY "${FRANKA_ROBOT_FAMILY}" PARENT_SCOPE)
  set(FRANKA_GIT_REVISION "${FRANKA_GIT_REVISION}" PARENT_SCOPE)
endfunction()
