# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

set(FLEXIV_RDK_SUPPORTED_VERSIONS 1.9.1 1.9.2 1.9.3 1.9.4)

function(nvidia_srl_controllers_select_flexiv_rdk_version SELECTED_VERSION)
  if(NOT "${SELECTED_VERSION}" MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
      "FLEXIV_RDK_VERSION must use major.minor.patch syntax; received "
      "'${SELECTED_VERSION}'.")
  endif()

  if("${SELECTED_VERSION}" STREQUAL "1.9.1")
    set(RDK_GIT_REVISION "5707121ffadc5a1577c5e0eabf2afa20b9842acb")
    set(RDK_ASSET_SHA256
        "41875278fd89af2db0834458d72f518e8916c26b34fdadb5fa85244caf2cd370")
    set(RDK_ASSET_NAME "libflexiv_rdk.x86_64-linux-gnu.a")
    set(RDK_INSTALLED_LIBRARY "libflexiv_rdk.a")
    set(RDK_INSTALL_KIND "static")
    set(RDK_REQUIRES_DEPENDENCIES TRUE)
  elseif("${SELECTED_VERSION}" STREQUAL "1.9.2")
    set(RDK_GIT_REVISION "20e2edf4f5776f8e97380a1fb14b709dffbf67d9")
    set(RDK_ASSET_SHA256
        "3b1c923b4bed6c1cc94ce774d57071906c5808a30302a5d7d7be40b349c69b58")
    set(RDK_ASSET_NAME "libflexiv_rdk.x86_64-linux-gnu.a")
    set(RDK_INSTALLED_LIBRARY "libflexiv_rdk.a")
    set(RDK_INSTALL_KIND "static")
    set(RDK_REQUIRES_DEPENDENCIES TRUE)
  elseif("${SELECTED_VERSION}" STREQUAL "1.9.3")
    set(RDK_GIT_REVISION "bbb7ffc1feb125f6078eb99d2048264ceaab65e8")
    set(RDK_ASSET_SHA256
        "b3314d3095dd4615391448ea5e471ee2414696fea92e1232accafa54f2a39cfc")
    set(RDK_ASSET_NAME "libflexiv_rdk.x86_64-linux-gnu.a")
    set(RDK_INSTALLED_LIBRARY "libflexiv_rdk.a")
    set(RDK_INSTALL_KIND "static")
    set(RDK_REQUIRES_DEPENDENCIES TRUE)
  elseif("${SELECTED_VERSION}" STREQUAL "1.9.4")
    set(RDK_GIT_REVISION "9439b408d3815b93843de5bceea2bf5b951164b6")
    set(RDK_ASSET_SHA256
        "1b2f4b6155c913b2955fd8110bf4064ab74dcab700dfb16fd13b0ad3c113d1c8")
    set(RDK_ASSET_NAME "libflexiv_rdk.x86_64-linux-gnu.so")
    set(RDK_INSTALLED_LIBRARY "libflexiv_rdk.so")
    set(RDK_INSTALL_KIND "shared")
    set(RDK_REQUIRES_DEPENDENCIES FALSE)
  else()
    list(JOIN FLEXIV_RDK_SUPPORTED_VERSIONS ", " SUPPORTED_VERSION_TEXT)
    message(FATAL_ERROR
      "Unsupported FLEXIV_RDK_VERSION '${SELECTED_VERSION}'. Supported exact versions: "
      "${SUPPORTED_VERSION_TEXT}.")
  endif()

  set(FLEXIV_RDK_GIT_REVISION "${RDK_GIT_REVISION}" PARENT_SCOPE)
  set(FLEXIV_RDK_LINUX_X86_64_SHA256 "${RDK_ASSET_SHA256}" PARENT_SCOPE)
  set(FLEXIV_RDK_ASSET_NAME "${RDK_ASSET_NAME}" PARENT_SCOPE)
  set(FLEXIV_RDK_INSTALLED_LIBRARY "${RDK_INSTALLED_LIBRARY}" PARENT_SCOPE)
  set(FLEXIV_RDK_INSTALL_KIND "${RDK_INSTALL_KIND}" PARENT_SCOPE)
  set(FLEXIV_RDK_REQUIRES_DEPENDENCIES "${RDK_REQUIRES_DEPENDENCIES}" PARENT_SCOPE)
endfunction()
