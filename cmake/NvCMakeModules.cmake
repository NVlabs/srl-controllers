# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

include(CMakePrintHelpers)

include(${CMAKE_CURRENT_LIST_DIR}/modules/NvAddExample.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvAddLibrary.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvAddPybindModule.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvAddTest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvAddTool.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvConfigureCoverage.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvConfigureCPack.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvConfigureDocs.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvConfigureInstall.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvConfigureProject.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvFindPackage.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvGetVersion.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvPrintVariables.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvSetTargetCompileOptions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/modules/NvSetCMakeOptions.cmake)
