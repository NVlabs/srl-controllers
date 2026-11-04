# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)
if(NOT DEFINED CONTROLLERS_DOC_HTML_DIR)
  message(FATAL_ERROR "CONTROLLERS_DOC_HTML_DIR is required")
endif()

# Doxygen flattens Markdown pages and HTML_EXTRA_FILES, but leaves relative JSON
# links verbatim. Keep checkout Markdown links intact and resolve only the three
# copied evidence assets in generated pages.
foreach(ASSET public_api_inventory.json example-validation.json hardware_profile_contract.json)
  if(NOT EXISTS "${CONTROLLERS_DOC_HTML_DIR}/${ASSET}")
    message(FATAL_ERROR "Missing documentation evidence asset: ${ASSET}")
  endif()
endforeach()
file(GLOB HTML_PAGES "${CONTROLLERS_DOC_HTML_DIR}/*.html")
foreach(HTML_PAGE IN LISTS HTML_PAGES)
  file(READ "${HTML_PAGE}" CONTENT)
  string(REGEX REPLACE
    "href=\"([^\"]*/)?(public_api_inventory[.]json|example-validation[.]json|hardware_profile_contract[.]json)\""
    "href=\"\\2\"" RESOLVED "${CONTENT}")
  if(NOT RESOLVED STREQUAL CONTENT)
    file(WRITE "${HTML_PAGE}" "${RESOLVED}")
  endif()
endforeach()
