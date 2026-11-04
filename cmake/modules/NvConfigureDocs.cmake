# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22


# ==============================================================================
# Set up documentation generation
macro(nv_configure_docs)
  find_package(Doxygen)

  # NOTE: C. Scott recommends explicitly setting the DOCDIR location to put
  # documentation for this project in a namespace hierarchy in case a higher
  # level CMakeList.txt has already called include(GNUInstallDirs), but that
  # does not seem to apply for us because we only create docs for the highest
  # level project.  From C. Scott, "Professional CMake 3rd Edition", p.306
  set(CMAKE_INSTALL_DOCDIR "${CMAKE_INSTALL_DATAROOTDIR}/docs")

  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(DOXYGEN_FOUND)
      # With this setup a Doxyfile is not needed, all Doxygen configuration
      # variables are set here
      # For more info see: https://cmake.org/cmake/help/latest/module/FindDoxygen.html
      set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen")
      set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
      set(DOXYGEN_GENERATE_XML "NO")
      set(DOXYGEN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/nvidia/srl/controllers")
      set(DOXYGEN_EXCLUDE_PATTERNS "*/*_test.cpp" "*/*_test.py" "*/__pycache__/*")
      set(DOXYGEN_HTML_EXTRA_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/public_api_inventory.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/example-validation.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/isaaclab/hardware_profile_contract.json")

      # set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/header.html")
      # set(DOXYGEN_HTML_FOOTER "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/footer.html")
      # set(DOXYGEN_HTML_EXTRA_STYLESHEET "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/customdoxygen.css")
      # set(DOXYGEN_HTML_EXTRA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/doxy-boot.js")
      set(DOXYGEN_HTML_COLORSTYLE_HUE "82")
      set(DOXYGEN_HTML_COLORSTYLE_SAT "100")
      set(DOXYGEN_HTML_COLORSTYLE_GAMMA "100")
      set(DOXYGEN_EXTRACT_ALL "YES")


      set(TARGET_NAME docs)
      doxygen_add_docs(${TARGET_NAME} README.md CHANGELOG.md CONTRIBUTING.md CLAUDE.md
        THIRD_PARTY_NOTICES.md docs examples python/src src include)
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
          "-DCONTROLLERS_DOC_HTML_DIR=${DOXYGEN_OUTPUT_DIRECTORY}/html"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/NvStageDocAssets.cmake"
        COMMENT "Resolve JSON evidence links in generated documentation")
      list(APPEND NV_CUSTOM_TARGETS ${TARGET_NAME})

      # NOTE: Uncomment to have docs installed as well.
      # install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs"
      #   DESTINATION "${CMAKE_INSTALL_DOCDIR}/.."
      #   OPTIONAL
      # )
    else()
        message(WARNING "Doxygen not found. Doxygen needs to be installed to generate the doxygen documentation")
    endif()
  endif()
endmacro()
