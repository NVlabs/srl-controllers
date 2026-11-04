# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Add a tool target based on the source file provided and listed
# libraries, and target to NV_TOOL_TARGETS.
#
# The name of the resulting tool file will be the file name without the
# extenion.
#
# The name of the target will be the current NV_NAMESPACE plus the file name
# without the extension.
#
# Args:
#   SRC_FILE (str) - Tool source file path.
#   LINK_LIBS (list) - List of libraries to privately link to the
#       tool target.
macro(nv_add_tool SRC_FILE)
  set(LINK_LIBS ${ARGN})

  nv_add_tool_target(${SRC_FILE} "${LINK_LIBS}" TARGET_NAME)
  list(APPEND NV_TOOL_TARGETS ${TARGET_NAME})
endmacro()


# ==============================================================================
# Add a tool target based on the source file provided and listed
# libraries.
#
# The name of the resulting tool file will be the file name without the
# extenion.
#
# The name of the target will be the current NV_NAMESPACE plus the file name
# without the extension.
#
# Args:
#   SRC_FILE (str) - Tool source file path.
#   LINK_LIBS (list) - List of libraries to privately link to the
#       tool target.
#
# Return:
#   OUTPUT (optional) (str) - Target name.
function(nv_add_tool_target SRC_FILE LINK_LIBS)
  # Set target name
  get_filename_component(SRC_NAME "${SRC_FILE}" NAME_WE)
  string(JOIN "_" TARGET_NAME ${NV_NAMESPACE} ${SRC_NAME})

  # Add tool
  add_executable(${TARGET_NAME} "${SRC_FILE}")

  # Link libraries
  # NOTE: Best practice is to default to PRIVATE, only use PUBLIC or INTERFACE if the
  # relationship genuinely requires it.
  # From C. Scott, https://www.mail-archive.com/cmake@cmake.org/msg61604.html
  target_link_libraries(${TARGET_NAME} PRIVATE "${LINK_LIBS}")

  # Set the output file name for the tool
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${SRC_NAME})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(RELATIVE_PATH NV_TOOL_LIBRARY_PATH
      "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
      INSTALL_RPATH "$ORIGIN/${NV_TOOL_LIBRARY_PATH}")
    set_property(TARGET ${TARGET_NAME} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(ENABLE_CUDA AND TARGET CUDA::cudart)
      set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
        INSTALL_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
    endif()
  endif()

  # Set target compile options
  nv_set_target_compile_options(${TARGET_NAME})

  # NOTE: Refer to C. Scott, "Professional CMake 3rd Edition", p.308 for
  # install(TARGETS ...) setup
  install(TARGETS ${TARGET_NAME} EXPORT ${NV_TARGETS_EXPORT_SET}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Throw an error if more than one output variable is provided
  list(LENGTH ARGN N_OUPUTS)
  if(N_OUPUTS GREATER 1)
    message(FATAL_ERROR "Function invoked with incorrect number of arguments")
  endif()
  # Return target name, if output variable is provided
  if(ARGN)
    LIST(GET ARGN 0 OUTPUT)
    set(${OUTPUT} ${TARGET_NAME} PARENT_SCOPE)
  endif()

endfunction()
