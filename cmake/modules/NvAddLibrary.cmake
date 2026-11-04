# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Add a shared library target and export it with the project package.
macro(nv_add_shared_library LIB_NAME SRC_FILES PUBLIC_LINK_LIBS PRIVATE_LINK_LIBS)
  nv_add_shared_library_target(
    ${LIB_NAME} "${SRC_FILES}" "${PUBLIC_LINK_LIBS}" "${PRIVATE_LINK_LIBS}" TARGET_NAME
  )
  list(APPEND NV_LIBRARY_TARGETS ${TARGET_NAME})
endmacro()

# ==============================================================================
# Create the shared library target for nv_add_shared_library.
function(nv_add_shared_library_target LIB_NAME SRC_FILES PUBLIC_LINK_LIBS PRIVATE_LINK_LIBS)
  string(JOIN "_" TARGET_NAME ${NV_NAMESPACE} ${LIB_NAME})
  string(JOIN "::" TARGET_ALIAS ${NV_NAMESPACE} ${LIB_NAME})

  if(VERBOSITY STRGREATER_EQUAL 2)
    message(STATUS "Adding shared library ${TARGET_NAME}")
    cmake_print_variables(SRC_FILES PUBLIC_LINK_LIBS PRIVATE_LINK_LIBS)
  endif()

  add_library(${TARGET_NAME} SHARED ${SRC_FILES})
  target_link_libraries(${TARGET_NAME}
    PUBLIC ${PUBLIC_LINK_LIBS}
    PRIVATE ${PRIVATE_LINK_LIBS}
  )
  set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME ${LIB_NAME})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    set_property(TARGET ${TARGET_NAME} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(ENABLE_CUDA AND TARGET CUDA::cudart)
      set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
        INSTALL_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
    endif()
  endif()
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${NV_SRC_DIR}>
      $<BUILD_INTERFACE:${NV_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  )
  nv_set_target_compile_options(${TARGET_NAME})
  add_library(${TARGET_ALIAS} ALIAS ${TARGET_NAME})

  install(TARGETS ${TARGET_NAME} EXPORT ${NV_TARGETS_EXPORT_SET}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  list(LENGTH ARGN NUM_OUTPUTS)
  if(NUM_OUTPUTS GREATER 1)
    message(FATAL_ERROR "Function invoked with incorrect number of arguments")
  endif()
  if(ARGN)
    list(GET ARGN 0 OUTPUT)
    set(${OUTPUT} ${TARGET_NAME} PARENT_SCOPE)
  endif()
endfunction()
