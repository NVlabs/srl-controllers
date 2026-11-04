# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ==============================================================================
# Add pybind targets based on given source files and linked libraries.
#
# NOTE: The pybind11::module target and the library target automatically get added
# as linked libraries.
#
# See `nv_add_pybind_module_target` for more details.
#
# Args:
#   LIB_NAME (str) - The name of the library that these bindings are create for.
#   PYBIND_SRC_FILES (list) - List of pybind source file paths.
#   PYBIND_LINK_LIBS (list) - List of libraries to privately link to the
#       pybind target.
macro(nv_add_pybind_module LIB_NAME PYBIND_SRC_FILES PYBIND_LINK_LIBS)
  nv_add_pybind_module_target(${LIB_NAME} "${PYBIND_SRC_FILES}" "${PYBIND_LINK_LIBS}" TARGET_NAME)
  list(APPEND NV_PYBIND_TARGETS ${TARGET_NAME})
endmacro()


# ==============================================================================
# Add a pybind target base on given source files and linked libraries.
#
# NOTE: The pybind11::module target and the library target automatically get added
# as linked libraries.
#
# Args:
#   LIB_NAME (str) - The name of the library that these bindings are create for.
#   PYBIND_SRC_FILES (list) - List of pybind source file paths.
#   PYBIND_LINK_LIBS (list) - List of libraries to privately link to the
#       pybind target.
function(nv_add_pybind_module_target LIB_NAME PYBIND_SRC_FILES PYBIND_LINK_LIBS)

  # Set the pybind target name
  string(JOIN "_" TARGET_NAME ${NV_NAMESPACE} ${LIB_NAME} "cpp")

  # Set the library target name
  string(JOIN "_" LIB_TARGET_NAME ${NV_NAMESPACE} ${LIB_NAME})

  # Append additional link libs
  list(APPEND PYBIND_LINK_LIBS "pybind11::module" ${LIB_TARGET_NAME})

  if(VERBOSITY STRGREATER_EQUAL 2)
    message(STATUS "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    message(STATUS "From nv_add_pybind_module_target()")
    message(STATUS "--------------------------------------------------")
    cmake_print_variables(TARGET_NAME)
    cmake_print_variables(NV_NAMESPACE)
    cmake_print_variables(LIB_NAME)
    cmake_print_variables(PYBIND_SRC_FILES)
    cmake_print_variables(PYBIND_LINK_LIBS)
    message(STATUS "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    message(STATUS "")
  endif()

  # Create pybind11 module target
  pybind11_add_module(${TARGET_NAME} SHARED "${PYBIND_SRC_FILES}")

  # Link dependency libraries to the target
  # NOTE: Best practice is to default to PRIVATE, only use PUBLIC or INTERFACE if the
  # relationship genuinely requires it.
  # From C. Scott, https://www.mail-archive.com/cmake@cmake.org/msg61604.html
  target_link_libraries(${TARGET_NAME} PUBLIC "${PYBIND_LINK_LIBS}")

  # NOTE (roflaherty): This ensures that Linux generates a proper `.so` file by
  # removing any default prefixes. The filename format is
  # `<module_name>.<interpreter_version>-<architecture>.so`
  # (e.g. `<module_name>.cpython-310-x86_64-linux-gnu.so`).
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
  endif()

  # Symlink the library from the build directory to the python namespace directory
  if(NVIDIASRLCONTROLLERS_PYTHON_SOURCE_SYMLINK)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ln -sf $<TARGET_FILE:${TARGET_NAME}> "${NV_PYTHON_PRIVATE_PKG_DIR}"
    )
  endif()

endfunction()
