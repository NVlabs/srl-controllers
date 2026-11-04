# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Used to set target compiler options consistently across the
# project. It will only set target compiler options if the project is the
# top most project. This means that these options will not get set if the
# project is included from a `add_subdirectory`. Additionally, the scope of the
# option is set to `PRIVATE`, which means that if the target is imported into
# another project these options will not be applied.
function(nv_set_target_compile_options TARGET_NAME)
  # Currently not being used.
  # Example usage: `set(DISABLED_WARNINGS -Wno-unused-parameter)`
  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(OPT_OPTIONS /O2)
    set(WARNING_OPTIONS /W4)  # basically equivalent to `-Wall`; MSVC does support `/Wall` but
                              # additional warnings are mostly informational/useless.
    if(WERROR)
      set(WERROR_OPTION /WX)  # equivalent to `-Werror`
    else()
      set(WERROR_OPTION)
    endif()
    set(DISABLED_WARNINGS
      /wd4101  # unreferenced local variable
      /wd4244  # '=': conversion from 'int' to 'char', possible loss of data
      /wd4245  # 'argument': conversion from 'int' to 'size_t'
      /wd4251  # class ... needs to have dll-interface to be used by clients of class ...
      /wd4267  # 'initializing': conversion from 'size_t' to 'int', possible loss of data
      /wd4275  # non dll-interface class ... used as base for dll-interface class ...
      /wd4305  # 'initializing': truncation from 'double' to 'float'
      /wd4324  # 'std::_Optional_destruct_base<_Ty,true>': structure was padded due to alignment specifier
      /wd4389  # '==': signed/unsigned mismatch
      /wd4456  # declaration of ... hides previous local declaration
      /wd4457  # declaration of ... hides function parameter
      /wd4458  # declaration of ... hides class member
      /wd4820  # padding added after data member
    )
  else()
    set(OPT_OPTIONS -O3)
    set(WARNING_OPTIONS -Wall -Wextra)
    if(WERROR)
      set(WERROR_OPTION -Werror)
    else()
      set(WERROR_OPTION)
    endif()
    set(DISABLED_WARNINGS)
  endif()

  # Set target options only if the top most project
  if(CMAKE_SOURCE_DIR STREQUAL ${PROJECT_NAME}_SOURCE_DIR)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "Coverage")
      target_compile_options(${TARGET_NAME} PRIVATE ${OPT_OPTIONS} ${WARNING_OPTIONS} ${WERROR_OPTION} ${DISABLED_WARNINGS})
    elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      target_compile_options(${TARGET_NAME} PRIVATE ${WARNING_OPTIONS} ${DISABLED_WARNINGS})
    endif()
  endif()
endfunction()
