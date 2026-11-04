# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# List library source files here (e.g. foo.cpp)
set(SRC_FILES
  controllers.cpp
  composition.cpp
  example_specs.cpp
  version.cpp
  tensor/cartesian_blocks.cpp
  tensor/bimanual_task_kinematics.cpp
  tensor/torque_filter.cpp
  tensor/energy_tank.cpp
  tensor/execution_context.cpp
  tensor/generic_blocks.cpp
  tensor/high_level_controllers.cpp
  tensor/joint_blocks.cpp
  tensor/operational_space_mass.cpp
  tensor/task_impedance.cpp
  tensor/task_wrench_to_joint_torque.cpp
  tensor/safety.cpp
)
set(CPU_SRC_FILES ${SRC_FILES})

if(ENABLE_CUDA)
  set_source_files_properties(tensor/high_level_controllers_cuda.cu PROPERTIES
    COMPILE_OPTIONS "--expt-relaxed-constexpr"
  )
  list(APPEND SRC_FILES
    composition_cuda.cu
    tensor/cartesian_blocks_cuda.cu
    tensor/bimanual_task_kinematics_cuda.cu
    tensor/torque_filter_cuda.cu
    tensor/energy_tank_cuda.cu
    tensor/generic_blocks_cuda.cu
    tensor/high_level_controllers_cuda.cu
    tensor/joint_blocks_cuda.cu
    tensor/operational_space_mass_cuda.cu
    tensor/task_impedance_cuda.cu
    tensor/task_wrench_to_joint_torque_cuda.cu
    tensor/safety_cuda.cu
  )
endif()

# List public dependencies for the library here (e.g. Eigen3::Eigen)
set(PUBLIC_LINK_LIBS "")

# List private dependencies for the library here
set(PRIVATE_LINK_LIBS "")

if(ENABLE_CUDA)
  list(APPEND PRIVATE_LINK_LIBS CUDA::cudart)
endif()

set(PYBIND_SRC_FILES
  controllers_python.cpp
  equation_adapter.cpp
)
set(PYBIND_LINK_LIBS nvidia::srl::controllers)
if(ENABLE_CUDA)
  list(APPEND PYBIND_SRC_FILES equation_adapter_cuda.cu)
  list(APPEND PYBIND_LINK_LIBS CUDA::cudart)
endif()


# ==============================================================================
# DO NOT EDIT THE CONTENT BELOW THIS LINE
# ==============================================================================

if(NOT SRC_FILES STREQUAL "")
  # Add shared library
  nv_add_shared_library(${LIB_NAME} "${SRC_FILES}" "${PUBLIC_LINK_LIBS}" "${PRIVATE_LINK_LIBS}")

  target_include_directories(nvidia_srl_controllers
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(nvidia_srl_controllers
    PRIVATE NVIDIA_SRL_CONTROLLERS_BUILDING_LIBRARY=1
  )
  set_target_properties(nvidia_srl_controllers PROPERTIES
    CUDA_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
  )
  if(ENABLE_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
     CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(ENABLE_ALLOCATION_GUARD_TEST "Build the unsanitized CPU allocation regression gate" ON)
    if(ENABLE_ALLOCATION_GUARD_TEST)
      foreach(FLAG_VARIABLE
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL
          CMAKE_EXE_LINKER_FLAGS)
        if("${${FLAG_VARIABLE}}" MATCHES "(^|[ ;])-fsanitize[= ]")
          message(FATAL_ERROR
            "Allocation interception must run separately from sanitizers; configure the sanitizer "
            "build with -DENABLE_ALLOCATION_GUARD_TEST=OFF and run this gate in an unsanitized build.")
        endif()
      endforeach()
      add_library(nvidia_srl_controllers_allocation_cpu STATIC
        ${CPU_SRC_FILES}
        ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/allocation_guard_test.cpp
      )
      foreach(CPU_PROPERTY INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES)
        get_target_property(CPU_PROPERTY_VALUE nvidia_srl_controllers ${CPU_PROPERTY})
        if(CPU_PROPERTY_VALUE)
          set_property(TARGET nvidia_srl_controllers_allocation_cpu
            PROPERTY ${CPU_PROPERTY} "${CPU_PROPERTY_VALUE}")
        endif()
      endforeach()
      target_include_directories(nvidia_srl_controllers_allocation_cpu INTERFACE
        $<BUILD_INTERFACE:${NV_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
      )
      target_compile_definitions(nvidia_srl_controllers_allocation_cpu
        PRIVATE NVIDIA_SRL_CONTROLLERS_ALLOCATION_CALIBRATION=1)
      target_compile_options(nvidia_srl_controllers_allocation_cpu
        PRIVATE -UNVIDIASRLCONTROLLERS_ENABLE_CUDA)
      set_target_properties(nvidia_srl_controllers_allocation_cpu PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
      )
    endif()
  endif()
  if(ENABLE_CUDA)
    target_compile_definitions(nvidia_srl_controllers
      PUBLIC NVIDIASRLCONTROLLERS_ENABLE_CUDA=1
      PRIVATE NVIDIASRLCONTROLLERS_CUDA_BUILD_SM=${NVIDIASRLCONTROLLERS_CUDA_BUILD_SM}
      PRIVATE NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE=${NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE_VALUE}
    )
    target_include_directories(nvidia_srl_controllers BEFORE
      PRIVATE ${NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_INCLUDE_ROOT}
    )
  endif()

  if(ENABLE_CUDA AND ENABLE_TOOLS AND ENABLE_TENSOR_JOINT_IMPEDANCE_CUDA_BENCHMARK)
    # Keep raw-kernel tooling private and uninstalled, with the same CUDA/profile settings.
    # Tools use this backend instead of the shared library to retain one execution context.
    add_library(nvidia_srl_controllers_cuda_tools_backend STATIC ${SRC_FILES})
    foreach(TOOLS_PROPERTY INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FEATURES
        INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_COMPILE_OPTIONS INTERFACE_COMPILE_FEATURES)
      get_target_property(TOOLS_PROPERTY_VALUE nvidia_srl_controllers ${TOOLS_PROPERTY})
      if(TOOLS_PROPERTY_VALUE)
        set_property(TARGET nvidia_srl_controllers_cuda_tools_backend
          PROPERTY ${TOOLS_PROPERTY} "${TOOLS_PROPERTY_VALUE}")
      endif()
    endforeach()
    target_link_libraries(nvidia_srl_controllers_cuda_tools_backend
      PUBLIC ${PUBLIC_LINK_LIBS}
      PRIVATE ${PRIVATE_LINK_LIBS}
    )
    set_target_properties(nvidia_srl_controllers_cuda_tools_backend PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES
    )
  endif()

  # Update parent scope variables
  set(NV_LIBRARY_TARGETS ${NV_LIBRARY_TARGETS} PARENT_SCOPE)
endif()

# Add Pybind module
if(ENABLE_PYBIND AND NOT PYBIND_SRC_FILES STREQUAL "")
  # Add python library
  nv_add_pybind_module(${LIB_NAME} "${PYBIND_SRC_FILES}" "${PYBIND_LINK_LIBS}")
  target_include_directories(nvidia_srl_controllers_cpp PRIVATE ${CMAKE_SOURCE_DIR})

  if(ENABLE_CUDA)
    target_compile_options(nvidia_srl_controllers_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(nvidia_srl_controllers_cpp
      PRIVATE NVIDIASRLCONTROLLERS_ENABLE_CUDA=1
    )
  endif()

  # Update parent scope variables
  set(NV_PYBIND_TARGETS ${NV_PYBIND_TARGETS} PARENT_SCOPE)
endif()

if(NVIDIASRLCONTROLLERS_PYTHON_WHEEL)
  if(APPLE)
    set(CONTROLLERS_WHEEL_RPATH "@loader_path")
  else()
    set(CONTROLLERS_WHEEL_RPATH "$ORIGIN")
  endif()
  set_target_properties(nvidia_srl_controllers nvidia_srl_controllers_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH YES
    INSTALL_RPATH "${CONTROLLERS_WHEEL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH NO)
endif()
