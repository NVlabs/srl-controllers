# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# NOTE: Best practice is not to use file globing and to list the files out explicitly.
# From C. Scott, "Professional CMake 3rd Edition", p.176

# List source test files here (e.g. foo_test.cpp)
set(TEST_FILES
${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/version_test.cpp
${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/controllers_test.cpp
)

list(APPEND TEST_FILES
  ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/cartesian_fields_test.cpp
  ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/bimanual_controllers_test.cpp
  ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/component_core_test.cpp
  ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/public_composition_test.cpp
  ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/composite_requirements_test.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND TEST_FILES
    ${CMAKE_SOURCE_DIR}/tests/nvidia/srl/controllers/rt_bootstrap_test.cpp)
endif()

# List dependencies for the test files here
# NOTE: GTest::gtest gets added automatically and does not need to be listed
# here.
set(TEST_LINK_LIBS
  nvidia::srl::controllers
)

if(ENABLE_CUDA)
  list(APPEND TEST_LINK_LIBS CUDA::cudart)
endif()



# The actual physical helper has no vendor dependency; Eigen remains test-only.
if(ENABLE_TESTS)
  find_package(Eigen3 3.4 QUIET)
  if(Eigen3_FOUND)
    foreach(MODEL fr3 panda)
      add_executable(franka_admittance_oracle_${MODEL} franka_admittance_oracle.cpp)
      target_include_directories(franka_admittance_oracle_${MODEL} PRIVATE "${CMAKE_SOURCE_DIR}")
      target_link_libraries(franka_admittance_oracle_${MODEL} PRIVATE
        nvidia::srl::controllers Eigen3::Eigen)
      if(MODEL STREQUAL "panda")
        target_compile_definitions(franka_admittance_oracle_${MODEL} PRIVATE NVIDIA_SRL_FRANKA_FER)
      endif()
      add_test(NAME franka_admittance_oracle_${MODEL} COMMAND franka_admittance_oracle_${MODEL})
    endforeach()
  endif()
endif()

# ==============================================================================
# DO NOT EDIT THE CONTENT BELOW THIS LINE
# ==============================================================================

# Add tests if necessary
if(ENABLE_TESTS AND NOT TEST_FILES STREQUAL "")
  nv_add_tests(${LIB_NAME} "${TEST_FILES}" "${TEST_LINK_LIBS}")
  target_link_libraries(nvidia_srl_controllers_composite_requirements_test PRIVATE GTest::gtest_main)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    target_link_libraries(nvidia_srl_controllers_rt_bootstrap_test PRIVATE Threads::Threads)
    target_link_options(nvidia_srl_controllers_rt_bootstrap_test PRIVATE "LINKER:--wrap=sigwait")
    if(TARGET nvidia_srl_controllers_allocation_cpu)
      add_executable(nvidia_srl_controllers_allocation_guard_test allocation_guard_test.cpp)
      target_link_libraries(nvidia_srl_controllers_allocation_guard_test
        PRIVATE nvidia_srl_controllers_allocation_cpu GTest::gtest Threads::Threads)
      nv_set_target_compile_options(nvidia_srl_controllers_allocation_guard_test)
      target_compile_options(nvidia_srl_controllers_allocation_guard_test
        PRIVATE -UNVIDIASRLCONTROLLERS_ENABLE_CUDA)
      set_target_properties(nvidia_srl_controllers_allocation_guard_test
        PROPERTIES OUTPUT_NAME allocation_guard_test)
      set(ALLOCATION_WRAP_SYMBOLS malloc calloc realloc aligned_alloc posix_memalign)
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ALLOCATION_SIZE_MANGLE m)
      else()
        set(ALLOCATION_SIZE_MANGLE j)
      endif()
      foreach(NEW_SYMBOL _Znw${ALLOCATION_SIZE_MANGLE} _Zna${ALLOCATION_SIZE_MANGLE})
        list(APPEND ALLOCATION_WRAP_SYMBOLS
          ${NEW_SYMBOL} ${NEW_SYMBOL}RKSt9nothrow_t
          ${NEW_SYMBOL}St11align_val_t ${NEW_SYMBOL}St11align_val_tRKSt9nothrow_t)
      endforeach()
      foreach(ALLOCATION_SYMBOL IN LISTS ALLOCATION_WRAP_SYMBOLS)
        target_link_options(nvidia_srl_controllers_allocation_guard_test
          PRIVATE "LINKER:--wrap=${ALLOCATION_SYMBOL}")
      endforeach()
      gtest_discover_tests(nvidia_srl_controllers_allocation_guard_test
        DISCOVERY_TIMEOUT 20 DISCOVERY_MODE PRE_TEST PROPERTIES LABELS allocation_guard TIMEOUT 30)
      list(APPEND NV_TEST_TARGETS nvidia_srl_controllers_allocation_guard_test)
    endif()
  endif()

  target_include_directories(nvidia_srl_controllers_cartesian_fields_test PRIVATE ${CMAKE_SOURCE_DIR})
endif()
