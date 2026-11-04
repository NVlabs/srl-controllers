# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.18)

if(ENABLE_CUDA_CONSUMER)
  project(nvidia_srl_controllers_cuda_package_consumer LANGUAGES CXX CUDA)
else()
  project(nvidia_srl_controllers_package_consumer LANGUAGES CXX)
endif()

find_package(nvidia-srl-controllers CONFIG REQUIRED)

if(ENABLE_CUDA_CONSUMER)
  if(NOT DEFINED EXPECTED_CUDA_SM OR "${EXPECTED_CUDA_SM}" STREQUAL "")
    message(FATAL_ERROR "EXPECTED_CUDA_SM is required for the CUDA package consumer")
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_executable(controllers_package_consumer cuda_main.cpp equations_cuda.cu)
  target_compile_definitions(controllers_package_consumer
    PRIVATE CONTROLLERS_EXPECTED_CUDA_SM=${EXPECTED_CUDA_SM}
  )
else()
  add_executable(controllers_package_consumer main.cpp)
endif()
target_compile_features(controllers_package_consumer PRIVATE cxx_std_17)
target_link_libraries(controllers_package_consumer PRIVATE nvidia::srl::controllers)
include(GNUInstallDirs)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(RELATIVE_PATH CONSUMER_LIBRARY_PATH
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
  set_target_properties(controllers_package_consumer PROPERTIES
    INSTALL_RPATH "$ORIGIN/${CONSUMER_LIBRARY_PATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
  target_link_libraries(controllers_package_consumer PRIVATE ${CMAKE_DL_LIBS})
endif()
if(ENABLE_CUDA_CONSUMER)
  target_link_libraries(controllers_package_consumer PRIVATE CUDA::cudart)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET controllers_package_consumer APPEND PROPERTY
      INSTALL_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
  endif()
endif()
install(TARGETS controllers_package_consumer RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
