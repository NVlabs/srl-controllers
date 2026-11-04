# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
cmake_minimum_required(VERSION 3.27)
project(custom_controller LANGUAGES CXX CUDA)
find_package(nvidia-srl-controllers CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_executable(custom_state custom_state.cu)
target_compile_features(custom_state PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(custom_state PRIVATE nvidia::srl::controllers CUDA::cudart)
