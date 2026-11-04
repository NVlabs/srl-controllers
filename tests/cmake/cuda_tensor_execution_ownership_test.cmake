# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED TENSOR_SOURCE_DIR OR "${TENSOR_SOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR "TENSOR_SOURCE_DIR must be set")
endif()

file(GLOB CUDA_TENSOR_SOURCES
  "${TENSOR_SOURCE_DIR}/*_cuda.cu"
  "${TENSOR_SOURCE_DIR}/*.cpp"
  "${TENSOR_SOURCE_DIR}/../composition*.cu"
  "${TENSOR_SOURCE_DIR}/../equation_adapter*.cu"
)
if(CUDA_TENSOR_SOURCES STREQUAL "")
  message(FATAL_ERROR "No CUDA tensor sources were found")
endif()

foreach(CUDA_TENSOR_SOURCE IN LISTS CUDA_TENSOR_SOURCES)
  file(READ "${CUDA_TENSOR_SOURCE}" CUDA_TENSOR_CONTENT)
  string(REGEX MATCH "cuda(Malloc|Free|StreamCreate|StreamDestroy|StreamSynchronize|DeviceSynchronize|Memcpy|Graph[A-Za-z]*)"
    FORBIDDEN_CUDA_OWNERSHIP_CALL
    "${CUDA_TENSOR_CONTENT}"
  )
  if(NOT FORBIDDEN_CUDA_OWNERSHIP_CALL STREQUAL "")
    message(FATAL_ERROR
      "${CUDA_TENSOR_SOURCE} owns CUDA execution resources through "
      "${FORBIDDEN_CUDA_OWNERSHIP_CALL}; tensor execution must use caller-owned resources."
    )
  endif()
endforeach()
