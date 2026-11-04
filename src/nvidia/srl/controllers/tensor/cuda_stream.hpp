/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cuda_stream.hpp
 * @brief CUDA stream selection for tensor kernels.
 */

#pragma once

#include <cuda_runtime.h>

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

inline cudaStream_t active_cuda_stream() noexcept {
  return static_cast<cudaStream_t>(tensor_execution_context().cuda_stream);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
