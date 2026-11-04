// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include <cuda_runtime.h>

#include <algorithm>
#include "nvidia/srl/controllers/composition.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace {
__global__ void merge_kernel(StatusView input, MutableStatusView output, bool first) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < output.size;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output.flags[row] = normalize_status(input.flags[row] | (first ? 0U : output.flags[row]));
}
}  // namespace
Result merge_status_cuda(const StatusView* inputs,
                         std::size_t count,
                         MutableStatusView output,
                         ExecutionContext context) noexcept {
  const auto stream = reinterpret_cast<cudaStream_t>(context.cuda_stream);
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>(65535, (output.size + 127) / 128));
  for (std::size_t i = 0; i < count; ++i) {
    merge_kernel<<<blocks, 128, 0, stream>>>(inputs[i], output, i == 0);
    if (cudaPeekAtLastError() != cudaSuccess)
      return {ResultCode::kInvalidArgument, "status merge kernel launch failed"};
  }
  return {};
}
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
