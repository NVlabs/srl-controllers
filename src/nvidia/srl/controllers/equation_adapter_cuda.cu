// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include <cuda_runtime.h>

#include <algorithm>
#include "nvidia/srl/controllers/equation_evaluation.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
namespace equation_adapter {
namespace {
template <typename Scalar>
__global__ void equation_kernel(Call call) {
  for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < call.batch_size;
       row += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    evaluate<Scalar>(call, row);
}
}  // namespace
Result execute_cuda(const Call& call, ExecutionContext context) noexcept {
  static_assert(sizeof(Call) < 4096, "equation descriptors must fit CUDA kernel parameters");
  const auto blocks =
      static_cast<unsigned>(std::min<std::size_t>(65535, (call.batch_size + 127) / 128));
  const auto stream = reinterpret_cast<cudaStream_t>(context.cuda_stream);
  if (call.dtype == DType::kFloat32)
    equation_kernel<float><<<blocks, 128, 0, stream>>>(call);
  else
    equation_kernel<double><<<blocks, 128, 0, stream>>>(call);
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{}
             : Result{ResultCode::kInvalidArgument, "equation kernel launch failed"};
}
}  // namespace equation_adapter
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
