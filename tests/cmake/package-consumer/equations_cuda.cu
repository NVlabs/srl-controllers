// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include <cuda_runtime.h>
#include "nvidia/srl/controllers/equations.hpp"

__global__ void custom_equation(double* command) {
  command[0] = nvidia::srl::controllers::equations::proportional_value(2., 3.);
}

bool installed_equations_run_on_cuda() {
  double* device = nullptr;
  double host = 0;
  if (cudaMalloc(&device, sizeof(double)) != cudaSuccess) return false;
  custom_equation<<<1, 1>>>(device);
  const auto copied = cudaMemcpy(&host, device, sizeof(double), cudaMemcpyDeviceToHost);
  const auto released = cudaFree(device);
  return copied == cudaSuccess && released == cudaSuccess && host == 6.;
}
