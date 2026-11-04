// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
// Synthetic custom state and nonlinear control, compiled against installed public headers.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include "nvidia/srl/controllers/composition.hpp"

namespace c = nvidia::srl::controllers;

struct AdaptiveState {
  double integrated_error;
  double previous_error;
};

__global__ void custom_law(AdaptiveState* state, double* command, std::uint64_t* status) {
  const unsigned row = threadIdx.x;
  if (row >= 2) return;
  const double error = .2;
  const auto integral = c::equations::integrator_trapezoid_value(
      state[row].integrated_error, error, state[row].previous_error, .01, -1., 1.);
  state[row] = {integral.value, error};
  command[row] =
      c::equations::proportional_value(2., error) + integral.value - error * error * error;
  status[row] = integral.saturated ? static_cast<std::uint64_t>(c::StatusFlag::kSaturation) : 0;
}

void checked(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <typename T>
struct Allocation {
  T* pointer = nullptr;
  Allocation() { checked(cudaMalloc(&pointer, 2 * sizeof(T))); }
  ~Allocation() { cudaFree(pointer); }
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
};

int main() {
  Allocation<AdaptiveState> state;
  Allocation<double> command, last_valid;
  Allocation<std::uint64_t> status, holds;
  Allocation<std::uint8_t> valid;
  cudaStream_t stream{};
  checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const c::ExecutionContext context{stream};
  const c::SafetyStateView history{c::write_view(last_valid.pointer, c::Device::kCuda, 2, 1),
                                   c::write_view(valid.pointer, c::Device::kCuda, 2),
                                   c::write_view(holds.pointer, c::Device::kCuda, 2)};
  const c::MutableStatusView flags{status.pointer, 2, c::Device::kCuda};
  checked(cudaMemsetAsync(state.pointer, 0, 2 * sizeof(AdaptiveState), stream));
  if (!c::reset_safety_state(2, history, {}, flags, context).ok()) return 1;
  custom_law<<<1, 32, 0, stream>>>(state.pointer, command.pointer, status.pointer);
  checked(cudaPeekAtLastError());
  if (!c::apply_safe_output_policy(
           c::write_view(command.pointer, c::Device::kCuda, 2, 1), flags, history, {}, context)
           .ok())
    return 1;
  std::array<double, 2> host{};
  checked(
      cudaMemcpyAsync(host.data(), command.pointer, sizeof(host), cudaMemcpyDeviceToHost, stream));
  checked(cudaStreamSynchronize(stream));
  checked(cudaStreamDestroy(stream));
  if (!(host[0] > .3929 && host[0] < .3931 && host[0] == host[1])) return 1;
  std::cout << "Synthetic command: " << host[0] << '\n';
}
