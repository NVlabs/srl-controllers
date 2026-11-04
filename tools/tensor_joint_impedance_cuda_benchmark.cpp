/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file tensor_joint_impedance_cuda_benchmark.cpp
 * @brief Measures the public CUDA joint-impedance auto-dispatch API.
 */

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "nvidia/srl/controllers/tensor/joint_blocks.hpp"

namespace {

namespace tensor = nvidia::srl::controllers::detail::tensor;

constexpr std::size_t kJointDim = 7;
constexpr int kWarmupIterations = 10;
constexpr int kMeasuredIterations = 100;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t size) : size_(size) {
    error_ = cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * size_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  bool ok() const { return error_ == cudaSuccess; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }

  bool copy_from_host(const std::vector<T>& values, const cudaStream_t stream) {
    return values.size() <= size_ &&
           cudaMemcpyAsync(
               data_, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
  }

 private:
  T* data_{nullptr};
  std::size_t size_{0};
  cudaError_t error_{cudaSuccess};
};

std::array<std::int64_t, tensor::kMaxTensorRank> shape1(const std::size_t n) {
  return {static_cast<std::int64_t>(n), 0, 0, 0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> shape2(const std::size_t n, const std::size_t d) {
  return {static_cast<std::int64_t>(n), static_cast<std::int64_t>(d), 0, 0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> shape3(const std::size_t n,
                                                        const std::size_t rows,
                                                        const std::size_t cols) {
  return {static_cast<std::int64_t>(n),
          static_cast<std::int64_t>(rows),
          static_cast<std::int64_t>(cols),
          0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> strides1() { return {1, 0, 0, 0}; }

std::array<std::int64_t, tensor::kMaxTensorRank> strides2(const std::size_t d) {
  return {static_cast<std::int64_t>(d), 1, 0, 0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> strides3(const std::size_t rows,
                                                          const std::size_t cols) {
  return {static_cast<std::int64_t>(rows * cols), static_cast<std::int64_t>(cols), 1, 0};
}

tensor::TensorView device_vector_view(const DeviceBuffer<float>& values, const std::size_t count) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  1,
                                  shape1(count),
                                  strides1());
}

tensor::TensorView device_matrix_view(const DeviceBuffer<float>& values,
                                      const std::size_t num_envs) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  2,
                                  shape2(num_envs, kJointDim),
                                  strides2(kJointDim));
}

tensor::TensorView device_tensor3_view(const DeviceBuffer<float>& values,
                                       const std::size_t num_envs) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  3,
                                  shape3(num_envs, kJointDim, kJointDim),
                                  strides3(kJointDim, kJointDim));
}

tensor::MutableTensorView device_mutable_matrix_view(DeviceBuffer<float>& values,
                                                     const std::size_t num_envs) {
  return tensor::make_mutable_tensor_view(values.data(),
                                          tensor::Device::kCuda,
                                          tensor::Layout::kRowMajor,
                                          2,
                                          shape2(num_envs, kJointDim),
                                          strides2(kJointDim));
}

tensor::StatusView device_status_view(const DeviceBuffer<std::uint64_t>& values) {
  return tensor::StatusView{values.data(), values.size(), tensor::Device::kCuda};
}

tensor::MutableStatusView device_mutable_status_view(DeviceBuffer<std::uint64_t>& values) {
  return tensor::MutableStatusView{values.data(), values.size(), tensor::Device::kCuda};
}

std::vector<float> batched_matrix(const std::size_t num_envs,
                                  const float diagonal,
                                  const float off_diagonal) {
  std::vector<float> values(num_envs * kJointDim * kJointDim, 0.0F);
  for (std::size_t env = 0; env < num_envs; ++env) {
    for (std::size_t row = 0; row < kJointDim; ++row) {
      for (std::size_t col = 0; col < kJointDim; ++col) {
        values[(env * kJointDim + row) * kJointDim + col] =
            row == col ? diagonal + 0.1F * static_cast<float>(row) : off_diagonal;
      }
    }
  }
  return values;
}

bool report_cuda_error(const char* action, const cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }
  std::cerr << action << ": " << cudaGetErrorString(error) << '\n';
  return false;
}

bool run_public_auto(const std::size_t num_envs) {
  DeviceBuffer<float> position_error(num_envs * kJointDim);
  DeviceBuffer<float> velocity_error(num_envs * kJointDim);
  DeviceBuffer<float> desired_acceleration(num_envs * kJointDim);
  DeviceBuffer<float> mass(num_envs * kJointDim * kJointDim);
  DeviceBuffer<float> stiffness(num_envs * kJointDim * kJointDim);
  DeviceBuffer<float> damping_factors(kJointDim);
  DeviceBuffer<float> torque(num_envs * kJointDim);
  DeviceBuffer<std::uint64_t> parameter_status(num_envs);
  DeviceBuffer<std::uint64_t> status(num_envs);
  if (!position_error.ok() || !velocity_error.ok() || !desired_acceleration.ok() || !mass.ok() ||
      !stiffness.ok() || !damping_factors.ok() || !torque.ok() || !parameter_status.ok() ||
      !status.ok()) {
    std::cerr << "cudaMalloc failed for " << num_envs << " environments\n";
    return false;
  }

  cudaStream_t stream{};
  if (!report_cuda_error("cudaStreamCreateWithFlags",
                         cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
    return false;
  }
  const std::vector<float> positions(num_envs * kJointDim, 0.1F);
  const std::vector<float> velocities(num_envs * kJointDim, -0.05F);
  const std::vector<float> accelerations(num_envs * kJointDim, 0.03F);
  const std::vector<float> mass_values = batched_matrix(num_envs, 2.0F, 0.01F);
  const std::vector<float> stiffness_values = batched_matrix(num_envs, 8.0F, 0.02F);
  const std::vector<float> damping(kJointDim, 1.0F);
  const bool copied =
      position_error.copy_from_host(positions, stream) &&
      velocity_error.copy_from_host(velocities, stream) &&
      desired_acceleration.copy_from_host(accelerations, stream) &&
      mass.copy_from_host(mass_values, stream) &&
      stiffness.copy_from_host(stiffness_values, stream) &&
      damping_factors.copy_from_host(damping, stream) &&
      report_cuda_error(
          "cudaMemsetAsync(parameter_status)",
          cudaMemsetAsync(parameter_status.data(), 0, sizeof(std::uint64_t) * num_envs, stream)) &&
      report_cuda_error(
          "cudaMemsetAsync(status)",
          cudaMemsetAsync(status.data(), 0, sizeof(std::uint64_t) * num_envs, stream));
  if (!copied) {
    cudaStreamDestroy(stream);
    return false;
  }

  tensor::ParameterView entries[] = {
      {"stiffness_matrix", device_tensor3_view(stiffness, num_envs), false},
      {"damping_factors", device_vector_view(damping_factors, kJointDim), true}};
  const tensor::ParameterSetView parameters{entries, 2};
  {
    tensor::ScopedTensorExecutionContext context{tensor::TensorExecutionContext{stream}};
    const tensor::Result apply = tensor::apply_joint_impedance_controller_parameters(
        parameters, num_envs, kJointDim, device_mutable_status_view(parameter_status));
    if (!apply.ok()) {
      std::cerr << "parameter apply failed: " << apply.message << '\n';
      cudaStreamDestroy(stream);
      return false;
    }
    const auto step = [&] {
      return tensor::step_joint_impedance_controller(
          device_matrix_view(position_error, num_envs),
          device_matrix_view(velocity_error, num_envs),
          device_matrix_view(desired_acceleration, num_envs),
          device_tensor3_view(mass, num_envs),
          parameters,
          device_status_view(parameter_status),
          device_mutable_matrix_view(torque, num_envs),
          device_mutable_status_view(status));
    };
    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      const tensor::Result result = step();
      if (!result.ok()) {
        std::cerr << "warmup failed: " << result.message << '\n';
        cudaStreamDestroy(stream);
        return false;
      }
    }
    if (!report_cuda_error("warmup synchronization", cudaStreamSynchronize(stream))) {
      cudaStreamDestroy(stream);
      return false;
    }

    cudaEvent_t begin{};
    cudaEvent_t end{};
    if (!report_cuda_error("cudaEventCreate(begin)", cudaEventCreate(&begin)) ||
        !report_cuda_error("cudaEventCreate(end)", cudaEventCreate(&end))) {
      if (begin != nullptr) {
        cudaEventDestroy(begin);
      }
      cudaStreamDestroy(stream);
      return false;
    }
    cudaEventRecord(begin, stream);
    for (int iteration = 0; iteration < kMeasuredIterations; ++iteration) {
      const tensor::Result result = step();
      if (!result.ok()) {
        std::cerr << "measurement failed: " << result.message << '\n';
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);
        return false;
      }
    }
    cudaEventRecord(end, stream);
    if (!report_cuda_error("benchmark synchronization", cudaEventSynchronize(end))) {
      cudaEventDestroy(begin);
      cudaEventDestroy(end);
      cudaStreamDestroy(stream);
      return false;
    }
    float elapsed_ms = 0.0F;
    if (!report_cuda_error("cudaEventElapsedTime", cudaEventElapsedTime(&elapsed_ms, begin, end))) {
      cudaEventDestroy(begin);
      cudaEventDestroy(end);
      cudaStreamDestroy(stream);
      return false;
    }
    std::cout << "public-auto,public-auto," << kJointDim << ',' << num_envs << ',' << std::fixed
              << std::setprecision(3)
              << (1000.0F * elapsed_ms / static_cast<float>(kMeasuredIterations)) << '\n';
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }
  return report_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
}

}  // namespace

int main() {
  int device_count = 0;
  if (!report_cuda_error("cudaGetDeviceCount", cudaGetDeviceCount(&device_count)) ||
      device_count == 0) {
    return EXIT_FAILURE;
  }
  std::cout << "api,dispatch,joint_dim,num_envs,mean_step_us\n";
  for (const std::size_t num_envs : {1U, 100U, 1000U}) {
    if (!run_public_auto(num_envs)) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
