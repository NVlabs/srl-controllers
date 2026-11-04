// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nvidia/srl/controllers/controllers.hpp"
#include "public_surface.hpp"  // NOLINT(build/include_subdir): standalone installed consumer.

namespace {

namespace controllers = nvidia::srl::controllers;

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    error_ = cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
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
  std::size_t size() const { return count_; }

 private:
  T* data_{nullptr};
  std::size_t count_{0};
  cudaError_t error_{cudaSuccess};
};

}  // namespace

bool installed_equations_run_on_cuda();

int main() {
  if (!installed_equations_run_on_cuda()) return 1;
  if (!public_surface_is_available()) return 1;
  cudaDeviceProp device{};
  if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) {
    std::cerr << "Unable to query the CUDA device.\n";
    return 1;
  }
  if (device.major * 10 + device.minor != CONTROLLERS_EXPECTED_CUDA_SM) {
    std::cerr << "Installed package was compiled for SM " << CONTROLLERS_EXPECTED_CUDA_SM
              << " but the runtime device is SM " << device.major * 10 + device.minor << ".\n";
    return 1;
  }

  constexpr std::size_t kBatch = 1;
  constexpr std::size_t kJointDim = 7;
  std::vector<float> position(kJointDim, 0.1F);
  std::vector<float> zeros(kJointDim, 0.0F);
  std::vector<float> mass(kJointDim * kJointDim, 0.0F);
  std::vector<float> stiffness(kJointDim * kJointDim, 0.0F);
  for (std::size_t dimension = 0; dimension < kJointDim; ++dimension) {
    mass[dimension * kJointDim + dimension] = 1.0F;
    stiffness[dimension * kJointDim + dimension] = 2.0F;
  }
  const std::vector<float> damping(kJointDim, 1.0F);

  DeviceBuffer<float> device_position(position.size());
  DeviceBuffer<float> device_velocity(zeros.size());
  DeviceBuffer<float> device_acceleration(zeros.size());
  DeviceBuffer<float> device_mass(mass.size());
  DeviceBuffer<float> device_stiffness(stiffness.size());
  DeviceBuffer<float> device_damping(damping.size());
  DeviceBuffer<float> device_torque(position.size());
  DeviceBuffer<float> last_valid_torque(position.size());
  DeviceBuffer<std::uint8_t> valid(kBatch);
  DeviceBuffer<std::uint64_t> hold_count(kBatch);
  DeviceBuffer<std::uint64_t> parameter_status(kBatch);
  DeviceBuffer<std::uint64_t> status(kBatch);
  if (!device_position.ok() || !device_velocity.ok() || !device_acceleration.ok() ||
      !device_mass.ok() || !device_stiffness.ok() || !device_damping.ok() || !device_torque.ok() ||
      !last_valid_torque.ok() || !valid.ok() || !hold_count.ok() || !parameter_status.ok() ||
      !status.ok()) {
    std::cerr << "Unable to allocate caller-owned CUDA buffers.\n";
    return 1;
  }

  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::cerr << "Unable to create the caller-owned CUDA stream.\n";
    return 1;
  }
  const auto cleanup = [&stream] { cudaStreamDestroy(stream); };
  if (cudaMemcpyAsync(device_position.data(),
                      position.data(),
                      position.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device_velocity.data(),
                      zeros.data(),
                      zeros.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device_acceleration.data(),
                      zeros.data(),
                      zeros.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device_mass.data(),
                      mass.data(),
                      mass.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device_stiffness.data(),
                      stiffness.data(),
                      stiffness.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device_damping.data(),
                      damping.data(),
                      damping.size() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemsetAsync(parameter_status.data(), 0, sizeof(std::uint64_t), stream) != cudaSuccess ||
      cudaMemsetAsync(status.data(), 0, sizeof(std::uint64_t), stream) != cudaSuccess) {
    std::cerr << "Unable to initialize caller-owned CUDA buffers.\n";
    cleanup();
    return 1;
  }

  const controllers::JointImpedanceController::Parameters parameters{
      controllers::shared(controllers::read_view(
          device_stiffness, controllers::Device::kCuda, kJointDim, kJointDim)),
      controllers::shared(
          controllers::read_view(device_damping, controllers::Device::kCuda, kJointDim))};
  const controllers::JointImpedanceController::Storage storage{
      controllers::write_view(device_torque, controllers::Device::kCuda, kBatch, kJointDim),
      controllers::write_status(parameter_status, controllers::Device::kCuda),
      controllers::write_status(status, controllers::Device::kCuda),
      controllers::make_safety_state(
          controllers::write_view(last_valid_torque, controllers::Device::kCuda, kBatch, kJointDim),
          controllers::write_view(valid, controllers::Device::kCuda, kBatch),
          controllers::write_view(hold_count, controllers::Device::kCuda, kBatch))};
  const controllers::JointImpedanceController::Inputs inputs{
      controllers::read_view(device_position, controllers::Device::kCuda, kBatch, kJointDim),
      controllers::read_view(device_velocity, controllers::Device::kCuda, kBatch, kJointDim),
      controllers::read_view(device_acceleration, controllers::Device::kCuda, kBatch, kJointDim),
      controllers::read_view(
          device_mass, controllers::Device::kCuda, kBatch, kJointDim, kJointDim)};
  const controllers::ExecutionContext context{stream};
  controllers::JointImpedanceController controller;
  const controllers::Result prepare = controller.prepare(
      {controllers::Device::kCuda, controllers::DType::kFloat32, kBatch, kJointDim, {}},
      parameters,
      storage,
      context);
  const controllers::Result reset = controller.reset({}, context);
  const controllers::Result step = controller.step(inputs, context);
  if (!prepare.ok() || !reset.ok() || !step.ok()) {
    std::cerr << "Installed CUDA controller API call failed.\n";
    cleanup();
    return 1;
  }

  std::vector<float> torque(kJointDim);
  std::uint64_t host_parameter_status = 0;
  std::uint64_t host_status = 0;
  if (cudaMemcpyAsync(torque.data(),
                      device_torque.data(),
                      torque.size() * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(&host_parameter_status,
                      parameter_status.data(),
                      sizeof(host_parameter_status),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(
          &host_status, status.data(), sizeof(host_status), cudaMemcpyDeviceToHost, stream) !=
          cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "Installed CUDA controller API did not complete successfully.\n";
    cleanup();
    return 1;
  }
  cleanup();
  if (host_parameter_status != 0U || host_status != 0U) {
    std::cerr << "Installed CUDA controller API returned a fault status.\n";
    return 1;
  }
  for (const float value : torque) {
    if (value < 0.199F || value > 0.201F) {
      std::cerr << "Installed CUDA controller API returned an unexpected torque.\n";
      return 1;
    }
  }
  return 0;
}
