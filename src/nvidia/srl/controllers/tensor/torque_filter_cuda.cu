/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file torque_filter_cuda.cu
 * @brief CUDA kernels for the tensor torque filter.
 */

#include "tensor/torque_filter_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/cuda_stream.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 128;
constexpr const char* kLowerLimit = "lower_limit";
constexpr const char* kUpperLimit = "upper_limit";
constexpr const char* kMaximumRate = "maximum_rate";
constexpr const char* kPreviousCommand = "previous_command";
constexpr const char* kInitialized = "initialized";
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDtFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kSaturationFlag = static_cast<std::uint64_t>(StatusFlag::kSaturation);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);

template <typename Scalar>
struct MatrixView {
  const Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t dim_stride{0};
};

template <typename Scalar>
struct MutableMatrixView {
  Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t dim_stride{0};
};

template <typename Scalar>
struct ScalarBatchView {
  const Scalar* data{nullptr};
  int rank{0};
  std::int64_t stride{0};
};

template <typename Scalar>
struct VectorParameterView {
  const Scalar* data{nullptr};
  bool shared{false};
  std::int64_t env_stride{0};
  std::int64_t dim_stride{0};
};

template <typename Scalar>
MatrixView<Scalar> matrix_view(const TensorView& view) noexcept {
  return MatrixView<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
MutableMatrixView<Scalar> matrix_view(const MutableTensorView& view) noexcept {
  return MutableMatrixView<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
VectorParameterView<Scalar> parameter_view(const ParameterView* parameter) noexcept {
  return VectorParameterView<Scalar>{
      static_cast<const Scalar*>(parameter->values.data),
      parameter->shared,
      parameter->shared ? 0 : parameter->values.strides[0],
      parameter->shared ? parameter->values.strides[0] : parameter->values.strides[1]};
}

template <typename Scalar>
__device__ Scalar read_matrix(const MatrixView<Scalar> view,
                              const std::size_t env,
                              const std::size_t dim) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   dim * static_cast<std::size_t>(view.dim_stride)];
}

template <typename Scalar>
__device__ Scalar read_matrix(const MutableMatrixView<Scalar> view,
                              const std::size_t env,
                              const std::size_t dim) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   dim * static_cast<std::size_t>(view.dim_stride)];
}

template <typename Scalar>
__device__ void write_matrix(const MutableMatrixView<Scalar> view,
                             const std::size_t env,
                             const std::size_t dim,
                             const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.env_stride) +
            dim * static_cast<std::size_t>(view.dim_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_parameter(const VectorParameterView<Scalar> view,
                                 const std::size_t env,
                                 const std::size_t dim) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   dim * static_cast<std::size_t>(view.dim_stride)];
}

template <typename Scalar>
__device__ bool parameters_are_valid(const VectorParameterView<Scalar> lower,
                                     const VectorParameterView<Scalar> upper,
                                     const VectorParameterView<Scalar> maximum_rate,
                                     const std::size_t env,
                                     const std::size_t signal_dim) {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar lower_value = read_parameter(lower, env, dim);
    const Scalar upper_value = read_parameter(upper, env, dim);
    const Scalar rate_value = read_parameter(maximum_rate, env, dim);
    if (!isfinite(lower_value) || !isfinite(upper_value) || !isfinite(rate_value) ||
        lower_value > upper_value || rate_value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__global__ void apply_kernel(VectorParameterView<Scalar> lower,
                             VectorParameterView<Scalar> upper,
                             VectorParameterView<Scalar> maximum_rate,
                             std::size_t num_envs,
                             std::size_t signal_dim,
                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env < num_envs) {
    status[env] = parameters_are_valid(lower, upper, maximum_rate, env, signal_dim)
                      ? 0U
                      : kInvalidParameterFlag;
  }
}

template <typename Scalar>
__global__ void reset_kernel(MutableMatrixView<Scalar> previous,
                             std::uint8_t* initialized,
                             std::int64_t initialized_stride,
                             const std::uint8_t* reset_mask,
                             std::int64_t reset_stride,
                             std::size_t num_envs,
                             std::size_t signal_dim,
                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool selected =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_stride)] != 0U;
  if (!selected) {
    status[env] = 0U;
    return;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(previous, env, dim, static_cast<Scalar>(0));
  }
  initialized[env * static_cast<std::size_t>(initialized_stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void step_kernel(MatrixView<Scalar> command,
                            ScalarBatchView<Scalar> dt,
                            VectorParameterView<Scalar> lower,
                            VectorParameterView<Scalar> upper,
                            VectorParameterView<Scalar> maximum_rate,
                            const std::uint64_t* parameter_status,
                            MutableMatrixView<Scalar> previous,
                            std::uint8_t* initialized,
                            std::int64_t initialized_stride,
                            MutableMatrixView<Scalar> output,
                            std::size_t num_envs,
                            std::size_t signal_dim,
                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const Scalar dt_value = dt.data[dt.rank == 0 ? 0 : env * static_cast<std::size_t>(dt.stride)];
  if (!parameters_are_valid(lower, upper, maximum_rate, env, signal_dim)) {
    flags |= kInvalidParameterFlag;
  }
  if (!isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!isfinite(read_matrix(command, env, dim))) {
      flags |= kNonFiniteInputFlag;
      break;
    }
  }
  if (flags != 0U) {
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags | kSafeModeFlag;
    return;
  }
  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized_stride);
  const bool has_previous = initialized[initialized_offset] != 0U;
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar lower_value = read_parameter(lower, env, dim);
    const Scalar upper_value = read_parameter(upper, env, dim);
    const Scalar raw = read_matrix(command, env, dim);
    const Scalar old = has_previous ? read_matrix(previous, env, dim) : static_cast<Scalar>(0);
    bool saturated = false;
    const Scalar filtered = detail::torque_filter_value(raw,
                                                       old,
                                                       lower_value,
                                                       upper_value,
                                                       read_parameter(maximum_rate, env, dim),
                                                       dt_value,
                                                       has_previous,
                                                       &saturated);
    if (saturated) {
      flags |= kSaturationFlag;
    }
    write_matrix(output, env, dim, filtered);
    write_matrix(previous, env, dim, filtered);
  }
  initialized[initialized_offset] = 1U;
  status[env] = flags;
}

Result launch_result() noexcept {
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{ResultCode::kOk, "ok"}
             : Result{ResultCode::kInvalidArgument, "CUDA torque-filter kernel launch failed"};
}

template <typename Scalar>
Result launch_apply(const ParameterSetView& parameters,
                    const std::size_t num_envs,
                    const std::size_t signal_dim,
                    const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  apply_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      parameter_view<Scalar>(find_parameter(parameters, kLowerLimit)),
      parameter_view<Scalar>(find_parameter(parameters, kUpperLimit)),
      parameter_view<Scalar>(find_parameter(parameters, kMaximumRate)),
      num_envs,
      signal_dim,
      status.flags);
  return launch_result();
}

template <typename Scalar>
Result launch_reset(const std::size_t num_envs,
                    StateView& state,
                    const TensorView& reset_mask,
                    const MutableStatusView status) noexcept {
  StateEntryView* previous = find_state(state, kPreviousCommand);
  StateEntryView* initialized = find_state(state, kInitialized);
  const std::size_t signal_dim = static_cast<std::size_t>(previous->values.shape[1]);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  reset_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      matrix_view<Scalar>(previous->values),
      static_cast<std::uint8_t*>(initialized->values.data),
      initialized->values.strides[0],
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      signal_dim,
      status.flags);
  return launch_result();
}

template <typename Scalar>
Result launch_step(const TensorView& command,
                   const TensorView& dt,
                   const ParameterSetView& parameters,
                   const StatusView parameter_status,
                   StateView& state,
                   const MutableTensorView& output,
                   const MutableStatusView status) noexcept {
  StateEntryView* previous = find_state(state, kPreviousCommand);
  StateEntryView* initialized = find_state(state, kInitialized);
  const std::size_t num_envs = static_cast<std::size_t>(command.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(command.shape[1]);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  step_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      matrix_view<Scalar>(command),
      ScalarBatchView<Scalar>{
          static_cast<const Scalar*>(dt.data), dt.rank, dt.rank == 0 ? 0 : dt.strides[0]},
      parameter_view<Scalar>(find_parameter(parameters, kLowerLimit)),
      parameter_view<Scalar>(find_parameter(parameters, kUpperLimit)),
      parameter_view<Scalar>(find_parameter(parameters, kMaximumRate)),
      parameter_status.flags,
      matrix_view<Scalar>(previous->values),
      static_cast<std::uint8_t*>(initialized->values.data),
      initialized->values.strides[0],
      matrix_view<Scalar>(output),
      num_envs,
      signal_dim,
      status.flags);
  return launch_result();
}

}  // namespace

Result apply_torque_filter_parameters_cuda(const ParameterSetView& parameters,
                                           const std::size_t num_envs,
                                           const std::size_t signal_dim,
                                           const MutableStatusView parameter_status) noexcept {
  const DType dtype = find_parameter(parameters, kLowerLimit)->values.dtype;
  return dtype == DType::kFloat32
             ? launch_apply<float>(parameters, num_envs, signal_dim, parameter_status)
             : launch_apply<double>(parameters, num_envs, signal_dim, parameter_status);
}

Result reset_torque_filter_cuda(const std::size_t num_envs,
                                StateView& state,
                                const TensorView& reset_mask,
                                const MutableStatusView status) noexcept {
  const DType dtype = find_state(state, kPreviousCommand)->values.dtype;
  return dtype == DType::kFloat32 ? launch_reset<float>(num_envs, state, reset_mask, status)
                                  : launch_reset<double>(num_envs, state, reset_mask, status);
}

Result step_torque_filter_cuda(const TensorView& command,
                               const TensorView& dt,
                               const ParameterSetView& parameters,
                               const StatusView parameter_status,
                               StateView& state,
                               const MutableTensorView& filtered_command,
                               const MutableStatusView status) noexcept {
  return command.dtype == DType::kFloat32
             ? launch_step<float>(
                   command, dt, parameters, parameter_status, state, filtered_command, status)
             : launch_step<double>(
                   command, dt, parameters, parameter_status, state, filtered_command, status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
