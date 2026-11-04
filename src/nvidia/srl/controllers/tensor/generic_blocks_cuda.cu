/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file generic_blocks_cuda.cu
 * @brief CUDA tensor kernels for generic controller blocks.
 */

#include "tensor/generic_blocks_cuda.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/math_core.hpp"

#include "tensor/cuda_component_tuning_profiles.hpp"
#include "tensor/cuda_stream.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kGain = "gain";
constexpr const char* kU0 = "u_0";
constexpr const char* kUpperLimit = "upper_limit";
constexpr const char* kLowerLimit = "lower_limit";
constexpr const char* kPreviousSignal = "previous_signal";
constexpr const char* kIntegratedSignal = "integrated_signal";
constexpr const char* kInitialized = "initialized";
constexpr std::uint64_t kUnsupportedModeFlag =
    static_cast<std::uint64_t>(StatusFlag::kUnsupportedMode);
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDtFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kSaturationFlag = static_cast<std::uint64_t>(StatusFlag::kSaturation);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kSafeTriggers = kUnsupportedModeFlag | kInvalidParameterFlag |
                                        kInvalidInputFlag | kNonFiniteInputFlag | kInvalidDtFlag;

struct CudaLaunchConfig {
  int blocks{0};
  int threads{0};
};

template <typename Scalar>
struct CudaMatrixView {
  const Scalar* data{nullptr};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaMutableMatrixView {
  Scalar* data{nullptr};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaScalarBatchView {
  const Scalar* data{nullptr};
  int rank{0};
  std::int64_t stride{0};
};

struct CudaMutableBoolVectorView {
  std::uint8_t* data{nullptr};
  std::int64_t stride{0};
};

template <typename Scalar>
struct CudaVectorParameterView {
  const Scalar* data{nullptr};
  bool shared{false};
  int rank{0};
  std::int64_t env_stride{0};
  std::int64_t dim_stride{0};
};

template <typename Scalar>
__device__ Scalar read_matrix(const CudaMatrixView<Scalar> view,
                              const std::size_t env,
                              const std::size_t dim) {
  return view.data[env * static_cast<std::size_t>(view.row_stride) +
                   dim * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ void write_matrix(const CudaMutableMatrixView<Scalar> view,
                             const std::size_t env,
                             const std::size_t dim,
                             const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.row_stride) +
            dim * static_cast<std::size_t>(view.col_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_gain(const CudaVectorParameterView<Scalar> gain,
                            const std::size_t env,
                            const std::size_t dim) {
  if (gain.shared || gain.rank == 1) {
    return gain.data[dim * static_cast<std::size_t>(gain.dim_stride)];
  }
  return gain.data[env * static_cast<std::size_t>(gain.env_stride) +
                   dim * static_cast<std::size_t>(gain.dim_stride)];
}

template <typename Scalar>
__device__ Scalar read_scalar_batch(const CudaScalarBatchView<Scalar> view, const std::size_t env) {
  if (view.rank == 0) {
    return view.data[0];
  }
  return view.data[env * static_cast<std::size_t>(view.stride)];
}

template <typename Scalar>
__device__ Scalar read_vector_parameter(const CudaVectorParameterView<Scalar> parameter,
                                        const std::size_t env,
                                        const std::size_t dim,
                                        const Scalar default_value) {
  if (parameter.data == nullptr) {
    return default_value;
  }
  if (parameter.shared || parameter.rank == 1) {
    return parameter.data[dim * static_cast<std::size_t>(parameter.dim_stride)];
  }
  return parameter.data[env * static_cast<std::size_t>(parameter.env_stride) +
                        dim * static_cast<std::size_t>(parameter.dim_stride)];
}

template <typename Scalar>
__device__ bool is_finite(const Scalar value) {
  return isfinite(value);
}

template <typename Scalar>
__device__ constexpr Scalar scalar_max();

template <>
__device__ constexpr float scalar_max<float>() {
  return FLT_MAX;
}

template <>
__device__ constexpr double scalar_max<double>() {
  return DBL_MAX;
}

template <typename Scalar>
__global__ void apply_proportional_parameters_kernel(CudaVectorParameterView<Scalar> gain,
                                                     std::size_t num_envs,
                                                     std::size_t signal_dim,
                                                     std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!is_finite(read_gain(gain, env, dim))) {
      flags |= kInvalidParameterFlag;
      break;
    }
  }
  parameter_status[env] = flags;
}

__global__ void reset_proportional_kernel(const std::uint8_t* reset_mask,
                                          std::int64_t reset_mask_stride,
                                          std::size_t num_envs,
                                          std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool reset =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_mask_stride)] != 0U;
  status[env] = reset ? kResetAppliedFlag : 0U;
}

template <typename Scalar>
__global__ void step_proportional_kernel(CudaMatrixView<Scalar> input,
                                         CudaVectorParameterView<Scalar> gain,
                                         const std::uint64_t* parameter_status,
                                         CudaMutableMatrixView<Scalar> output,
                                         std::size_t num_envs,
                                         std::size_t signal_dim,
                                         std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!is_finite(read_gain(gain, env, dim))) {
      flags |= kInvalidParameterFlag;
    }
    if (!is_finite(read_matrix(input, env, dim))) {
      flags |= kNonFiniteInputFlag;
    }
  }

  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(
        output,
        env,
        dim,
        detail::proportional_value(read_gain(gain, env, dim), read_matrix(input, env, dim)));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void apply_integrator_parameters_kernel(CudaVectorParameterView<Scalar> u0,
                                                   CudaVectorParameterView<Scalar> upper,
                                                   CudaVectorParameterView<Scalar> lower,
                                                   std::size_t num_envs,
                                                   std::size_t signal_dim,
                                                   std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar u0_value = read_vector_parameter(u0, env, dim, static_cast<Scalar>(0));
    const Scalar upper_value = read_vector_parameter(upper, env, dim, scalar_max<Scalar>());
    const Scalar lower_value = read_vector_parameter(lower, env, dim, -scalar_max<Scalar>());
    if (!is_finite(u0_value) || !is_finite(upper_value) || !is_finite(lower_value) ||
        upper_value < lower_value) {
      flags |= kInvalidParameterFlag;
      break;
    }
  }
  parameter_status[env] = flags;
}

__global__ void clear_status_kernel(std::size_t num_envs, std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  status[env] = 0U;
}

template <typename Scalar>
__global__ void reset_integrator_kernel(CudaMutableMatrixView<Scalar> previous,
                                        CudaMutableMatrixView<Scalar> integrated,
                                        CudaMutableBoolVectorView initialized,
                                        const std::uint8_t* reset_mask,
                                        std::int64_t reset_mask_stride,
                                        std::size_t num_envs,
                                        std::size_t signal_dim,
                                        std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool reset =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_mask_stride)] != 0U;
  if (!reset) {
    status[env] = 0U;
    return;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(previous, env, dim, static_cast<Scalar>(0));
    write_matrix(integrated, env, dim, static_cast<Scalar>(0));
  }
  initialized.data[env * static_cast<std::size_t>(initialized.stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void reset_differentiator_kernel(CudaMutableMatrixView<Scalar> previous,
                                            CudaMutableBoolVectorView initialized,
                                            const std::uint8_t* reset_mask,
                                            std::int64_t reset_mask_stride,
                                            std::size_t num_envs,
                                            std::size_t signal_dim,
                                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool reset =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_mask_stride)] != 0U;
  if (!reset) {
    status[env] = 0U;
    return;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(previous, env, dim, static_cast<Scalar>(0));
  }
  initialized.data[env * static_cast<std::size_t>(initialized.stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void step_integrator_kernel(CudaMatrixView<Scalar> input,
                                       CudaScalarBatchView<Scalar> dt,
                                       CudaVectorParameterView<Scalar> u0,
                                       CudaVectorParameterView<Scalar> upper,
                                       CudaVectorParameterView<Scalar> lower,
                                       const std::uint64_t* parameter_status,
                                       CudaMutableMatrixView<Scalar> previous,
                                       CudaMutableMatrixView<Scalar> integrated,
                                       CudaMutableBoolVectorView initialized,
                                       CudaMutableMatrixView<Scalar> output,
                                       std::size_t num_envs,
                                       std::size_t signal_dim,
                                       std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const Scalar dt_value = read_scalar_batch(dt, env);
  if (!is_finite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar input_value = read_matrix(input, env, dim);
    const Scalar u0_value = read_vector_parameter(u0, env, dim, static_cast<Scalar>(0));
    const Scalar upper_value = read_vector_parameter(upper, env, dim, scalar_max<Scalar>());
    const Scalar lower_value = read_vector_parameter(lower, env, dim, -scalar_max<Scalar>());
    if (!is_finite(input_value)) {
      flags |= kNonFiniteInputFlag;
    }
    if (!is_finite(u0_value) || !is_finite(upper_value) || !is_finite(lower_value) ||
        upper_value < lower_value) {
      flags |= kInvalidParameterFlag;
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags;
    return;
  }

  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized.stride);
  if (initialized.data[initialized_offset] == 0U) {
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      const Scalar input_value = read_matrix(input, env, dim);
      write_matrix(previous, env, dim, input_value);
      write_matrix(
          integrated, env, dim, read_vector_parameter(u0, env, dim, static_cast<Scalar>(0)));
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    initialized.data[initialized_offset] = 1U;
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar upper_value = read_vector_parameter(upper, env, dim, scalar_max<Scalar>());
    const Scalar lower_value = read_vector_parameter(lower, env, dim, -scalar_max<Scalar>());
    const Scalar input_value = read_matrix(input, env, dim);
    const Scalar old_integrated = read_matrix(
        CudaMatrixView<Scalar>{integrated.data, integrated.row_stride, integrated.col_stride},
        env,
        dim);
    const Scalar old_previous = read_matrix(
        CudaMatrixView<Scalar>{previous.data, previous.row_stride, previous.col_stride}, env, dim);
    const auto update = detail::integrator_trapezoid_value(
        old_integrated, input_value, old_previous, dt_value, lower_value, upper_value);
    if (update.saturated) {
      flags |= kSaturationFlag;
    }
    write_matrix(integrated, env, dim, update.value);
    write_matrix(output, env, dim, update.value);
    write_matrix(previous, env, dim, input_value);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_differentiator_kernel(CudaMatrixView<Scalar> input,
                                           CudaScalarBatchView<Scalar> dt,
                                           const std::uint64_t* parameter_status,
                                           CudaMutableMatrixView<Scalar> previous,
                                           CudaMutableBoolVectorView initialized,
                                           CudaMutableMatrixView<Scalar> output,
                                           std::size_t num_envs,
                                           std::size_t signal_dim,
                                           std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const Scalar dt_value = read_scalar_batch(dt, env);
  if (!is_finite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!is_finite(read_matrix(input, env, dim))) {
      flags |= kNonFiniteInputFlag;
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags;
    return;
  }

  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized.stride);
  if (initialized.data[initialized_offset] == 0U) {
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      write_matrix(previous, env, dim, read_matrix(input, env, dim));
      write_matrix(output, env, dim, static_cast<Scalar>(0));
    }
    initialized.data[initialized_offset] = 1U;
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar input_value = read_matrix(input, env, dim);
    const Scalar previous_value = read_matrix(
        CudaMatrixView<Scalar>{previous.data, previous.row_stride, previous.col_stride}, env, dim);
    write_matrix(
        output, env, dim, detail::differentiator_value(input_value, previous_value, dt_value));
    write_matrix(previous, env, dim, input_value);
  }
  status[env] = flags;
}

inline Result cuda_launch_result() {
  const cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return Result{ResultCode::kInvalidArgument, "CUDA kernel launch failed"};
  }
  return Result{ResultCode::kOk, "ok"};
}

template <CudaTensorLaunchFamily Family, typename Scalar>
inline CudaLaunchConfig launch_config(const std::size_t num_envs) {
  constexpr int kThreads = cuda_tensor_tuned_threads<Family, Scalar>();
  return CudaLaunchConfig{static_cast<int>((num_envs + kThreads - 1) / kThreads), kThreads};
}

inline Result validate_cuda_status(const MutableStatusView& status, const std::size_t num_envs) {
  if (status.flags == nullptr) {
    return Result{ResultCode::kInvalidView, "status flags are null"};
  }
  if (status.size < num_envs) {
    return Result{ResultCode::kInvalidView, "status size is smaller than num_envs"};
  }
  if (status.device != Device::kCuda) {
    return Result{ResultCode::kInvalidView, "status must be a CUDA status view"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_status(const StatusView& status,
                                   const std::size_t num_envs,
                                   const bool allow_empty) {
  if (status.flags == nullptr) {
    return allow_empty ? Result{ResultCode::kOk, "ok"}
                       : Result{ResultCode::kInvalidView, "status flags are null"};
  }
  if (status.size < num_envs) {
    return Result{ResultCode::kInvalidView, "status size is smaller than num_envs"};
  }
  if (status.device != Device::kCuda) {
    return Result{ResultCode::kInvalidView, "status must be a CUDA status view"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline bool is_cuda_matrix(const TensorView& view) {
  return view.data != nullptr && view.device == Device::kCuda &&
         (view.dtype == DType::kFloat32 || view.dtype == DType::kFloat64) && view.rank == 2 &&
         view.layout == Layout::kRowMajor && view.strides[1] == 1 && view.shape[0] > 0 &&
         view.shape[1] > 0 && view.strides[0] >= view.shape[1];
}

inline bool is_cuda_matrix(const MutableTensorView& view) {
  return view.data != nullptr && view.device == Device::kCuda &&
         (view.dtype == DType::kFloat32 || view.dtype == DType::kFloat64) && view.rank == 2 &&
         view.layout == Layout::kRowMajor && view.strides[1] == 1 && view.shape[0] > 0 &&
         view.shape[1] > 0 && view.strides[0] >= view.shape[1];
}

inline Result validate_cuda_output(const MutableTensorView& output,
                                   const DType dtype,
                                   const std::size_t num_envs,
                                   const std::size_t signal_dim) {
  if (!is_cuda_matrix(output)) {
    return Result{ResultCode::kInvalidView,
                  "output tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  if (output.dtype != dtype || output.shape[0] != static_cast<std::int64_t>(num_envs) ||
      output.shape[1] != static_cast<std::int64_t>(signal_dim)) {
    return Result{ResultCode::kInvalidView, "output tensor shape or dtype does not match input"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_reset_mask(const TensorView& reset_mask, const std::size_t num_envs) {
  if (reset_mask.data == nullptr) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (reset_mask.device != Device::kCuda || reset_mask.dtype != DType::kBool ||
      reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs) ||
      reset_mask.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "reset mask must be CUDA bool [num_envs]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_scalar_batch(const TensorView& input,
                                         const char* name,
                                         const DType dtype,
                                         const std::size_t num_envs) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype) {
    return Result{ResultCode::kInvalidView, "scalar batch tensor must be CUDA and match dtype"};
  }
  if (input.rank == 0) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "scalar batch tensor must be scalar or [num_envs]"};
}

inline Result validate_cuda_gain(const ParameterView* gain,
                                 const DType dtype,
                                 const std::size_t num_envs,
                                 const std::size_t signal_dim) {
  if (gain == nullptr || gain->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "proportional gain parameter is missing"};
  }
  const TensorView& values = gain->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "gain parameter must be a CUDA tensor matching input dtype"};
  }
  if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
      values.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(signal_dim) && values.strides[1] == 1 &&
      values.strides[0] >= values.shape[1]) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView,
                "gain parameter must be [signal_dim] or [num_envs, signal_dim]"};
}

inline Result validate_cuda_vector_parameter(const ParameterView* parameter,
                                             const char* missing_message,
                                             const DType dtype,
                                             const std::size_t num_envs,
                                             const std::size_t signal_dim,
                                             const bool required) {
  if (parameter == nullptr) {
    return required ? Result{ResultCode::kInvalidView, missing_message}
                    : Result{ResultCode::kOk, "ok"};
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "parameter data is null"};
  }
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "vector parameter must be a CUDA tensor matching component dtype"};
  }
  if (parameter->shared) {
    if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
        values.strides[0] >= 1) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView, "shared vector parameter must have [signal_dim]"};
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(signal_dim) && values.strides[1] == 1 &&
      values.strides[0] >= values.shape[1]) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView,
                "batched vector parameter must have [num_envs, signal_dim]"};
}

inline Result validate_cuda_parameter_set_devices(const ParameterSetView& parameters) {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    const TensorView& values = parameters.parameters[i].values;
    if (values.data != nullptr && values.device != Device::kCuda) {
      return Result{ResultCode::kInvalidView, "CUDA execution requires CUDA parameter tensors"};
    }
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_state_matrix(const MutableTensorView& values,
                                         const char* message,
                                         const DType dtype,
                                         const std::size_t num_envs,
                                         const std::size_t signal_dim) {
  if (!is_cuda_matrix(values)) {
    return Result{ResultCode::kInvalidView, message};
  }
  if (values.dtype != dtype || values.shape[0] != static_cast<std::int64_t>(num_envs) ||
      values.shape[1] != static_cast<std::int64_t>(signal_dim)) {
    return Result{ResultCode::kInvalidView, "state tensor shape or dtype does not match input"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_initialized_state(const MutableTensorView& values,
                                              const std::size_t num_envs) {
  if (values.data == nullptr || values.device != Device::kCuda || values.dtype != DType::kBool ||
      values.rank != 1 || values.shape[0] != static_cast<std::int64_t>(num_envs) ||
      values.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "initialized state must be CUDA bool [num_envs]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_integrator_state(StateView& state,
                                             const DType dtype,
                                             const std::size_t num_envs,
                                             const std::size_t signal_dim) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous == nullptr || integrated == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView,
                  "integrator state requires previous_signal, integrated_signal, initialized"};
  }
  Result result = validate_cuda_state_matrix(previous->values,
                                             "previous_signal must be CUDA [num_envs, signal_dim]",
                                             dtype,
                                             num_envs,
                                             signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_matrix(integrated->values,
                                      "integrated_signal must be CUDA [num_envs, signal_dim]",
                                      dtype,
                                      num_envs,
                                      signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_initialized_state(initialized->values, num_envs);
}

inline Result validate_cuda_differentiator_state(StateView& state,
                                                 const DType dtype,
                                                 const std::size_t num_envs,
                                                 const std::size_t signal_dim) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView,
                  "differentiator state requires previous_signal and initialized"};
  }
  Result result = validate_cuda_state_matrix(previous->values,
                                             "previous_signal must be CUDA [num_envs, signal_dim]",
                                             dtype,
                                             num_envs,
                                             signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_initialized_state(initialized->values, num_envs);
}

template <typename Scalar>
CudaMatrixView<Scalar> as_cuda_matrix(const TensorView& view) {
  return CudaMatrixView<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaMutableMatrixView<Scalar> as_cuda_mutable_matrix(const MutableTensorView& view) {
  return CudaMutableMatrixView<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaScalarBatchView<Scalar> as_cuda_scalar_batch(const TensorView& view) {
  return CudaScalarBatchView<Scalar>{
      static_cast<const Scalar*>(view.data), view.rank, view.rank == 0 ? 0 : view.strides[0]};
}

inline CudaMutableBoolVectorView as_cuda_mutable_bool_vector(const MutableTensorView& view) {
  return CudaMutableBoolVectorView{static_cast<std::uint8_t*>(view.data), view.strides[0]};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> as_cuda_gain(const ParameterView& gain) {
  const TensorView& values = gain.values;
  const bool shared = gain.shared || values.rank == 1;
  return CudaVectorParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         shared,
                                         values.rank,
                                         values.rank == 2 ? values.strides[0] : 0,
                                         values.rank == 2 ? values.strides[1] : values.strides[0]};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> as_cuda_vector_parameter(const ParameterView* parameter) {
  if (parameter == nullptr) {
    return CudaVectorParameterView<Scalar>{};
  }
  const TensorView& values = parameter->values;
  const bool shared = parameter->shared || values.rank == 1;
  return CudaVectorParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         shared,
                                         values.rank,
                                         values.rank == 2 ? values.strides[0] : 0,
                                         values.rank == 2 ? values.strides[1] : values.strides[0]};
}

template <typename Scalar>
Result launch_apply(const ParameterView& gain,
                    const std::size_t num_envs,
                    const std::size_t signal_dim,
                    MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericApplyProportional, Scalar>(num_envs);
  apply_proportional_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_gain<Scalar>(gain), num_envs, signal_dim, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_clear_status(const std::size_t num_envs, MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericClearStatus, Scalar>(num_envs);
  clear_status_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(num_envs,
                                                                                  status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_integrator(const ParameterSetView& parameters,
                               const std::size_t num_envs,
                               const std::size_t signal_dim,
                               MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericApplyIntegrator, Scalar>(num_envs);
  apply_integrator_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kU0)),
          as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kUpperLimit)),
          as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kLowerLimit)),
          num_envs,
          signal_dim,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step(const TensorView& input_signal,
                   const ParameterView& gain,
                   StatusView parameter_status,
                   const MutableTensorView& output_signal,
                   MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericStepProportional, Scalar>(num_envs);
  step_proportional_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(input_signal),
      as_cuda_gain<Scalar>(gain),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(output_signal),
      num_envs,
      signal_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_integrator(const std::size_t num_envs,
                               const std::size_t signal_dim,
                               StateView& state,
                               const TensorView& reset_mask,
                               MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericResetIntegrator, Scalar>(num_envs);
  reset_integrator_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_mutable_matrix<Scalar>(previous->values),
      as_cuda_mutable_matrix<Scalar>(integrated->values),
      as_cuda_mutable_bool_vector(initialized->values),
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      signal_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_differentiator(const std::size_t num_envs,
                                   const std::size_t signal_dim,
                                   StateView& state,
                                   const TensorView& reset_mask,
                                   MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericResetDifferentiator, Scalar>(num_envs);
  reset_differentiator_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_mutable_matrix<Scalar>(previous->values),
      as_cuda_mutable_bool_vector(initialized->values),
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      signal_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_integrator(const TensorView& input_signal,
                              const TensorView& dt,
                              const ParameterSetView& parameters,
                              StatusView parameter_status,
                              StateView& state,
                              const MutableTensorView& output_signal,
                              MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* integrated = find_state(state, kIntegratedSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericStepIntegrator, Scalar>(num_envs);
  step_integrator_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(input_signal),
      as_cuda_scalar_batch<Scalar>(dt),
      as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kU0)),
      as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kUpperLimit)),
      as_cuda_vector_parameter<Scalar>(find_parameter(parameters, kLowerLimit)),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(previous->values),
      as_cuda_mutable_matrix<Scalar>(integrated->values),
      as_cuda_mutable_bool_vector(initialized->values),
      as_cuda_mutable_matrix<Scalar>(output_signal),
      num_envs,
      signal_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_differentiator(const TensorView& input_signal,
                                  const TensorView& dt,
                                  StatusView parameter_status,
                                  StateView& state,
                                  const MutableTensorView& output_signal,
                                  MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  StateEntryView* initialized = find_state(state, kInitialized);
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kGenericStepDifferentiator, Scalar>(num_envs);
  step_differentiator_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(input_signal),
      as_cuda_scalar_batch<Scalar>(dt),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(previous->values),
      as_cuda_mutable_bool_vector(initialized->values),
      as_cuda_mutable_matrix<Scalar>(output_signal),
      num_envs,
      signal_dim,
      status.flags);
  return cuda_launch_result();
}

}  // namespace

Result apply_generic_proportional_block_parameters_cuda(const ParameterSetView& parameters,
                                                        const std::size_t num_envs,
                                                        const std::size_t signal_dim,
                                                        MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* gain = find_parameter(parameters, kGain);
  const DType dtype = gain == nullptr ? DType::kUnknown : gain->values.dtype;
  result = validate_cuda_gain(gain, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply<float>(*gain, num_envs, signal_dim, parameter_status)
             : launch_apply<double>(*gain, num_envs, signal_dim, parameter_status);
}

Result reset_generic_proportional_block_cuda(const std::size_t num_envs,
                                             const TensorView& reset_mask,
                                             MutableStatusView status) {
  Result result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  constexpr int kThreads =
      cuda_tensor_tuned_threads<CudaTensorLaunchFamily::kGenericResetProportional,
                                DType::kUnknown>();
  const CudaLaunchConfig config{
      static_cast<int>((num_envs + static_cast<std::size_t>(kThreads) - 1U) /
                       static_cast<std::size_t>(kThreads)),
      kThreads};
  reset_proportional_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      status.flags);
  return cuda_launch_result();
}

Result step_generic_proportional_block_cuda(const TensorView& input_signal,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            const MutableTensorView& output_signal,
                                            MutableStatusView status) {
  if (!is_cuda_matrix(input_signal)) {
    return Result{ResultCode::kInvalidView,
                  "input tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  Result result = validate_cuda_output(output_signal, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* gain = find_parameter(parameters, kGain);
  result = validate_cuda_gain(gain, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return input_signal.dtype == DType::kFloat32
             ? launch_step<float>(input_signal, *gain, parameter_status, output_signal, status)
             : launch_step<double>(input_signal, *gain, parameter_status, output_signal, status);
}

Result apply_generic_integrator_block_parameters_cuda(const ParameterSetView& parameters,
                                                      const std::size_t num_envs,
                                                      const std::size_t signal_dim,
                                                      MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_parameter_set_devices(parameters);
  if (!result.ok()) {
    return result;
  }
  DType dtype = DType::kFloat64;
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.dtype == DType::kFloat32 ||
        parameters.parameters[i].values.dtype == DType::kFloat64) {
      dtype = parameters.parameters[i].values.dtype;
      break;
    }
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kU0),
                                          "u_0 parameter is missing",
                                          dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kUpperLimit),
                                          "upper_limit parameter is missing",
                                          dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kLowerLimit),
                                          "lower_limit parameter is missing",
                                          dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_integrator<float>(parameters, num_envs, signal_dim, parameter_status)
             : launch_apply_integrator<double>(parameters, num_envs, signal_dim, parameter_status);
}

Result reset_generic_integrator_block_cuda(const std::size_t num_envs,
                                           StateView& state,
                                           const TensorView& reset_mask,
                                           MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  if (previous == nullptr) {
    return Result{ResultCode::kInvalidView, "integrator state requires previous_signal"};
  }
  const DType dtype = previous->values.dtype;
  const std::size_t signal_dim =
      previous->values.shape[1] > 0 ? static_cast<std::size_t>(previous->values.shape[1]) : 0U;
  Result result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_integrator_state(state, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_integrator<float>(num_envs, signal_dim, state, reset_mask, status)
             : launch_reset_integrator<double>(num_envs, signal_dim, state, reset_mask, status);
}

Result step_generic_integrator_block_cuda(const TensorView& input_signal,
                                          const TensorView& dt,
                                          const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          StateView& state,
                                          const MutableTensorView& output_signal,
                                          MutableStatusView status) {
  if (!is_cuda_matrix(input_signal)) {
    return Result{ResultCode::kInvalidView,
                  "input tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  Result result = validate_cuda_output(output_signal, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch(dt, "dt tensor data is null", input_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_parameter_set_devices(parameters);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kU0),
                                          "u_0 parameter is missing",
                                          input_signal.dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kUpperLimit),
                                          "upper_limit parameter is missing",
                                          input_signal.dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, kLowerLimit),
                                          "lower_limit parameter is missing",
                                          input_signal.dtype,
                                          num_envs,
                                          signal_dim,
                                          false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_integrator_state(state, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return input_signal.dtype == DType::kFloat32
             ? launch_step_integrator<float>(
                   input_signal, dt, parameters, parameter_status, state, output_signal, status)
             : launch_step_integrator<double>(
                   input_signal, dt, parameters, parameter_status, state, output_signal, status);
}

Result apply_generic_differentiator_block_parameters_cuda(const std::size_t num_envs,
                                                          MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  return launch_clear_status<float>(num_envs, parameter_status);
}

Result reset_generic_differentiator_block_cuda(const std::size_t num_envs,
                                               StateView& state,
                                               const TensorView& reset_mask,
                                               MutableStatusView status) {
  StateEntryView* previous = find_state(state, kPreviousSignal);
  if (previous == nullptr) {
    return Result{ResultCode::kInvalidView, "differentiator state requires previous_signal"};
  }
  const DType dtype = previous->values.dtype;
  const std::size_t signal_dim =
      previous->values.shape[1] > 0 ? static_cast<std::size_t>(previous->values.shape[1]) : 0U;
  Result result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_differentiator_state(state, dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_differentiator<float>(num_envs, signal_dim, state, reset_mask, status)
             : launch_reset_differentiator<double>(num_envs, signal_dim, state, reset_mask, status);
}

Result step_generic_differentiator_block_cuda(const TensorView& input_signal,
                                              const TensorView& dt,
                                              StatusView parameter_status,
                                              StateView& state,
                                              const MutableTensorView& output_signal,
                                              MutableStatusView status) {
  if (!is_cuda_matrix(input_signal)) {
    return Result{ResultCode::kInvalidView,
                  "input tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(input_signal.shape[0]);
  const std::size_t signal_dim = static_cast<std::size_t>(input_signal.shape[1]);
  Result result = validate_cuda_output(output_signal, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch(dt, "dt tensor data is null", input_signal.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_differentiator_state(state, input_signal.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return input_signal.dtype == DType::kFloat32
             ? launch_step_differentiator<float>(
                   input_signal, dt, parameter_status, state, output_signal, status)
             : launch_step_differentiator<double>(
                   input_signal, dt, parameter_status, state, output_signal, status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
