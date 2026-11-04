/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file operational_space_mass_cuda.cu
 * @brief CUDA tensor kernel for operational-space mass calculation.
 */

#include "tensor/operational_space_mass_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"
#include "nvidia/srl/controllers/tensor/operational_space_mass.hpp"

#include "tensor/cuda_stream.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 64;
constexpr const char* kRegularization = "regularization";
constexpr const char* kMinimumPivot = "minimum_pivot";
constexpr const char* kJointMassFactor = "joint_mass_factor";
constexpr const char* kMassInverseJacobianTranspose = "mass_inverse_jacobian_transpose";
constexpr const char* kCartesianInverseMassFactor = "cartesian_inverse_mass_factor";
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);

template <typename Scalar>
struct CudaTensor3View {
  Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaConstTensor3View {
  const Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaScalarParameter {
  const Scalar* data{nullptr};
  bool shared{false};
  std::int64_t stride{0};
};

template <typename Scalar>
CudaConstTensor3View<Scalar> cuda_view(const TensorView& view) noexcept {
  return CudaConstTensor3View<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaTensor3View<Scalar> cuda_view(const MutableTensorView& view) noexcept {
  return CudaTensor3View<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaScalarParameter<Scalar> cuda_parameter(const ParameterView* parameter) noexcept {
  if (parameter == nullptr) {
    return CudaScalarParameter<Scalar>{};
  }
  return CudaScalarParameter<Scalar>{
      static_cast<const Scalar*>(parameter->values.data),
      parameter->shared,
      parameter->values.rank == 0 ? 0 : parameter->values.strides[0]};
}

template <typename Scalar>
__device__ Scalar read_tensor3(const CudaConstTensor3View<Scalar> view,
                               const std::size_t env,
                               const std::size_t row,
                               const std::size_t col) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   row * static_cast<std::size_t>(view.row_stride) +
                   col * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ Scalar read_tensor3(const CudaTensor3View<Scalar> view,
                               const std::size_t env,
                               const std::size_t row,
                               const std::size_t col) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   row * static_cast<std::size_t>(view.row_stride) +
                   col * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ void write_tensor3(const CudaTensor3View<Scalar> view,
                              const std::size_t env,
                              const std::size_t row,
                              const std::size_t col,
                              const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.env_stride) +
            row * static_cast<std::size_t>(view.row_stride) +
            col * static_cast<std::size_t>(view.col_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_parameter(const CudaScalarParameter<Scalar> parameter,
                                 const std::size_t env,
                                 const Scalar default_value) {
  if (parameter.data == nullptr) {
    return default_value;
  }
  return parameter.data[parameter.shared ? 0 : env * static_cast<std::size_t>(parameter.stride)];
}

template <typename Scalar>
__device__ constexpr Scalar default_minimum_pivot();

template <>
__device__ constexpr float default_minimum_pivot<float>() {
  return 1.0e-6F;
}

template <>
__device__ constexpr double default_minimum_pivot<double>() {
  return 1.0e-12;
}

template <typename Scalar>
__device__ bool parameters_are_valid(const CudaScalarParameter<Scalar> regularization,
                                     const CudaScalarParameter<Scalar> minimum_pivot,
                                     const std::size_t env) {
  const Scalar regularization_value = read_parameter(regularization, env, static_cast<Scalar>(-1));
  const Scalar minimum_pivot_value =
      read_parameter(minimum_pivot, env, default_minimum_pivot<Scalar>());
  return detail::operational_space_mass_parameters_valid(regularization_value, minimum_pivot_value);
}

template <typename Scalar>
__global__ void apply_parameters_kernel(CudaScalarParameter<Scalar> regularization,
                                        CudaScalarParameter<Scalar> minimum_pivot,
                                        std::size_t num_envs,
                                        std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env < num_envs) {
    status[env] =
        parameters_are_valid(regularization, minimum_pivot, env) ? 0U : kInvalidParameterFlag;
  }
}

__global__ void reset_kernel(const std::uint8_t* reset_mask,
                             std::int64_t reset_stride,
                             std::size_t num_envs,
                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool selected =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_stride)] != 0U;
  status[env] = selected ? kResetAppliedFlag : 0U;
}

template <typename Scalar>
__device__ void zero_output(const CudaTensor3View<Scalar> output,
                            const std::size_t env,
                            const std::size_t task_dim) {
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < task_dim; ++col) {
      write_tensor3(output, env, row, col, static_cast<Scalar>(0));
    }
  }
}

template <typename Scalar>
__global__ void step_kernel(CudaConstTensor3View<Scalar> mass,
                            CudaConstTensor3View<Scalar> jacobian,
                            CudaScalarParameter<Scalar> regularization,
                            CudaScalarParameter<Scalar> minimum_pivot,
                            const std::uint64_t* parameter_status,
                            CudaTensor3View<Scalar> joint_factor,
                            CudaTensor3View<Scalar> inverse_jacobian,
                            CudaTensor3View<Scalar> cartesian_factor,
                            CudaTensor3View<Scalar> output,
                            std::size_t num_envs,
                            std::size_t joint_dim,
                            std::size_t task_dim,
                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!parameters_are_valid(regularization, minimum_pivot, env)) {
    flags |= kInvalidParameterFlag;
  }
  for (std::size_t row = 0; row < joint_dim && flags == 0U; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!isfinite(read_tensor3(mass, env, row, col))) {
        flags |= kNonFiniteInputFlag;
        break;
      }
    }
  }
  for (std::size_t row = 0; row < task_dim && flags == 0U; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!isfinite(read_tensor3(jacobian, env, row, col))) {
        flags |= kNonFiniteInputFlag;
        break;
      }
    }
  }
  if (flags != 0U) {
    zero_output(output, env, task_dim);
    status[env] = flags | kSafeModeFlag;
    return;
  }

  const Scalar pivot = read_parameter(minimum_pivot, env, default_minimum_pivot<Scalar>());
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      const Scalar forward = read_tensor3(mass, env, row, col);
      const Scalar transpose = read_tensor3(mass, env, col, row);
      if (!detail::symmetric_pair_is_valid(forward, transpose, pivot)) {
        flags |= kInvalidInputFlag;
        break;
      }
      Scalar value = detail::symmetric_average(forward, transpose);
      for (std::size_t inner = 0; inner < col; ++inner) {
        value -= read_tensor3(joint_factor, env, row, inner) *
                 read_tensor3(joint_factor, env, col, inner);
      }
      if (row == col) {
        if (!isfinite(value) || value <= pivot) {
          flags |= kInvalidInputFlag;
          break;
        }
        write_tensor3(joint_factor, env, row, col, sqrt(value));
      } else {
        write_tensor3(
            joint_factor, env, row, col, value / read_tensor3(joint_factor, env, col, col));
      }
    }
    if (flags != 0U) {
      break;
    }
    for (std::size_t col = row + 1; col < joint_dim; ++col) {
      write_tensor3(joint_factor, env, row, col, static_cast<Scalar>(0));
    }
  }
  if (flags != 0U) {
    zero_output(output, env, task_dim);
    status[env] = flags | kSafeModeFlag;
    return;
  }

  for (std::size_t cartesian_col = 0; cartesian_col < task_dim; ++cartesian_col) {
    for (std::size_t row = 0; row < joint_dim; ++row) {
      Scalar value = read_tensor3(jacobian, env, cartesian_col, row);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -= read_tensor3(joint_factor, env, row, inner) *
                 read_tensor3(inverse_jacobian, env, inner, cartesian_col);
      }
      write_tensor3(inverse_jacobian,
                    env,
                    row,
                    cartesian_col,
                    value / read_tensor3(joint_factor, env, row, row));
    }
    for (std::size_t reverse = joint_dim; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      Scalar value = read_tensor3(inverse_jacobian, env, row, cartesian_col);
      for (std::size_t inner = row + 1; inner < joint_dim; ++inner) {
        value -= read_tensor3(joint_factor, env, inner, row) *
                 read_tensor3(inverse_jacobian, env, inner, cartesian_col);
      }
      write_tensor3(inverse_jacobian,
                    env,
                    row,
                    cartesian_col,
                    value / read_tensor3(joint_factor, env, row, row));
    }
  }

  const Scalar regularization_value = read_parameter(regularization, env, static_cast<Scalar>(-1));
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      Scalar forward = static_cast<Scalar>(0);
      Scalar transpose = static_cast<Scalar>(0);
      for (std::size_t joint = 0; joint < joint_dim; ++joint) {
        forward += read_tensor3(jacobian, env, row, joint) *
                   read_tensor3(inverse_jacobian, env, joint, col);
        transpose += read_tensor3(jacobian, env, col, joint) *
                     read_tensor3(inverse_jacobian, env, joint, row);
      }
      Scalar value = detail::operational_space_inverse_mass_value(
          forward, transpose, regularization_value, row == col);
      for (std::size_t inner = 0; inner < col; ++inner) {
        value -= read_tensor3(cartesian_factor, env, row, inner) *
                 read_tensor3(cartesian_factor, env, col, inner);
      }
      if (row == col) {
        if (!isfinite(value) || value <= pivot) {
          flags |= kInvalidInputFlag;
          break;
        }
        write_tensor3(cartesian_factor, env, row, col, sqrt(value));
      } else {
        write_tensor3(
            cartesian_factor, env, row, col, value / read_tensor3(cartesian_factor, env, col, col));
      }
    }
    if (flags != 0U) {
      break;
    }
    for (std::size_t col = row + 1; col < task_dim; ++col) {
      write_tensor3(cartesian_factor, env, row, col, static_cast<Scalar>(0));
    }
  }
  if (flags != 0U) {
    zero_output(output, env, task_dim);
    status[env] = flags | kSafeModeFlag;
    return;
  }

  for (std::size_t col = 0; col < task_dim; ++col) {
    for (std::size_t row = 0; row < task_dim; ++row) {
      Scalar value = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -=
            read_tensor3(cartesian_factor, env, row, inner) * read_tensor3(output, env, inner, col);
      }
      write_tensor3(output, env, row, col, value / read_tensor3(cartesian_factor, env, row, row));
    }
    for (std::size_t reverse = task_dim; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      Scalar value = read_tensor3(output, env, row, col);
      for (std::size_t inner = row + 1; inner < task_dim; ++inner) {
        value -=
            read_tensor3(cartesian_factor, env, inner, row) * read_tensor3(output, env, inner, col);
      }
      value /= read_tensor3(cartesian_factor, env, row, row);
      if (!isfinite(value)) {
        flags |= kInvalidInputFlag;
        break;
      }
      write_tensor3(output, env, row, col, value);
    }
    if (flags != 0U) {
      break;
    }
  }
  if (flags != 0U) {
    zero_output(output, env, task_dim);
    status[env] = flags | kSafeModeFlag;
    return;
  }
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < row; ++col) {
      const Scalar symmetric = detail::symmetric_average(read_tensor3(output, env, row, col),
                                                         read_tensor3(output, env, col, row));
      write_tensor3(output, env, row, col, symmetric);
      write_tensor3(output, env, col, row, symmetric);
    }
  }
  status[env] = 0U;
}

Result launch_result() noexcept {
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{ResultCode::kOk, "ok"}
             : Result{ResultCode::kInvalidArgument, "CUDA operational-space kernel launch failed"};
}

template <typename Scalar>
Result launch_apply(const ParameterSetView& parameters,
                    const std::size_t num_envs,
                    const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  apply_parameters_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      cuda_parameter<Scalar>(find_parameter(parameters, kRegularization)),
      cuda_parameter<Scalar>(find_parameter(parameters, kMinimumPivot)),
      num_envs,
      status.flags);
  return launch_result();
}

template <typename Scalar>
Result launch_step(const TensorView& joint_mass_matrix,
                   const TensorView& jacobian,
                   const ParameterSetView& parameters,
                   const StatusView parameter_status,
                   ScratchView& scratch,
                   const MutableTensorView& output,
                   const MutableStatusView status) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(joint_mass_matrix.shape[0]);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  step_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      cuda_view<Scalar>(joint_mass_matrix),
      cuda_view<Scalar>(jacobian),
      cuda_parameter<Scalar>(find_parameter(parameters, kRegularization)),
      cuda_parameter<Scalar>(find_parameter(parameters, kMinimumPivot)),
      parameter_status.flags,
      cuda_view<Scalar>(find_scratch(scratch, kJointMassFactor)->values),
      cuda_view<Scalar>(find_scratch(scratch, kMassInverseJacobianTranspose)->values),
      cuda_view<Scalar>(find_scratch(scratch, kCartesianInverseMassFactor)->values),
      cuda_view<Scalar>(output),
      num_envs,
      static_cast<std::size_t>(joint_mass_matrix.shape[1]),
      static_cast<std::size_t>(jacobian.shape[1]),
      status.flags);
  return launch_result();
}

}  // namespace

Result apply_operational_space_mass_parameters_cuda(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    const MutableStatusView parameter_status) noexcept {
  const ParameterView* regularization = find_parameter(parameters, kRegularization);
  return regularization->values.dtype == DType::kFloat32
             ? launch_apply<float>(parameters, num_envs, parameter_status)
             : launch_apply<double>(parameters, num_envs, parameter_status);
}

Result reset_operational_space_mass_cuda(const std::size_t num_envs,
                                         const TensorView& reset_mask,
                                         const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  reset_kernel<<<blocks, kThreads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      status.flags);
  return launch_result();
}

Result step_operational_space_mass_cuda(const TensorView& joint_mass_matrix,
                                        const TensorView& jacobian,
                                        const ParameterSetView& parameters,
                                        const StatusView parameter_status,
                                        ScratchView& scratch,
                                        const MutableTensorView& cartesian_mass_matrix,
                                        const MutableStatusView status) noexcept {
  return joint_mass_matrix.dtype == DType::kFloat32 ? launch_step<float>(joint_mass_matrix,
                                                                         jacobian,
                                                                         parameters,
                                                                         parameter_status,
                                                                         scratch,
                                                                         cartesian_mass_matrix,
                                                                         status)
                                                    : launch_step<double>(joint_mass_matrix,
                                                                          jacobian,
                                                                          parameters,
                                                                          parameter_status,
                                                                          scratch,
                                                                          cartesian_mass_matrix,
                                                                          status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
