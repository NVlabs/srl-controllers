/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_impedance_cuda.cu
 * @brief CUDA backend for bounded variable-dimension task impedance.
 */

#include "tensor/task_impedance_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/cuda_stream.hpp"
#include "tensor/symmetric_matrix_policy.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 32;
constexpr std::size_t kMaxTaskDim = 12;
constexpr std::size_t kMatrixSize = kMaxTaskDim * kMaxTaskDim;
constexpr std::size_t kWorkspaceMatrixCount = 15;
constexpr std::size_t kWorkspaceVectorCount = 7;
constexpr const char* kStiffnessMatrix = "stiffness_matrix";
constexpr const char* kDampingFactors = "damping_factors";
constexpr const char* kPreviousMassMatrix = "previous_mass_matrix";
constexpr const char* kPreviousJacobian = "previous_jacobian";
constexpr const char* kInitialized = "initialized";
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDtFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kSafeTriggers =
    kInvalidParameterFlag | kInvalidInputFlag | kNonFiniteInputFlag | kInvalidDtFlag;

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
struct CudaTensor3View {
  const Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaMutableTensor3View {
  Scalar* data{nullptr};
  std::int64_t env_stride{0};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaScalarBatchView {
  const Scalar* data{nullptr};
  int rank{0};
  std::int64_t env_stride{0};
};

template <typename Scalar>
struct CudaMatrixParameterView {
  const Scalar* data{nullptr};
  bool shared{false};
  int rank{0};
  std::int64_t env_stride{0};
  std::int64_t row_stride{0};
  std::int64_t col_stride{0};
};

template <typename Scalar>
struct CudaVectorParameterView {
  const Scalar* data{nullptr};
  bool shared{false};
  int rank{0};
  std::int64_t env_stride{0};
  std::int64_t value_stride{0};
};

template <typename Scalar>
struct TaskWorkspace {
  Scalar matrices[kWorkspaceMatrixCount][kMatrixSize];
  Scalar vectors[kWorkspaceVectorCount][kMaxTaskDim];
  std::uint64_t flags;
};

template <typename Scalar>
CudaMatrixView<Scalar> cuda_matrix(const TensorView& view) noexcept {
  return CudaMatrixView<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaMutableMatrixView<Scalar> cuda_matrix(const MutableTensorView& view) noexcept {
  return CudaMutableMatrixView<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaTensor3View<Scalar> cuda_tensor3(const TensorView& view) noexcept {
  return CudaTensor3View<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaMutableTensor3View<Scalar> cuda_tensor3(const MutableTensorView& view) noexcept {
  return CudaMutableTensor3View<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaScalarBatchView<Scalar> cuda_scalar_batch(const TensorView& view) noexcept {
  return CudaScalarBatchView<Scalar>{
      static_cast<const Scalar*>(view.data), view.rank, view.rank == 0 ? 0 : view.strides[0]};
}

template <typename Scalar>
CudaMatrixParameterView<Scalar> cuda_matrix_parameter(const ParameterView* parameter) noexcept {
  const TensorView& view = parameter->values;
  return CudaMatrixParameterView<Scalar>{static_cast<const Scalar*>(view.data),
                                         parameter->shared,
                                         view.rank,
                                         view.rank == 3 ? view.strides[0] : 0,
                                         view.rank == 3 ? view.strides[1] : view.strides[0],
                                         view.rank == 3 ? view.strides[2] : view.strides[1]};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> cuda_vector_parameter(const ParameterView* parameter) noexcept {
  const TensorView& view = parameter->values;
  return CudaVectorParameterView<Scalar>{static_cast<const Scalar*>(view.data),
                                         parameter->shared,
                                         view.rank,
                                         view.rank == 2 ? view.strides[0] : 0,
                                         view.rank == 2 ? view.strides[1] : view.strides[0]};
}

template <typename Scalar>
__device__ Scalar read_matrix(const CudaMatrixView<Scalar> view,
                              const std::size_t env,
                              const std::size_t col) {
  return view.data[env * static_cast<std::size_t>(view.row_stride) +
                   col * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ void write_matrix(const CudaMutableMatrixView<Scalar> view,
                             const std::size_t env,
                             const std::size_t col,
                             const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.row_stride) +
            col * static_cast<std::size_t>(view.col_stride)] = value;
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
__device__ Scalar read_tensor3(const CudaMutableTensor3View<Scalar> view,
                               const std::size_t env,
                               const std::size_t row,
                               const std::size_t col) {
  return view.data[env * static_cast<std::size_t>(view.env_stride) +
                   row * static_cast<std::size_t>(view.row_stride) +
                   col * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ void write_tensor3(const CudaMutableTensor3View<Scalar> view,
                              const std::size_t env,
                              const std::size_t row,
                              const std::size_t col,
                              const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.env_stride) +
            row * static_cast<std::size_t>(view.row_stride) +
            col * static_cast<std::size_t>(view.col_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_scalar(const CudaScalarBatchView<Scalar> view, const std::size_t env) {
  return view.data[view.rank == 0 ? 0 : env * static_cast<std::size_t>(view.env_stride)];
}

template <typename Scalar>
__device__ Scalar read_parameter(const CudaMatrixParameterView<Scalar> view,
                                 const std::size_t env,
                                 const std::size_t row,
                                 const std::size_t col) {
  const std::size_t env_offset =
      view.shared || view.rank == 2 ? 0 : env * static_cast<std::size_t>(view.env_stride);
  return view.data[env_offset + row * static_cast<std::size_t>(view.row_stride) +
                   col * static_cast<std::size_t>(view.col_stride)];
}

template <typename Scalar>
__device__ Scalar read_parameter(const CudaVectorParameterView<Scalar> view,
                                 const std::size_t env,
                                 const std::size_t index) {
  const std::size_t env_offset =
      view.shared || view.rank == 1 ? 0 : env * static_cast<std::size_t>(view.env_stride);
  return view.data[env_offset + index * static_cast<std::size_t>(view.value_stride)];
}

__device__ void set_flag(std::uint64_t* flags, const std::uint64_t flag) {
  // CUDA exposes its 64-bit atomic OR through the unsigned long long overload.
  // NOLINTNEXTLINE(runtime/int)
  atomicOr(reinterpret_cast<unsigned long long*>(flags), static_cast<unsigned long long>(flag));
}

template <typename Scalar>
__device__ void set_identity(Scalar* matrix, const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      matrix[row * dim + col] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
__device__ void transpose(const Scalar* input, Scalar* output, const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      output[row * dim + col] = input[col * dim + row];
    }
  }
}

template <typename Scalar>
__device__ void multiply(const Scalar* left,
                         const Scalar* right,
                         Scalar* output,
                         const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += left[row * dim + inner] * right[inner * dim + col];
      }
      output[row * dim + col] = value;
    }
  }
}

template <typename Scalar>
__device__ bool validate_and_symmetrize(Scalar* matrix, const std::size_t dim) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t index = 0; index < dim * dim; ++index) {
    if (!isfinite(matrix[index])) {
      return false;
    }
    scale = fmax(scale, fabs(matrix[index]));
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(dim) * scale;
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar forward = matrix[row * dim + col];
      const Scalar transpose_value = matrix[col * dim + row];
      if (fabs(forward - transpose_value) > tolerance) {
        return false;
      }
      const Scalar value = static_cast<Scalar>(0.5) * (forward + transpose_value);
      matrix[row * dim + col] = value;
      matrix[col * dim + row] = value;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool symmetric_eigen(const Scalar* input,
                                Scalar* eigenvectors,
                                Scalar* eigenvalues,
                                Scalar* matrix_scale,
                                Scalar* matrix,
                                const std::size_t dim) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t index = 0; index < dim * dim; ++index) {
    if (!isfinite(input[index])) {
      return false;
    }
    scale = fmax(scale, fabs(input[index]));
  }
  if (!isfinite(scale)) {
    return false;
  }
  scale = scale == static_cast<Scalar>(0) ? static_cast<Scalar>(1) : scale;
  for (std::size_t index = 0; index < dim * dim; ++index) {
    matrix[index] = input[index] / scale;
  }
  set_identity(eigenvectors, dim);
  bool has_converged = false;
  const std::size_t maximum_iterations = 16 * dim * dim;
  for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = dim > 1 ? 1 : 0;
    Scalar maximum = dim > 1 ? fabs(matrix[1]) : static_cast<Scalar>(0);
    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = row + 1; col < dim; ++col) {
        const Scalar candidate = fabs(matrix[row * dim + col]);
        if (candidate > maximum) {
          maximum = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }
    if (maximum <= internal::SymmetricMatrixPolicy<Scalar>::jacobi_relative_tolerance(dim)) {
      has_converged = true;
      break;
    }
    const Scalar app = matrix[pivot_row * dim + pivot_row];
    const Scalar aqq = matrix[pivot_col * dim + pivot_col];
    const Scalar apq = matrix[pivot_row * dim + pivot_col];
    const Scalar angle = static_cast<Scalar>(0.5) * atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar cosine = cos(angle);
    const Scalar sine = sin(angle);
    for (std::size_t index = 0; index < dim; ++index) {
      if (index == pivot_row || index == pivot_col) {
        continue;
      }
      const Scalar aip = matrix[index * dim + pivot_row];
      const Scalar aiq = matrix[index * dim + pivot_col];
      matrix[index * dim + pivot_row] = cosine * aip - sine * aiq;
      matrix[pivot_row * dim + index] = matrix[index * dim + pivot_row];
      matrix[index * dim + pivot_col] = sine * aip + cosine * aiq;
      matrix[pivot_col * dim + index] = matrix[index * dim + pivot_col];
    }
    matrix[pivot_row * dim + pivot_row] =
        cosine * cosine * app - static_cast<Scalar>(2) * sine * cosine * apq + sine * sine * aqq;
    matrix[pivot_col * dim + pivot_col] =
        sine * sine * app + static_cast<Scalar>(2) * sine * cosine * apq + cosine * cosine * aqq;
    matrix[pivot_row * dim + pivot_col] = static_cast<Scalar>(0);
    matrix[pivot_col * dim + pivot_row] = static_cast<Scalar>(0);
    for (std::size_t row = 0; row < dim; ++row) {
      const Scalar vip = eigenvectors[row * dim + pivot_row];
      const Scalar viq = eigenvectors[row * dim + pivot_col];
      eigenvectors[row * dim + pivot_row] = cosine * vip - sine * viq;
      eigenvectors[row * dim + pivot_col] = sine * vip + cosine * viq;
    }
  }
  if (!has_converged) {
    return false;
  }
  for (std::size_t index = 0; index < dim; ++index) {
    eigenvalues[index] = matrix[index * dim + index];
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
__device__ bool matrix_roots(const Scalar* input,
                             Scalar* output,
                             Scalar* inverse_output,
                             const bool compute_inverse,
                             const bool strict_positive,
                             const std::size_t dim,
                             Scalar* eigenvectors,
                             Scalar* eigenvalues,
                             Scalar* matrix,
                             Scalar* scaled,
                             Scalar* inverse_scaled,
                             Scalar* eigenvectors_transpose) {
  Scalar scale{};
  if (!symmetric_eigen(input, eigenvectors, eigenvalues, &scale, matrix, dim)) {
    return false;
  }
  Scalar maximum = static_cast<Scalar>(0);
  for (std::size_t index = 0; index < dim; ++index) {
    maximum = fmax(maximum, fabs(eigenvalues[index]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(dim) * maximum;
  for (std::size_t col = 0; col < dim; ++col) {
    if ((strict_positive && eigenvalues[col] <= threshold) || eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar root = sqrt(scale) * sqrt(fmax(eigenvalues[col], static_cast<Scalar>(0)));
    if (compute_inverse && (!isfinite(root) || root <= static_cast<Scalar>(0))) {
      return false;
    }
    for (std::size_t row = 0; row < dim; ++row) {
      scaled[row * dim + col] = eigenvectors[row * dim + col] * root;
      inverse_scaled[row * dim + col] =
          compute_inverse ? eigenvectors[row * dim + col] / root : static_cast<Scalar>(0);
    }
  }
  transpose(eigenvectors, eigenvectors_transpose, dim);
  multiply(scaled, eigenvectors_transpose, output, dim);
  if (compute_inverse) {
    multiply(inverse_scaled, eigenvectors_transpose, inverse_output, dim);
  }
  return true;
}

template <typename Scalar>
__device__ bool calculate_damping(TaskWorkspace<Scalar>* workspace, const std::size_t dim) {
  Scalar* mass = workspace->matrices[0];
  Scalar* stiffness = workspace->matrices[1];
  Scalar* damping = workspace->matrices[2];
  Scalar* mass_root = workspace->matrices[3];
  Scalar* mass_inverse_root = workspace->matrices[4];
  Scalar* inverse_root_transpose = workspace->matrices[5];
  Scalar* temporary = workspace->matrices[6];
  Scalar* transformed_stiffness = workspace->matrices[7];
  Scalar* transformed_root = workspace->matrices[8];
  Scalar* scratch_a = workspace->matrices[9];
  Scalar* scratch_b = workspace->matrices[10];
  Scalar* scratch_c = workspace->matrices[11];
  Scalar* scratch_d = workspace->matrices[12];
  Scalar* scratch_e = workspace->matrices[13];
  Scalar* scratch_f = workspace->matrices[14];
  Scalar* eigenvalues = workspace->vectors[1];
  if (!validate_and_symmetrize(mass, dim) || !validate_and_symmetrize(stiffness, dim)) {
    return false;
  }
  if (!matrix_roots(mass,
                    mass_root,
                    mass_inverse_root,
                    true,
                    true,
                    dim,
                    scratch_a,
                    eigenvalues,
                    scratch_b,
                    scratch_c,
                    scratch_d,
                    scratch_e)) {
    return false;
  }
  transpose(mass_inverse_root, inverse_root_transpose, dim);
  multiply(mass_inverse_root, stiffness, temporary, dim);
  multiply(temporary, inverse_root_transpose, transformed_stiffness, dim);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar value = static_cast<Scalar>(0.5) * (transformed_stiffness[row * dim + col] +
                                                       transformed_stiffness[col * dim + row]);
      transformed_stiffness[row * dim + col] = value;
      transformed_stiffness[col * dim + row] = value;
    }
  }
  if (!matrix_roots(transformed_stiffness,
                    transformed_root,
                    scratch_a,
                    false,
                    false,
                    dim,
                    scratch_b,
                    eigenvalues,
                    scratch_c,
                    scratch_d,
                    scratch_e,
                    scratch_f)) {
    return false;
  }
  Scalar* scaled_root = scratch_a;
  Scalar* left = scratch_b;
  Scalar* mass_root_transpose = scratch_c;
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      scaled_root[row * dim + col] = workspace->vectors[0][row] * transformed_root[row * dim + col];
    }
  }
  multiply(mass_root, scaled_root, left, dim);
  transpose(mass_root, mass_root_transpose, dim);
  multiply(left, mass_root_transpose, damping, dim);
  for (std::size_t index = 0; index < dim * dim; ++index) {
    damping[index] = detail::damping_double_diagonalization_value(damping[index]);
    if (!isfinite(damping[index])) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__global__ void apply_parameters_kernel(CudaMatrixParameterView<Scalar> stiffness,
                                        CudaVectorParameterView<Scalar> damping,
                                        const std::size_t num_envs,
                                        const std::size_t task_dim,
                                        std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < task_dim; ++col) {
      if (!isfinite(read_parameter(stiffness, env, row, col))) {
        flags |= kInvalidParameterFlag;
      }
    }
    const Scalar factor = read_parameter(damping, env, row);
    if (!isfinite(factor) || factor < static_cast<Scalar>(0)) {
      flags |= kInvalidParameterFlag;
    }
  }
  status[env] = flags;
}

__global__ void reset_kernel(const std::uint8_t* reset_mask,
                             const std::int64_t reset_stride,
                             std::uint8_t* initialized,
                             const std::int64_t initialized_stride,
                             const std::size_t num_envs,
                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool is_selected =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_stride)] != 0U;
  if (is_selected) {
    initialized[env * static_cast<std::size_t>(initialized_stride)] = 0U;
  }
  status[env] = is_selected ? kResetAppliedFlag : 0U;
}

template <typename Scalar>
__global__ void step_kernel(CudaMatrixView<Scalar> pose_error,
                            CudaMatrixView<Scalar> velocity_error,
                            CudaMatrixView<Scalar> desired_acceleration,
                            CudaMatrixView<Scalar> joint_velocity,
                            CudaTensor3View<Scalar> mass_matrix,
                            CudaTensor3View<Scalar> jacobian,
                            CudaScalarBatchView<Scalar> dt,
                            CudaMatrixParameterView<Scalar> stiffness_parameter,
                            CudaVectorParameterView<Scalar> damping_parameter,
                            const std::uint64_t* parameter_status,
                            CudaMutableTensor3View<Scalar> previous_mass,
                            CudaMutableTensor3View<Scalar> previous_jacobian,
                            std::uint8_t* initialized,
                            const std::int64_t initialized_stride,
                            CudaMutableMatrixView<Scalar> wrench,
                            const std::size_t num_envs,
                            const std::size_t task_dim,
                            const std::size_t joint_dim,
                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x;
  if (env >= num_envs) {
    return;
  }
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* workspace = reinterpret_cast<TaskWorkspace<Scalar>*>(shared_bytes);
  if (threadIdx.x == 0) {
    workspace->flags = parameter_status == nullptr ? 0U : parameter_status[env];
  }
  __syncthreads();

  for (std::size_t index = threadIdx.x; index < task_dim * task_dim; index += blockDim.x) {
    const std::size_t row = index / task_dim;
    const std::size_t col = index % task_dim;
    const Scalar mass_value = read_tensor3(mass_matrix, env, row, col);
    const Scalar stiffness_value = read_parameter(stiffness_parameter, env, row, col);
    workspace->matrices[0][index] = mass_value;
    workspace->matrices[1][index] = stiffness_value;
    if (!isfinite(mass_value)) {
      set_flag(&workspace->flags, kNonFiniteInputFlag);
    }
    if (!isfinite(stiffness_value)) {
      set_flag(&workspace->flags, kInvalidParameterFlag);
    }
  }
  for (std::size_t row = threadIdx.x; row < task_dim; row += blockDim.x) {
    const Scalar factor = read_parameter(damping_parameter, env, row);
    workspace->vectors[0][row] = factor;
    if (!isfinite(factor) || factor < static_cast<Scalar>(0)) {
      set_flag(&workspace->flags, kInvalidParameterFlag);
    }
    const Scalar pose = read_matrix(pose_error, env, row);
    const Scalar velocity = read_matrix(velocity_error, env, row);
    const Scalar acceleration = read_matrix(desired_acceleration, env, row);
    if (!isfinite(pose) || !isfinite(velocity) || !isfinite(acceleration)) {
      set_flag(&workspace->flags, kNonFiniteInputFlag);
    }
    if (workspace->matrices[0][row * task_dim + row] <= static_cast<Scalar>(0) ||
        workspace->matrices[1][row * task_dim + row] < static_cast<Scalar>(0)) {
      set_flag(&workspace->flags, kInvalidInputFlag);
    }
  }
  for (std::size_t joint = threadIdx.x; joint < joint_dim; joint += blockDim.x) {
    if (!isfinite(read_matrix(joint_velocity, env, joint))) {
      set_flag(&workspace->flags, kNonFiniteInputFlag);
    }
  }
  for (std::size_t index = threadIdx.x; index < task_dim * joint_dim; index += blockDim.x) {
    const std::size_t row = index / joint_dim;
    const std::size_t col = index % joint_dim;
    if (!isfinite(read_tensor3(jacobian, env, row, col))) {
      set_flag(&workspace->flags, kNonFiniteInputFlag);
    }
  }
  if (threadIdx.x == 0) {
    const Scalar dt_value = read_scalar(dt, env);
    if (!isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      workspace->flags |= kInvalidDtFlag;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && (workspace->flags & kSafeTriggers) == 0U) {
    if (!calculate_damping(workspace, task_dim)) {
      workspace->flags |= kInvalidInputFlag;
    } else {
      Scalar* stiffness_wrench = workspace->vectors[2];
      Scalar* damping_wrench = workspace->vectors[3];
      Scalar* acceleration_wrench = workspace->vectors[4];
      Scalar* inertial_wrench = workspace->vectors[5];
      Scalar* jacobian_derivative_dq = workspace->vectors[6];
      for (std::size_t row = 0; row < task_dim; ++row) {
        stiffness_wrench[row] = static_cast<Scalar>(0);
        damping_wrench[row] = static_cast<Scalar>(0);
        acceleration_wrench[row] = static_cast<Scalar>(0);
        inertial_wrench[row] = static_cast<Scalar>(0);
        jacobian_derivative_dq[row] = static_cast<Scalar>(0);
        for (std::size_t col = 0; col < task_dim; ++col) {
          stiffness_wrench[row] +=
              workspace->matrices[1][row * task_dim + col] * read_matrix(pose_error, env, col);
          damping_wrench[row] +=
              workspace->matrices[2][row * task_dim + col] * read_matrix(velocity_error, env, col);
          acceleration_wrench[row] += workspace->matrices[0][row * task_dim + col] *
                                      read_matrix(desired_acceleration, env, col);
        }
      }
      const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized_stride);
      if (initialized[initialized_offset] != 0U) {
        const Scalar dt_value = read_scalar(dt, env);
        for (std::size_t row = 0; row < task_dim; ++row) {
          for (std::size_t joint = 0; joint < joint_dim; ++joint) {
            jacobian_derivative_dq[row] +=
                detail::matrix_derivative_value(read_tensor3(jacobian, env, row, joint),
                                                read_tensor3(previous_jacobian, env, row, joint),
                                                dt_value) *
                read_matrix(joint_velocity, env, joint);
          }
        }
        for (std::size_t row = 0; row < task_dim; ++row) {
          Scalar mass_derivative_term = static_cast<Scalar>(0);
          Scalar jacobian_term = static_cast<Scalar>(0);
          for (std::size_t col = 0; col < task_dim; ++col) {
            mass_derivative_term +=
                detail::matrix_derivative_value(read_tensor3(mass_matrix, env, row, col),
                                                read_tensor3(previous_mass, env, row, col),
                                                dt_value) *
                read_matrix(velocity_error, env, col);
            jacobian_term +=
                workspace->matrices[0][row * task_dim + col] * jacobian_derivative_dq[col];
          }
          inertial_wrench[row] =
              detail::inertial_coupling_value(mass_derivative_term, jacobian_term);
        }
      }
      for (std::size_t row = 0; row < task_dim; ++row) {
        stiffness_wrench[row] = detail::cartesian_impedance_value(stiffness_wrench[row],
                                                                  damping_wrench[row],
                                                                  inertial_wrench[row],
                                                                  acceleration_wrench[row]);
        if (!isfinite(stiffness_wrench[row])) {
          workspace->flags |= kInvalidInputFlag;
        }
      }
    }
  }
  __syncthreads();

  if ((workspace->flags & kSafeTriggers) != 0U) {
    for (std::size_t row = threadIdx.x; row < task_dim; row += blockDim.x) {
      write_matrix(wrench, env, row, static_cast<Scalar>(0));
    }
    if (threadIdx.x == 0) {
      status[env] = workspace->flags | kSafeModeFlag;
    }
    return;
  }
  for (std::size_t index = threadIdx.x; index < task_dim * task_dim; index += blockDim.x) {
    const std::size_t row = index / task_dim;
    const std::size_t col = index % task_dim;
    write_tensor3(previous_mass, env, row, col, read_tensor3(mass_matrix, env, row, col));
  }
  for (std::size_t index = threadIdx.x; index < task_dim * joint_dim; index += blockDim.x) {
    const std::size_t row = index / joint_dim;
    const std::size_t col = index % joint_dim;
    write_tensor3(previous_jacobian, env, row, col, read_tensor3(jacobian, env, row, col));
  }
  for (std::size_t row = threadIdx.x; row < task_dim; row += blockDim.x) {
    write_matrix(wrench, env, row, workspace->vectors[2][row]);
  }
  if (threadIdx.x == 0) {
    initialized[env * static_cast<std::size_t>(initialized_stride)] = 1U;
    status[env] = workspace->flags;
  }
}

Result launch_result() noexcept {
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{ResultCode::kOk, "ok"}
             : Result{ResultCode::kInvalidArgument, "CUDA task impedance kernel launch failed"};
}

template <typename Scalar>
Result launch_apply(const ParameterSetView& parameters,
                    const std::size_t num_envs,
                    const std::size_t task_dim,
                    const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  apply_parameters_kernel<Scalar><<<blocks, kThreads, 0, active_cuda_stream()>>>(
      cuda_matrix_parameter<Scalar>(find_parameter(parameters, kStiffnessMatrix)),
      cuda_vector_parameter<Scalar>(find_parameter(parameters, kDampingFactors)),
      num_envs,
      task_dim,
      status.flags);
  return launch_result();
}

template <typename Scalar>
Result launch_step(const TensorView& pose_error,
                   const TensorView& velocity_error,
                   const TensorView& desired_acceleration,
                   const TensorView& joint_velocity,
                   const TensorView& mass_matrix,
                   const TensorView& jacobian,
                   const TensorView& dt,
                   const ParameterSetView& parameters,
                   const StatusView parameter_status,
                   StateView& state,
                   const MutableTensorView& wrench,
                   const MutableStatusView status) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(pose_error.shape[0]);
  const std::size_t task_dim = static_cast<std::size_t>(pose_error.shape[1]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_velocity.shape[1]);
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  step_kernel<Scalar><<<static_cast<int>(num_envs),
                        kThreads,
                        sizeof(TaskWorkspace<Scalar>),
                        active_cuda_stream()>>>(
      cuda_matrix<Scalar>(pose_error),
      cuda_matrix<Scalar>(velocity_error),
      cuda_matrix<Scalar>(desired_acceleration),
      cuda_matrix<Scalar>(joint_velocity),
      cuda_tensor3<Scalar>(mass_matrix),
      cuda_tensor3<Scalar>(jacobian),
      cuda_scalar_batch<Scalar>(dt),
      cuda_matrix_parameter<Scalar>(find_parameter(parameters, kStiffnessMatrix)),
      cuda_vector_parameter<Scalar>(find_parameter(parameters, kDampingFactors)),
      parameter_status.flags,
      cuda_tensor3<Scalar>(previous_mass->values),
      cuda_tensor3<Scalar>(previous_jacobian->values),
      static_cast<std::uint8_t*>(initialized->values.data),
      initialized->values.strides[0],
      cuda_matrix<Scalar>(wrench),
      num_envs,
      task_dim,
      joint_dim,
      status.flags);
  return launch_result();
}

}  // namespace

Result apply_task_impedance_parameters_cuda(const ParameterSetView& parameters,
                                            const std::size_t num_envs,
                                            const std::size_t task_dim,
                                            const MutableStatusView parameter_status) noexcept {
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  return stiffness->values.dtype == DType::kFloat32
             ? launch_apply<float>(parameters, num_envs, task_dim, parameter_status)
             : launch_apply<double>(parameters, num_envs, task_dim, parameter_status);
}

Result reset_task_impedance_controller_cuda(const std::size_t num_envs,
                                            const std::size_t task_dim,
                                            const std::size_t joint_dim,
                                            StateView& state,
                                            const TensorView& reset_mask,
                                            const MutableStatusView status) noexcept {
  (void)task_dim;
  (void)joint_dim;
  StateEntryView* initialized = find_state(state, kInitialized);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  reset_kernel<<<blocks, kThreads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      static_cast<std::uint8_t*>(initialized->values.data),
      initialized->values.strides[0],
      num_envs,
      status.flags);
  return launch_result();
}

Result step_task_impedance_controller_cuda(const TensorView& task_pose_error,
                                           const TensorView& task_velocity_error,
                                           const TensorView& desired_acceleration,
                                           const TensorView& joint_velocity,
                                           const TensorView& task_mass_matrix,
                                           const TensorView& jacobian,
                                           const TensorView& dt,
                                           const ParameterSetView& parameters,
                                           const StatusView parameter_status,
                                           StateView& state,
                                           const MutableTensorView& task_wrench,
                                           const MutableStatusView status) noexcept {
  return task_pose_error.dtype == DType::kFloat32 ? launch_step<float>(task_pose_error,
                                                                       task_velocity_error,
                                                                       desired_acceleration,
                                                                       joint_velocity,
                                                                       task_mass_matrix,
                                                                       jacobian,
                                                                       dt,
                                                                       parameters,
                                                                       parameter_status,
                                                                       state,
                                                                       task_wrench,
                                                                       status)
                                                  : launch_step<double>(task_pose_error,
                                                                        task_velocity_error,
                                                                        desired_acceleration,
                                                                        joint_velocity,
                                                                        task_mass_matrix,
                                                                        jacobian,
                                                                        dt,
                                                                        parameters,
                                                                        parameter_status,
                                                                        state,
                                                                        task_wrench,
                                                                        status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
