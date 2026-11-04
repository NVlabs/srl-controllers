/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_blocks_cuda.cu
 * @brief CUDA tensor kernels for joint controller blocks.
 */

#include "tensor/joint_blocks_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/cuda_component_tuning_profiles.hpp"
#include "tensor/cuda_stream.hpp"
#include "tensor/symmetric_matrix_policy.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr std::uint64_t kUnsupportedModeFlag =
    static_cast<std::uint64_t>(StatusFlag::kUnsupportedMode);
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kSafeTriggers =
    kUnsupportedModeFlag | kInvalidParameterFlag | kInvalidInputFlag | kNonFiniteInputFlag;
constexpr std::size_t kMaxJointDim = 32;
constexpr std::size_t kJointMatrixElements = kMaxJointDim * kMaxJointDim;
constexpr std::size_t kMaxTaskDim = 12;

#ifndef NVIDIASRLCONTROLLERS_CUDA_BUILD_SM
#error "NVIDIASRLCONTROLLERS_CUDA_BUILD_SM must be provided by the CUDA build"
#endif

constexpr int kCudaBuildSm = NVIDIASRLCONTROLLERS_CUDA_BUILD_SM;

struct CudaTuningProfile {
  int sm{0};
  DType dtype{DType::kUnknown};
  std::size_t joint_dim{0};
  std::size_t capacity{0};
  int warps_per_block{0};
};

// The qualification tool emits this table from dense deployment workloads. CMake copies the
// selected file into the build tree so a qualification runner can rebuild and validate exactly
// the profile it selected without exposing tuning controls through the public API.
#define NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE(SM, DTYPE, DIM, CAPACITY, WARPS) \
  CudaTuningProfile{SM, DType::DTYPE, DIM, CAPACITY, WARPS},
constexpr CudaTuningProfile kCudaTuningProfiles[] = {
#include "tensor/joint_impedance_cuda_tuning_profiles.inc"
};
#undef NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE

constexpr const CudaTuningProfile* find_cuda_tuning_profile(const DType dtype,
                                                            const std::size_t joint_dim) {
  for (const CudaTuningProfile& profile : kCudaTuningProfiles) {
    if (profile.sm == kCudaBuildSm && profile.dtype == dtype && profile.joint_dim == joint_dim) {
      return &profile;
    }
  }
  return nullptr;
}

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
struct CudaTensor3View {
  const Scalar* data{nullptr};
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
  std::int64_t dim_stride{0};
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
__device__ Scalar read_matrix(const CudaMatrixView<Scalar> view,
                              const std::size_t env,
                              const std::size_t dim) {
  return view.data[env * static_cast<std::size_t>(view.row_stride) +
                   dim * static_cast<std::size_t>(view.col_stride)];
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
__device__ void write_matrix(const CudaMutableMatrixView<Scalar> view,
                             const std::size_t env,
                             const std::size_t dim,
                             const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.row_stride) +
            dim * static_cast<std::size_t>(view.col_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_diagonal(const CudaVectorParameterView<Scalar> diagonal,
                                const std::size_t env,
                                const std::size_t dim) {
  if (diagonal.shared || diagonal.rank == 1) {
    return diagonal.data[dim * static_cast<std::size_t>(diagonal.dim_stride)];
  }
  return diagonal.data[env * static_cast<std::size_t>(diagonal.env_stride) +
                       dim * static_cast<std::size_t>(diagonal.dim_stride)];
}

template <typename Scalar>
__device__ Scalar read_parameter_matrix(const CudaMatrixParameterView<Scalar> matrix,
                                        const std::size_t env,
                                        const std::size_t row,
                                        const std::size_t col) {
  if (matrix.shared || matrix.rank == 2) {
    return matrix.data[row * static_cast<std::size_t>(matrix.row_stride) +
                       col * static_cast<std::size_t>(matrix.col_stride)];
  }
  return matrix.data[env * static_cast<std::size_t>(matrix.env_stride) +
                     row * static_cast<std::size_t>(matrix.row_stride) +
                     col * static_cast<std::size_t>(matrix.col_stride)];
}

template <typename Scalar>
__device__ bool is_finite(const Scalar value) {
  return isfinite(value);
}

template <typename Scalar>
__device__ bool matrix_input_is_finite(const CudaMatrixView<Scalar> input,
                                       const std::size_t env,
                                       const std::size_t joint_dim) {
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    if (!is_finite(read_matrix(input, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool tensor3_square_is_finite(const CudaTensor3View<Scalar> input,
                                         const std::size_t env,
                                         const std::size_t joint_dim) {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!is_finite(read_tensor3(input, env, row, col))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool tensor3_matrix_is_finite(const CudaTensor3View<Scalar> input,
                                         const std::size_t env,
                                         const std::size_t rows,
                                         const std::size_t cols) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (!is_finite(read_tensor3(input, env, row, col))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool damping_factors_are_valid(const CudaVectorParameterView<Scalar> damping_factors,
                                          const std::size_t env,
                                          const std::size_t joint_dim) {
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    const Scalar value = read_diagonal(damping_factors, env, dim);
    if (!is_finite(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool dense_parameter_is_valid(const CudaMatrixParameterView<Scalar> matrix,
                                         const std::size_t env,
                                         const std::size_t joint_dim) {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!is_finite(read_parameter_matrix(matrix, env, row, col))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ void zero_joint_output(const CudaMutableMatrixView<Scalar> output,
                                  const std::size_t env,
                                  const std::size_t joint_dim) {
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    write_matrix(output, env, dim, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
__device__ void read_tensor3_square(const CudaTensor3View<Scalar> view,
                                    const std::size_t env,
                                    Scalar matrix[kMaxJointDim][kMaxJointDim],
                                    const std::size_t joint_dim) {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      matrix[row][col] = read_tensor3(view, env, row, col);
    }
  }
}

__device__ std::size_t flat_index(const std::size_t row,
                                  const std::size_t col,
                                  const std::size_t matrix_stride = kMaxJointDim) {
  return row * matrix_stride + col;
}

template <typename Scalar>
__device__ Scalar flat_read(const Scalar* matrix,
                            const std::size_t row,
                            const std::size_t col,
                            const std::size_t matrix_stride = kMaxJointDim) {
  return matrix[flat_index(row, col, matrix_stride)];
}

template <typename Scalar>
__device__ void flat_write(Scalar* matrix,
                           const std::size_t row,
                           const std::size_t col,
                           const Scalar value,
                           const std::size_t matrix_stride = kMaxJointDim) {
  matrix[flat_index(row, col, matrix_stride)] = value;
}

template <typename Scalar>
__device__ void set_identity_flat(Scalar* matrix,
                                  const std::size_t dim,
                                  const std::size_t matrix_stride = kMaxJointDim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      flat_write(matrix,
                 row,
                 col,
                 row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0),
                 matrix_stride);
    }
  }
}

template <typename Scalar>
__device__ bool copy_tensor3_square_to_flat(const CudaTensor3View<Scalar> view,
                                            const std::size_t env,
                                            Scalar* output,
                                            const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const Scalar value = read_tensor3(view, env, row, col);
      if (!is_finite(value)) {
        return false;
      }
      flat_write(output, row, col, value);
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool copy_parameter_square_to_flat(const CudaMatrixParameterView<Scalar> view,
                                              const std::size_t env,
                                              Scalar* output,
                                              const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const Scalar value = read_parameter_matrix(view, env, row, col);
      if (!is_finite(value)) {
        return false;
      }
      flat_write(output, row, col, value);
    }
  }
  return true;
}

template <typename Scalar>
__device__ void copy_flat(const Scalar* input,
                          Scalar* output,
                          const std::size_t dim,
                          const std::size_t matrix_stride = kMaxJointDim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      flat_write(output, row, col, flat_read(input, row, col, matrix_stride), matrix_stride);
    }
  }
}

template <typename Scalar>
__device__ bool symmetric_eigen_work_flat(Scalar* matrix,
                                          Scalar* eigenvectors,
                                          Scalar* eigenvalues,
                                          const std::size_t dim,
                                          Scalar* matrix_scale) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const Scalar value = flat_read(matrix, row, col);
      if (!is_finite(value)) {
        return false;
      }
      scale = fmax(scale, fabs(value));
    }
  }
  if (!is_finite(scale)) {
    return false;
  }
  if (scale == static_cast<Scalar>(0)) {
    scale = static_cast<Scalar>(1);
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      flat_write(matrix, row, col, flat_read(matrix, row, col) / scale);
    }
  }

  set_identity_flat(eigenvectors, dim);
  const std::size_t ten_dim_squared = 10 * dim * dim;
  const std::size_t max_iterations = ten_dim_squared > 80 ? ten_dim_squared : 80;
  bool converged = dim <= 1;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = dim > 1 ? 1 : 0;
    Scalar max_offdiag = dim > 1 ? fabs(flat_read(matrix, pivot_row, pivot_col)) : 0;
    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = row + 1; col < dim; ++col) {
        const Scalar candidate = fabs(flat_read(matrix, row, col));
        if (candidate > max_offdiag) {
          max_offdiag = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }
    if (max_offdiag <= internal::SymmetricMatrixPolicy<Scalar>::jacobi_relative_tolerance(dim)) {
      converged = true;
      break;
    }

    const Scalar app = flat_read(matrix, pivot_row, pivot_row);
    const Scalar aqq = flat_read(matrix, pivot_col, pivot_col);
    const Scalar apq = flat_read(matrix, pivot_row, pivot_col);
    const Scalar angle = static_cast<Scalar>(0.5) * atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar c = cos(angle);
    const Scalar s = sin(angle);

    for (std::size_t idx = 0; idx < dim; ++idx) {
      if (idx == pivot_row || idx == pivot_col) {
        continue;
      }
      const Scalar aip = flat_read(matrix, idx, pivot_row);
      const Scalar aiq = flat_read(matrix, idx, pivot_col);
      flat_write(matrix, idx, pivot_row, c * aip - s * aiq);
      flat_write(matrix, pivot_row, idx, flat_read(matrix, idx, pivot_row));
      flat_write(matrix, idx, pivot_col, s * aip + c * aiq);
      flat_write(matrix, pivot_col, idx, flat_read(matrix, idx, pivot_col));
    }
    flat_write(matrix,
               pivot_row,
               pivot_row,
               c * c * app - static_cast<Scalar>(2) * s * c * apq + s * s * aqq);
    flat_write(matrix,
               pivot_col,
               pivot_col,
               s * s * app + static_cast<Scalar>(2) * s * c * apq + c * c * aqq);
    flat_write(matrix, pivot_row, pivot_col, static_cast<Scalar>(0));
    flat_write(matrix, pivot_col, pivot_row, static_cast<Scalar>(0));

    for (std::size_t row = 0; row < dim; ++row) {
      const Scalar vip = flat_read(eigenvectors, row, pivot_row);
      const Scalar viq = flat_read(eigenvectors, row, pivot_col);
      flat_write(eigenvectors, row, pivot_row, c * vip - s * viq);
      flat_write(eigenvectors, row, pivot_col, s * vip + c * viq);
    }
  }

  if (!converged) {
    return false;
  }

  for (std::size_t dim_index = 0; dim_index < dim; ++dim_index) {
    eigenvalues[dim_index] = flat_read(matrix, dim_index, dim_index);
    if (!is_finite(eigenvalues[dim_index])) {
      return false;
    }
    for (std::size_t row = 0; row < dim; ++row) {
      if (!is_finite(flat_read(eigenvectors, row, dim_index))) {
        return false;
      }
    }
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
__device__ bool validate_and_symmetrize_external_flat(Scalar* matrix, const std::size_t dim) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      const Scalar value = flat_read(matrix, row, col);
      if (!is_finite(value)) {
        return false;
      }
      scale = fmax(scale, fabs(value));
    }
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(dim) * scale;
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar upper = flat_read(matrix, row, col);
      const Scalar lower = flat_read(matrix, col, row);
      if (fabs(upper - lower) > tolerance) {
        return false;
      }
      const Scalar value = static_cast<Scalar>(0.5) * upper + static_cast<Scalar>(0.5) * lower;
      flat_write(matrix, row, col, value);
      flat_write(matrix, col, row, value);
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool symmetrize_derived_flat(Scalar* matrix, const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!is_finite(flat_read(matrix, row, col))) {
        return false;
      }
    }
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar value = static_cast<Scalar>(0.5) * flat_read(matrix, row, col) +
                           static_cast<Scalar>(0.5) * flat_read(matrix, col, row);
      flat_write(matrix, row, col, value);
      flat_write(matrix, col, row, value);
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool matrix_roots_from_work_flat(Scalar* work_matrix,
                                            Scalar* eigenvectors,
                                            Scalar* eigenvalues,
                                            Scalar* output,
                                            Scalar* inverse_output,
                                            const std::size_t dim,
                                            const bool strict_positive) {
  Scalar scale{};
  if (!symmetric_eigen_work_flat(work_matrix, eigenvectors, eigenvalues, dim, &scale)) {
    return false;
  }
  Scalar max_eigenvalue = static_cast<Scalar>(0);
  for (std::size_t col = 0; col < dim; ++col) {
    max_eigenvalue = fmax(max_eigenvalue, fabs(eigenvalues[col]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(dim) * max_eigenvalue;
  for (std::size_t col = 0; col < dim; ++col) {
    if (strict_positive) {
      if (eigenvalues[col] <= threshold) {
        return false;
      }
    } else if (eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar normalized = fmax(eigenvalues[col], static_cast<Scalar>(0));
    const Scalar sqrt_value = sqrt(scale) * sqrt(normalized);
    for (std::size_t row = 0; row < dim; ++row) {
      flat_write(work_matrix, row, col, flat_read(eigenvectors, row, col) * sqrt_value);
    }
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += flat_read(work_matrix, row, inner) * flat_read(eigenvectors, col, inner);
      }
      flat_write(output, row, col, value);
      if (inverse_output != nullptr) {
        Scalar inverse_value = static_cast<Scalar>(0);
        for (std::size_t inner = 0; inner < dim; ++inner) {
          inverse_value += flat_read(eigenvectors, row, inner) *
                           flat_read(eigenvectors, col, inner) /
                           (sqrt(scale) * sqrt(eigenvalues[inner]));
        }
        flat_write(inverse_output, row, col, inverse_value);
      }
      if (!is_finite(value) ||
          (inverse_output != nullptr && !is_finite(flat_read(inverse_output, row, col)))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool matrix_sqrt_tensor3_flat(const CudaTensor3View<Scalar> input,
                                         const std::size_t env,
                                         Scalar* output,
                                         Scalar* work_matrix,
                                         Scalar* eigenvectors,
                                         Scalar* eigenvalues,
                                         const std::size_t dim,
                                         const bool strict_positive) {
  if (!copy_tensor3_square_to_flat(input, env, work_matrix, dim)) {
    return false;
  }
  return matrix_roots_from_work_flat(work_matrix,
                                     eigenvectors,
                                     eigenvalues,
                                     output,
                                     static_cast<Scalar*>(nullptr),
                                     dim,
                                     strict_positive);
}

template <typename Scalar>
__device__ bool matrix_sqrt_flat(const Scalar* input,
                                 Scalar* output,
                                 Scalar* work_matrix,
                                 Scalar* eigenvectors,
                                 Scalar* eigenvalues,
                                 const std::size_t dim,
                                 const bool strict_positive) {
  copy_flat(input, work_matrix, dim);
  return matrix_roots_from_work_flat(work_matrix,
                                     eigenvectors,
                                     eigenvalues,
                                     output,
                                     static_cast<Scalar*>(nullptr),
                                     dim,
                                     strict_positive);
}

template <typename Scalar>
__device__ bool inverse_flat(const Scalar* input,
                             Scalar* output,
                             Scalar* work_matrix,
                             const std::size_t dim,
                             const std::size_t matrix_stride = kMaxJointDim) {
  copy_flat(input, work_matrix, dim, matrix_stride);
  set_identity_flat(output, dim, matrix_stride);
  for (std::size_t pivot = 0; pivot < dim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = fabs(flat_read(work_matrix, pivot, pivot, matrix_stride));
    for (std::size_t row = pivot + 1; row < dim; ++row) {
      const Scalar candidate = fabs(flat_read(work_matrix, row, pivot, matrix_stride));
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }
    if (pivot_abs <= static_cast<Scalar>(1e-12)) {
      return false;
    }
    if (pivot_row != pivot) {
      for (std::size_t col = 0; col < dim; ++col) {
        const Scalar matrix_temporary = flat_read(work_matrix, pivot, col, matrix_stride);
        flat_write(work_matrix,
                   pivot,
                   col,
                   flat_read(work_matrix, pivot_row, col, matrix_stride),
                   matrix_stride);
        flat_write(work_matrix, pivot_row, col, matrix_temporary, matrix_stride);
        const Scalar output_temporary = flat_read(output, pivot, col, matrix_stride);
        flat_write(
            output, pivot, col, flat_read(output, pivot_row, col, matrix_stride), matrix_stride);
        flat_write(output, pivot_row, col, output_temporary, matrix_stride);
      }
    }
    const Scalar pivot_value = flat_read(work_matrix, pivot, pivot, matrix_stride);
    for (std::size_t col = 0; col < dim; ++col) {
      flat_write(work_matrix,
                 pivot,
                 col,
                 flat_read(work_matrix, pivot, col, matrix_stride) / pivot_value,
                 matrix_stride);
      flat_write(output,
                 pivot,
                 col,
                 flat_read(output, pivot, col, matrix_stride) / pivot_value,
                 matrix_stride);
    }
    for (std::size_t row = 0; row < dim; ++row) {
      if (row == pivot) {
        continue;
      }
      const Scalar factor = flat_read(work_matrix, row, pivot, matrix_stride);
      for (std::size_t col = 0; col < dim; ++col) {
        flat_write(work_matrix,
                   row,
                   col,
                   flat_read(work_matrix, row, col, matrix_stride) -
                       factor * flat_read(work_matrix, pivot, col, matrix_stride),
                   matrix_stride);
        flat_write(output,
                   row,
                   col,
                   flat_read(output, row, col, matrix_stride) -
                       factor * flat_read(output, pivot, col, matrix_stride),
                   matrix_stride);
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ void multiply_flat_tensor3_right(const Scalar* left,
                                            const CudaTensor3View<Scalar> right,
                                            const std::size_t env,
                                            Scalar* output,
                                            const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += flat_read(left, row, inner) * read_tensor3(right, env, inner, col);
      }
      flat_write(output, row, col, value);
    }
  }
}

template <typename Scalar>
__device__ void multiply_flat_parameter_right(const Scalar* left,
                                              const CudaMatrixParameterView<Scalar> right,
                                              const std::size_t env,
                                              Scalar* output,
                                              const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += flat_read(left, row, inner) * read_parameter_matrix(right, env, inner, col);
      }
      flat_write(output, row, col, value);
    }
  }
}

template <typename Scalar>
__device__ void multiply_flat_right_transpose(const Scalar* left,
                                              const Scalar* right,
                                              Scalar* output,
                                              const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += flat_read(left, row, inner) * flat_read(right, col, inner);
      }
      flat_write(output, row, col, value);
    }
  }
}

template <typename Scalar>
__device__ void multiply_flat(const Scalar* left,
                              const Scalar* right,
                              Scalar* output,
                              const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += flat_read(left, row, inner) * flat_read(right, inner, col);
      }
      flat_write(output, row, col, value);
    }
  }
}

template <typename Scalar>
__device__ bool compute_double_diagonalization_torque_flat(
    const CudaTensor3View<Scalar> joint_mass_matrix,
    const CudaTensor3View<Scalar> stiffness_matrix,
    const CudaVectorParameterView<Scalar> damping_factors,
    const CudaMatrixView<Scalar> joint_velocity_error,
    const std::size_t env,
    const std::size_t joint_dim,
    Scalar* mass_sqrt,
    Scalar* mass_sqrt_inv,
    Scalar* transformed_stiffness_sqrt,
    Scalar* transformed_stiffness,
    Scalar* work_matrix,
    Scalar* eigenvalues,
    Scalar* velocity,
    Scalar* right_vector,
    Scalar* temp_vector,
    Scalar* torque) {
  if (!copy_tensor3_square_to_flat(joint_mass_matrix, env, work_matrix, joint_dim) ||
      !validate_and_symmetrize_external_flat(work_matrix, joint_dim) ||
      !matrix_roots_from_work_flat(work_matrix,
                                   transformed_stiffness,
                                   eigenvalues,
                                   mass_sqrt,
                                   mass_sqrt_inv,
                                   joint_dim,
                                   true)) {
    return false;
  }
  if (!copy_tensor3_square_to_flat(stiffness_matrix, env, transformed_stiffness_sqrt, joint_dim) ||
      !validate_and_symmetrize_external_flat(transformed_stiffness_sqrt, joint_dim)) {
    return false;
  }
  multiply_flat(mass_sqrt_inv, transformed_stiffness_sqrt, work_matrix, joint_dim);
  multiply_flat_right_transpose(work_matrix, mass_sqrt_inv, transformed_stiffness_sqrt, joint_dim);
  if (!symmetrize_derived_flat(transformed_stiffness_sqrt, joint_dim)) {
    return false;
  }
  if (!matrix_sqrt_flat(transformed_stiffness_sqrt,
                        transformed_stiffness_sqrt,
                        work_matrix,
                        transformed_stiffness,
                        eigenvalues,
                        joint_dim,
                        true)) {
    return false;
  }
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    velocity[dim] = read_matrix(joint_velocity_error, env, dim);
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(mass_sqrt, col, row) * velocity[col];
    }
    right_vector[row] = value;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(transformed_stiffness_sqrt, row, col) * right_vector[col];
    }
    temp_vector[row] = read_diagonal(damping_factors, env, row) * value;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(mass_sqrt, row, col) * temp_vector[col];
    }
    torque[row] = detail::damping_double_diagonalization_value(value);
  }
  return true;
}

// The generic controller path above deliberately keeps its historical 32-wide
// layout.  The specialized path below uses a capacity-sized layout so small
// robots do not reserve the generic 32 x 32 workspace for every environment.
template <typename Scalar, std::size_t Capacity>
__device__ Scalar specialized_read(const Scalar* matrix,
                                   const std::size_t row,
                                   const std::size_t col) {
  return matrix[row * Capacity + col];
}

template <typename Scalar, std::size_t Capacity>
__device__ void specialized_write(Scalar* matrix,
                                  const std::size_t row,
                                  const std::size_t col,
                                  const Scalar value) {
  matrix[row * Capacity + col] = value;
}

template <typename Scalar>
__device__ Scalar specialized_warp_max(Scalar value, const unsigned int warp_mask) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmax(value, __shfl_down_sync(warp_mask, value, offset));
  }
  return __shfl_sync(warp_mask, value, 0);
}

__device__ bool specialized_warp_all(const bool predicate, const unsigned int warp_mask) {
  return __all_sync(warp_mask, predicate) != 0;
}

template <typename Scalar, std::size_t Capacity>
__device__ bool specialized_validate_and_symmetrize(Scalar* matrix,
                                                    const std::size_t dim,
                                                    const int lane,
                                                    const unsigned int warp_mask) {
  Scalar local_scale = static_cast<Scalar>(0);
  bool is_valid = true;
  const std::size_t element_count = dim * dim;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const Scalar value = specialized_read<Scalar, Capacity>(matrix, index / dim, index % dim);
    is_valid = is_valid && is_finite(value);
    local_scale = fmax(local_scale, fabs(value));
  }
  const Scalar scale = specialized_warp_max(local_scale, warp_mask);
  if (!specialized_warp_all(is_valid, warp_mask)) {
    return false;
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(dim) * scale;
  is_valid = true;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    if (col > row) {
      const Scalar upper = specialized_read<Scalar, Capacity>(matrix, row, col);
      const Scalar lower = specialized_read<Scalar, Capacity>(matrix, col, row);
      is_valid = is_valid && fabs(upper - lower) <= tolerance;
      const Scalar value = static_cast<Scalar>(0.5) * (upper + lower);
      specialized_write<Scalar, Capacity>(matrix, row, col, value);
      specialized_write<Scalar, Capacity>(matrix, col, row, value);
    }
  }
  const bool is_warp_valid = specialized_warp_all(is_valid, warp_mask);
  __syncwarp(warp_mask);
  return is_warp_valid;
}

template <typename Scalar, std::size_t Capacity>
__device__ bool specialized_symmetrize(Scalar* matrix,
                                       const std::size_t dim,
                                       const int lane,
                                       const unsigned int warp_mask) {
  bool is_valid = true;
  const std::size_t element_count = dim * dim;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    is_valid =
        is_valid && is_finite(specialized_read<Scalar, Capacity>(matrix, index / dim, index % dim));
  }
  if (!specialized_warp_all(is_valid, warp_mask)) {
    return false;
  }
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    if (col > row) {
      const Scalar value =
          static_cast<Scalar>(0.5) * (specialized_read<Scalar, Capacity>(matrix, row, col) +
                                      specialized_read<Scalar, Capacity>(matrix, col, row));
      specialized_write<Scalar, Capacity>(matrix, row, col, value);
      specialized_write<Scalar, Capacity>(matrix, col, row, value);
    }
  }
  __syncwarp(warp_mask);
  return true;
}

template <typename Scalar, std::size_t Capacity>
__device__ bool specialized_symmetric_eigen(Scalar* matrix,
                                            Scalar* eigenvectors,
                                            Scalar* eigenvalues,
                                            const std::size_t dim,
                                            Scalar* matrix_scale,
                                            const int lane,
                                            const unsigned int warp_mask) {
  Scalar local_scale = static_cast<Scalar>(0);
  bool is_valid = true;
  const std::size_t element_count = dim * dim;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    const Scalar value = specialized_read<Scalar, Capacity>(matrix, row, col);
    is_valid = is_valid && is_finite(value);
    local_scale = fmax(local_scale, fabs(value));
    specialized_write<Scalar, Capacity>(
        eigenvectors, row, col, row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0));
  }
  Scalar scale = specialized_warp_max(local_scale, warp_mask);
  if (!specialized_warp_all(is_valid, warp_mask) || !is_finite(scale)) {
    return false;
  }
  scale = scale == static_cast<Scalar>(0) ? static_cast<Scalar>(1) : scale;
  __syncwarp(warp_mask);
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    specialized_write<Scalar, Capacity>(
        matrix, row, col, specialized_read<Scalar, Capacity>(matrix, row, col) / scale);
  }
  __syncwarp(warp_mask);
  bool converged = dim <= 1;
  const std::size_t scaled_iterations = 10 * dim * dim;
  const std::size_t max_iterations =
      scaled_iterations > static_cast<std::size_t>(80) ? scaled_iterations : 80;
  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    Scalar maximum = static_cast<Scalar>(-1);
    unsigned int pivot_index = ~0U;
    for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
      const std::size_t row = index / dim;
      const std::size_t col = index % dim;
      if (col > row) {
        const Scalar candidate = fabs(specialized_read<Scalar, Capacity>(matrix, row, col));
        if (candidate > maximum ||
            (candidate == maximum && static_cast<unsigned int>(index) < pivot_index)) {
          maximum = candidate;
          pivot_index = static_cast<unsigned int>(index);
        }
      }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      const Scalar candidate = __shfl_down_sync(warp_mask, maximum, offset);
      const unsigned int candidate_index = __shfl_down_sync(warp_mask, pivot_index, offset);
      if (candidate > maximum || (candidate == maximum && candidate_index < pivot_index)) {
        maximum = candidate;
        pivot_index = candidate_index;
      }
    }
    maximum = __shfl_sync(warp_mask, maximum, 0);
    pivot_index = __shfl_sync(warp_mask, pivot_index, 0);
    if (maximum <= internal::SymmetricMatrixPolicy<Scalar>::jacobi_relative_tolerance(dim)) {
      converged = true;
      break;
    }
    const std::size_t pivot_row = static_cast<std::size_t>(pivot_index) / dim;
    const std::size_t pivot_col = static_cast<std::size_t>(pivot_index) % dim;
    const Scalar app = specialized_read<Scalar, Capacity>(matrix, pivot_row, pivot_row);
    const Scalar aqq = specialized_read<Scalar, Capacity>(matrix, pivot_col, pivot_col);
    const Scalar apq = specialized_read<Scalar, Capacity>(matrix, pivot_row, pivot_col);
    Scalar cosine = static_cast<Scalar>(0);
    Scalar sine = static_cast<Scalar>(0);
    if (lane == 0) {
      const Scalar angle =
          static_cast<Scalar>(0.5) * atan2(static_cast<Scalar>(2) * apq, aqq - app);
      cosine = cos(angle);
      sine = sin(angle);
    }
    cosine = __shfl_sync(warp_mask, cosine, 0);
    sine = __shfl_sync(warp_mask, sine, 0);
    for (std::size_t index = static_cast<std::size_t>(lane); index < dim; index += 32U) {
      if (index == pivot_row || index == pivot_col) {
        continue;
      }
      const Scalar aip = specialized_read<Scalar, Capacity>(matrix, index, pivot_row);
      const Scalar aiq = specialized_read<Scalar, Capacity>(matrix, index, pivot_col);
      specialized_write<Scalar, Capacity>(matrix, index, pivot_row, cosine * aip - sine * aiq);
      specialized_write<Scalar, Capacity>(
          matrix, pivot_row, index, specialized_read<Scalar, Capacity>(matrix, index, pivot_row));
      specialized_write<Scalar, Capacity>(matrix, index, pivot_col, sine * aip + cosine * aiq);
      specialized_write<Scalar, Capacity>(
          matrix, pivot_col, index, specialized_read<Scalar, Capacity>(matrix, index, pivot_col));
    }
    __syncwarp(warp_mask);
    if (lane == 0) {
      specialized_write<Scalar, Capacity>(
          matrix,
          pivot_row,
          pivot_row,
          cosine * cosine * app - static_cast<Scalar>(2) * sine * cosine * apq + sine * sine * aqq);
      specialized_write<Scalar, Capacity>(
          matrix,
          pivot_col,
          pivot_col,
          sine * sine * app + static_cast<Scalar>(2) * sine * cosine * apq + cosine * cosine * aqq);
      specialized_write<Scalar, Capacity>(matrix, pivot_row, pivot_col, static_cast<Scalar>(0));
      specialized_write<Scalar, Capacity>(matrix, pivot_col, pivot_row, static_cast<Scalar>(0));
    }
    __syncwarp(warp_mask);
    for (std::size_t row = static_cast<std::size_t>(lane); row < dim; row += 32U) {
      const Scalar vip = specialized_read<Scalar, Capacity>(eigenvectors, row, pivot_row);
      const Scalar viq = specialized_read<Scalar, Capacity>(eigenvectors, row, pivot_col);
      specialized_write<Scalar, Capacity>(eigenvectors, row, pivot_row, cosine * vip - sine * viq);
      specialized_write<Scalar, Capacity>(eigenvectors, row, pivot_col, sine * vip + cosine * viq);
    }
    __syncwarp(warp_mask);
  }
  if (!converged) {
    return false;
  }
  is_valid = true;
  for (std::size_t col = static_cast<std::size_t>(lane); col < dim; col += 32U) {
    eigenvalues[col] = specialized_read<Scalar, Capacity>(matrix, col, col);
    is_valid = is_valid && is_finite(eigenvalues[col]);
  }
  *matrix_scale = scale;
  __syncwarp(warp_mask);
  return specialized_warp_all(is_valid, warp_mask);
}

template <typename Scalar, std::size_t Capacity>
__device__ bool specialized_matrix_roots(Scalar* work,
                                         Scalar* eigenvectors,
                                         Scalar* eigenvalues,
                                         Scalar* output,
                                         Scalar* inverse_output,
                                         const std::size_t dim,
                                         const bool strict_positive,
                                         const int lane,
                                         const unsigned int warp_mask) {
  Scalar scale{};
  if (!specialized_symmetric_eigen<Scalar, Capacity>(
          work, eigenvectors, eigenvalues, dim, &scale, lane, warp_mask)) {
    return false;
  }
  Scalar local_maximum = static_cast<Scalar>(0);
  for (std::size_t col = static_cast<std::size_t>(lane); col < dim; col += 32U) {
    local_maximum = fmax(local_maximum, fabs(eigenvalues[col]));
  }
  const Scalar maximum = specialized_warp_max(local_maximum, warp_mask);
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(dim) * maximum;
  bool is_valid = true;
  for (std::size_t col = static_cast<std::size_t>(lane); col < dim; col += 32U) {
    if ((strict_positive && eigenvalues[col] <= threshold) ||
        (!strict_positive && eigenvalues[col] < -threshold)) {
      is_valid = false;
    }
  }
  if (!specialized_warp_all(is_valid, warp_mask)) {
    return false;
  }
  const std::size_t element_count = dim * dim;
  const Scalar scale_root = sqrt(scale);
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    const Scalar root = scale_root * sqrt(fmax(eigenvalues[col], static_cast<Scalar>(0)));
    specialized_write<Scalar, Capacity>(
        work, row, col, specialized_read<Scalar, Capacity>(eigenvectors, row, col) * root);
  }
  __syncwarp(warp_mask);
  is_valid = true;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    Scalar value = static_cast<Scalar>(0);
    Scalar inverse_value = static_cast<Scalar>(0);
    for (std::size_t inner = 0; inner < dim; ++inner) {
      value += specialized_read<Scalar, Capacity>(work, row, inner) *
               specialized_read<Scalar, Capacity>(eigenvectors, col, inner);
      if (inverse_output != nullptr) {
        inverse_value += specialized_read<Scalar, Capacity>(eigenvectors, row, inner) *
                         specialized_read<Scalar, Capacity>(eigenvectors, col, inner) /
                         (scale_root * sqrt(eigenvalues[inner]));
      }
    }
    is_valid =
        is_valid && is_finite(value) && (inverse_output == nullptr || is_finite(inverse_value));
    specialized_write<Scalar, Capacity>(output, row, col, value);
    if (inverse_output != nullptr) {
      specialized_write<Scalar, Capacity>(inverse_output, row, col, inverse_value);
    }
  }
  const bool is_warp_valid = specialized_warp_all(is_valid, warp_mask);
  __syncwarp(warp_mask);
  return is_warp_valid;
}

template <typename Scalar, std::size_t Capacity>
__device__ void specialized_multiply(const Scalar* left,
                                     const Scalar* right,
                                     Scalar* output,
                                     const std::size_t dim,
                                     const bool right_transposed,
                                     const int lane,
                                     const unsigned int warp_mask) {
  const std::size_t element_count = dim * dim;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t inner = 0; inner < dim; ++inner) {
      value += specialized_read<Scalar, Capacity>(left, row, inner) *
               specialized_read<Scalar, Capacity>(
                   right, right_transposed ? col : inner, right_transposed ? inner : col);
    }
    specialized_write<Scalar, Capacity>(output, row, col, value);
  }
  __syncwarp(warp_mask);
}

template <typename Scalar, std::size_t Capacity>
__device__ bool specialized_double_diagonalization(
    const CudaTensor3View<Scalar> mass_view,
    const CudaMatrixParameterView<Scalar> stiffness_view,
    const CudaVectorParameterView<Scalar> damping_factors,
    const CudaMatrixView<Scalar> velocity_error,
    const std::size_t env,
    const std::size_t dim,
    Scalar* mass_sqrt,
    Scalar* mass_sqrt_inverse,
    Scalar* transformed_stiffness_sqrt,
    Scalar* transformed_stiffness,
    Scalar* work,
    Scalar* eigenvalues,
    Scalar* velocity,
    Scalar* right_vector,
    Scalar* temporary,
    Scalar* torque,
    const int lane,
    const unsigned int warp_mask) {
  const std::size_t element_count = dim * dim;
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    specialized_write<Scalar, Capacity>(work, row, col, read_tensor3(mass_view, env, row, col));
  }
  __syncwarp(warp_mask);
  if (!specialized_validate_and_symmetrize<Scalar, Capacity>(work, dim, lane, warp_mask) ||
      !specialized_matrix_roots<Scalar, Capacity>(work,
                                                  transformed_stiffness,
                                                  eigenvalues,
                                                  mass_sqrt,
                                                  mass_sqrt_inverse,
                                                  dim,
                                                  true,
                                                  lane,
                                                  warp_mask)) {
    return false;
  }
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    specialized_write<Scalar, Capacity>(
        transformed_stiffness_sqrt, row, col, read_parameter_matrix(stiffness_view, env, row, col));
  }
  __syncwarp(warp_mask);
  if (!specialized_validate_and_symmetrize<Scalar, Capacity>(
          transformed_stiffness_sqrt, dim, lane, warp_mask)) {
    return false;
  }
  specialized_multiply<Scalar, Capacity>(
      mass_sqrt_inverse, transformed_stiffness_sqrt, work, dim, false, lane, warp_mask);
  specialized_multiply<Scalar, Capacity>(
      work, mass_sqrt_inverse, transformed_stiffness_sqrt, dim, true, lane, warp_mask);
  if (!specialized_symmetrize<Scalar, Capacity>(transformed_stiffness_sqrt, dim, lane, warp_mask)) {
    return false;
  }
  for (std::size_t index = static_cast<std::size_t>(lane); index < element_count; index += 32U) {
    const std::size_t row = index / dim;
    const std::size_t col = index % dim;
    specialized_write<Scalar, Capacity>(
        work, row, col, specialized_read<Scalar, Capacity>(transformed_stiffness_sqrt, row, col));
  }
  __syncwarp(warp_mask);
  if (!specialized_matrix_roots<Scalar, Capacity>(work,
                                                  transformed_stiffness,
                                                  eigenvalues,
                                                  transformed_stiffness_sqrt,
                                                  nullptr,
                                                  dim,
                                                  true,
                                                  lane,
                                                  warp_mask)) {
    return false;
  }
  for (std::size_t col = static_cast<std::size_t>(lane); col < dim; col += 32U) {
    velocity[col] = read_matrix(velocity_error, env, col);
  }
  __syncwarp(warp_mask);
  for (std::size_t row = static_cast<std::size_t>(lane); row < dim; row += 32U) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < dim; ++col) {
      value += specialized_read<Scalar, Capacity>(mass_sqrt, col, row) * velocity[col];
    }
    right_vector[row] = value;
  }
  __syncwarp(warp_mask);
  for (std::size_t row = static_cast<std::size_t>(lane); row < dim; row += 32U) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < dim; ++col) {
      value += specialized_read<Scalar, Capacity>(transformed_stiffness_sqrt, row, col) *
               right_vector[col];
    }
    temporary[row] = read_diagonal(damping_factors, env, row) * value;
  }
  __syncwarp(warp_mask);
  for (std::size_t row = static_cast<std::size_t>(lane); row < dim; row += 32U) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < dim; ++col) {
      value += specialized_read<Scalar, Capacity>(mass_sqrt, row, col) * temporary[col];
    }
    torque[row] = detail::damping_double_diagonalization_value(value);
  }
  __syncwarp(warp_mask);
  return true;
}

template <typename Scalar>
__device__ bool compute_double_diagonalization_torque_flat(
    const CudaTensor3View<Scalar> joint_mass_matrix,
    const CudaMatrixParameterView<Scalar> stiffness_matrix,
    const CudaVectorParameterView<Scalar> damping_factors,
    const CudaMatrixView<Scalar> joint_velocity_error,
    const std::size_t env,
    const std::size_t joint_dim,
    Scalar* mass_sqrt,
    Scalar* mass_sqrt_inv,
    Scalar* transformed_stiffness_sqrt,
    Scalar* transformed_stiffness,
    Scalar* work_matrix,
    Scalar* eigenvalues,
    Scalar* velocity,
    Scalar* right_vector,
    Scalar* temp_vector,
    Scalar* torque) {
  if (!copy_tensor3_square_to_flat(joint_mass_matrix, env, work_matrix, joint_dim) ||
      !validate_and_symmetrize_external_flat(work_matrix, joint_dim) ||
      !matrix_roots_from_work_flat(work_matrix,
                                   transformed_stiffness,
                                   eigenvalues,
                                   mass_sqrt,
                                   mass_sqrt_inv,
                                   joint_dim,
                                   true)) {
    return false;
  }
  if (!copy_parameter_square_to_flat(
          stiffness_matrix, env, transformed_stiffness_sqrt, joint_dim) ||
      !validate_and_symmetrize_external_flat(transformed_stiffness_sqrt, joint_dim)) {
    return false;
  }
  multiply_flat(mass_sqrt_inv, transformed_stiffness_sqrt, work_matrix, joint_dim);
  multiply_flat_right_transpose(work_matrix, mass_sqrt_inv, transformed_stiffness_sqrt, joint_dim);
  if (!symmetrize_derived_flat(transformed_stiffness_sqrt, joint_dim)) {
    return false;
  }
  if (!matrix_sqrt_flat(transformed_stiffness_sqrt,
                        transformed_stiffness_sqrt,
                        work_matrix,
                        transformed_stiffness,
                        eigenvalues,
                        joint_dim,
                        true)) {
    return false;
  }
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    velocity[dim] = read_matrix(joint_velocity_error, env, dim);
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(mass_sqrt, col, row) * velocity[col];
    }
    right_vector[row] = value;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(transformed_stiffness_sqrt, row, col) * right_vector[col];
    }
    temp_vector[row] = read_diagonal(damping_factors, env, row) * value;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(mass_sqrt, row, col) * temp_vector[col];
    }
    torque[row] = detail::damping_double_diagonalization_value(value);
  }
  return true;
}

template <typename Scalar>
__global__ void apply_diagonal_parameters_kernel(CudaVectorParameterView<Scalar> diagonal,
                                                 std::size_t num_envs,
                                                 std::size_t joint_dim,
                                                 std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    const Scalar value = read_diagonal(diagonal, env, dim);
    if (!is_finite(value) || value < static_cast<Scalar>(0)) {
      flags |= kInvalidParameterFlag;
      break;
    }
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_dense_parameters_kernel(CudaMatrixParameterView<Scalar> matrix,
                                              std::size_t num_envs,
                                              std::size_t joint_dim,
                                              std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!is_finite(read_parameter_matrix(matrix, env, row, col))) {
        flags |= kInvalidParameterFlag;
        break;
      }
    }
    if (flags != 0U) {
      break;
    }
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_impedance_parameters_kernel(CudaMatrixParameterView<Scalar> stiffness_matrix,
                                                  CudaVectorParameterView<Scalar> damping_factors,
                                                  std::size_t num_envs,
                                                  std::size_t joint_dim,
                                                  std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!dense_parameter_is_valid(stiffness_matrix, env, joint_dim) ||
      !damping_factors_are_valid(damping_factors, env, joint_dim)) {
    flags |= kInvalidParameterFlag;
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_nullspace_parameters_kernel(CudaMatrixParameterView<Scalar> stiffness_matrix,
                                                  CudaMatrixParameterView<Scalar> damping_matrix,
                                                  std::size_t num_envs,
                                                  std::size_t joint_dim,
                                                  std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!dense_parameter_is_valid(stiffness_matrix, env, joint_dim) ||
      !dense_parameter_is_valid(damping_matrix, env, joint_dim)) {
    flags |= kInvalidParameterFlag;
  }
  parameter_status[env] = flags;
}

__global__ void reset_stateless_joint_kernel(const std::uint8_t* reset_mask,
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
__global__ void step_diagonal_kernel(CudaMatrixView<Scalar> joint_error,
                                     CudaVectorParameterView<Scalar> diagonal,
                                     const std::uint64_t* parameter_status,
                                     CudaMutableMatrixView<Scalar> joint_torque,
                                     std::size_t num_envs,
                                     std::size_t joint_dim,
                                     std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    const Scalar gain = read_diagonal(diagonal, env, dim);
    const Scalar error = read_matrix(joint_error, env, dim);
    if (!is_finite(gain) || gain < static_cast<Scalar>(0)) {
      flags |= kInvalidParameterFlag;
    }
    if (!is_finite(error)) {
      flags |= kNonFiniteInputFlag;
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      write_matrix(joint_torque, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    write_matrix(joint_torque,
                 env,
                 dim,
                 detail::joint_diagonal_feedback_value(read_diagonal(diagonal, env, dim),
                                                       read_matrix(joint_error, env, dim)));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_dense_kernel(CudaMatrixView<Scalar> joint_error,
                                  CudaMatrixParameterView<Scalar> matrix,
                                  const std::uint64_t* parameter_status,
                                  CudaMutableMatrixView<Scalar> joint_torque,
                                  std::size_t num_envs,
                                  std::size_t joint_dim,
                                  std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    if (!is_finite(read_matrix(joint_error, env, dim))) {
      flags |= kNonFiniteInputFlag;
    }
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      if (!is_finite(read_parameter_matrix(matrix, env, row, col))) {
        flags |= kInvalidParameterFlag;
      }
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      write_matrix(joint_torque, env, dim, static_cast<Scalar>(0));
    }
    status[env] = flags;
    return;
  }

  const Scalar* matrix_data = matrix.data;
  if (!matrix.shared && matrix.rank != 2) {
    matrix_data += static_cast<std::ptrdiff_t>(env) * matrix.env_stride;
  }
  const detail::ConstMatrixView<Scalar> matrix_view{
      matrix_data, joint_dim, joint_dim, matrix.row_stride, matrix.col_stride};
  const detail::ConstVectorView<Scalar> error_view{
      joint_error.data + static_cast<std::ptrdiff_t>(env) * joint_error.row_stride,
      joint_dim,
      joint_error.col_stride};
  for (std::size_t row = 0; row < joint_dim; ++row) {
    write_matrix(joint_torque, env, row, detail::dense_feedback_row(matrix_view, error_view, row));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_double_diagonalization_kernel(CudaMatrixView<Scalar> joint_velocity_error,
                                                   CudaTensor3View<Scalar> joint_mass_matrix,
                                                   CudaTensor3View<Scalar> stiffness_matrix,
                                                   CudaVectorParameterView<Scalar> damping_factors,
                                                   const std::uint64_t* parameter_status,
                                                   CudaMutableMatrixView<Scalar> joint_torque,
                                                   std::size_t num_envs,
                                                   std::size_t joint_dim,
                                                   std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!damping_factors_are_valid(damping_factors, env, joint_dim)) {
    flags |= kInvalidParameterFlag;
  }
  if (!matrix_input_is_finite(joint_velocity_error, env, joint_dim) ||
      !tensor3_square_is_finite(joint_mass_matrix, env, joint_dim) ||
      !tensor3_square_is_finite(stiffness_matrix, env, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }

  extern __shared__ std::uint64_t shared_storage[];
  Scalar* storage = reinterpret_cast<Scalar*>(shared_storage);
  Scalar* mass_sqrt = storage;
  Scalar* mass_sqrt_inv = mass_sqrt + kJointMatrixElements;
  Scalar* transformed_stiffness_sqrt = mass_sqrt_inv + kJointMatrixElements;
  Scalar* transformed_stiffness = transformed_stiffness_sqrt + kJointMatrixElements;
  Scalar* work_matrix = transformed_stiffness + kJointMatrixElements;
  Scalar* eigenvalues = work_matrix + kJointMatrixElements;
  Scalar* velocity = eigenvalues + kMaxJointDim;
  Scalar* right_vector = velocity + kMaxJointDim;
  Scalar* temp_vector = right_vector + kMaxJointDim;
  Scalar* torque = temp_vector + kMaxJointDim;
  if (!compute_double_diagonalization_torque_flat(joint_mass_matrix,
                                                  stiffness_matrix,
                                                  damping_factors,
                                                  joint_velocity_error,
                                                  env,
                                                  joint_dim,
                                                  mass_sqrt,
                                                  mass_sqrt_inv,
                                                  transformed_stiffness_sqrt,
                                                  transformed_stiffness,
                                                  work_matrix,
                                                  eigenvalues,
                                                  velocity,
                                                  right_vector,
                                                  temp_vector,
                                                  torque)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    if (!is_finite(torque[dim])) {
      flags |= kInvalidInputFlag | kSafeModeFlag;
      zero_joint_output(joint_torque, env, joint_dim);
      status[env] = flags;
      return;
    }
    write_matrix(joint_torque, env, dim, torque[dim]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_joint_impedance_kernel(CudaMatrixView<Scalar> joint_position_error,
                                            CudaMatrixView<Scalar> joint_velocity_error,
                                            CudaMatrixView<Scalar> desired_acceleration,
                                            CudaTensor3View<Scalar> joint_mass_matrix,
                                            CudaMatrixParameterView<Scalar> stiffness_matrix,
                                            CudaVectorParameterView<Scalar> damping_factors,
                                            const std::uint64_t* parameter_status,
                                            CudaMutableMatrixView<Scalar> joint_torque,
                                            std::size_t num_envs,
                                            std::size_t joint_dim,
                                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!dense_parameter_is_valid(stiffness_matrix, env, joint_dim) ||
      !damping_factors_are_valid(damping_factors, env, joint_dim)) {
    flags |= kInvalidParameterFlag;
  }
  if (!matrix_input_is_finite(joint_position_error, env, joint_dim) ||
      !matrix_input_is_finite(joint_velocity_error, env, joint_dim) ||
      !matrix_input_is_finite(desired_acceleration, env, joint_dim) ||
      !tensor3_square_is_finite(joint_mass_matrix, env, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }

  extern __shared__ std::uint64_t shared_storage[];
  Scalar* storage = reinterpret_cast<Scalar*>(shared_storage);
  Scalar* mass_sqrt = storage;
  Scalar* mass_sqrt_inv = mass_sqrt + kJointMatrixElements;
  Scalar* transformed_stiffness_sqrt = mass_sqrt_inv + kJointMatrixElements;
  Scalar* transformed_stiffness = transformed_stiffness_sqrt + kJointMatrixElements;
  Scalar* work_matrix = transformed_stiffness + kJointMatrixElements;
  Scalar* eigenvalues = work_matrix + kJointMatrixElements;
  Scalar* velocity = eigenvalues + kMaxJointDim;
  Scalar* right_vector = velocity + kMaxJointDim;
  Scalar* temp_vector = right_vector + kMaxJointDim;
  Scalar* damping_torque = temp_vector + kMaxJointDim;
  if (!compute_double_diagonalization_torque_flat(joint_mass_matrix,
                                                  stiffness_matrix,
                                                  damping_factors,
                                                  joint_velocity_error,
                                                  env,
                                                  joint_dim,
                                                  mass_sqrt,
                                                  mass_sqrt_inv,
                                                  transformed_stiffness_sqrt,
                                                  transformed_stiffness,
                                                  work_matrix,
                                                  eigenvalues,
                                                  velocity,
                                                  right_vector,
                                                  temp_vector,
                                                  damping_torque)) {
    flags |= kInvalidInputFlag | kInvalidParameterFlag | kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar stiffness_torque = static_cast<Scalar>(0);
    Scalar acceleration_torque = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      const Scalar stiffness =
          row == col
              ? read_parameter_matrix(stiffness_matrix, env, row, col)
              : static_cast<Scalar>(0.5) * read_parameter_matrix(stiffness_matrix, env, row, col) +
                    static_cast<Scalar>(0.5) *
                        read_parameter_matrix(stiffness_matrix, env, col, row);
      const Scalar mass =
          row == col
              ? read_tensor3(joint_mass_matrix, env, row, col)
              : static_cast<Scalar>(0.5) * read_tensor3(joint_mass_matrix, env, row, col) +
                    static_cast<Scalar>(0.5) * read_tensor3(joint_mass_matrix, env, col, row);
      stiffness_torque += stiffness * read_matrix(joint_position_error, env, col);
      acceleration_torque += mass * read_matrix(desired_acceleration, env, col);
    }
    temp_vector[row] =
        detail::joint_impedance_value(stiffness_torque, damping_torque[row], acceleration_torque);
    if (!is_finite(temp_vector[row])) {
      flags |= kInvalidInputFlag | kInvalidParameterFlag | kSafeModeFlag;
      zero_joint_output(joint_torque, env, joint_dim);
      status[env] = flags;
      return;
    }
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    write_matrix(joint_torque, env, row, temp_vector[row]);
  }
  status[env] = flags;
}

template <typename Scalar, std::size_t Capacity, int WarpsPerBlock>
constexpr std::size_t specialized_impedance_shared_bytes() {
  constexpr std::size_t kMatrixBufferCount = 5;
  constexpr std::size_t kVectorBufferCount = 5;
  constexpr std::size_t kScalarsPerWarp =
      kMatrixBufferCount * Capacity * Capacity + kVectorBufferCount * Capacity;
  constexpr std::size_t kBytes = WarpsPerBlock * kScalarsPerWarp * sizeof(Scalar);
  static_assert(kBytes <= 48U * 1024U, "specialized impedance CTA exceeds 48 KiB shared memory");
  return kBytes;
}

template <typename Scalar, std::size_t Capacity, std::size_t ExactDim, int WarpsPerBlock>
__global__ void step_joint_impedance_specialized_kernel(
    CudaMatrixView<Scalar> joint_position_error,
    CudaMatrixView<Scalar> joint_velocity_error,
    CudaMatrixView<Scalar> desired_acceleration,
    CudaTensor3View<Scalar> joint_mass_matrix,
    CudaMatrixParameterView<Scalar> stiffness_matrix,
    CudaVectorParameterView<Scalar> damping_factors,
    const std::uint64_t* parameter_status,
    CudaMutableMatrixView<Scalar> joint_torque,
    const std::size_t num_envs,
    const std::size_t runtime_joint_dim,
    const std::size_t environment_offset,
    std::uint64_t* status) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr unsigned int kWarpMask = 0xffffffffU;
  const std::size_t env = environment_offset +
                          blockIdx.x * static_cast<std::size_t>(WarpsPerBlock) +
                          static_cast<std::size_t>(warp);
  if (env >= num_envs) {
    return;
  }
  const std::size_t joint_dim = ExactDim == 0 ? runtime_joint_dim : ExactDim;
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  bool are_parameters_valid = true;
  for (std::size_t index = static_cast<std::size_t>(lane); index < joint_dim * joint_dim;
       index += 32U) {
    are_parameters_valid =
        are_parameters_valid && is_finite(read_parameter_matrix(
                                    stiffness_matrix, env, index / joint_dim, index % joint_dim));
  }
  for (std::size_t dim = static_cast<std::size_t>(lane); dim < joint_dim; dim += 32U) {
    const Scalar damping = read_diagonal(damping_factors, env, dim);
    are_parameters_valid =
        are_parameters_valid && is_finite(damping) && damping >= static_cast<Scalar>(0);
  }
  if (!specialized_warp_all(are_parameters_valid, kWarpMask)) {
    flags |= kInvalidParameterFlag;
  }
  bool are_inputs_finite = true;
  for (std::size_t dim = static_cast<std::size_t>(lane); dim < joint_dim; dim += 32U) {
    are_inputs_finite = are_inputs_finite &&
                        is_finite(read_matrix(joint_position_error, env, dim)) &&
                        is_finite(read_matrix(joint_velocity_error, env, dim)) &&
                        is_finite(read_matrix(desired_acceleration, env, dim));
  }
  for (std::size_t index = static_cast<std::size_t>(lane); index < joint_dim * joint_dim;
       index += 32U) {
    are_inputs_finite =
        are_inputs_finite &&
        is_finite(read_tensor3(joint_mass_matrix, env, index / joint_dim, index % joint_dim));
  }
  if (!specialized_warp_all(are_inputs_finite, kWarpMask)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    for (std::size_t dim = static_cast<std::size_t>(lane); dim < joint_dim; dim += 32U) {
      write_matrix(joint_torque, env, dim, static_cast<Scalar>(0));
    }
    if (lane == 0) {
      status[env] = flags;
    }
    return;
  }

  extern __shared__ std::uint64_t shared_storage[];
  Scalar* storage = reinterpret_cast<Scalar*>(shared_storage);
  constexpr std::size_t kMatrixBufferCount = 5;
  constexpr std::size_t kVectorBufferCount = 5;
  constexpr std::size_t kScalarsPerWarp =
      kMatrixBufferCount * Capacity * Capacity + kVectorBufferCount * Capacity;
  Scalar* const warp_storage = storage + static_cast<std::size_t>(warp) * kScalarsPerWarp;
  Scalar* const mass_sqrt = warp_storage;
  Scalar* const mass_sqrt_inverse = mass_sqrt + Capacity * Capacity;
  Scalar* const transformed_stiffness_sqrt = mass_sqrt_inverse + Capacity * Capacity;
  Scalar* const transformed_stiffness = transformed_stiffness_sqrt + Capacity * Capacity;
  Scalar* const work = transformed_stiffness + Capacity * Capacity;
  Scalar* const eigenvalues = work + Capacity * Capacity;
  Scalar* const velocity = eigenvalues + Capacity;
  Scalar* const right_vector = velocity + Capacity;
  Scalar* const temporary = right_vector + Capacity;
  Scalar* const damping_torque = temporary + Capacity;
  if (!specialized_double_diagonalization<Scalar, Capacity>(joint_mass_matrix,
                                                            stiffness_matrix,
                                                            damping_factors,
                                                            joint_velocity_error,
                                                            env,
                                                            joint_dim,
                                                            mass_sqrt,
                                                            mass_sqrt_inverse,
                                                            transformed_stiffness_sqrt,
                                                            transformed_stiffness,
                                                            work,
                                                            eigenvalues,
                                                            velocity,
                                                            right_vector,
                                                            temporary,
                                                            damping_torque,
                                                            lane,
                                                            kWarpMask)) {
    flags |= kInvalidInputFlag | kInvalidParameterFlag | kSafeModeFlag;
    for (std::size_t dim = static_cast<std::size_t>(lane); dim < joint_dim; dim += 32U) {
      write_matrix(joint_torque, env, dim, static_cast<Scalar>(0));
    }
    if (lane == 0) {
      status[env] = flags;
    }
    return;
  }
  bool is_torque_finite = true;
  for (std::size_t row = static_cast<std::size_t>(lane); row < joint_dim; row += 32U) {
    Scalar stiffness_torque = static_cast<Scalar>(0);
    Scalar acceleration_torque = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      const Scalar stiffness =
          row == col
              ? read_parameter_matrix(stiffness_matrix, env, row, col)
              : static_cast<Scalar>(0.5) * (read_parameter_matrix(stiffness_matrix, env, row, col) +
                                            read_parameter_matrix(stiffness_matrix, env, col, row));
      const Scalar mass =
          row == col ? read_tensor3(joint_mass_matrix, env, row, col)
                     : static_cast<Scalar>(0.5) * (read_tensor3(joint_mass_matrix, env, row, col) +
                                                   read_tensor3(joint_mass_matrix, env, col, row));
      stiffness_torque += stiffness * read_matrix(joint_position_error, env, col);
      acceleration_torque += mass * read_matrix(desired_acceleration, env, col);
    }
    temporary[row] =
        detail::joint_impedance_value(stiffness_torque, damping_torque[row], acceleration_torque);
    is_torque_finite = is_torque_finite && is_finite(temporary[row]);
  }
  if (!specialized_warp_all(is_torque_finite, kWarpMask)) {
    flags |= kInvalidInputFlag | kInvalidParameterFlag | kSafeModeFlag;
    for (std::size_t dim = static_cast<std::size_t>(lane); dim < joint_dim; dim += 32U) {
      write_matrix(joint_torque, env, dim, static_cast<Scalar>(0));
    }
    if (lane == 0) {
      status[env] = flags;
    }
    return;
  }
  __syncwarp(kWarpMask);
  for (std::size_t row = static_cast<std::size_t>(lane); row < joint_dim; row += 32U) {
    write_matrix(joint_torque, env, row, temporary[row]);
  }
  if (lane == 0) {
    status[env] = flags;
  }
}

template <typename Scalar>
__global__ void step_nullspace_kernel(CudaMatrixView<Scalar> joint_position_error,
                                      CudaMatrixView<Scalar> joint_velocity,
                                      CudaTensor3View<Scalar> joint_mass_matrix,
                                      CudaTensor3View<Scalar> jacobian,
                                      CudaMatrixParameterView<Scalar> stiffness_matrix,
                                      CudaMatrixParameterView<Scalar> damping_matrix,
                                      const std::uint64_t* parameter_status,
                                      CudaMutableMatrixView<Scalar> joint_torque,
                                      std::size_t num_envs,
                                      std::size_t joint_dim,
                                      std::size_t task_dim,
                                      std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!dense_parameter_is_valid(stiffness_matrix, env, joint_dim) ||
      !dense_parameter_is_valid(damping_matrix, env, joint_dim)) {
    flags |= kInvalidParameterFlag;
  }
  if (!matrix_input_is_finite(joint_position_error, env, joint_dim) ||
      !matrix_input_is_finite(joint_velocity, env, joint_dim) ||
      !tensor3_square_is_finite(joint_mass_matrix, env, joint_dim) ||
      !tensor3_matrix_is_finite(jacobian, env, task_dim, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }

  extern __shared__ std::uint64_t shared_storage[];
  Scalar* storage = reinterpret_cast<Scalar*>(shared_storage);
  Scalar* mass = storage;
  Scalar* mass_inv = mass + kJointMatrixElements;
  Scalar* work_matrix = mass_inv + kJointMatrixElements;
  Scalar* task_matrix = work_matrix + kJointMatrixElements;
  Scalar* task_matrix_inv = task_matrix + kJointMatrixElements;
  Scalar* desired_torque = task_matrix_inv + kJointMatrixElements;
  Scalar* task_desired = desired_torque + kMaxJointDim;
  Scalar* task_lambda = task_desired + kMaxJointDim;
  if (!copy_tensor3_square_to_flat(joint_mass_matrix, env, mass, joint_dim) ||
      !inverse_flat(mass, mass_inv, work_matrix, joint_dim)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }

  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar stiffness_torque = static_cast<Scalar>(0);
    Scalar damping_torque = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      stiffness_torque += read_parameter_matrix(stiffness_matrix, env, row, col) *
                          read_matrix(joint_position_error, env, col);
      damping_torque -= read_parameter_matrix(damping_matrix, env, row, col) *
                        read_matrix(joint_velocity, env, col);
    }
    desired_torque[row] = detail::nullspace_feedback_value(stiffness_torque, damping_torque);
  }

  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < task_dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t left_joint = 0; left_joint < joint_dim; ++left_joint) {
        const Scalar jac_left = read_tensor3(jacobian, env, row, left_joint);
        for (std::size_t right_joint = 0; right_joint < joint_dim; ++right_joint) {
          value += jac_left * flat_read(mass_inv, left_joint, right_joint) *
                   read_tensor3(jacobian, env, col, right_joint);
        }
      }
      flat_write(task_matrix, row, col, value);
    }
  }
  if (!inverse_flat(task_matrix, task_matrix_inv, work_matrix, task_dim)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_joint_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }
  // Torque-space projection N^T tau_0 = tau_0 - J^T Lambda J M^-1 tau_0.
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += flat_read(mass_inv, row, col) * desired_torque[col];
    }
    task_desired[row] = value;
  }
  for (std::size_t row = 0; row < task_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < joint_dim; ++col) {
      value += read_tensor3(jacobian, env, row, col) * task_desired[col];
    }
    task_lambda[row] = value;
  }
  for (std::size_t row = 0; row < task_dim; ++row) {
    Scalar value = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < task_dim; ++col) {
      value += flat_read(task_matrix_inv, row, col) * task_lambda[col];
    }
    work_matrix[row] = value;
  }
  for (std::size_t row = 0; row < joint_dim; ++row) {
    Scalar correction = static_cast<Scalar>(0);
    for (std::size_t task_row = 0; task_row < task_dim; ++task_row) {
      correction += read_tensor3(jacobian, env, task_row, row) * work_matrix[task_row];
    }
    write_matrix(joint_torque,
                 env,
                 row,
                 detail::nullspace_projected_torque_value(desired_torque[row], correction));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_bimanual_independent_nullspace_kernel(
    CudaMatrixView<Scalar> joint_position_error,
    CudaMatrixView<Scalar> joint_velocity,
    CudaTensor3View<Scalar> joint_mass_matrix,
    CudaTensor3View<Scalar> arm1_jacobian,
    CudaTensor3View<Scalar> arm2_jacobian,
    CudaMatrixParameterView<Scalar> stiffness_matrix,
    CudaMatrixParameterView<Scalar> damping_matrix,
    const std::uint64_t* parameter_status,
    CudaMutableMatrixView<Scalar> joint_torque,
    std::size_t num_envs,
    std::uint64_t* status) {
  constexpr std::size_t kArmJoints = 7;
  constexpr std::size_t kArmTasks = 6;
  constexpr std::size_t kJoints = 14;
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!dense_parameter_is_valid(stiffness_matrix, env, kJoints) ||
      !dense_parameter_is_valid(damping_matrix, env, kJoints)) {
    flags |= kInvalidParameterFlag;
  }
  if (!matrix_input_is_finite(joint_position_error, env, kJoints) ||
      !matrix_input_is_finite(joint_velocity, env, kJoints) ||
      !tensor3_square_is_finite(joint_mass_matrix, env, kJoints) ||
      !tensor3_matrix_is_finite(arm1_jacobian, env, kArmTasks, kArmJoints) ||
      !tensor3_matrix_is_finite(arm2_jacobian, env, kArmTasks, kArmJoints)) {
    flags |= kNonFiniteInputFlag;
  }
  for (std::size_t row = 0; row < kArmJoints; ++row) {
    for (std::size_t col = 0; col < kArmJoints; ++col) {
      if (read_parameter_matrix(stiffness_matrix, env, row, kArmJoints + col) !=
              static_cast<Scalar>(0) ||
          read_parameter_matrix(stiffness_matrix, env, kArmJoints + row, col) !=
              static_cast<Scalar>(0) ||
          read_parameter_matrix(damping_matrix, env, row, kArmJoints + col) !=
              static_cast<Scalar>(0) ||
          read_parameter_matrix(damping_matrix, env, kArmJoints + row, col) !=
              static_cast<Scalar>(0)) {
        flags |= kInvalidParameterFlag;
      }
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    zero_joint_output(joint_torque, env, kJoints);
    status[env] = flags | kSafeModeFlag;
    return;
  }
  // Compact per-arm scratch uses a seven-column stride, including the 6x6 task solve.
  Scalar mass[kArmJoints * kArmJoints]{};
  Scalar mass_inverse[kArmJoints * kArmJoints]{};
  Scalar work[kArmJoints * kArmJoints]{};
  Scalar task_matrix[kArmJoints * kArmJoints]{};
  Scalar task_inverse[kArmJoints * kArmJoints]{};
  Scalar desired_torque[kArmJoints]{};
  Scalar mass_inverse_torque[kArmJoints]{};
  Scalar task_torque[kArmTasks]{};
  for (std::size_t arm = 0; arm < 2; ++arm) {
    const std::size_t offset = arm * kArmJoints;
    const CudaTensor3View<Scalar> jacobian = arm == 0 ? arm1_jacobian : arm2_jacobian;
    for (std::size_t row = 0; row < kArmJoints; ++row) {
      desired_torque[row] = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kArmJoints; ++col) {
        flat_write(mass,
                   row,
                   col,
                   read_tensor3(joint_mass_matrix, env, offset + row, offset + col),
                   kArmJoints);
        desired_torque[row] +=
            read_parameter_matrix(stiffness_matrix, env, offset + row, offset + col) *
                read_matrix(joint_position_error, env, offset + col) -
            read_parameter_matrix(damping_matrix, env, offset + row, offset + col) *
                read_matrix(joint_velocity, env, offset + col);
      }
    }
    if (!inverse_flat(mass, mass_inverse, work, kArmJoints, kArmJoints)) {
      flags |= kInvalidInputFlag | kSafeModeFlag;
      break;
    }
    for (std::size_t row = 0; row < kArmTasks; ++row) {
      for (std::size_t col = 0; col < kArmTasks; ++col) {
        Scalar value = static_cast<Scalar>(0);
        for (std::size_t left = 0; left < kArmJoints; ++left) {
          for (std::size_t right = 0; right < kArmJoints; ++right) {
            value += read_tensor3(jacobian, env, row, left) *
                     flat_read(mass_inverse, left, right, kArmJoints) *
                     read_tensor3(jacobian, env, col, right);
          }
        }
        flat_write(task_matrix, row, col, value, kArmJoints);
      }
    }
    if (!inverse_flat(task_matrix, task_inverse, work, kArmTasks, kArmJoints)) {
      flags |= kInvalidInputFlag | kSafeModeFlag;
      break;
    }
    for (std::size_t row = 0; row < kArmJoints; ++row) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kArmJoints; ++col) {
        value += flat_read(mass_inverse, row, col, kArmJoints) * desired_torque[col];
      }
      mass_inverse_torque[row] = value;
    }
    for (std::size_t row = 0; row < kArmTasks; ++row) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kArmJoints; ++col) {
        value += read_tensor3(jacobian, env, row, col) * mass_inverse_torque[col];
      }
      task_torque[row] = value;
    }
    for (std::size_t row = 0; row < kArmTasks; ++row) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kArmTasks; ++col) {
        value += flat_read(task_inverse, row, col, kArmJoints) * task_torque[col];
      }
      work[row] = value;
    }
    for (std::size_t joint = 0; joint < kArmJoints; ++joint) {
      Scalar correction = static_cast<Scalar>(0);
      for (std::size_t row = 0; row < kArmTasks; ++row) {
        correction += read_tensor3(jacobian, env, row, joint) * work[row];
      }
      write_matrix(joint_torque, env, offset + joint, desired_torque[joint] - correction);
    }
  }
  if ((flags & kSafeModeFlag) != 0U) {
    zero_joint_output(joint_torque, env, kJoints);
  }
  status[env] = flags;
}

inline Result cuda_launch_result() {
  const cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return Result{ResultCode::kInvalidArgument, cudaGetErrorString(error)};
  }
  return Result{ResultCode::kOk, "ok"};
}

template <CudaTensorLaunchFamily Family, typename Scalar>
inline CudaLaunchConfig launch_config(const std::size_t num_envs) {
  constexpr int kThreads = cuda_tensor_tuned_threads<Family, Scalar>();
  return CudaLaunchConfig{static_cast<int>((num_envs + kThreads - 1) / kThreads), kThreads};
}

template <typename Scalar>
constexpr std::size_t double_diagonalization_shared_bytes() {
  constexpr std::size_t kMatrixBufferCount = 5;
  constexpr std::size_t kVectorBufferCount = 5;
  constexpr std::size_t kScalarCount =
      kMatrixBufferCount * kJointMatrixElements + kVectorBufferCount * kMaxJointDim;
  return kScalarCount * sizeof(Scalar);
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
                                   const std::size_t joint_dim) {
  if (!is_cuda_matrix(output)) {
    return Result{ResultCode::kInvalidView,
                  "output tensor must be CUDA row-major [num_envs, joint_dim]"};
  }
  if (output.dtype != dtype || output.shape[0] != static_cast<std::int64_t>(num_envs) ||
      output.shape[1] != static_cast<std::int64_t>(joint_dim)) {
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

inline Result validate_cuda_diagonal(const ParameterView* diagonal,
                                     const DType dtype,
                                     const std::size_t num_envs,
                                     const std::size_t joint_dim) {
  if (diagonal == nullptr || diagonal->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "diagonal joint parameter is missing"};
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat64) {
    return Result{ResultCode::kInvalidView, "diagonal joint parameter dtype must be float"};
  }
  const TensorView& values = diagonal->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "diagonal joint parameter must be a CUDA tensor matching input dtype"};
  }
  if (diagonal->shared) {
    if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(joint_dim) &&
        values.strides[0] >= 1) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView, "shared diagonal parameter must have [joint_dim]"};
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(joint_dim) && values.strides[1] == 1 &&
      values.strides[0] >= values.shape[1]) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView,
                "batched diagonal parameter must have [num_envs, joint_dim]"};
}

inline Result validate_cuda_dense(const ParameterView* matrix,
                                  const DType dtype,
                                  const std::size_t num_envs,
                                  const std::size_t joint_dim) {
  if (matrix == nullptr || matrix->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "dense joint parameter is missing"};
  }
  if (dtype != DType::kFloat32 && dtype != DType::kFloat64) {
    return Result{ResultCode::kInvalidView, "dense joint parameter dtype must be float"};
  }
  const TensorView& values = matrix->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "dense joint parameter must be a CUDA tensor matching input dtype"};
  }
  if (matrix->shared) {
    if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(joint_dim) &&
        values.shape[1] == static_cast<std::int64_t>(joint_dim) && values.strides[1] == 1 &&
        values.strides[0] >= values.shape[1]) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView,
                  "shared dense parameter must have [joint_dim, joint_dim]"};
  }
  if (values.rank == 3 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(joint_dim) &&
      values.shape[2] == static_cast<std::int64_t>(joint_dim) && values.strides[2] == 1 &&
      values.strides[1] >= values.shape[2] &&
      values.strides[0] >= values.shape[1] * values.strides[1]) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView,
                "batched dense parameter must have [num_envs, joint_dim, joint_dim]"};
}

inline Result validate_cuda_tensor3_square(const TensorView& view,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const std::size_t joint_dim) {
  if (view.data == nullptr || view.device != Device::kCuda || view.dtype != dtype ||
      view.layout != Layout::kRowMajor || view.rank != 3 ||
      view.shape[0] != static_cast<std::int64_t>(num_envs) ||
      view.shape[1] != static_cast<std::int64_t>(joint_dim) ||
      view.shape[2] != static_cast<std::int64_t>(joint_dim) || view.strides[2] != 1 ||
      view.strides[1] < view.shape[2] || view.strides[0] < view.shape[1] * view.strides[1]) {
    return Result{ResultCode::kInvalidView,
                  "joint square tensor must be CUDA row-major [num_envs, joint_dim, joint_dim]"};
  }
  if (joint_dim == 0 || joint_dim > kMaxJointDim) {
    return Result{ResultCode::kInvalidView,
                  "joint_dim exceeds tensor joint double-diagonalization bound"};
  }
  return Result{ResultCode::kOk, name};
}

inline Result validate_cuda_tensor3_matrix(const TensorView& view,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const std::size_t rows,
                                           const std::size_t cols) {
  if (view.data == nullptr || view.device != Device::kCuda || view.dtype != dtype ||
      view.layout != Layout::kRowMajor || view.rank != 3 ||
      view.shape[0] != static_cast<std::int64_t>(num_envs) ||
      view.shape[1] != static_cast<std::int64_t>(rows) ||
      view.shape[2] != static_cast<std::int64_t>(cols) || view.strides[2] != 1 ||
      view.strides[1] < view.shape[2] || view.strides[0] < view.shape[1] * view.strides[1]) {
    return Result{ResultCode::kInvalidView,
                  "joint matrix tensor must be CUDA row-major [num_envs, rows, cols]"};
  }
  return Result{ResultCode::kOk, name};
}

template <typename Scalar>
CudaMatrixView<Scalar> as_cuda_matrix(const TensorView& view) {
  return CudaMatrixView<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaTensor3View<Scalar> as_cuda_tensor3(const TensorView& view) {
  return CudaTensor3View<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaMutableMatrixView<Scalar> as_cuda_mutable_matrix(const MutableTensorView& view) {
  return CudaMutableMatrixView<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1]};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> as_cuda_diagonal(const ParameterView& diagonal) {
  const TensorView& values = diagonal.values;
  const bool shared = diagonal.shared || values.rank == 1;
  return CudaVectorParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         shared,
                                         values.rank,
                                         values.rank == 2 ? values.strides[0] : 0,
                                         values.rank == 2 ? values.strides[1] : values.strides[0]};
}

template <typename Scalar>
CudaMatrixParameterView<Scalar> as_cuda_dense(const ParameterView& matrix) {
  const TensorView& values = matrix.values;
  const bool shared = matrix.shared || values.rank == 2;
  return CudaMatrixParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         shared,
                                         values.rank,
                                         values.rank == 3 ? values.strides[0] : 0,
                                         values.rank == 3 ? values.strides[1] : values.strides[0],
                                         values.rank == 3 ? values.strides[2] : values.strides[1]};
}

template <typename Scalar>
Result launch_apply_diagonal(const ParameterView& diagonal,
                             const std::size_t num_envs,
                             const std::size_t joint_dim,
                             MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointApplyDiagonal, Scalar>(num_envs);
  apply_diagonal_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_diagonal<Scalar>(diagonal), num_envs, joint_dim, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_dense(const ParameterView& matrix,
                          const std::size_t num_envs,
                          const std::size_t joint_dim,
                          MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointApplyDense, Scalar>(num_envs);
  apply_dense_parameters_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_dense<Scalar>(matrix), num_envs, joint_dim, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_impedance(const ParameterView& stiffness_matrix,
                              const ParameterView& damping_factors,
                              const std::size_t num_envs,
                              const std::size_t joint_dim,
                              MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointApplyImpedance, Scalar>(num_envs);
  apply_impedance_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_dense<Scalar>(stiffness_matrix),
          as_cuda_diagonal<Scalar>(damping_factors),
          num_envs,
          joint_dim,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_nullspace(const ParameterView& stiffness_matrix,
                              const ParameterView& damping_matrix,
                              const std::size_t num_envs,
                              const std::size_t joint_dim,
                              MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointApplyNullspace, Scalar>(num_envs);
  apply_nullspace_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_dense<Scalar>(stiffness_matrix),
          as_cuda_dense<Scalar>(damping_matrix),
          num_envs,
          joint_dim,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_diagonal(const TensorView& joint_error,
                            const ParameterView& diagonal,
                            StatusView parameter_status,
                            const MutableTensorView& joint_torque,
                            MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepDiagonal, Scalar>(num_envs);
  step_diagonal_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(joint_error),
      as_cuda_diagonal<Scalar>(diagonal),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(joint_torque),
      num_envs,
      joint_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_dense(const TensorView& joint_error,
                         const ParameterView& matrix,
                         StatusView parameter_status,
                         const MutableTensorView& joint_torque,
                         MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepDense, Scalar>(num_envs);
  step_dense_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(joint_error),
      as_cuda_dense<Scalar>(matrix),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(joint_torque),
      num_envs,
      joint_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_double_diagonalization(const TensorView& joint_velocity_error,
                                          const TensorView& joint_mass_matrix,
                                          const TensorView& stiffness_matrix,
                                          const ParameterView& damping_factors,
                                          StatusView parameter_status,
                                          const MutableTensorView& joint_torque,
                                          MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_velocity_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_velocity_error.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepDoubleDiagonalization, Scalar>(num_envs);
  step_double_diagonalization_kernel<Scalar>
      <<<config.blocks,
         config.threads,
         double_diagonalization_shared_bytes<Scalar>(),
         active_cuda_stream()>>>(as_cuda_matrix<Scalar>(joint_velocity_error),
                                 as_cuda_tensor3<Scalar>(joint_mass_matrix),
                                 as_cuda_tensor3<Scalar>(stiffness_matrix),
                                 as_cuda_diagonal<Scalar>(damping_factors),
                                 parameter_status.flags,
                                 as_cuda_mutable_matrix<Scalar>(joint_torque),
                                 num_envs,
                                 joint_dim,
                                 status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_impedance_generic(const TensorView& joint_position_error,
                                     const TensorView& joint_velocity_error,
                                     const TensorView& desired_acceleration,
                                     const TensorView& joint_mass_matrix,
                                     const ParameterView& stiffness_matrix,
                                     const ParameterView& damping_factors,
                                     StatusView parameter_status,
                                     const MutableTensorView& joint_torque,
                                     MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepImpedanceGeneric, Scalar>(num_envs);
  step_joint_impedance_kernel<Scalar>
      <<<config.blocks,
         config.threads,
         double_diagonalization_shared_bytes<Scalar>(),
         active_cuda_stream()>>>(as_cuda_matrix<Scalar>(joint_position_error),
                                 as_cuda_matrix<Scalar>(joint_velocity_error),
                                 as_cuda_matrix<Scalar>(desired_acceleration),
                                 as_cuda_tensor3<Scalar>(joint_mass_matrix),
                                 as_cuda_dense<Scalar>(stiffness_matrix),
                                 as_cuda_diagonal<Scalar>(damping_factors),
                                 parameter_status.flags,
                                 as_cuda_mutable_matrix<Scalar>(joint_torque),
                                 num_envs,
                                 joint_dim,
                                 status.flags);
  return cuda_launch_result();
}

template <typename Scalar, std::size_t Capacity, std::size_t ExactDim, int WarpsPerBlock>
Result launch_step_impedance_specialized(const TensorView& joint_position_error,
                                         const TensorView& joint_velocity_error,
                                         const TensorView& desired_acceleration,
                                         const TensorView& joint_mass_matrix,
                                         const ParameterView& stiffness_matrix,
                                         const ParameterView& damping_factors,
                                         StatusView parameter_status,
                                         const MutableTensorView& joint_torque,
                                         MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  const std::size_t full_envs =
      num_envs / static_cast<std::size_t>(WarpsPerBlock) * static_cast<std::size_t>(WarpsPerBlock);
  if (full_envs != 0U) {
    step_joint_impedance_specialized_kernel<Scalar, Capacity, ExactDim, WarpsPerBlock>
        <<<static_cast<int>(full_envs / static_cast<std::size_t>(WarpsPerBlock)),
           WarpsPerBlock * 32,
           specialized_impedance_shared_bytes<Scalar, Capacity, WarpsPerBlock>(),
           active_cuda_stream()>>>(as_cuda_matrix<Scalar>(joint_position_error),
                                   as_cuda_matrix<Scalar>(joint_velocity_error),
                                   as_cuda_matrix<Scalar>(desired_acceleration),
                                   as_cuda_tensor3<Scalar>(joint_mass_matrix),
                                   as_cuda_dense<Scalar>(stiffness_matrix),
                                   as_cuda_diagonal<Scalar>(damping_factors),
                                   parameter_status.flags,
                                   as_cuda_mutable_matrix<Scalar>(joint_torque),
                                   num_envs,
                                   joint_dim,
                                   0U,
                                   status.flags);
  }
  const std::size_t tail_envs = num_envs - full_envs;
  if (tail_envs != 0U) {
    step_joint_impedance_specialized_kernel<Scalar, Capacity, ExactDim, 1>
        <<<static_cast<int>(tail_envs),
           32,
           specialized_impedance_shared_bytes<Scalar, Capacity, 1>(),
           active_cuda_stream()>>>(as_cuda_matrix<Scalar>(joint_position_error),
                                   as_cuda_matrix<Scalar>(joint_velocity_error),
                                   as_cuda_matrix<Scalar>(desired_acceleration),
                                   as_cuda_tensor3<Scalar>(joint_mass_matrix),
                                   as_cuda_dense<Scalar>(stiffness_matrix),
                                   as_cuda_diagonal<Scalar>(damping_factors),
                                   parameter_status.flags,
                                   as_cuda_mutable_matrix<Scalar>(joint_torque),
                                   num_envs,
                                   joint_dim,
                                   full_envs,
                                   status.flags);
  }
  return cuda_launch_result();
}

template <typename Scalar, std::size_t Capacity, std::size_t ExactDim, int WarpsPerBlock>
Result launch_step_impedance_with_plan(const internal::JointImpedanceCudaLaunchPlan& plan,
                                       const TensorView& joint_position_error,
                                       const TensorView& joint_velocity_error,
                                       const TensorView& desired_acceleration,
                                       const TensorView& joint_mass_matrix,
                                       const ParameterView& stiffness_matrix,
                                       const ParameterView& damping_factors,
                                       StatusView parameter_status,
                                       const MutableTensorView& joint_torque,
                                       MutableStatusView status) {
  if (plan.warps_per_block != WarpsPerBlock) {
    return Result{ResultCode::kUnsupported,
                  "tensor joint-impedance launch plan has an invalid warp count"};
  }
  return launch_step_impedance_specialized<Scalar, Capacity, ExactDim, WarpsPerBlock>(
      joint_position_error,
      joint_velocity_error,
      desired_acceleration,
      joint_mass_matrix,
      stiffness_matrix,
      damping_factors,
      parameter_status,
      joint_torque,
      status);
}

template <typename Scalar, std::size_t Capacity, std::size_t ExactDim>
Result launch_step_impedance_small_profile(const internal::JointImpedanceCudaLaunchPlan& plan,
                                           const TensorView& joint_position_error,
                                           const TensorView& joint_velocity_error,
                                           const TensorView& desired_acceleration,
                                           const TensorView& joint_mass_matrix,
                                           const ParameterView& stiffness_matrix,
                                           const ParameterView& damping_factors,
                                           StatusView parameter_status,
                                           const MutableTensorView& joint_torque,
                                           MutableStatusView status) {
  switch (plan.warps_per_block) {
    case 1:
      return launch_step_impedance_with_plan<Scalar, Capacity, ExactDim, 1>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 2:
      return launch_step_impedance_with_plan<Scalar, Capacity, ExactDim, 2>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 4:
      return launch_step_impedance_with_plan<Scalar, Capacity, ExactDim, 4>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 8:
      return launch_step_impedance_with_plan<Scalar, Capacity, ExactDim, 8>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    default:
      return Result{ResultCode::kUnsupported,
                    "tensor joint-impedance profile has an invalid small-kernel warp count"};
  }
}

template <typename Scalar, std::size_t Capacity>
Result launch_step_impedance_medium_profile(const internal::JointImpedanceCudaLaunchPlan& plan,
                                            const TensorView& joint_position_error,
                                            const TensorView& joint_velocity_error,
                                            const TensorView& desired_acceleration,
                                            const TensorView& joint_mass_matrix,
                                            const ParameterView& stiffness_matrix,
                                            const ParameterView& damping_factors,
                                            StatusView parameter_status,
                                            const MutableTensorView& joint_torque,
                                            MutableStatusView status) {
  switch (plan.warps_per_block) {
    case 1:
      return launch_step_impedance_with_plan<Scalar, Capacity, Capacity, 1>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 2:
      return launch_step_impedance_with_plan<Scalar, Capacity, Capacity, 2>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 4:
      return launch_step_impedance_with_plan<Scalar, Capacity, Capacity, 4>(plan,
                                                                            joint_position_error,
                                                                            joint_velocity_error,
                                                                            desired_acceleration,
                                                                            joint_mass_matrix,
                                                                            stiffness_matrix,
                                                                            damping_factors,
                                                                            parameter_status,
                                                                            joint_torque,
                                                                            status);
    case 8:
      if constexpr (sizeof(Scalar) == sizeof(float)) {
        return launch_step_impedance_with_plan<Scalar, Capacity, Capacity, 8>(plan,
                                                                              joint_position_error,
                                                                              joint_velocity_error,
                                                                              desired_acceleration,
                                                                              joint_mass_matrix,
                                                                              stiffness_matrix,
                                                                              damping_factors,
                                                                              parameter_status,
                                                                              joint_torque,
                                                                              status);
      }
      break;
    default:
      break;
  }
  return Result{ResultCode::kUnsupported,
                "tensor joint-impedance profile has an invalid medium-kernel warp count"};
}

template <typename Scalar, std::size_t Capacity>
Result launch_step_impedance_large_profile(const internal::JointImpedanceCudaLaunchPlan& plan,
                                           const TensorView& joint_position_error,
                                           const TensorView& joint_velocity_error,
                                           const TensorView& desired_acceleration,
                                           const TensorView& joint_mass_matrix,
                                           const ParameterView& stiffness_matrix,
                                           const ParameterView& damping_factors,
                                           StatusView parameter_status,
                                           const MutableTensorView& joint_torque,
                                           MutableStatusView status) {
  switch (plan.warps_per_block) {
    case 1:
      return launch_step_impedance_with_plan<Scalar, Capacity, 0, 1>(plan,
                                                                     joint_position_error,
                                                                     joint_velocity_error,
                                                                     desired_acceleration,
                                                                     joint_mass_matrix,
                                                                     stiffness_matrix,
                                                                     damping_factors,
                                                                     parameter_status,
                                                                     joint_torque,
                                                                     status);
    case 2:
      return launch_step_impedance_with_plan<Scalar, Capacity, 0, 2>(plan,
                                                                     joint_position_error,
                                                                     joint_velocity_error,
                                                                     desired_acceleration,
                                                                     joint_mass_matrix,
                                                                     stiffness_matrix,
                                                                     damping_factors,
                                                                     parameter_status,
                                                                     joint_torque,
                                                                     status);
    case 4:
      if constexpr (sizeof(Scalar) == sizeof(float)) {
        return launch_step_impedance_with_plan<Scalar, Capacity, 0, 4>(plan,
                                                                       joint_position_error,
                                                                       joint_velocity_error,
                                                                       desired_acceleration,
                                                                       joint_mass_matrix,
                                                                       stiffness_matrix,
                                                                       damping_factors,
                                                                       parameter_status,
                                                                       joint_torque,
                                                                       status);
      }
      break;
    default:
      break;
  }
  return Result{ResultCode::kUnsupported,
                "tensor joint-impedance profile has an invalid large-kernel warp count"};
}

template <typename Scalar, std::size_t Capacity>
Result launch_step_impedance_maximum_profile(const internal::JointImpedanceCudaLaunchPlan& plan,
                                             const TensorView& joint_position_error,
                                             const TensorView& joint_velocity_error,
                                             const TensorView& desired_acceleration,
                                             const TensorView& joint_mass_matrix,
                                             const ParameterView& stiffness_matrix,
                                             const ParameterView& damping_factors,
                                             StatusView parameter_status,
                                             const MutableTensorView& joint_torque,
                                             MutableStatusView status) {
  if (plan.warps_per_block == 1) {
    return launch_step_impedance_with_plan<Scalar, Capacity, 0, 1>(plan,
                                                                   joint_position_error,
                                                                   joint_velocity_error,
                                                                   desired_acceleration,
                                                                   joint_mass_matrix,
                                                                   stiffness_matrix,
                                                                   damping_factors,
                                                                   parameter_status,
                                                                   joint_torque,
                                                                   status);
  }
  if constexpr (sizeof(Scalar) == sizeof(float)) {
    if (plan.warps_per_block == 2) {
      return launch_step_impedance_with_plan<Scalar, Capacity, 0, 2>(plan,
                                                                     joint_position_error,
                                                                     joint_velocity_error,
                                                                     desired_acceleration,
                                                                     joint_mass_matrix,
                                                                     stiffness_matrix,
                                                                     damping_factors,
                                                                     parameter_status,
                                                                     joint_torque,
                                                                     status);
    }
  }
  return Result{ResultCode::kUnsupported,
                "tensor joint-impedance profile has an invalid maximum-kernel warp count"};
}

template <typename Scalar>
Result launch_step_impedance(const TensorView& joint_position_error,
                             const TensorView& joint_velocity_error,
                             const TensorView& desired_acceleration,
                             const TensorView& joint_mass_matrix,
                             const ParameterView& stiffness_matrix,
                             const ParameterView& damping_factors,
                             StatusView parameter_status,
                             const MutableTensorView& joint_torque,
                             MutableStatusView status,
                             const internal::JointImpedanceCudaRoute route,
                             const internal::JointImpedanceCudaLaunchPlan* requested_plan) {
  if (route == internal::JointImpedanceCudaRoute::kGeneric) {
    return launch_step_impedance_generic<Scalar>(joint_position_error,
                                                 joint_velocity_error,
                                                 desired_acceleration,
                                                 joint_mass_matrix,
                                                 stiffness_matrix,
                                                 damping_factors,
                                                 parameter_status,
                                                 joint_torque,
                                                 status);
  }
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  constexpr DType kDType = sizeof(Scalar) == sizeof(float) ? DType::kFloat32 : DType::kFloat64;
  const internal::JointImpedanceCudaLaunchPlan plan =
      requested_plan == nullptr ? internal::joint_impedance_cuda_launch_plan(kDType, joint_dim)
                                : *requested_plan;
  if (plan.route != internal::JointImpedanceCudaRoute::kOptimized) {
    if (route == internal::JointImpedanceCudaRoute::kOptimized) {
      return Result{ResultCode::kInvalidView,
                    "joint_dim has no optimized tensor joint-impedance kernel"};
    }
    return launch_step_impedance_generic<Scalar>(joint_position_error,
                                                 joint_velocity_error,
                                                 desired_acceleration,
                                                 joint_mass_matrix,
                                                 stiffness_matrix,
                                                 damping_factors,
                                                 parameter_status,
                                                 joint_torque,
                                                 status);
  }
  switch (plan.capacity) {
    case 3:
      return launch_step_impedance_small_profile<Scalar, 3, 3>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 5:
      return launch_step_impedance_small_profile<Scalar, 5, 5>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 6:
      return launch_step_impedance_small_profile<Scalar, 6, 6>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 7:
      return launch_step_impedance_small_profile<Scalar, 7, 7>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 8:
      return launch_step_impedance_small_profile<Scalar, 8, 8>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 12:
      return launch_step_impedance_medium_profile<Scalar, 12>(plan,
                                                              joint_position_error,
                                                              joint_velocity_error,
                                                              desired_acceleration,
                                                              joint_mass_matrix,
                                                              stiffness_matrix,
                                                              damping_factors,
                                                              parameter_status,
                                                              joint_torque,
                                                              status);
    case 14:
      return launch_step_impedance_medium_profile<Scalar, 14>(plan,
                                                              joint_position_error,
                                                              joint_velocity_error,
                                                              desired_acceleration,
                                                              joint_mass_matrix,
                                                              stiffness_matrix,
                                                              damping_factors,
                                                              parameter_status,
                                                              joint_torque,
                                                              status);
    case 16:
      return launch_step_impedance_medium_profile<Scalar, 16>(plan,
                                                              joint_position_error,
                                                              joint_velocity_error,
                                                              desired_acceleration,
                                                              joint_mass_matrix,
                                                              stiffness_matrix,
                                                              damping_factors,
                                                              parameter_status,
                                                              joint_torque,
                                                              status);
    case 20:
      return launch_step_impedance_large_profile<Scalar, 20>(plan,
                                                             joint_position_error,
                                                             joint_velocity_error,
                                                             desired_acceleration,
                                                             joint_mass_matrix,
                                                             stiffness_matrix,
                                                             damping_factors,
                                                             parameter_status,
                                                             joint_torque,
                                                             status);
    case 24:
      return launch_step_impedance_large_profile<Scalar, 24>(plan,
                                                             joint_position_error,
                                                             joint_velocity_error,
                                                             desired_acceleration,
                                                             joint_mass_matrix,
                                                             stiffness_matrix,
                                                             damping_factors,
                                                             parameter_status,
                                                             joint_torque,
                                                             status);
    case 28:
      return launch_step_impedance_maximum_profile<Scalar, 28>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    case 32:
      return launch_step_impedance_maximum_profile<Scalar, 32>(plan,
                                                               joint_position_error,
                                                               joint_velocity_error,
                                                               desired_acceleration,
                                                               joint_mass_matrix,
                                                               stiffness_matrix,
                                                               damping_factors,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
    default:
      return Result{ResultCode::kUnsupported,
                    "tensor joint-impedance launch plan has an invalid capacity"};
  }
}

template <typename Scalar>
Result launch_step_nullspace(const TensorView& joint_position_error,
                             const TensorView& joint_velocity,
                             const TensorView& joint_mass_matrix,
                             const TensorView& jacobian,
                             const ParameterView& stiffness_matrix,
                             const ParameterView& damping_matrix,
                             StatusView parameter_status,
                             const MutableTensorView& joint_torque,
                             MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  const std::size_t task_dim = static_cast<std::size_t>(jacobian.shape[1]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepNullspace, Scalar>(num_envs);
  step_nullspace_kernel<Scalar>
      <<<config.blocks,
         config.threads,
         double_diagonalization_shared_bytes<Scalar>(),
         active_cuda_stream()>>>(as_cuda_matrix<Scalar>(joint_position_error),
                                 as_cuda_matrix<Scalar>(joint_velocity),
                                 as_cuda_tensor3<Scalar>(joint_mass_matrix),
                                 as_cuda_tensor3<Scalar>(jacobian),
                                 as_cuda_dense<Scalar>(stiffness_matrix),
                                 as_cuda_dense<Scalar>(damping_matrix),
                                 parameter_status.flags,
                                 as_cuda_mutable_matrix<Scalar>(joint_torque),
                                 num_envs,
                                 joint_dim,
                                 task_dim,
                                 status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_bimanual_independent_nullspace(const TensorView& joint_position_error,
                                                  const TensorView& joint_velocity,
                                                  const TensorView& joint_mass_matrix,
                                                  const TensorView& arm1_jacobian,
                                                  const TensorView& arm2_jacobian,
                                                  const ParameterView& stiffness_matrix,
                                                  const ParameterView& damping_matrix,
                                                  const StatusView parameter_status,
                                                  const MutableTensorView& joint_torque,
                                                  const MutableStatusView status) {
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kJointStepNullspace, Scalar>(num_envs);
  step_bimanual_independent_nullspace_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix<Scalar>(joint_position_error),
          as_cuda_matrix<Scalar>(joint_velocity),
          as_cuda_tensor3<Scalar>(joint_mass_matrix),
          as_cuda_tensor3<Scalar>(arm1_jacobian),
          as_cuda_tensor3<Scalar>(arm2_jacobian),
          as_cuda_dense<Scalar>(stiffness_matrix),
          as_cuda_dense<Scalar>(damping_matrix),
          parameter_status.flags,
          as_cuda_mutable_matrix<Scalar>(joint_torque),
          num_envs,
          status.flags);
  return cuda_launch_result();
}

}  // namespace

Result apply_joint_diagonal_parameters_cuda(const ParameterSetView& parameters,
                                            const char* parameter_name,
                                            const std::size_t num_envs,
                                            const std::size_t joint_dim,
                                            MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* diagonal = find_parameter(parameters, parameter_name);
  const DType dtype = diagonal == nullptr ? DType::kUnknown : diagonal->values.dtype;
  result = validate_cuda_diagonal(diagonal, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_diagonal<float>(*diagonal, num_envs, joint_dim, parameter_status)
             : launch_apply_diagonal<double>(*diagonal, num_envs, joint_dim, parameter_status);
}

Result apply_joint_dense_parameters_cuda(const ParameterSetView& parameters,
                                         const char* parameter_name,
                                         const std::size_t num_envs,
                                         const std::size_t joint_dim,
                                         MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  const DType dtype = matrix == nullptr ? DType::kUnknown : matrix->values.dtype;
  result = validate_cuda_dense(matrix, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_dense<float>(*matrix, num_envs, joint_dim, parameter_status)
             : launch_apply_dense<double>(*matrix, num_envs, joint_dim, parameter_status);
}

Result apply_joint_damping_factors_parameters_cuda(const ParameterSetView& parameters,
                                                   const std::size_t num_envs,
                                                   const std::size_t joint_dim,
                                                   MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, "damping_factors");
  const DType dtype = damping_factors == nullptr ? DType::kUnknown : damping_factors->values.dtype;
  result = validate_cuda_diagonal(damping_factors, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_diagonal<float>(*damping_factors, num_envs, joint_dim, parameter_status)
             : launch_apply_diagonal<double>(
                   *damping_factors, num_envs, joint_dim, parameter_status);
}

Result apply_joint_impedance_parameters_cuda(const ParameterSetView& parameters,
                                             const std::size_t num_envs,
                                             const std::size_t joint_dim,
                                             MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* stiffness_matrix = find_parameter(parameters, "stiffness_matrix");
  const DType dtype =
      stiffness_matrix == nullptr ? DType::kUnknown : stiffness_matrix->values.dtype;
  result = validate_cuda_dense(stiffness_matrix, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, "damping_factors");
  result = validate_cuda_diagonal(damping_factors, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_impedance<float>(
                   *stiffness_matrix, *damping_factors, num_envs, joint_dim, parameter_status)
             : launch_apply_impedance<double>(
                   *stiffness_matrix, *damping_factors, num_envs, joint_dim, parameter_status);
}

Result apply_nullspace_parameters_cuda(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       const std::size_t joint_dim,
                                       MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* stiffness_matrix = find_parameter(parameters, "stiffness_matrix");
  const DType dtype =
      stiffness_matrix == nullptr ? DType::kUnknown : stiffness_matrix->values.dtype;
  result = validate_cuda_dense(stiffness_matrix, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_matrix = find_parameter(parameters, "damping_matrix");
  result = validate_cuda_dense(damping_matrix, dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_nullspace<float>(
                   *stiffness_matrix, *damping_matrix, num_envs, joint_dim, parameter_status)
             : launch_apply_nullspace<double>(
                   *stiffness_matrix, *damping_matrix, num_envs, joint_dim, parameter_status);
}

Result reset_stateless_joint_cuda(const std::size_t num_envs,
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
      cuda_tensor_tuned_threads<CudaTensorLaunchFamily::kJointResetStateless, DType::kUnknown>();
  const CudaLaunchConfig config{
      static_cast<int>((num_envs + static_cast<std::size_t>(kThreads) - 1U) /
                       static_cast<std::size_t>(kThreads)),
      kThreads};
  reset_stateless_joint_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      status.flags);
  return cuda_launch_result();
}

Result step_joint_diagonal_cuda(const TensorView& joint_error,
                                const ParameterSetView& parameters,
                                const char* parameter_name,
                                StatusView parameter_status,
                                const MutableTensorView& joint_torque,
                                MutableStatusView status) {
  if (!is_cuda_matrix(joint_error)) {
    return Result{ResultCode::kInvalidView,
                  "joint error tensor must be CUDA row-major [num_envs, joint_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  Result result = validate_cuda_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
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
  const ParameterView* diagonal = find_parameter(parameters, parameter_name);
  result = validate_cuda_diagonal(diagonal, joint_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return joint_error.dtype == DType::kFloat32
             ? launch_step_diagonal<float>(
                   joint_error, *diagonal, parameter_status, joint_torque, status)
             : launch_step_diagonal<double>(
                   joint_error, *diagonal, parameter_status, joint_torque, status);
}

Result step_joint_dense_cuda(const TensorView& joint_error,
                             const ParameterSetView& parameters,
                             const char* parameter_name,
                             StatusView parameter_status,
                             const MutableTensorView& joint_torque,
                             MutableStatusView status) {
  if (!is_cuda_matrix(joint_error)) {
    return Result{ResultCode::kInvalidView,
                  "joint error tensor must be CUDA row-major [num_envs, joint_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  Result result = validate_cuda_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
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
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  result = validate_cuda_dense(matrix, joint_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return joint_error.dtype == DType::kFloat32
             ? launch_step_dense<float>(
                   joint_error, *matrix, parameter_status, joint_torque, status)
             : launch_step_dense<double>(
                   joint_error, *matrix, parameter_status, joint_torque, status);
}

Result step_joint_damping_double_diagonalization_cuda(const TensorView& joint_velocity_error,
                                                      const TensorView& joint_mass_matrix,
                                                      const TensorView& stiffness_matrix,
                                                      const ParameterSetView& parameters,
                                                      StatusView parameter_status,
                                                      const MutableTensorView& joint_torque,
                                                      MutableStatusView status) {
  if (!is_cuda_matrix(joint_velocity_error)) {
    return Result{ResultCode::kInvalidView,
                  "joint_velocity_error must be CUDA row-major [num_envs, joint_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_velocity_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_velocity_error.shape[1]);
  if (joint_dim > kMaxJointDim) {
    return Result{ResultCode::kInvalidView,
                  "joint_dim exceeds tensor joint double-diagonalization bound"};
  }
  Result result = validate_cuda_tensor3_square(
      joint_mass_matrix, "joint_mass_matrix", joint_velocity_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_square(
      stiffness_matrix, "stiffness_matrix", joint_velocity_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(joint_torque, joint_velocity_error.dtype, num_envs, joint_dim);
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
  const ParameterView* damping_factors = find_parameter(parameters, "damping_factors");
  result = validate_cuda_diagonal(damping_factors, joint_velocity_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return joint_velocity_error.dtype == DType::kFloat32
             ? launch_step_double_diagonalization<float>(joint_velocity_error,
                                                         joint_mass_matrix,
                                                         stiffness_matrix,
                                                         *damping_factors,
                                                         parameter_status,
                                                         joint_torque,
                                                         status)
             : launch_step_double_diagonalization<double>(joint_velocity_error,
                                                          joint_mass_matrix,
                                                          stiffness_matrix,
                                                          *damping_factors,
                                                          parameter_status,
                                                          joint_torque,
                                                          status);
}

Result step_joint_impedance_controller_cuda_impl(
    const TensorView& joint_position_error,
    const TensorView& joint_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_mass_matrix,
    const ParameterSetView& parameters,
    StatusView parameter_status,
    const MutableTensorView& joint_torque,
    MutableStatusView status,
    const internal::JointImpedanceCudaRoute route,
    const internal::JointImpedanceCudaLaunchPlan* requested_plan = nullptr) {
  if (!is_cuda_matrix(joint_position_error)) {
    return Result{ResultCode::kInvalidView,
                  "joint_position_error must be CUDA row-major [num_envs, joint_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  if (joint_dim > kMaxJointDim) {
    return Result{ResultCode::kInvalidView, "joint_dim exceeds tensor joint impedance bound"};
  }
  if (!is_cuda_matrix(joint_velocity_error) ||
      joint_velocity_error.dtype != joint_position_error.dtype ||
      joint_velocity_error.shape[0] != joint_position_error.shape[0] ||
      joint_velocity_error.shape[1] != joint_position_error.shape[1]) {
    return Result{ResultCode::kInvalidView, "joint_velocity_error must match joint_position_error"};
  }
  if (!is_cuda_matrix(desired_acceleration) ||
      desired_acceleration.dtype != joint_position_error.dtype ||
      desired_acceleration.shape[0] != joint_position_error.shape[0] ||
      desired_acceleration.shape[1] != joint_position_error.shape[1]) {
    return Result{ResultCode::kInvalidView, "desired_acceleration must match joint_position_error"};
  }
  Result result = validate_cuda_tensor3_square(
      joint_mass_matrix, "joint_mass_matrix", joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(joint_torque, joint_position_error.dtype, num_envs, joint_dim);
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
  const ParameterView* stiffness_matrix = find_parameter(parameters, "stiffness_matrix");
  result = validate_cuda_dense(stiffness_matrix, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, "damping_factors");
  result = validate_cuda_diagonal(damping_factors, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return joint_position_error.dtype == DType::kFloat32
             ? launch_step_impedance<float>(joint_position_error,
                                            joint_velocity_error,
                                            desired_acceleration,
                                            joint_mass_matrix,
                                            *stiffness_matrix,
                                            *damping_factors,
                                            parameter_status,
                                            joint_torque,
                                            status,
                                            route,
                                            requested_plan)
             : launch_step_impedance<double>(joint_position_error,
                                             joint_velocity_error,
                                             desired_acceleration,
                                             joint_mass_matrix,
                                             *stiffness_matrix,
                                             *damping_factors,
                                             parameter_status,
                                             joint_torque,
                                             status,
                                             route,
                                             requested_plan);
}

Result step_joint_impedance_controller_cuda(const TensorView& joint_position_error,
                                            const TensorView& joint_velocity_error,
                                            const TensorView& desired_acceleration,
                                            const TensorView& joint_mass_matrix,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            const MutableTensorView& joint_torque,
                                            MutableStatusView status) {
  return step_joint_impedance_controller_cuda_impl(joint_position_error,
                                                   joint_velocity_error,
                                                   desired_acceleration,
                                                   joint_mass_matrix,
                                                   parameters,
                                                   parameter_status,
                                                   joint_torque,
                                                   status,
                                                   internal::JointImpedanceCudaRoute::kAuto);
}

namespace internal {

int joint_impedance_cuda_build_sm() noexcept { return kCudaBuildSm; }

JointImpedanceCudaLaunchPlan joint_impedance_cuda_launch_plan(
    const DType dtype, const std::size_t joint_dim) noexcept {
  const CudaTuningProfile* const profile = find_cuda_tuning_profile(dtype, joint_dim);
  if (profile != nullptr) {
    return JointImpedanceCudaLaunchPlan{
        JointImpedanceCudaRoute::kOptimized, profile->capacity, profile->warps_per_block};
  }
  return JointImpedanceCudaLaunchPlan{};
}

Result step_joint_impedance_controller_cuda_with_route(const TensorView& joint_position_error,
                                                       const TensorView& joint_velocity_error,
                                                       const TensorView& desired_acceleration,
                                                       const TensorView& joint_mass_matrix,
                                                       const ParameterSetView& parameters,
                                                       StatusView parameter_status,
                                                       const MutableTensorView& joint_torque,
                                                       MutableStatusView status,
                                                       const JointImpedanceCudaRoute route) {
  return step_joint_impedance_controller_cuda_impl(joint_position_error,
                                                   joint_velocity_error,
                                                   desired_acceleration,
                                                   joint_mass_matrix,
                                                   parameters,
                                                   parameter_status,
                                                   joint_torque,
                                                   status,
                                                   route);
}

Result step_joint_impedance_controller_cuda_with_launch_plan(
    const TensorView& joint_position_error,
    const TensorView& joint_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_mass_matrix,
    const ParameterSetView& parameters,
    StatusView parameter_status,
    const MutableTensorView& joint_torque,
    MutableStatusView status,
    const JointImpedanceCudaLaunchPlan launch_plan) {
  return step_joint_impedance_controller_cuda_impl(joint_position_error,
                                                   joint_velocity_error,
                                                   desired_acceleration,
                                                   joint_mass_matrix,
                                                   parameters,
                                                   parameter_status,
                                                   joint_torque,
                                                   status,
                                                   launch_plan.route,
                                                   &launch_plan);
}

}  // namespace internal

Result step_nullspace_controller_cuda(const TensorView& joint_position_error,
                                      const TensorView& joint_velocity,
                                      const TensorView& joint_mass_matrix,
                                      const TensorView& jacobian,
                                      const ParameterSetView& parameters,
                                      StatusView parameter_status,
                                      const MutableTensorView& joint_torque,
                                      MutableStatusView status) {
  if (!is_cuda_matrix(joint_position_error)) {
    return Result{ResultCode::kInvalidView,
                  "joint_position_error must be CUDA row-major [num_envs, joint_dim]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const std::size_t joint_dim = static_cast<std::size_t>(joint_position_error.shape[1]);
  if (joint_dim > kMaxJointDim) {
    return Result{ResultCode::kInvalidView, "joint_dim exceeds tensor nullspace bound"};
  }
  if (!is_cuda_matrix(joint_velocity) || joint_velocity.dtype != joint_position_error.dtype ||
      joint_velocity.shape[0] != joint_position_error.shape[0] ||
      joint_velocity.shape[1] != joint_position_error.shape[1]) {
    return Result{ResultCode::kInvalidView, "joint_velocity must match joint_position_error"};
  }
  Result result = validate_cuda_tensor3_square(
      joint_mass_matrix, "joint_mass_matrix", joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  const std::size_t task_dim =
      jacobian.shape[1] > 0 ? static_cast<std::size_t>(jacobian.shape[1]) : 0;
  if (task_dim == 0 || task_dim > kMaxTaskDim || task_dim > joint_dim) {
    return Result{ResultCode::kInvalidView,
                  "nullspace task_dim must be in [1, min(12, joint_dim)]"};
  }
  result = validate_cuda_tensor3_matrix(
      jacobian, "jacobian", joint_position_error.dtype, num_envs, task_dim, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(joint_torque, joint_position_error.dtype, num_envs, joint_dim);
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
  const ParameterView* stiffness_matrix = find_parameter(parameters, "stiffness_matrix");
  result = validate_cuda_dense(stiffness_matrix, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_matrix = find_parameter(parameters, "damping_matrix");
  result = validate_cuda_dense(damping_matrix, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  return joint_position_error.dtype == DType::kFloat32
             ? launch_step_nullspace<float>(joint_position_error,
                                            joint_velocity,
                                            joint_mass_matrix,
                                            jacobian,
                                            *stiffness_matrix,
                                            *damping_matrix,
                                            parameter_status,
                                            joint_torque,
                                            status)
             : launch_step_nullspace<double>(joint_position_error,
                                             joint_velocity,
                                             joint_mass_matrix,
                                             jacobian,
                                             *stiffness_matrix,
                                             *damping_matrix,
                                             parameter_status,
                                             joint_torque,
                                             status);
}

Result step_bimanual_independent_nullspace_controller_cuda(const TensorView& joint_position_error,
                                                           const TensorView& joint_velocity,
                                                           const TensorView& joint_mass_matrix,
                                                           const TensorView& arm1_jacobian,
                                                           const TensorView& arm2_jacobian,
                                                           const ParameterSetView& parameters,
                                                           const StatusView parameter_status,
                                                           const MutableTensorView& joint_torque,
                                                           const MutableStatusView status) {
  constexpr std::size_t kJoints = 14;
  constexpr std::size_t kArmJoints = 7;
  constexpr std::size_t kArmTasks = 6;
  if (!is_cuda_matrix(joint_position_error) || joint_position_error.shape[1] != kJoints) {
    return Result{ResultCode::kInvalidView,
                  "joint_position_error must be CUDA row-major [num_envs,14]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  if (!is_cuda_matrix(joint_velocity) || joint_velocity.dtype != joint_position_error.dtype ||
      joint_velocity.shape[0] != joint_position_error.shape[0] ||
      joint_velocity.shape[1] != kJoints) {
    return Result{ResultCode::kInvalidView, "joint_velocity must match joint_position_error"};
  }
  Result result = validate_cuda_tensor3_square(
      joint_mass_matrix, "joint_mass_matrix", joint_position_error.dtype, num_envs, kJoints);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(
      arm1_jacobian, "arm1_jacobian", joint_position_error.dtype, num_envs, kArmTasks, kArmJoints);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(
      arm2_jacobian, "arm2_jacobian", joint_position_error.dtype, num_envs, kArmTasks, kArmJoints);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(joint_torque, joint_position_error.dtype, num_envs, kJoints);
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
  const ParameterView* stiffness_matrix = find_parameter(parameters, "stiffness_matrix");
  result = validate_cuda_dense(stiffness_matrix, joint_position_error.dtype, num_envs, kJoints);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_matrix = find_parameter(parameters, "damping_matrix");
  result = validate_cuda_dense(damping_matrix, joint_position_error.dtype, num_envs, kJoints);
  if (!result.ok()) {
    return result;
  }
  return joint_position_error.dtype == DType::kFloat32
             ? launch_step_bimanual_independent_nullspace<float>(joint_position_error,
                                                                 joint_velocity,
                                                                 joint_mass_matrix,
                                                                 arm1_jacobian,
                                                                 arm2_jacobian,
                                                                 *stiffness_matrix,
                                                                 *damping_matrix,
                                                                 parameter_status,
                                                                 joint_torque,
                                                                 status)
             : launch_step_bimanual_independent_nullspace<double>(joint_position_error,
                                                                  joint_velocity,
                                                                  joint_mass_matrix,
                                                                  arm1_jacobian,
                                                                  arm2_jacobian,
                                                                  *stiffness_matrix,
                                                                  *damping_matrix,
                                                                  parameter_status,
                                                                  joint_torque,
                                                                  status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
