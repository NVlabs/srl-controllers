/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_blocks_cuda.cu
 * @brief CUDA tensor kernels for Cartesian controller blocks.
 */

#include "tensor/cartesian_blocks_cuda.hpp"

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

constexpr std::size_t kCartesianDim = 6;
constexpr std::size_t kPoseRows = 4;
constexpr std::size_t kPoseCols = 4;
constexpr std::size_t kPointDim = 3;
constexpr std::uint64_t kUnsupportedModeFlag =
    static_cast<std::uint64_t>(StatusFlag::kUnsupportedMode);
constexpr std::uint64_t kInvalidParameterFlag =
    static_cast<std::uint64_t>(StatusFlag::kInvalidParameter);
constexpr std::uint64_t kInvalidInputFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInputFlag =
    static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kInvalidDtFlag = static_cast<std::uint64_t>(StatusFlag::kInvalidDt);
constexpr std::uint64_t kResetAppliedFlag = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSaturationFlag = static_cast<std::uint64_t>(StatusFlag::kSaturation);
constexpr std::uint64_t kSafeModeFlag = static_cast<std::uint64_t>(StatusFlag::kSafeMode);
constexpr std::uint64_t kSafeTriggers = kUnsupportedModeFlag | kInvalidParameterFlag |
                                        kInvalidInputFlag | kNonFiniteInputFlag | kInvalidDtFlag;
constexpr double kCommandLimit = 1e100;
constexpr double kDefaultLimit = 1e100;

template <typename Scalar>
__host__ __device__ constexpr Scalar default_limit();

template <>
__host__ __device__ constexpr float default_limit<float>() {
  return 1.0e30F;
}

template <>
__host__ __device__ constexpr double default_limit<double>() {
  return kDefaultLimit;
}

template <typename Scalar>
__host__ __device__ constexpr Scalar command_limit();

template <>
__host__ __device__ constexpr float command_limit<float>() {
  return 1.0e30F;
}

template <>
__host__ __device__ constexpr double command_limit<double>() {
  return kCommandLimit;
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

struct CudaBoolVectorView {
  std::uint8_t* data{nullptr};
  std::int64_t stride{0};
};

template <typename Scalar>
struct CudaScalarParameterView {
  const Scalar* data{nullptr};
  bool shared{false};
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
__device__ void write_matrix(const CudaMutableMatrixView<Scalar> view,
                             const std::size_t env,
                             const std::size_t dim,
                             const Scalar value) {
  view.data[env * static_cast<std::size_t>(view.row_stride) +
            dim * static_cast<std::size_t>(view.col_stride)] = value;
}

template <typename Scalar>
__device__ Scalar read_scalar_batch(const CudaScalarBatchView<Scalar> view, const std::size_t env) {
  if (view.rank == 0) {
    return view.data[0];
  }
  return view.data[env * static_cast<std::size_t>(view.env_stride)];
}

template <typename Scalar>
__device__ Scalar read_scalar_parameter(const CudaScalarParameterView<Scalar> parameter,
                                        const std::size_t env) {
  if (parameter.shared) {
    return parameter.data[0];
  }
  return parameter.data[env * static_cast<std::size_t>(parameter.env_stride)];
}

template <typename Scalar>
__device__ Scalar read_matrix_parameter(const CudaMatrixParameterView<Scalar> matrix,
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
__device__ Scalar read_vector_parameter(const CudaVectorParameterView<Scalar> parameter,
                                        const std::size_t env,
                                        const std::size_t dim) {
  if (parameter.shared || parameter.rank == 1) {
    return parameter.data[dim * static_cast<std::size_t>(parameter.dim_stride)];
  }
  return parameter.data[env * static_cast<std::size_t>(parameter.env_stride) +
                        dim * static_cast<std::size_t>(parameter.dim_stride)];
}

template <typename Scalar>
__device__ Scalar read_optional_vector_parameter(CudaVectorParameterView<Scalar> parameter,
                                                 const std::size_t env,
                                                 const std::size_t dim,
                                                 const Scalar default_value) {
  if (parameter.data == nullptr) {
    return default_value;
  }
  return read_vector_parameter(parameter, env, dim);
}

template <typename Scalar>
__device__ detail::ConstVectorView<Scalar> core_vector_env(CudaMatrixView<Scalar> view,
                                                           const std::size_t env,
                                                           const std::size_t size) {
  return detail::ConstVectorView<Scalar>{
      view.data + static_cast<std::ptrdiff_t>(env) * view.row_stride, size, view.col_stride};
}

template <typename Scalar>
__device__ detail::ConstVectorView<Scalar> core_vector_env(CudaMutableMatrixView<Scalar> view,
                                                           const std::size_t env,
                                                           const std::size_t size) {
  return detail::ConstVectorView<Scalar>{
      view.data + static_cast<std::ptrdiff_t>(env) * view.row_stride, size, view.col_stride};
}

template <typename Scalar>
__device__ detail::ConstMatrixView<Scalar> core_tensor3_env(CudaMutableTensor3View<Scalar> view,
                                                            const std::size_t env) {
  return detail::ConstMatrixView<Scalar>{
      view.data + static_cast<std::ptrdiff_t>(env) * view.env_stride,
      kCartesianDim,
      kCartesianDim,
      view.row_stride,
      view.col_stride};
}

template <typename Scalar>
__device__ detail::ConstMatrixView<Scalar> core_matrix_parameter(
    CudaMatrixParameterView<Scalar> matrix, const std::size_t env) {
  const Scalar* data = matrix.data;
  if (!matrix.shared && matrix.rank != 2) {
    data += static_cast<std::ptrdiff_t>(env) * matrix.env_stride;
  }
  return detail::ConstMatrixView<Scalar>{
      data, kCartesianDim, kCartesianDim, matrix.row_stride, matrix.col_stride};
}

template <typename Scalar>
__device__ bool is_finite(const Scalar value) {
  return isfinite(value);
}

template <typename Scalar>
__device__ bool matrix_parameter_is_finite(CudaMatrixParameterView<Scalar> matrix,
                                           const std::size_t env) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!is_finite(read_matrix_parameter(matrix, env, row, col))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool vector_parameters_are_valid(CudaVectorParameterView<Scalar> maximum,
                                            CudaVectorParameterView<Scalar> shaping,
                                            const std::size_t env,
                                            const std::size_t signal_dim) {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar maximum_value = read_vector_parameter(maximum, env, dim);
    const Scalar shaping_value = read_vector_parameter(shaping, env, dim);
    if (!is_finite(maximum_value) || !is_finite(shaping_value) || maximum_value <= shaping_value) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool vector3_parameter_is_finite(CudaVectorParameterView<Scalar> parameter,
                                            const std::size_t env) {
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    if (!is_finite(read_vector_parameter(parameter, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool vector_parameter_is_finite(CudaVectorParameterView<Scalar> parameter,
                                           const std::size_t env) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!is_finite(read_vector_parameter(parameter, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool optional_limit_parameter_is_valid(CudaVectorParameterView<Scalar> parameter,
                                                  const std::size_t env) {
  if (parameter.data == nullptr) {
    return true;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar value = read_vector_parameter(parameter, env, dim);
    if (isnan(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool stiffness_limits_are_valid(CudaMatrixParameterView<Scalar> initial,
                                           CudaVectorParameterView<Scalar> maximum,
                                           CudaVectorParameterView<Scalar> minimum,
                                           CudaVectorParameterView<Scalar> maximum_rate,
                                           const std::size_t env) {
  if (!matrix_parameter_is_finite(initial, env)) {
    return false;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar max_value = read_vector_parameter(maximum, env, dim);
    const Scalar min_value =
        read_optional_vector_parameter(minimum, env, dim, static_cast<Scalar>(0));
    const Scalar rate_value = read_vector_parameter(maximum_rate, env, dim);
    if (!is_finite(max_value) || !is_finite(min_value) || !is_finite(rate_value) ||
        max_value < static_cast<Scalar>(0) || min_value < static_cast<Scalar>(0) ||
        rate_value < static_cast<Scalar>(0) || max_value < min_value) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool force_parameters_are_valid(CudaVectorParameterView<Scalar> proportional_gain,
                                           CudaVectorParameterView<Scalar> differentiator_gain,
                                           CudaVectorParameterView<Scalar> integral_gain,
                                           CudaVectorParameterView<Scalar> u0,
                                           CudaVectorParameterView<Scalar> upper,
                                           CudaVectorParameterView<Scalar> lower,
                                           CudaVectorParameterView<Scalar> maximum,
                                           CudaVectorParameterView<Scalar> shaping,
                                           const bool use_shaping_function,
                                           const std::size_t env) {
  if (!vector_parameter_is_finite(proportional_gain, env) ||
      !vector_parameter_is_finite(differentiator_gain, env) ||
      !vector_parameter_is_finite(integral_gain, env)) {
    return false;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar upper_value =
        read_optional_vector_parameter(upper, env, dim, default_limit<Scalar>());
    const Scalar lower_value =
        read_optional_vector_parameter(lower, env, dim, -default_limit<Scalar>());
    const Scalar u0_value = read_optional_vector_parameter(u0, env, dim, static_cast<Scalar>(0));
    if (!is_finite(upper_value) || !is_finite(lower_value) || !is_finite(u0_value) ||
        upper_value < lower_value) {
      return false;
    }
  }
  return !use_shaping_function || vector_parameters_are_valid(maximum, shaping, env, kCartesianDim);
}

template <typename Scalar>
__device__ Scalar norm3(const Scalar values[3]) {
  return sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

template <typename Scalar>
__device__ Scalar norm6(const Scalar values[6]) {
  Scalar sum = static_cast<Scalar>(0);
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    sum += values[dim] * values[dim];
  }
  return sqrt(sum);
}

template <typename Scalar>
__device__ bool so3_log_error(const CudaTensor3View<Scalar> current_pose,
                              const CudaTensor3View<Scalar> goal_pose,
                              const std::size_t env,
                              Scalar rotation_error[3]) {
  const Scalar* current =
      current_pose.data + static_cast<std::ptrdiff_t>(env) * current_pose.env_stride;
  const Scalar* goal = goal_pose.data + static_cast<std::ptrdiff_t>(env) * goal_pose.env_stride;
  return detail::so3_log_error(
      detail::ConstMatrixView<Scalar>{
          current, kPoseRows, kPoseCols, current_pose.row_stride, current_pose.col_stride},
      detail::ConstMatrixView<Scalar>{
          goal, kPoseRows, kPoseCols, goal_pose.row_stride, goal_pose.col_stride},
      detail::MutableVectorView<Scalar>{rotation_error, kPointDim, 1});
}

template <typename Scalar>
__device__ bool flow_parameters_are_valid(CudaScalarParameterView<Scalar> funnel_radius,
                                          CudaScalarParameterView<Scalar> max_distance,
                                          CudaScalarParameterView<Scalar> funnel_velocity,
                                          CudaVectorParameterView<Scalar> start_point,
                                          CudaVectorParameterView<Scalar> end_point,
                                          const std::size_t env) {
  if (!vector3_parameter_is_finite(start_point, env) ||
      !vector3_parameter_is_finite(end_point, env)) {
    return false;
  }
  const Scalar radius = read_scalar_parameter(funnel_radius, env);
  const Scalar max = read_scalar_parameter(max_distance, env);
  const Scalar velocity = read_scalar_parameter(funnel_velocity, env);
  Scalar direction[3]{};
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    direction[dim] =
        read_vector_parameter(end_point, env, dim) - read_vector_parameter(start_point, env, dim);
  }
  return is_finite(radius) && is_finite(max) && is_finite(velocity) &&
         radius >= static_cast<Scalar>(0) && max >= static_cast<Scalar>(0) &&
         velocity >= static_cast<Scalar>(0) && radius <= max &&
         norm3(direction) >= static_cast<Scalar>(1e-6);
}

template <typename Scalar>
__device__ bool inverse_apf_parameters_are_valid(CudaScalarParameterView<Scalar> alpha,
                                                 CudaScalarParameterView<Scalar> gamma,
                                                 CudaScalarParameterView<Scalar> d_min,
                                                 CudaScalarParameterView<Scalar> d_max,
                                                 CudaScalarParameterView<Scalar> deadzone,
                                                 const std::size_t env) {
  return equations::validate_inverse_apf_parameters(read_scalar_parameter(alpha, env),
                                                    read_scalar_parameter(gamma, env),
                                                    read_scalar_parameter(d_min, env),
                                                    read_scalar_parameter(d_max, env),
                                                    read_scalar_parameter(deadzone, env));
}

template <typename Scalar>
__device__ Scalar shaping_factor_value(const Scalar error,
                                       const Scalar maximum_distance,
                                       const Scalar shaping_distance) {
  return detail::shaping_factor_value(error, maximum_distance, shaping_distance);
}

template <typename Scalar>
__device__ bool tensor3_is_finite(CudaTensor3View<Scalar> input,
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
__device__ bool tensor3_is_finite(CudaMutableTensor3View<Scalar> input,
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
__device__ bool input_is_finite(CudaMatrixView<Scalar> input, const std::size_t env) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!is_finite(read_matrix(input, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool mutable_input_is_finite(CudaMutableMatrixView<Scalar> input,
                                        const std::size_t env) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar value = input.data[env * static_cast<std::size_t>(input.row_stride) +
                                    dim * static_cast<std::size_t>(input.col_stride)];
    if (!is_finite(value)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool input_is_finite(CudaMatrixView<Scalar> input,
                                const std::size_t env,
                                const std::size_t signal_dim) {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!is_finite(read_matrix(input, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool scalar_batch_is_valid_dt(CudaScalarBatchView<Scalar> dt, const std::size_t env) {
  const Scalar value = read_scalar_batch(dt, env);
  return is_finite(value) && value > static_cast<Scalar>(0);
}

template <typename Scalar>
__device__ void zero_output(CudaMutableMatrixView<Scalar> output, const std::size_t env) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(output, env, dim, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
__device__ void zero_output(CudaMutableMatrixView<Scalar> output,
                            const std::size_t env,
                            const std::size_t signal_dim) {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(output, env, dim, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
__device__ void read_parameter_matrix6(CudaMatrixParameterView<Scalar> view,
                                       const std::size_t env,
                                       Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = read_matrix_parameter(view, env, row, col);
    }
  }
}

template <typename Scalar>
__device__ void read_tensor3_matrix6(CudaTensor3View<Scalar> view,
                                     const std::size_t env,
                                     Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = read_tensor3(view, env, row, col);
    }
  }
}

template <typename Scalar>
__device__ void read_tensor3_matrix6(CudaMutableTensor3View<Scalar> view,
                                     const std::size_t env,
                                     Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = read_tensor3(view, env, row, col);
    }
  }
}

template <typename Scalar>
__device__ bool matrix6_has_positive_diagonal(const Scalar matrix[6][6]) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!is_finite(matrix[dim][dim]) || matrix[dim][dim] <= static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool matrix6_has_non_negative_diagonal(const Scalar matrix[6][6]) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!is_finite(matrix[dim][dim]) || matrix[dim][dim] < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool damping_factors_are_valid(CudaVectorParameterView<Scalar> damping_factors,
                                          const std::size_t env) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar value = read_vector_parameter(damping_factors, env, dim);
    if (!is_finite(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
__device__ void transpose_6x6(const Scalar input[6][6], Scalar output[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      output[row][col] = input[col][row];
    }
  }
}

template <typename Scalar>
__device__ void multiply_6x6(const Scalar left[6][6],
                             const Scalar right[6][6],
                             Scalar output[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < kCartesianDim; ++inner) {
        value += left[row][inner] * right[inner][col];
      }
      output[row][col] = value;
    }
  }
}

template <typename Scalar>
__device__ void scale_rows_6x6(const Scalar input[6][6],
                               const Scalar row_scale[6],
                               Scalar output[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      output[row][col] = row_scale[row] * input[row][col];
    }
  }
}

template <typename Scalar>
__device__ void set_identity_6x6(Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
__device__ bool symmetric_eigen_6x6(const Scalar input[6][6],
                                    Scalar eigenvectors[6][6],
                                    Scalar eigenvalues[6],
                                    Scalar* matrix_scale) {
  Scalar matrix[6][6]{};
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!is_finite(input[row][col])) {
        return false;
      }
      scale = fmax(scale, fabs(input[row][col]));
    }
  }
  if (!is_finite(scale)) {
    return false;
  }
  if (scale == static_cast<Scalar>(0)) {
    scale = static_cast<Scalar>(1);
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = input[row][col] / scale;
    }
  }

  set_identity_6x6(eigenvectors);
  bool converged = false;
  for (std::size_t iter = 0; iter < 360; ++iter) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = 1;
    Scalar max_offdiag = fabs(matrix[pivot_row][pivot_col]);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
        const Scalar candidate = fabs(matrix[row][col]);
        if (candidate > max_offdiag) {
          max_offdiag = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }
    if (max_offdiag <=
        internal::SymmetricMatrixPolicy<Scalar>::jacobi_relative_tolerance(kCartesianDim)) {
      converged = true;
      break;
    }

    const Scalar app = matrix[pivot_row][pivot_row];
    const Scalar aqq = matrix[pivot_col][pivot_col];
    const Scalar apq = matrix[pivot_row][pivot_col];
    const Scalar angle = static_cast<Scalar>(0.5) * atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar c = cos(angle);
    const Scalar s = sin(angle);

    for (std::size_t idx = 0; idx < kCartesianDim; ++idx) {
      if (idx == pivot_row || idx == pivot_col) {
        continue;
      }
      const Scalar aip = matrix[idx][pivot_row];
      const Scalar aiq = matrix[idx][pivot_col];
      matrix[idx][pivot_row] = c * aip - s * aiq;
      matrix[pivot_row][idx] = matrix[idx][pivot_row];
      matrix[idx][pivot_col] = s * aip + c * aiq;
      matrix[pivot_col][idx] = matrix[idx][pivot_col];
    }
    matrix[pivot_row][pivot_row] = c * c * app - static_cast<Scalar>(2) * s * c * apq + s * s * aqq;
    matrix[pivot_col][pivot_col] = s * s * app + static_cast<Scalar>(2) * s * c * apq + c * c * aqq;
    matrix[pivot_row][pivot_col] = static_cast<Scalar>(0);
    matrix[pivot_col][pivot_row] = static_cast<Scalar>(0);

    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const Scalar vip = eigenvectors[row][pivot_row];
      const Scalar viq = eigenvectors[row][pivot_col];
      eigenvectors[row][pivot_row] = c * vip - s * viq;
      eigenvectors[row][pivot_col] = s * vip + c * viq;
    }
  }
  if (!converged) {
    return false;
  }

  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    eigenvalues[dim] = matrix[dim][dim];
    if (!is_finite(eigenvalues[dim])) {
      return false;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      if (!is_finite(eigenvectors[row][dim])) {
        return false;
      }
    }
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
__device__ bool matrix_roots_6x6(const Scalar input[6][6],
                                 Scalar output[6][6],
                                 Scalar (*inverse_output)[6],
                                 const bool strict_positive) {
  Scalar eigenvectors[6][6]{};
  Scalar eigenvalues[6]{};
  Scalar scale{};
  if (!symmetric_eigen_6x6(input, eigenvectors, eigenvalues, &scale)) {
    return false;
  }
  Scalar scaled_eigenvectors[6][6]{};
  Scalar inverse_scaled_eigenvectors[6][6]{};
  Scalar max_eigenvalue = static_cast<Scalar>(0);
  for (std::size_t col = 0; col < kCartesianDim; ++col) {
    max_eigenvalue = fmax(max_eigenvalue, fabs(eigenvalues[col]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(kCartesianDim) *
      max_eigenvalue;
  for (std::size_t col = 0; col < kCartesianDim; ++col) {
    if ((strict_positive && eigenvalues[col] <= threshold) || eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar normalized = fmax(eigenvalues[col], static_cast<Scalar>(0));
    const Scalar sqrt_value = sqrt(scale) * sqrt(normalized);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      scaled_eigenvectors[row][col] = eigenvectors[row][col] * sqrt_value;
      if (inverse_output != nullptr) {
        inverse_scaled_eigenvectors[row][col] = eigenvectors[row][col] / sqrt_value;
      }
    }
  }
  Scalar eigenvectors_transpose[6][6]{};
  transpose_6x6(eigenvectors, eigenvectors_transpose);
  multiply_6x6(scaled_eigenvectors, eigenvectors_transpose, output);
  if (inverse_output != nullptr) {
    multiply_6x6(inverse_scaled_eigenvectors, eigenvectors_transpose, inverse_output);
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!is_finite(output[row][col]) ||
          (inverse_output != nullptr && !is_finite(inverse_output[row][col]))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool validate_and_symmetrize_external_6x6(Scalar matrix[6][6]) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!is_finite(matrix[row][col])) {
        return false;
      }
      scale = fmax(scale, fabs(matrix[row][col]));
    }
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(kCartesianDim) * scale;
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
      if (fabs(matrix[row][col] - matrix[col][row]) > tolerance) {
        return false;
      }
      const Scalar value =
          static_cast<Scalar>(0.5) * matrix[row][col] + static_cast<Scalar>(0.5) * matrix[col][row];
      matrix[row][col] = value;
      matrix[col][row] = value;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool symmetrize_derived_6x6(Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!is_finite(matrix[row][col])) {
        return false;
      }
    }
    for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
      const Scalar value =
          static_cast<Scalar>(0.5) * matrix[row][col] + static_cast<Scalar>(0.5) * matrix[col][row];
      matrix[row][col] = value;
      matrix[col][row] = value;
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool compute_damping_design_matrix(const bool factorization_design,
                                              Scalar mass_matrix[6][6],
                                              Scalar stiffness_matrix[6][6],
                                              const Scalar damping_factors[6],
                                              Scalar damping_matrix[6][6]) {
  if (!validate_and_symmetrize_external_6x6(mass_matrix) ||
      !validate_and_symmetrize_external_6x6(stiffness_matrix)) {
    return false;
  }
  Scalar mass_sqrt[6][6]{};
  Scalar mass_sqrt_inv[6][6]{};
  if (!matrix_roots_6x6(mass_matrix,
                        mass_sqrt,
                        factorization_design ? nullptr : mass_sqrt_inv,
                        !factorization_design)) {
    return false;
  }
  if (factorization_design) {
    Scalar stiffness_sqrt[6][6]{};
    if (!matrix_roots_6x6(
            stiffness_matrix, stiffness_sqrt, static_cast<Scalar(*)[6]>(nullptr), false)) {
      return false;
    }
    Scalar d_stiffness[6][6]{};
    Scalar d_mass[6][6]{};
    Scalar first[6][6]{};
    Scalar second[6][6]{};
    scale_rows_6x6(stiffness_sqrt, damping_factors, d_stiffness);
    scale_rows_6x6(mass_sqrt, damping_factors, d_mass);
    multiply_6x6(mass_sqrt, d_stiffness, first);
    multiply_6x6(stiffness_sqrt, d_mass, second);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        damping_matrix[row][col] =
            detail::damping_factorization_value(first[row][col], second[row][col]);
      }
    }
    return true;
  }

  Scalar mass_sqrt_inv_transpose[6][6]{};
  transpose_6x6(mass_sqrt_inv, mass_sqrt_inv_transpose);
  Scalar temp[6][6]{};
  Scalar transformed_stiffness[6][6]{};
  multiply_6x6(mass_sqrt_inv, stiffness_matrix, temp);
  multiply_6x6(temp, mass_sqrt_inv_transpose, transformed_stiffness);
  if (!symmetrize_derived_6x6(transformed_stiffness)) {
    return false;
  }
  Scalar transformed_stiffness_sqrt[6][6]{};
  if (!matrix_roots_6x6(transformed_stiffness,
                        transformed_stiffness_sqrt,
                        static_cast<Scalar(*)[6]>(nullptr),
                        false)) {
    return false;
  }
  Scalar d_stiffness[6][6]{};
  Scalar left[6][6]{};
  Scalar mass_sqrt_transpose[6][6]{};
  scale_rows_6x6(transformed_stiffness_sqrt, damping_factors, d_stiffness);
  multiply_6x6(mass_sqrt, d_stiffness, left);
  transpose_6x6(mass_sqrt, mass_sqrt_transpose);
  multiply_6x6(left, mass_sqrt_transpose, damping_matrix);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      damping_matrix[row][col] =
          detail::damping_double_diagonalization_value(damping_matrix[row][col]);
    }
  }
  return true;
}

template <typename Scalar>
__device__ bool solve_6x6(CudaMatrixParameterView<Scalar> matrix,
                          const std::size_t env,
                          CudaMatrixView<Scalar> rhs,
                          Scalar solution[6]) {
  Scalar augmented[6][7]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      augmented[row][col] = read_matrix_parameter(matrix, env, row, col);
    }
    augmented[row][6] = read_matrix(rhs, env, row);
  }

  for (std::size_t pivot = 0; pivot < kCartesianDim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < kCartesianDim; ++row) {
      const Scalar candidate = fabs(augmented[row][pivot]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }
    if (pivot_abs <= static_cast<Scalar>(1e-12)) {
      return false;
    }
    if (pivot_row != pivot) {
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        const Scalar temporary = augmented[pivot][col];
        augmented[pivot][col] = augmented[pivot_row][col];
        augmented[pivot_row][col] = temporary;
      }
    }
    const Scalar pivot_value = augmented[pivot][pivot];
    for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
      augmented[pivot][col] /= pivot_value;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      if (row == pivot) {
        continue;
      }
      const Scalar factor = augmented[row][pivot];
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        augmented[row][col] -= factor * augmented[pivot][col];
      }
    }
  }

  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    solution[row] = augmented[row][6];
  }
  return true;
}

template <typename Scalar>
__device__ bool solve_6x6_matrix_vector(const Scalar matrix[6][6],
                                        const Scalar rhs[6],
                                        Scalar solution[6]) {
  Scalar augmented[6][7]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      augmented[row][col] = matrix[row][col];
    }
    augmented[row][6] = rhs[row];
  }
  for (std::size_t pivot = 0; pivot < kCartesianDim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < kCartesianDim; ++row) {
      const Scalar candidate = fabs(augmented[row][pivot]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }
    if (pivot_abs <= static_cast<Scalar>(1e-12)) {
      return false;
    }
    if (pivot_row != pivot) {
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        const Scalar temporary = augmented[pivot][col];
        augmented[pivot][col] = augmented[pivot_row][col];
        augmented[pivot_row][col] = temporary;
      }
    }
    const Scalar pivot_value = augmented[pivot][pivot];
    for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
      augmented[pivot][col] /= pivot_value;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      if (row == pivot) {
        continue;
      }
      const Scalar factor = augmented[row][pivot];
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        augmented[row][col] -= factor * augmented[pivot][col];
      }
    }
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    solution[row] = augmented[row][6];
  }
  return true;
}

template <typename Scalar>
__global__ void apply_matrix6_parameters_kernel(CudaMatrixParameterView<Scalar> matrix,
                                                std::size_t num_envs,
                                                std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] = matrix_parameter_is_finite(matrix, env) ? 0U : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_shaping_parameters_kernel(CudaVectorParameterView<Scalar> maximum,
                                                CudaVectorParameterView<Scalar> shaping,
                                                std::size_t num_envs,
                                                std::size_t signal_dim,
                                                std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] =
      vector_parameters_are_valid(maximum, shaping, env, signal_dim) ? 0U : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_damping_design_parameters_kernel(
    CudaVectorParameterView<Scalar> damping_factors,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] =
      damping_factors_are_valid(damping_factors, env) ? 0U : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_flow_parameters_kernel(CudaScalarParameterView<Scalar> funnel_radius,
                                             CudaScalarParameterView<Scalar> max_distance,
                                             CudaScalarParameterView<Scalar> funnel_velocity,
                                             CudaVectorParameterView<Scalar> start_point,
                                             CudaVectorParameterView<Scalar> end_point,
                                             std::size_t num_envs,
                                             std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] =
      flow_parameters_are_valid(
          funnel_radius, max_distance, funnel_velocity, start_point, end_point, env)
          ? 0U
          : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_inverse_apf_parameters_kernel(CudaScalarParameterView<Scalar> alpha,
                                                    CudaScalarParameterView<Scalar> gamma,
                                                    CudaScalarParameterView<Scalar> d_min,
                                                    CudaScalarParameterView<Scalar> d_max,
                                                    CudaScalarParameterView<Scalar> deadzone,
                                                    std::size_t num_envs,
                                                    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] =
      inverse_apf_parameters_are_valid(alpha, gamma, d_min, d_max, deadzone, env)
          ? 0U
          : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_operational_space_parameters_kernel(
    CudaVectorParameterView<Scalar> position_gain,
    CudaVectorParameterView<Scalar> velocity_gain,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  bool gains_are_valid = vector_parameter_is_finite(position_gain, env) &&
                         vector_parameter_is_finite(velocity_gain, env);
  for (std::size_t dim = 0; dim < kCartesianDim && gains_are_valid; ++dim) {
    gains_are_valid = read_vector_parameter(position_gain, env, dim) >= static_cast<Scalar>(0) &&
                      read_vector_parameter(velocity_gain, env, dim) >= static_cast<Scalar>(0);
  }
  parameter_status[env] = gains_are_valid ? 0U : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_cartesian_impedance_parameters_kernel(
    CudaMatrixParameterView<Scalar> stiffness_matrix,
    CudaVectorParameterView<Scalar> damping_factors,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] = matrix_parameter_is_finite(stiffness_matrix, env) &&
                                  damping_factors_are_valid(damping_factors, env)
                              ? 0U
                              : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_cartesian_admittance_parameters_kernel(
    CudaMatrixParameterView<Scalar> stiffness_matrix,
    CudaMatrixParameterView<Scalar> damping_matrix,
    CudaMatrixParameterView<Scalar> virtual_mass_matrix,
    CudaVectorParameterView<Scalar> force_filter_threshold,
    CudaVectorParameterView<Scalar> acceleration_limit,
    CudaVectorParameterView<Scalar> twist_limit,
    CudaVectorParameterView<Scalar> pose_limit,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!matrix_parameter_is_finite(stiffness_matrix, env) ||
      !matrix_parameter_is_finite(damping_matrix, env) ||
      !matrix_parameter_is_finite(virtual_mass_matrix, env)) {
    flags |= kInvalidParameterFlag;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (isnan(read_vector_parameter(force_filter_threshold, env, dim))) {
      flags |= kInvalidParameterFlag;
    }
  }
  if (!optional_limit_parameter_is_valid(acceleration_limit, env) ||
      !optional_limit_parameter_is_valid(twist_limit, env) ||
      !optional_limit_parameter_is_valid(pose_limit, env)) {
    flags |= kInvalidParameterFlag;
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_variable_stiffness_parameters_kernel(
    CudaMatrixParameterView<Scalar> initial_stiffness,
    CudaVectorParameterView<Scalar> maximum_stiffness,
    CudaVectorParameterView<Scalar> minimum_stiffness,
    CudaVectorParameterView<Scalar> maximum_rate,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] =
      stiffness_limits_are_valid(
          initial_stiffness, maximum_stiffness, minimum_stiffness, maximum_rate, env)
          ? 0U
          : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_force_parameters_kernel(CudaVectorParameterView<Scalar> proportional_gain,
                                              CudaVectorParameterView<Scalar> differentiator_gain,
                                              CudaVectorParameterView<Scalar> integral_gain,
                                              CudaVectorParameterView<Scalar> u0,
                                              CudaVectorParameterView<Scalar> upper,
                                              CudaVectorParameterView<Scalar> lower,
                                              CudaVectorParameterView<Scalar> maximum,
                                              CudaVectorParameterView<Scalar> shaping,
                                              bool use_shaping_function,
                                              std::size_t num_envs,
                                              std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  parameter_status[env] = force_parameters_are_valid(proportional_gain,
                                                     differentiator_gain,
                                                     integral_gain,
                                                     u0,
                                                     upper,
                                                     lower,
                                                     maximum,
                                                     shaping,
                                                     use_shaping_function,
                                                     env)
                              ? 0U
                              : kInvalidParameterFlag;
}

template <typename Scalar>
__global__ void apply_unified_force_impedance_parameters_kernel(
    CudaMatrixParameterView<Scalar> stiffness_matrix,
    CudaVectorParameterView<Scalar> damping_factors,
    CudaVectorParameterView<Scalar> proportional_gain,
    CudaVectorParameterView<Scalar> differentiator_gain,
    CudaVectorParameterView<Scalar> integral_gain,
    CudaVectorParameterView<Scalar> u0,
    CudaVectorParameterView<Scalar> upper,
    CudaVectorParameterView<Scalar> lower,
    CudaVectorParameterView<Scalar> maximum,
    CudaVectorParameterView<Scalar> shaping,
    bool use_shaping_function,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!matrix_parameter_is_finite(stiffness_matrix, env) ||
      !damping_factors_are_valid(damping_factors, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!force_parameters_are_valid(proportional_gain,
                                  differentiator_gain,
                                  integral_gain,
                                  u0,
                                  upper,
                                  lower,
                                  maximum,
                                  shaping,
                                  use_shaping_function,
                                  env)) {
    flags |= kInvalidParameterFlag;
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_variable_impedance_parameters_kernel(
    CudaMatrixParameterView<Scalar> initial_stiffness,
    CudaVectorParameterView<Scalar> maximum_stiffness,
    CudaVectorParameterView<Scalar> minimum_stiffness,
    CudaVectorParameterView<Scalar> maximum_rate,
    CudaVectorParameterView<Scalar> damping_factors,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!stiffness_limits_are_valid(
          initial_stiffness, maximum_stiffness, minimum_stiffness, maximum_rate, env) ||
      !damping_factors_are_valid(damping_factors, env)) {
    flags |= kInvalidParameterFlag;
  }
  parameter_status[env] = flags;
}

template <typename Scalar>
__global__ void apply_variable_unified_parameters_kernel(
    CudaMatrixParameterView<Scalar> initial_stiffness,
    CudaVectorParameterView<Scalar> maximum_stiffness,
    CudaVectorParameterView<Scalar> minimum_stiffness,
    CudaVectorParameterView<Scalar> maximum_rate,
    CudaVectorParameterView<Scalar> damping_factors,
    CudaVectorParameterView<Scalar> proportional_gain,
    CudaVectorParameterView<Scalar> differentiator_gain,
    CudaVectorParameterView<Scalar> integral_gain,
    CudaVectorParameterView<Scalar> u0,
    CudaVectorParameterView<Scalar> upper,
    CudaVectorParameterView<Scalar> lower,
    CudaVectorParameterView<Scalar> maximum,
    CudaVectorParameterView<Scalar> shaping,
    bool use_shaping_function,
    std::size_t num_envs,
    std::uint64_t* parameter_status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = 0U;
  if (!stiffness_limits_are_valid(
          initial_stiffness, maximum_stiffness, minimum_stiffness, maximum_rate, env) ||
      !damping_factors_are_valid(damping_factors, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!force_parameters_are_valid(proportional_gain,
                                  differentiator_gain,
                                  integral_gain,
                                  u0,
                                  upper,
                                  lower,
                                  maximum,
                                  shaping,
                                  use_shaping_function,
                                  env)) {
    flags |= kInvalidParameterFlag;
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

__global__ void reset_stateless_kernel(const std::uint8_t* reset_mask,
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
__global__ void reset_inertial_kernel(const std::uint8_t* reset_mask,
                                      std::int64_t reset_mask_stride,
                                      CudaMutableTensor3View<Scalar> previous_mass,
                                      CudaMutableTensor3View<Scalar> previous_jacobian,
                                      CudaBoolVectorView initialized,
                                      std::size_t num_envs,
                                      std::size_t joint_dim,
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
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      write_tensor3(previous_mass, env, row, col, static_cast<Scalar>(0));
    }
    for (std::size_t col = 0; col < joint_dim; ++col) {
      write_tensor3(previous_jacobian, env, row, col, static_cast<Scalar>(0));
    }
  }
  initialized.data[env * static_cast<std::size_t>(initialized.stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void reset_admittance_kernel(const std::uint8_t* reset_mask,
                                        std::int64_t reset_mask_stride,
                                        CudaMutableMatrixView<Scalar> pose,
                                        CudaMutableMatrixView<Scalar> twist,
                                        CudaMutableMatrixView<Scalar> pose_previous,
                                        CudaMutableMatrixView<Scalar> twist_previous,
                                        CudaBoolVectorView pose_initialized,
                                        CudaBoolVectorView twist_initialized,
                                        std::size_t num_envs,
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
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(pose, env, dim, static_cast<Scalar>(0));
    write_matrix(twist, env, dim, static_cast<Scalar>(0));
    write_matrix(pose_previous, env, dim, static_cast<Scalar>(0));
    write_matrix(twist_previous, env, dim, static_cast<Scalar>(0));
  }
  pose_initialized.data[env * static_cast<std::size_t>(pose_initialized.stride)] = 0U;
  twist_initialized.data[env * static_cast<std::size_t>(twist_initialized.stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void reset_force_kernel(const std::uint8_t* reset_mask,
                                   std::int64_t reset_mask_stride,
                                   CudaMutableMatrixView<Scalar> differentiator_previous,
                                   CudaBoolVectorView differentiator_initialized,
                                   CudaMutableMatrixView<Scalar> integrator_previous,
                                   CudaMutableMatrixView<Scalar> integrated_error,
                                   CudaBoolVectorView integrator_initialized,
                                   std::size_t num_envs,
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
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(differentiator_previous, env, dim, static_cast<Scalar>(0));
    write_matrix(integrator_previous, env, dim, static_cast<Scalar>(0));
    write_matrix(integrated_error, env, dim, static_cast<Scalar>(0));
  }
  differentiator_initialized
      .data[env * static_cast<std::size_t>(differentiator_initialized.stride)] = 0U;
  integrator_initialized.data[env * static_cast<std::size_t>(integrator_initialized.stride)] = 0U;
  status[env] = kResetAppliedFlag;
}

template <typename Scalar>
__global__ void reset_variable_stiffness_kernel(const std::uint8_t* reset_mask,
                                                std::int64_t reset_mask_stride,
                                                CudaMatrixParameterView<Scalar> initial_stiffness,
                                                const std::uint64_t* parameter_status,
                                                CudaMutableTensor3View<Scalar> current_stiffness,
                                                CudaMutableTensor3View<Scalar> previous_stiffness,
                                                CudaBoolVectorView initialized,
                                                std::size_t num_envs,
                                                std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const bool reset =
      reset_mask == nullptr || reset_mask[env * static_cast<std::size_t>(reset_mask_stride)] != 0U;
  if (!reset) {
    status[env] = flags;
    return;
  }
  if (!matrix_parameter_is_finite(initial_stiffness, env)) {
    flags |= kInvalidParameterFlag;
  }
  if ((flags & kInvalidParameterFlag) != 0U) {
    flags |= kSafeModeFlag;
    status[env] = flags;
    return;
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      const Scalar value = read_matrix_parameter(initial_stiffness, env, row, col);
      write_tensor3(current_stiffness, env, row, col, value);
      write_tensor3(previous_stiffness, env, row, col, value);
    }
  }
  initialized.data[env * static_cast<std::size_t>(initialized.stride)] = 0U;
  status[env] = flags | kResetAppliedFlag;
}

template <typename Scalar>
__global__ void step_matrix6_kernel(CudaMatrixView<Scalar> input,
                                    CudaMatrixParameterView<Scalar> matrix,
                                    const std::uint64_t* parameter_status,
                                    CudaMutableMatrixView<Scalar> output,
                                    std::size_t num_envs,
                                    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!matrix_parameter_is_finite(matrix, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(input, env)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(output, env);
    status[env] = flags;
    return;
  }

  const auto matrix_view = core_matrix_parameter(matrix, env);
  const auto input_view = core_vector_env(input, env, kCartesianDim);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    write_matrix(output, env, row, detail::dense_feedback_row(matrix_view, input_view, row));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_shaping_kernel(CudaMatrixView<Scalar> error,
                                    CudaVectorParameterView<Scalar> maximum,
                                    CudaVectorParameterView<Scalar> shaping,
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
  if (!vector_parameters_are_valid(maximum, shaping, env, signal_dim)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(error, env, signal_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(output, env, signal_dim);
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix(output,
                 env,
                 dim,
                 shaping_factor_value(read_matrix(error, env, dim),
                                      read_vector_parameter(maximum, env, dim),
                                      read_vector_parameter(shaping, env, dim)));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_command_observer_kernel(CudaMatrixView<Scalar> command,
                                             const std::uint64_t* parameter_status,
                                             std::uint8_t* command_is_valid,
                                             std::int64_t output_stride,
                                             std::size_t num_envs,
                                             std::size_t signal_dim,
                                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  bool valid = true;
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar value = read_matrix(command, env, dim);
    valid = valid && detail::command_observer_accepts(value, command_limit<Scalar>());
  }
  if (!valid) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
  }
  command_is_valid[env * static_cast<std::size_t>(output_stride)] = valid ? 1U : 0U;
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_flow_kernel(CudaTensor3View<Scalar> cartesian_pose,
                                 CudaScalarParameterView<Scalar> funnel_radius,
                                 CudaScalarParameterView<Scalar> max_distance,
                                 CudaScalarParameterView<Scalar> funnel_velocity,
                                 CudaVectorParameterView<Scalar> start_point,
                                 CudaVectorParameterView<Scalar> end_point,
                                 const std::uint64_t* parameter_status,
                                 CudaMutableMatrixView<Scalar> cartesian_flow,
                                 std::size_t num_envs,
                                 std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!flow_parameters_are_valid(
          funnel_radius, max_distance, funnel_velocity, start_point, end_point, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!tensor3_is_finite(cartesian_pose, env, kPoseRows, kPoseCols)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_flow, env);
    status[env] = flags;
    return;
  }

  Scalar position[3]{};
  Scalar start[3]{};
  Scalar end[3]{};
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    position[dim] = read_tensor3(cartesian_pose, env, dim, 3);
    start[dim] = read_vector_parameter(start_point, env, dim);
    end[dim] = read_vector_parameter(end_point, env, dim);
  }
  Scalar direction[3]{};
  Scalar direction_to_projection[3]{};
  const detail::FlowFunnelGeometry<Scalar> geometry = detail::flow_funnel_geometry(
      detail::ConstVectorView<Scalar>{start, kPointDim, 1},
      detail::ConstVectorView<Scalar>{end, kPointDim, 1},
      detail::ConstVectorView<Scalar>{position, kPointDim, 1},
      detail::MutableVectorView<Scalar>{direction, kPointDim, 1},
      detail::MutableVectorView<Scalar>{direction_to_projection, kPointDim, 1});
  const Scalar configured_radius = read_scalar_parameter(funnel_radius, env);
  const Scalar configured_max = read_scalar_parameter(max_distance, env);
  const Scalar weight =
      detail::flow_funnel_weight(geometry.radius, configured_radius, configured_max);

  Scalar unit_flow[3]{};
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    unit_flow[dim] = detail::flow_funnel_direction_value(geometry.radius,
                                                         configured_radius,
                                                         configured_max,
                                                         weight,
                                                         direction[dim],
                                                         direction_to_projection[dim]);
  }

  const Scalar velocity = read_scalar_parameter(funnel_velocity, env);
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    write_matrix(cartesian_flow, env, dim, velocity * unit_flow[dim]);
    write_matrix(cartesian_flow, env, dim + kPointDim, static_cast<Scalar>(0));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_inverse_apf_kernel(CudaTensor3View<Scalar> current_pose,
                                        CudaTensor3View<Scalar> goal_pose,
                                        CudaMatrixView<Scalar> cartesian_twist,
                                        CudaScalarParameterView<Scalar> alpha,
                                        CudaScalarParameterView<Scalar> gamma,
                                        CudaScalarParameterView<Scalar> d_min,
                                        CudaScalarParameterView<Scalar> d_max,
                                        CudaScalarParameterView<Scalar> deadzone,
                                        const std::uint64_t* parameter_status,
                                        CudaMutableMatrixView<Scalar> cartesian_wrench,
                                        std::size_t num_envs,
                                        std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!inverse_apf_parameters_are_valid(alpha, gamma, d_min, d_max, deadzone, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!tensor3_is_finite(current_pose, env, kPoseRows, kPoseCols) ||
      !tensor3_is_finite(goal_pose, env, kPoseRows, kPoseCols) ||
      !input_is_finite(cartesian_twist, env)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }

  Scalar position_error[3]{};
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    position_error[dim] =
        read_tensor3(goal_pose, env, dim, 3) - read_tensor3(current_pose, env, dim, 3);
  }

  Scalar rotation_error[3]{};
  if (!so3_log_error(current_pose, goal_pose, env, rotation_error)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  Scalar error[6]{};
  for (std::size_t dim = 0; dim < kPointDim; ++dim) {
    error[dim] = position_error[dim];
    error[dim + kPointDim] = rotation_error[dim];
  }
  const Scalar error_norm = norm6(error);
  const Scalar d_min_value = read_scalar_parameter(d_min, env);
  const Scalar error_norm_squared = error_norm * error_norm;
  const Scalar damping = detail::inverse_apf_damping(error_norm_squared,
                                                     d_min_value,
                                                     read_scalar_parameter(d_max, env),
                                                     read_scalar_parameter(gamma, env));
  Scalar wrench[6]{};
  bool wrench_is_finite = isfinite(error_norm) && isfinite(damping);
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar deadzone_value = read_scalar_parameter(deadzone, env);
    wrench[dim] = detail::inverse_apf_wrench_value(error[dim],
                                                   read_matrix(cartesian_twist, env, dim),
                                                   error_norm,
                                                   error_norm_squared,
                                                   read_scalar_parameter(alpha, env),
                                                   deadzone_value,
                                                   damping);
    wrench_is_finite = wrench_is_finite && isfinite(wrench[dim]);
  }
  if (!wrench_is_finite) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(cartesian_wrench, env, dim, wrench[dim]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_damping_design_kernel(
    bool factorization_design,
    CudaMatrixView<Scalar> cartesian_velocity_error,
    CudaTensor3View<Scalar> cartesian_mass_matrix,
    CudaTensor3View<Scalar> stiffness_matrix,
    CudaVectorParameterView<Scalar> damping_factors_parameter,
    const std::uint64_t* parameter_status,
    CudaMutableMatrixView<Scalar> cartesian_wrench,
    std::size_t num_envs,
    std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!damping_factors_are_valid(damping_factors_parameter, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(cartesian_velocity_error, env) ||
      !tensor3_is_finite(cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
      !tensor3_is_finite(stiffness_matrix, env, kCartesianDim, kCartesianDim)) {
    flags |= kNonFiniteInputFlag;
  }

  Scalar mass[6][6]{};
  Scalar stiffness[6][6]{};
  read_tensor3_matrix6(cartesian_mass_matrix, env, mass);
  read_tensor3_matrix6(stiffness_matrix, env, stiffness);
  if (!matrix6_has_positive_diagonal(mass) || !matrix6_has_non_negative_diagonal(stiffness)) {
    flags |= kInvalidInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }

  Scalar damping_factors[6]{};
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    damping_factors[dim] = read_vector_parameter(damping_factors_parameter, env, dim);
  }
  Scalar damping_matrix[6][6]{};
  if (!compute_damping_design_matrix(
          factorization_design, mass, stiffness, damping_factors, damping_matrix)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }

  Scalar wrench[6]{};
  bool wrench_is_finite = true;
  const detail::ConstMatrixView<Scalar> damping_view{
      &damping_matrix[0][0], kCartesianDim, kCartesianDim, kCartesianDim, 1};
  const auto velocity_view = core_vector_env(cartesian_velocity_error, env, kCartesianDim);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    const Scalar value = detail::cartesian_damping_value(damping_view, velocity_view, row);
    wrench[row] = value;
    wrench_is_finite = wrench_is_finite && is_finite(value);
  }
  if (!wrench_is_finite) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    write_matrix(cartesian_wrench, env, row, wrench[row]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_virtual_mass_kernel(CudaMatrixView<Scalar> force,
                                         CudaMatrixParameterView<Scalar> matrix,
                                         const std::uint64_t* parameter_status,
                                         CudaMutableMatrixView<Scalar> output,
                                         std::size_t num_envs,
                                         std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }

  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  if (!matrix_parameter_is_finite(matrix, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(force, env)) {
    flags |= kNonFiniteInputFlag;
  }
  Scalar solution[6]{};
  if ((flags & kSafeTriggers) == 0U && !solve_6x6(matrix, env, force, solution)) {
    flags |= kInvalidParameterFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(output, env);
    status[env] = flags;
    return;
  }

  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(output, env, dim, solution[dim]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_operational_space_kernel(CudaMatrixView<Scalar> cartesian_pose_error,
                                              CudaMatrixView<Scalar> cartesian_velocity_error,
                                              CudaMatrixView<Scalar> desired_acceleration,
                                              CudaMatrixView<Scalar> jacobian_dot_joint_velocity,
                                              CudaTensor3View<Scalar> cartesian_mass_matrix,
                                              CudaTensor3View<Scalar> jacobian,
                                              CudaMatrixView<Scalar> joint_bias_torque,
                                              CudaVectorParameterView<Scalar> position_gain,
                                              CudaVectorParameterView<Scalar> velocity_gain,
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
  bool gains_are_valid = vector_parameter_is_finite(position_gain, env) &&
                         vector_parameter_is_finite(velocity_gain, env);
  for (std::size_t dim = 0; dim < kCartesianDim && gains_are_valid; ++dim) {
    gains_are_valid = read_vector_parameter(position_gain, env, dim) >= static_cast<Scalar>(0) &&
                      read_vector_parameter(velocity_gain, env, dim) >= static_cast<Scalar>(0);
  }
  if (!gains_are_valid) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(cartesian_pose_error, env) ||
      !input_is_finite(cartesian_velocity_error, env) ||
      !input_is_finite(desired_acceleration, env) ||
      !input_is_finite(jacobian_dot_joint_velocity, env) ||
      !tensor3_is_finite(cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
      !tensor3_is_finite(jacobian, env, kCartesianDim, joint_dim) ||
      !input_is_finite(joint_bias_torque, env, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (read_tensor3(cartesian_mass_matrix, env, dim, dim) < static_cast<Scalar>(0)) {
      flags |= kInvalidInputFlag;
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(joint_torque, env, joint_dim);
    status[env] = flags;
    return;
  }
  Scalar commanded_acceleration[6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    commanded_acceleration[row] = detail::operational_space_acceleration_value(
        read_matrix(desired_acceleration, env, row) -
            read_matrix(jacobian_dot_joint_velocity, env, row),
        detail::proportional_value(read_vector_parameter(position_gain, env, row),
                                   read_matrix(cartesian_pose_error, env, row)),
        detail::proportional_value(read_vector_parameter(velocity_gain, env, row),
                                   read_matrix(cartesian_velocity_error, env, row)));
  }
  Scalar wrench[6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      wrench[row] +=
          read_tensor3(cartesian_mass_matrix, env, row, col) * commanded_acceleration[col];
    }
  }
  bool torque_is_finite = true;
  for (std::size_t joint = 0; joint < joint_dim; ++joint) {
    Scalar torque = read_matrix(joint_bias_torque, env, joint);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      torque += read_tensor3(jacobian, env, row, joint) * wrench[row];
    }
    torque_is_finite = torque_is_finite && is_finite(torque);
    write_matrix(joint_torque, env, joint, torque);
  }
  if (!torque_is_finite) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(joint_torque, env, joint_dim);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_inertial_kernel(CudaMatrixView<Scalar> cartesian_velocity_error,
                                     CudaMatrixView<Scalar> joint_velocity,
                                     CudaTensor3View<Scalar> cartesian_mass_matrix,
                                     CudaTensor3View<Scalar> jacobian,
                                     CudaScalarBatchView<Scalar> dt,
                                     const std::uint64_t* parameter_status,
                                     CudaMutableTensor3View<Scalar> previous_mass,
                                     CudaMutableTensor3View<Scalar> previous_jacobian,
                                     CudaBoolVectorView initialized,
                                     CudaMutableMatrixView<Scalar> cartesian_wrench,
                                     std::size_t num_envs,
                                     std::size_t joint_dim,
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
  if (!input_is_finite(cartesian_velocity_error, env) ||
      !input_is_finite(joint_velocity, env, joint_dim) ||
      !tensor3_is_finite(cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
      !tensor3_is_finite(jacobian, env, kCartesianDim, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (read_tensor3(cartesian_mass_matrix, env, dim, dim) < static_cast<Scalar>(0)) {
      flags |= kInvalidInputFlag;
    }
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }

  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized.stride);
  if (initialized.data[initialized_offset] == 0U) {
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        write_tensor3(
            previous_mass, env, row, col, read_tensor3(cartesian_mass_matrix, env, row, col));
      }
      for (std::size_t col = 0; col < joint_dim; ++col) {
        write_tensor3(previous_jacobian, env, row, col, read_tensor3(jacobian, env, row, col));
      }
      write_matrix(cartesian_wrench, env, row, static_cast<Scalar>(0));
    }
    initialized.data[initialized_offset] = 1U;
    status[env] = flags;
    return;
  }

  Scalar jacobian_derivative_times_dq[6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      const Scalar d_jacobian =
          detail::matrix_derivative_value(read_tensor3(jacobian, env, row, joint),
                                          read_tensor3(previous_jacobian, env, row, joint),
                                          dt_value);
      jacobian_derivative_times_dq[row] += d_jacobian * read_matrix(joint_velocity, env, joint);
    }
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    Scalar mass_derivative_term = static_cast<Scalar>(0);
    Scalar jacobian_term = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      const Scalar d_mass =
          detail::matrix_derivative_value(read_tensor3(cartesian_mass_matrix, env, row, col),
                                          read_tensor3(previous_mass, env, row, col),
                                          dt_value);
      mass_derivative_term += d_mass * read_matrix(cartesian_velocity_error, env, col);
      jacobian_term +=
          read_tensor3(cartesian_mass_matrix, env, row, col) * jacobian_derivative_times_dq[col];
    }
    write_matrix(cartesian_wrench,
                 env,
                 row,
                 detail::inertial_coupling_value(mass_derivative_term, jacobian_term));
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      write_tensor3(
          previous_mass, env, row, col, read_tensor3(cartesian_mass_matrix, env, row, col));
    }
    for (std::size_t col = 0; col < joint_dim; ++col) {
      write_tensor3(previous_jacobian, env, row, col, read_tensor3(jacobian, env, row, col));
    }
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_cartesian_impedance_kernel(CudaMatrixView<Scalar> cartesian_pose_error,
                                                CudaMatrixView<Scalar> cartesian_velocity_error,
                                                CudaMatrixView<Scalar> desired_acceleration,
                                                CudaMatrixView<Scalar> joint_velocity,
                                                CudaTensor3View<Scalar> cartesian_mass_matrix,
                                                CudaTensor3View<Scalar> jacobian,
                                                CudaScalarBatchView<Scalar> dt,
                                                CudaMatrixParameterView<Scalar> stiffness_matrix,
                                                CudaVectorParameterView<Scalar> damping_factors,
                                                const std::uint64_t* parameter_status,
                                                CudaMutableTensor3View<Scalar> previous_mass,
                                                CudaMutableTensor3View<Scalar> previous_jacobian,
                                                CudaBoolVectorView initialized,
                                                CudaMutableMatrixView<Scalar> cartesian_wrench,
                                                std::size_t num_envs,
                                                std::size_t joint_dim,
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
  if (!matrix_parameter_is_finite(stiffness_matrix, env) ||
      !damping_factors_are_valid(damping_factors, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(cartesian_pose_error, env) ||
      !input_is_finite(cartesian_velocity_error, env) ||
      !input_is_finite(desired_acceleration, env) ||
      !input_is_finite(joint_velocity, env, joint_dim) ||
      !tensor3_is_finite(cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
      !tensor3_is_finite(jacobian, env, kCartesianDim, joint_dim)) {
    flags |= kNonFiniteInputFlag;
  }
  Scalar mass[6][6]{};
  Scalar stiffness[6][6]{};
  read_tensor3_matrix6(cartesian_mass_matrix, env, mass);
  read_parameter_matrix6(stiffness_matrix, env, stiffness);
  if (!matrix6_has_positive_diagonal(mass) || !matrix6_has_non_negative_diagonal(stiffness)) {
    flags |= kInvalidInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  Scalar damping[6]{};
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    damping[dim] = read_vector_parameter(damping_factors, env, dim);
  }
  Scalar damping_matrix[6][6]{};
  if (!compute_damping_design_matrix(false, mass, stiffness, damping, damping_matrix)) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  Scalar stiffness_wrench[6]{};
  Scalar damping_wrench[6]{};
  Scalar acceleration_wrench[6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      stiffness_wrench[row] += stiffness[row][col] * read_matrix(cartesian_pose_error, env, col);
      damping_wrench[row] +=
          damping_matrix[row][col] * read_matrix(cartesian_velocity_error, env, col);
      acceleration_wrench[row] += mass[row][col] * read_matrix(desired_acceleration, env, col);
    }
  }
  Scalar inertial_wrench[6]{};
  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized.stride);
  if (initialized.data[initialized_offset] == 0U) {
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        write_tensor3(
            previous_mass, env, row, col, read_tensor3(cartesian_mass_matrix, env, row, col));
      }
      for (std::size_t col = 0; col < joint_dim; ++col) {
        write_tensor3(previous_jacobian, env, row, col, read_tensor3(jacobian, env, row, col));
      }
    }
    initialized.data[initialized_offset] = 1U;
  } else {
    Scalar jacobian_derivative_times_dq[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t joint = 0; joint < joint_dim; ++joint) {
        const Scalar d_jacobian =
            detail::matrix_derivative_value(read_tensor3(jacobian, env, row, joint),
                                            read_tensor3(previous_jacobian, env, row, joint),
                                            dt_value);
        jacobian_derivative_times_dq[row] += d_jacobian * read_matrix(joint_velocity, env, joint);
      }
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      Scalar mass_derivative_term = static_cast<Scalar>(0);
      Scalar jacobian_term = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        const Scalar d_mass =
            detail::matrix_derivative_value(read_tensor3(cartesian_mass_matrix, env, row, col),
                                            read_tensor3(previous_mass, env, row, col),
                                            dt_value);
        mass_derivative_term += d_mass * read_matrix(cartesian_velocity_error, env, col);
        jacobian_term += mass[row][col] * jacobian_derivative_times_dq[col];
      }
      inertial_wrench[row] = detail::inertial_coupling_value(mass_derivative_term, jacobian_term);
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        write_tensor3(
            previous_mass, env, row, col, read_tensor3(cartesian_mass_matrix, env, row, col));
      }
      for (std::size_t col = 0; col < joint_dim; ++col) {
        write_tensor3(previous_jacobian, env, row, col, read_tensor3(jacobian, env, row, col));
      }
    }
  }
  Scalar wrench[6]{};
  bool wrench_is_finite = true;
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    wrench[dim] = detail::cartesian_impedance_value(
        stiffness_wrench[dim], damping_wrench[dim], inertial_wrench[dim], acceleration_wrench[dim]);
    wrench_is_finite = wrench_is_finite && is_finite(wrench[dim]);
  }
  if (!wrench_is_finite) {
    flags |= kInvalidInputFlag | kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(cartesian_wrench, env, dim, wrench[dim]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_admittance_kernel(CudaMatrixView<Scalar> external_wrench,
                                       CudaMatrixView<Scalar> feedforward_wrench,
                                       CudaScalarBatchView<Scalar> dt,
                                       CudaMatrixParameterView<Scalar> stiffness_matrix,
                                       CudaMatrixParameterView<Scalar> damping_matrix,
                                       CudaMatrixParameterView<Scalar> virtual_mass_matrix,
                                       CudaVectorParameterView<Scalar> force_filter_threshold,
                                       CudaVectorParameterView<Scalar> acceleration_limit,
                                       CudaVectorParameterView<Scalar> twist_limit,
                                       CudaVectorParameterView<Scalar> pose_limit,
                                       const std::uint64_t* parameter_status,
                                       CudaMutableMatrixView<Scalar> pose,
                                       CudaMutableMatrixView<Scalar> twist,
                                       CudaMutableMatrixView<Scalar> pose_previous,
                                       CudaMutableMatrixView<Scalar> twist_previous,
                                       CudaBoolVectorView pose_initialized,
                                       CudaBoolVectorView twist_initialized,
                                       CudaMutableMatrixView<Scalar> commanded_acceleration,
                                       std::size_t num_envs,
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
  if (!matrix_parameter_is_finite(stiffness_matrix, env) ||
      !matrix_parameter_is_finite(damping_matrix, env) ||
      !matrix_parameter_is_finite(virtual_mass_matrix, env) ||
      !optional_limit_parameter_is_valid(acceleration_limit, env) ||
      !optional_limit_parameter_is_valid(twist_limit, env) ||
      !optional_limit_parameter_is_valid(pose_limit, env)) {
    flags |= kInvalidParameterFlag;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (isnan(read_vector_parameter(force_filter_threshold, env, dim))) {
      flags |= kInvalidParameterFlag;
    }
  }
  if (!input_is_finite(external_wrench, env) || !input_is_finite(feedforward_wrench, env)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(commanded_acceleration, env);
    status[env] = flags;
    return;
  }

  const auto stiffness_view = core_matrix_parameter(stiffness_matrix, env);
  const auto damping_view = core_matrix_parameter(damping_matrix, env);
  const auto pose_view = core_vector_env(pose, env, kCartesianDim);
  const auto twist_view = core_vector_env(twist, env, kCartesianDim);
  Scalar generalized_force[6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    const Scalar threshold = read_vector_parameter(force_filter_threshold, env, row);
    const Scalar filtered_external =
        detail::deadband_value(read_matrix(external_wrench, env, row), threshold);
    generalized_force[row] = detail::admittance_force_balance_value(
        filtered_external,
        read_matrix(feedforward_wrench, env, row),
        detail::cartesian_stiffness_value(stiffness_view, pose_view, row),
        detail::cartesian_damping_value(damping_view, twist_view, row));
  }
  Scalar mass[6][6]{};
  read_parameter_matrix6(virtual_mass_matrix, env, mass);
  Scalar acceleration[6]{};
  if (!solve_6x6_matrix_vector(mass, generalized_force, acceleration)) {
    flags |= kInvalidParameterFlag | kSafeModeFlag;
    zero_output(commanded_acceleration, env);
    status[env] = flags;
    return;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar limit =
        read_optional_vector_parameter(acceleration_limit, env, dim, default_limit<Scalar>());
    const auto limited = detail::clamp_value(acceleration[dim], -limit, limit);
    acceleration[dim] = limited.value;
    if (limited.saturated) {
      flags |= kSaturationFlag;
    }
    write_matrix(commanded_acceleration, env, dim, acceleration[dim]);
  }

  const std::size_t twist_init_offset = env * static_cast<std::size_t>(twist_initialized.stride);
  Scalar new_twist[6]{};
  if (twist_initialized.data[twist_init_offset] == 0U) {
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      write_matrix(twist_previous, env, dim, acceleration[dim]);
      write_matrix(twist, env, dim, static_cast<Scalar>(0));
    }
    twist_initialized.data[twist_init_offset] = 1U;
  } else {
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar old_value = twist.data[env * static_cast<std::size_t>(twist.row_stride) +
                                          dim * static_cast<std::size_t>(twist.col_stride)];
      const Scalar previous_input =
          twist_previous.data[env * static_cast<std::size_t>(twist_previous.row_stride) +
                              dim * static_cast<std::size_t>(twist_previous.col_stride)];
      const Scalar limit =
          read_optional_vector_parameter(twist_limit, env, dim, default_limit<Scalar>());
      const auto update = detail::integrator_trapezoid_value(
          old_value, acceleration[dim], previous_input, dt_value, -limit, limit);
      if (update.saturated) {
        flags |= kSaturationFlag;
      }
      new_twist[dim] = update.value;
      write_matrix(twist, env, dim, update.value);
      write_matrix(twist_previous, env, dim, acceleration[dim]);
    }
  }

  const std::size_t pose_init_offset = env * static_cast<std::size_t>(pose_initialized.stride);
  if (pose_initialized.data[pose_init_offset] == 0U) {
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      write_matrix(pose_previous, env, dim, new_twist[dim]);
      write_matrix(pose, env, dim, static_cast<Scalar>(0));
    }
    pose_initialized.data[pose_init_offset] = 1U;
  } else {
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar old_value = pose.data[env * static_cast<std::size_t>(pose.row_stride) +
                                         dim * static_cast<std::size_t>(pose.col_stride)];
      const Scalar previous_input =
          pose_previous.data[env * static_cast<std::size_t>(pose_previous.row_stride) +
                             dim * static_cast<std::size_t>(pose_previous.col_stride)];
      const Scalar limit =
          read_optional_vector_parameter(pose_limit, env, dim, default_limit<Scalar>());
      const auto update = detail::integrator_trapezoid_value(
          old_value, new_twist[dim], previous_input, dt_value, -limit, limit);
      if (update.saturated) {
        flags |= kSaturationFlag;
      }
      write_matrix(pose, env, dim, update.value);
      write_matrix(pose_previous, env, dim, new_twist[dim]);
    }
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_variable_stiffness_kernel(CudaMatrixView<Scalar> cartesian_pose_error,
                                               CudaTensor3View<Scalar> stiffness_matrix,
                                               CudaScalarBatchView<Scalar> dt,
                                               CudaMatrixParameterView<Scalar> initial_stiffness,
                                               CudaVectorParameterView<Scalar> maximum_stiffness,
                                               CudaVectorParameterView<Scalar> minimum_stiffness,
                                               CudaVectorParameterView<Scalar> maximum_rate,
                                               const std::uint64_t* parameter_status,
                                               CudaMutableTensor3View<Scalar> current_stiffness,
                                               CudaMutableTensor3View<Scalar> previous_stiffness,
                                               CudaBoolVectorView initialized,
                                               CudaMutableMatrixView<Scalar> cartesian_wrench,
                                               std::size_t num_envs,
                                               std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const Scalar dt_value = read_scalar_batch(dt, env);
  if (!stiffness_limits_are_valid(
          initial_stiffness, maximum_stiffness, minimum_stiffness, maximum_rate, env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!input_is_finite(cartesian_pose_error, env) ||
      !tensor3_is_finite(stiffness_matrix, env, kCartesianDim, kCartesianDim)) {
    flags |= kNonFiniteInputFlag;
  }
  if (!is_finite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(cartesian_wrench, env);
    status[env] = flags;
    return;
  }
  const std::size_t initialized_offset = env * static_cast<std::size_t>(initialized.stride);
  if (initialized.data[initialized_offset] == 0U) {
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        const Scalar value = read_matrix_parameter(initial_stiffness, env, row, col);
        write_tensor3(current_stiffness, env, row, col, value);
        write_tensor3(previous_stiffness, env, row, col, value);
      }
    }
    initialized.data[initialized_offset] = 1U;
  } else {
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const Scalar max_stiffness = read_vector_parameter(maximum_stiffness, env, row);
      const Scalar min_stiffness =
          read_optional_vector_parameter(minimum_stiffness, env, row, static_cast<Scalar>(0));
      const Scalar max_rate = read_vector_parameter(maximum_rate, env, row);
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        const Scalar old_value = read_tensor3(current_stiffness, env, row, col);
        const Scalar target_value = read_tensor3(stiffness_matrix, env, row, col);
        const Scalar upper_limit = row == col ? max_stiffness : static_cast<Scalar>(0);
        const Scalar lower_limit = row == col ? min_stiffness : static_cast<Scalar>(0);
        const Scalar rate_limit = row == col ? max_rate : static_cast<Scalar>(0);
        const Scalar new_value = detail::variable_stiffness_value(
            old_value, target_value, lower_limit, upper_limit, rate_limit, dt_value);
        if (new_value != target_value) {
          flags |= kSaturationFlag;
        }
        write_tensor3(previous_stiffness, env, row, col, new_value);
        write_tensor3(current_stiffness, env, row, col, new_value);
      }
    }
  }
  const auto current_view = core_tensor3_env(current_stiffness, env);
  const auto error_view = core_vector_env(cartesian_pose_error, env, kCartesianDim);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    write_matrix(cartesian_wrench,
                 env,
                 row,
                 detail::cartesian_stiffness_value(current_view, error_view, row));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void step_force_kernel(CudaMatrixView<Scalar> external_wrench,
                                  CudaMatrixView<Scalar> desired_wrench,
                                  CudaMatrixView<Scalar> feedforward_wrench,
                                  CudaMatrixView<Scalar> pose_error,
                                  CudaScalarBatchView<Scalar> dt,
                                  bool use_shaping_function,
                                  CudaVectorParameterView<Scalar> proportional_gain,
                                  CudaVectorParameterView<Scalar> differentiator_gain,
                                  CudaVectorParameterView<Scalar> integral_gain,
                                  CudaVectorParameterView<Scalar> u0,
                                  CudaVectorParameterView<Scalar> upper,
                                  CudaVectorParameterView<Scalar> lower,
                                  CudaVectorParameterView<Scalar> maximum,
                                  CudaVectorParameterView<Scalar> shaping,
                                  const std::uint64_t* parameter_status,
                                  CudaMutableMatrixView<Scalar> differentiator_previous,
                                  CudaBoolVectorView differentiator_initialized,
                                  CudaMutableMatrixView<Scalar> integrator_previous,
                                  CudaMutableMatrixView<Scalar> integrated_error,
                                  CudaBoolVectorView integrator_initialized,
                                  CudaMutableMatrixView<Scalar> commanded_wrench,
                                  std::size_t num_envs,
                                  std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = parameter_status == nullptr ? 0U : parameter_status[env];
  const Scalar dt_value = read_scalar_batch(dt, env);
  if (!force_parameters_are_valid(proportional_gain,
                                  differentiator_gain,
                                  integral_gain,
                                  u0,
                                  upper,
                                  lower,
                                  maximum,
                                  shaping,
                                  use_shaping_function,
                                  env)) {
    flags |= kInvalidParameterFlag;
  }
  if (!is_finite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
    flags |= kInvalidDtFlag;
  }
  if (!input_is_finite(external_wrench, env) || !input_is_finite(desired_wrench, env) ||
      !input_is_finite(feedforward_wrench, env) || !input_is_finite(pose_error, env)) {
    flags |= kNonFiniteInputFlag;
  }
  if ((flags & kSafeTriggers) != 0U) {
    flags |= kSafeModeFlag;
    zero_output(commanded_wrench, env);
    status[env] = flags;
    return;
  }

  const std::size_t diff_initialized_offset =
      env * static_cast<std::size_t>(differentiator_initialized.stride);
  const std::size_t int_initialized_offset =
      env * static_cast<std::size_t>(integrator_initialized.stride);
  Scalar derivative[6]{};
  Scalar integral_output[6]{};
  Scalar error[6]{};
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    error[dim] = detail::force_error_value(read_matrix(desired_wrench, env, dim),
                                           read_matrix(external_wrench, env, dim));
    if (differentiator_initialized.data[diff_initialized_offset] != 0U) {
      const Scalar previous =
          differentiator_previous
              .data[env * static_cast<std::size_t>(differentiator_previous.row_stride) +
                    dim * static_cast<std::size_t>(differentiator_previous.col_stride)];
      derivative[dim] = detail::differentiator_value(error[dim], previous, dt_value);
    }
    if (integrator_initialized.data[int_initialized_offset] == 0U) {
      write_matrix(integrated_error,
                   env,
                   dim,
                   read_optional_vector_parameter(u0, env, dim, static_cast<Scalar>(0)));
    } else {
      const Scalar upper_value =
          read_optional_vector_parameter(upper, env, dim, default_limit<Scalar>());
      const Scalar lower_value =
          read_optional_vector_parameter(lower, env, dim, -default_limit<Scalar>());
      const Scalar previous =
          integrator_previous.data[env * static_cast<std::size_t>(integrator_previous.row_stride) +
                                   dim * static_cast<std::size_t>(integrator_previous.col_stride)];
      const Scalar integrated =
          integrated_error.data[env * static_cast<std::size_t>(integrated_error.row_stride) +
                                dim * static_cast<std::size_t>(integrated_error.col_stride)];
      const auto update = detail::integrator_trapezoid_value(
          integrated, error[dim], previous, dt_value, lower_value, upper_value);
      if (update.saturated) {
        flags |= kSaturationFlag;
      }
      write_matrix(integrated_error, env, dim, update.value);
      integral_output[dim] = update.value;
    }
  }
  differentiator_initialized.data[diff_initialized_offset] = 1U;
  integrator_initialized.data[int_initialized_offset] = 1U;

  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    Scalar scale = static_cast<Scalar>(1);
    if (use_shaping_function) {
      scale = shaping_factor_value(read_matrix(pose_error, env, dim),
                                   read_vector_parameter(maximum, env, dim),
                                   read_vector_parameter(shaping, env, dim));
    }
    const Scalar command = detail::cartesian_force_command_value(
        detail::proportional_value(read_vector_parameter(proportional_gain, env, dim), error[dim]),
        detail::proportional_value(read_vector_parameter(differentiator_gain, env, dim),
                                   derivative[dim]),
        detail::proportional_value(read_vector_parameter(integral_gain, env, dim),
                                   integral_output[dim]),
        read_matrix(desired_wrench, env, dim),
        read_matrix(feedforward_wrench, env, dim),
        scale);
    write_matrix(commanded_wrench, env, dim, command);
    write_matrix(differentiator_previous, env, dim, error[dim]);
    write_matrix(integrator_previous, env, dim, error[dim]);
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void combine_sum_kernel(CudaMatrixView<Scalar> left_wrench,
                                   const std::uint64_t* left_status,
                                   CudaMatrixView<Scalar> right_wrench,
                                   const std::uint64_t* right_status,
                                   CudaMutableMatrixView<Scalar> output,
                                   std::size_t num_envs,
                                   std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = (left_status == nullptr ? 0U : left_status[env]) |
                        (right_status == nullptr ? 0U : right_status[env]);
  if ((flags & kSafeModeFlag) != 0U) {
    zero_output(output, env);
    status[env] = flags;
    return;
  }
  if (!input_is_finite(left_wrench, env) || !input_is_finite(right_wrench, env)) {
    flags |= kNonFiniteInputFlag | kSafeModeFlag;
    zero_output(output, env);
    status[env] = flags;
    return;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    write_matrix(output,
                 env,
                 dim,
                 detail::unified_force_impedance_value(read_matrix(left_wrench, env, dim),
                                                       read_matrix(right_wrench, env, dim)));
  }
  status[env] = flags;
}

template <typename Scalar>
__global__ void combine_variable_impedance_kernel(CudaMatrixView<Scalar> stiffness_wrench,
                                                  const std::uint64_t* stiffness_status,
                                                  CudaMatrixView<Scalar> damping_wrench,
                                                  const std::uint64_t* damping_status,
                                                  CudaMatrixView<Scalar> inertial_wrench,
                                                  const std::uint64_t* inertial_status,
                                                  CudaMatrixView<Scalar> desired_acceleration,
                                                  CudaTensor3View<Scalar> cartesian_mass_matrix,
                                                  CudaMutableMatrixView<Scalar> output,
                                                  std::size_t num_envs,
                                                  std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  std::uint64_t flags = (stiffness_status == nullptr ? 0U : stiffness_status[env]) |
                        (damping_status == nullptr ? 0U : damping_status[env]) |
                        (inertial_status == nullptr ? 0U : inertial_status[env]);
  if (!input_is_finite(desired_acceleration, env) ||
      !tensor3_is_finite(cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
      !input_is_finite(stiffness_wrench, env) || !input_is_finite(damping_wrench, env) ||
      !input_is_finite(inertial_wrench, env)) {
    flags |= kNonFiniteInputFlag | kSafeModeFlag;
  }
  if ((flags & kSafeModeFlag) != 0U) {
    zero_output(output, env);
    status[env] = flags;
    return;
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    Scalar acceleration_wrench = static_cast<Scalar>(0);
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      acceleration_wrench += read_tensor3(cartesian_mass_matrix, env, row, col) *
                             read_matrix(desired_acceleration, env, col);
    }
    write_matrix(output,
                 env,
                 row,
                 detail::cartesian_impedance_value(read_matrix(stiffness_wrench, env, row),
                                                   read_matrix(damping_wrench, env, row),
                                                   read_matrix(inertial_wrench, env, row),
                                                   acceleration_wrench));
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

template <CudaTensorLaunchFamily Family>
inline CudaLaunchConfig launch_untyped_config(const std::size_t num_envs) {
  constexpr int kThreads = cuda_tensor_tuned_threads<Family, DType::kUnknown>();
  return CudaLaunchConfig{static_cast<int>((num_envs + kThreads - 1) / kThreads), kThreads};
}

inline bool is_float_dtype(const DType dtype) {
  return dtype == DType::kFloat32 || dtype == DType::kFloat64;
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

inline bool is_cuda_cartesian_matrix(const TensorView& view) {
  return view.data != nullptr && view.device == Device::kCuda &&
         (view.dtype == DType::kFloat32 || view.dtype == DType::kFloat64) && view.rank == 2 &&
         view.layout == Layout::kRowMajor && view.shape[0] > 0 &&
         view.shape[1] == static_cast<std::int64_t>(kCartesianDim) && view.strides[1] == 1 &&
         view.strides[0] >= view.shape[1];
}

inline bool is_cuda_cartesian_matrix(const MutableTensorView& view) {
  return view.data != nullptr && view.device == Device::kCuda &&
         (view.dtype == DType::kFloat32 || view.dtype == DType::kFloat64) && view.rank == 2 &&
         view.layout == Layout::kRowMajor && view.shape[0] > 0 &&
         view.shape[1] == static_cast<std::int64_t>(kCartesianDim) && view.strides[1] == 1 &&
         view.strides[0] >= view.shape[1];
}

inline Result validate_cuda_output(const MutableTensorView& output,
                                   const DType dtype,
                                   const std::size_t num_envs) {
  if (!is_cuda_cartesian_matrix(output)) {
    return Result{ResultCode::kInvalidView, "output tensor must be CUDA row-major [num_envs, 6]"};
  }
  if (output.dtype != dtype || output.shape[0] != static_cast<std::int64_t>(num_envs)) {
    return Result{ResultCode::kInvalidView, "output tensor shape or dtype does not match input"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_matrix_signal(const TensorView& input,
                                          const char* name,
                                          std::size_t* num_envs,
                                          std::size_t* signal_dim) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || !is_float_dtype(input.dtype) || input.rank != 2 ||
      input.layout != Layout::kRowMajor || input.shape[0] <= 0 || input.shape[1] <= 0 ||
      input.strides[1] != 1 || input.strides[0] < input.shape[1]) {
    return Result{ResultCode::kInvalidView,
                  "input tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  *num_envs = static_cast<std::size_t>(input.shape[0]);
  *signal_dim = static_cast<std::size_t>(input.shape[1]);
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_matrix_output(const MutableTensorView& output,
                                          const DType dtype,
                                          const std::size_t num_envs,
                                          const std::size_t signal_dim) {
  if (output.data == nullptr) {
    return Result{ResultCode::kInvalidView, "output tensor data is null"};
  }
  if (output.device != Device::kCuda || output.dtype != dtype || output.rank != 2 ||
      output.layout != Layout::kRowMajor ||
      output.shape[0] != static_cast<std::int64_t>(num_envs) ||
      output.shape[1] != static_cast<std::int64_t>(signal_dim) || output.strides[1] != 1 ||
      output.strides[0] < output.shape[1]) {
    return Result{ResultCode::kInvalidView,
                  "output tensor must be CUDA row-major [num_envs, signal_dim]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_tensor3_matrix6(const TensorView& input,
                                            const char* name,
                                            const DType dtype,
                                            const std::size_t num_envs) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype || input.rank != 3 ||
      input.layout != Layout::kRowMajor || input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      input.shape[2] != static_cast<std::int64_t>(kCartesianDim) || input.strides[2] != 1 ||
      input.strides[1] < input.shape[2] || input.strides[0] < input.shape[1] * input.strides[1]) {
    return Result{ResultCode::kInvalidView,
                  "matrix tensor must be CUDA row-major [num_envs, 6, 6]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_tensor3_matrix6(const MutableTensorView& input,
                                            const char* name,
                                            const DType dtype,
                                            const std::size_t num_envs) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype || input.rank != 3 ||
      input.layout != Layout::kRowMajor || input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      input.shape[2] != static_cast<std::int64_t>(kCartesianDim) || input.strides[2] != 1 ||
      input.strides[1] < input.shape[2] || input.strides[0] < input.shape[1] * input.strides[1]) {
    return Result{ResultCode::kInvalidView, "matrix state must be CUDA row-major [num_envs, 6, 6]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_tensor3_matrix(const TensorView& input,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const std::size_t rows,
                                           const std::size_t cols) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype || input.rank != 3 ||
      input.layout != Layout::kRowMajor || input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(rows) ||
      input.shape[2] != static_cast<std::int64_t>(cols) || input.strides[2] != 1 ||
      input.strides[1] < input.shape[2] || input.strides[0] < input.shape[1] * input.strides[1]) {
    return Result{ResultCode::kInvalidView,
                  "matrix tensor must be CUDA row-major [num_envs, rows, cols]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_tensor3_matrix(const MutableTensorView& input,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const std::size_t rows,
                                           const std::size_t cols) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != dtype || input.rank != 3 ||
      input.layout != Layout::kRowMajor || input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(rows) ||
      input.shape[2] != static_cast<std::int64_t>(cols) || input.strides[2] != 1 ||
      input.strides[1] < input.shape[2] || input.strides[0] < input.shape[1] * input.strides[1]) {
    return Result{ResultCode::kInvalidView,
                  "matrix state must be CUDA row-major [num_envs, rows, cols]"};
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
  if (input.device != Device::kCuda || input.dtype != dtype || input.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "scalar batch must be a CUDA tensor matching input dtype"};
  }
  if (input.rank == 0) {
    return Result{ResultCode::kOk, "ok"};
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "scalar batch must be scalar or [num_envs]"};
}

inline Result validate_cuda_state_matrix(const MutableTensorView& input,
                                         const char* name,
                                         const DType dtype,
                                         const std::size_t num_envs) {
  if (!is_cuda_cartesian_matrix(input)) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.dtype != dtype || input.shape[0] != static_cast<std::int64_t>(num_envs)) {
    return Result{ResultCode::kInvalidView, "state matrix shape or dtype does not match input"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_state_bool(const MutableTensorView& input,
                                       const char* name,
                                       const std::size_t num_envs) {
  if (input.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (input.device != Device::kCuda || input.dtype != DType::kBool || input.rank != 1 ||
      input.shape[0] < static_cast<std::int64_t>(num_envs) || input.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "state bool must be CUDA bool [num_envs]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_pose(const TensorView& pose,
                                 const char* name,
                                 const DType dtype,
                                 const std::size_t num_envs) {
  if (pose.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (pose.device != Device::kCuda || pose.dtype != dtype || pose.rank != 3 ||
      pose.layout != Layout::kRowMajor || pose.shape[0] != static_cast<std::int64_t>(num_envs) ||
      pose.shape[1] != static_cast<std::int64_t>(kPoseRows) ||
      pose.shape[2] != static_cast<std::int64_t>(kPoseCols) || pose.strides[2] != 1 ||
      pose.strides[1] < pose.shape[2] || pose.strides[0] < pose.shape[1] * pose.strides[1]) {
    return Result{ResultCode::kInvalidView, "pose tensor must be CUDA row-major [num_envs, 4, 4]"};
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_pose_with_num_envs(const TensorView& pose,
                                               const char* name,
                                               std::size_t* num_envs) {
  if (pose.data == nullptr) {
    return Result{ResultCode::kInvalidView, name};
  }
  if (pose.device != Device::kCuda || !is_float_dtype(pose.dtype) || pose.rank != 3 ||
      pose.layout != Layout::kRowMajor || pose.shape[0] <= 0 ||
      pose.shape[1] != static_cast<std::int64_t>(kPoseRows) ||
      pose.shape[2] != static_cast<std::int64_t>(kPoseCols) || pose.strides[2] != 1 ||
      pose.strides[1] < pose.shape[2] || pose.strides[0] < pose.shape[1] * pose.strides[1]) {
    return Result{ResultCode::kInvalidView, "pose tensor must be CUDA row-major [num_envs, 4, 4]"};
  }
  *num_envs = static_cast<std::size_t>(pose.shape[0]);
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_bool_output(const MutableTensorView& output,
                                        const std::size_t num_envs) {
  if (output.data == nullptr) {
    return Result{ResultCode::kInvalidView, "bool output data is null"};
  }
  if (output.device != Device::kCuda || output.dtype != DType::kBool || output.rank != 1 ||
      output.shape[0] < static_cast<std::int64_t>(num_envs) || output.strides[0] < 1) {
    return Result{ResultCode::kInvalidView, "bool output must be CUDA [num_envs]"};
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

inline Result validate_cuda_scalar_parameter(const ParameterView* parameter,
                                             const char* missing_message,
                                             const DType dtype,
                                             const std::size_t num_envs) {
  if (parameter == nullptr || parameter->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, missing_message};
  }
  if (!is_float_dtype(dtype)) {
    return Result{ResultCode::kInvalidView, "scalar parameter dtype must be float32 or float64"};
  }
  const TensorView& values = parameter->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "scalar parameter must be a CUDA tensor matching input dtype"};
  }
  if (parameter->shared) {
    if (values.rank == 0 || (values.rank == 1 && values.shape[0] == 1 && values.strides[0] >= 1)) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView, "shared scalar parameter must be scalar or [1]"};
  }
  if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.strides[0] >= 1) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "batched scalar parameter must have [num_envs]"};
}

inline Result validate_cuda_vector_parameter(const ParameterView* parameter,
                                             const char* missing_message,
                                             const DType dtype,
                                             const std::size_t num_envs,
                                             const std::size_t signal_dim) {
  if (parameter == nullptr || parameter->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, missing_message};
  }
  if (!is_float_dtype(dtype)) {
    return Result{ResultCode::kInvalidView, "vector parameter dtype must be float32 or float64"};
  }
  const TensorView& values = parameter->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "vector parameter must be a CUDA tensor matching input dtype"};
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

inline Result validate_cuda_optional_vector_parameter(const ParameterView* parameter,
                                                      const DType dtype,
                                                      const std::size_t num_envs,
                                                      const std::size_t signal_dim) {
  if (parameter == nullptr) {
    return Result{ResultCode::kOk, "ok"};
  }
  return validate_cuda_vector_parameter(
      parameter, "optional vector parameter is invalid", dtype, num_envs, signal_dim);
}

inline Result validate_cuda_matrix_parameter(const ParameterView* matrix,
                                             const DType dtype,
                                             const std::size_t num_envs) {
  if (matrix == nullptr || matrix->values.data == nullptr) {
    return Result{ResultCode::kInvalidView, "Cartesian matrix parameter is missing"};
  }
  if (!is_float_dtype(dtype)) {
    return Result{ResultCode::kInvalidView,
                  "Cartesian matrix parameter dtype must be float32 or float64"};
  }
  const TensorView& values = matrix->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return Result{ResultCode::kInvalidView,
                  "Cartesian matrix parameter must be a CUDA tensor matching input dtype"};
  }
  if (matrix->shared) {
    if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(kCartesianDim) &&
        values.shape[1] == static_cast<std::int64_t>(kCartesianDim) && values.strides[1] == 1 &&
        values.strides[0] >= values.shape[1]) {
      return Result{ResultCode::kOk, "ok"};
    }
    return Result{ResultCode::kInvalidView, "shared Cartesian matrix must have [6, 6]"};
  }
  if (values.rank == 3 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(kCartesianDim) &&
      values.shape[2] == static_cast<std::int64_t>(kCartesianDim) && values.strides[2] == 1 &&
      values.strides[1] >= values.shape[2] &&
      values.strides[0] >= values.shape[1] * values.strides[1]) {
    return Result{ResultCode::kOk, "ok"};
  }
  return Result{ResultCode::kInvalidView, "batched Cartesian matrix must have [num_envs, 6, 6]"};
}

inline Result validate_cuda_flow_parameters(const ParameterSetView& parameters,
                                            const DType dtype,
                                            const std::size_t num_envs) {
  Result result = validate_cuda_scalar_parameter(find_parameter(parameters, "funnel_radius"),
                                                 "funnel_radius parameter is missing",
                                                 dtype,
                                                 num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_parameter(find_parameter(parameters, "max_distance"),
                                          "max_distance parameter is missing",
                                          dtype,
                                          num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_parameter(find_parameter(parameters, "funnel_velocity"),
                                          "funnel_velocity parameter is missing",
                                          dtype,
                                          num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, "funnel_start_point"),
                                          "funnel_start_point parameter is missing",
                                          dtype,
                                          num_envs,
                                          kPointDim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter(find_parameter(parameters, "funnel_end_point"),
                                        "funnel_end_point parameter is missing",
                                        dtype,
                                        num_envs,
                                        kPointDim);
}

inline Result validate_cuda_inverse_apf_parameters(const ParameterSetView& parameters,
                                                   const DType dtype,
                                                   const std::size_t num_envs) {
  const char* names[] = {"alpha", "gamma", "d_min", "d_max", "deadzone"};
  for (const char* name : names) {
    Result result = validate_cuda_scalar_parameter(
        find_parameter(parameters, name), "scalar parameter is missing", dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_impedance_parameters(const ParameterSetView& parameters,
                                                 const DType dtype,
                                                 const std::size_t num_envs) {
  Result result = validate_cuda_matrix_parameter(
      find_parameter(parameters, "stiffness_matrix"), dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter(find_parameter(parameters, "damping_factors"),
                                        "damping_factors parameter is missing",
                                        dtype,
                                        num_envs,
                                        kCartesianDim);
}

inline Result validate_cuda_operational_space_parameters(const ParameterSetView& parameters,
                                                         const DType dtype,
                                                         const std::size_t num_envs) {
  Result result = validate_cuda_vector_parameter(find_parameter(parameters, "position_gain"),
                                                 "position_gain parameter is missing",
                                                 dtype,
                                                 num_envs,
                                                 kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter(find_parameter(parameters, "velocity_gain"),
                                        "velocity_gain parameter is missing",
                                        dtype,
                                        num_envs,
                                        kCartesianDim);
}

inline Result validate_cuda_admittance_parameters(const ParameterSetView& parameters,
                                                  const DType dtype,
                                                  const std::size_t num_envs) {
  const char* matrix_names[] = {"stiffness_matrix", "damping_matrix", "virtual_mass_matrix"};
  for (const char* name : matrix_names) {
    Result result =
        validate_cuda_matrix_parameter(find_parameter(parameters, name), dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  Result result =
      validate_cuda_vector_parameter(find_parameter(parameters, "force_filter_threshold"),
                                     "force_filter_threshold parameter is missing",
                                     dtype,
                                     num_envs,
                                     kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  const char* optional_names[] = {"acceleration_limit", "twist_limit", "pose_limit"};
  for (const char* name : optional_names) {
    result = validate_cuda_optional_vector_parameter(
        find_parameter(parameters, name), dtype, num_envs, kCartesianDim);
    if (!result.ok()) {
      return result;
    }
  }
  return Result{ResultCode::kOk, "ok"};
}

inline Result validate_cuda_variable_stiffness_parameters(const ParameterSetView& parameters,
                                                          const DType dtype,
                                                          const std::size_t num_envs) {
  Result result = validate_cuda_matrix_parameter(
      find_parameter(parameters, "initial_stiffness_matrix"), dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, "maximum_stiffness"),
                                          "maximum_stiffness parameter is missing",
                                          dtype,
                                          num_envs,
                                          kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_optional_vector_parameter(
      find_parameter(parameters, "minimum_stiffness"), dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter(find_parameter(parameters, "maximum_stiffness_rate"),
                                        "maximum_stiffness_rate parameter is missing",
                                        dtype,
                                        num_envs,
                                        kCartesianDim);
}

inline Result validate_cuda_force_parameters(const ParameterSetView& parameters,
                                             const DType dtype,
                                             const std::size_t num_envs,
                                             const bool use_shaping_function) {
  const char* gain_names[] = {"proportional_gain", "differentiator_gain", "integral_gain"};
  for (const char* name : gain_names) {
    Result result = validate_cuda_vector_parameter(find_parameter(parameters, name),
                                                   "force controller gain is missing",
                                                   dtype,
                                                   num_envs,
                                                   kCartesianDim);
    if (!result.ok()) {
      return result;
    }
  }
  const char* optional_names[] = {"u_0", "upper_limit", "lower_limit"};
  for (const char* name : optional_names) {
    Result result = validate_cuda_optional_vector_parameter(
        find_parameter(parameters, name), dtype, num_envs, kCartesianDim);
    if (!result.ok()) {
      return result;
    }
  }
  if (!use_shaping_function) {
    return Result{ResultCode::kOk, "ok"};
  }
  Result result = validate_cuda_vector_parameter(find_parameter(parameters, "maximum_distance"),
                                                 "maximum_distance parameter is missing",
                                                 dtype,
                                                 num_envs,
                                                 kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter(find_parameter(parameters, "shaping_distance"),
                                        "shaping_distance parameter is missing",
                                        dtype,
                                        num_envs,
                                        kCartesianDim);
}

template <typename Scalar>
CudaTensor3View<Scalar> as_cuda_tensor3(const TensorView& view) {
  return CudaTensor3View<Scalar>{
      static_cast<const Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
}

template <typename Scalar>
CudaMutableTensor3View<Scalar> as_cuda_mutable_tensor3(const MutableTensorView& view) {
  return CudaMutableTensor3View<Scalar>{
      static_cast<Scalar*>(view.data), view.strides[0], view.strides[1], view.strides[2]};
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
      static_cast<const Scalar*>(view.data), view.rank, view.rank == 1 ? view.strides[0] : 0};
}

inline CudaBoolVectorView as_cuda_bool_vector(const MutableTensorView& view) {
  return CudaBoolVectorView{static_cast<std::uint8_t*>(view.data), view.strides[0]};
}

template <typename Scalar>
CudaMatrixParameterView<Scalar> as_cuda_matrix_parameter(const ParameterView& matrix) {
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
CudaScalarParameterView<Scalar> as_cuda_scalar_parameter(const ParameterView& parameter) {
  const TensorView& values = parameter.values;
  return CudaScalarParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         parameter.shared,
                                         values.rank == 1 ? values.strides[0] : 0};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> as_cuda_vector_parameter(const ParameterView& parameter) {
  const TensorView& values = parameter.values;
  const bool shared = parameter.shared || values.rank == 1;
  return CudaVectorParameterView<Scalar>{static_cast<const Scalar*>(values.data),
                                         shared,
                                         values.rank,
                                         values.rank == 2 ? values.strides[0] : 0,
                                         values.rank == 2 ? values.strides[1] : values.strides[0]};
}

template <typename Scalar>
CudaVectorParameterView<Scalar> as_cuda_optional_vector_parameter(const ParameterView* parameter) {
  if (parameter == nullptr) {
    return CudaVectorParameterView<Scalar>{nullptr, true, 0, 0, 0};
  }
  return as_cuda_vector_parameter<Scalar>(*parameter);
}

template <typename Scalar>
Result launch_apply_matrix6(const ParameterView& matrix,
                            const std::size_t num_envs,
                            MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyMatrix6, Scalar>(num_envs);
  apply_matrix6_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(matrix), num_envs, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_shaping(const ParameterView& maximum,
                            const ParameterView& shaping,
                            const std::size_t num_envs,
                            const std::size_t signal_dim,
                            MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyShaping, Scalar>(num_envs);
  apply_shaping_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_vector_parameter<Scalar>(maximum),
          as_cuda_vector_parameter<Scalar>(shaping),
          num_envs,
          signal_dim,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_damping_design(const ParameterView& damping_factors,
                                   const std::size_t num_envs,
                                   MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyDampingDesign, Scalar>(num_envs);
  apply_damping_design_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_vector_parameter<Scalar>(damping_factors), num_envs, parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_flow(const ParameterSetView& parameters,
                         const std::size_t num_envs,
                         MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyFlow, Scalar>(num_envs);
  apply_flow_parameters_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "funnel_radius")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "max_distance")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "funnel_velocity")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "funnel_start_point")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "funnel_end_point")),
      num_envs,
      parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_inverse_apf(const ParameterSetView& parameters,
                                const std::size_t num_envs,
                                MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyInverseApf, Scalar>(num_envs);
  apply_inverse_apf_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "alpha")),
          as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "gamma")),
          as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "d_min")),
          as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "d_max")),
          as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "deadzone")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_impedance(const ParameterSetView& parameters,
                              const std::size_t num_envs,
                              MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyImpedance, Scalar>(num_envs);
  apply_cartesian_impedance_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "damping_factors")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_operational_space(const ParameterSetView& parameters,
                                      const std::size_t num_envs,
                                      MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyOperationalSpace, Scalar>(num_envs);
  apply_operational_space_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "position_gain")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "velocity_gain")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_admittance(const ParameterSetView& parameters,
                               const std::size_t num_envs,
                               MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyAdmittance, Scalar>(num_envs);
  apply_cartesian_admittance_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "stiffness_matrix")),
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "damping_matrix")),
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "virtual_mass_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "force_filter_threshold")),
          as_cuda_optional_vector_parameter<Scalar>(
              find_parameter(parameters, "acceleration_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "twist_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "pose_limit")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_variable_stiffness(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyVariableStiffness, Scalar>(num_envs);
  apply_variable_stiffness_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "initial_stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness")),
          as_cuda_optional_vector_parameter<Scalar>(
              find_parameter(parameters, "minimum_stiffness")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness_rate")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_force(const ParameterSetView& parameters,
                          const std::size_t num_envs,
                          const bool use_shaping_function,
                          MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyForce, Scalar>(num_envs);
  apply_force_parameters_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "proportional_gain")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "differentiator_gain")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "integral_gain")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "u_0")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "upper_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "lower_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "maximum_distance")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "shaping_distance")),
      use_shaping_function,
      num_envs,
      parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_unified_force_impedance(const ParameterSetView& parameters,
                                            const std::size_t num_envs,
                                            const bool use_shaping_function,
                                            MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyUnifiedForceImpedance, Scalar>(num_envs);
  apply_unified_force_impedance_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "damping_factors")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "proportional_gain")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "differentiator_gain")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "integral_gain")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "u_0")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "upper_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "lower_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "maximum_distance")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "shaping_distance")),
          use_shaping_function,
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_variable_impedance(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyVariableImpedance, Scalar>(num_envs);
  apply_variable_impedance_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "initial_stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness")),
          as_cuda_optional_vector_parameter<Scalar>(
              find_parameter(parameters, "minimum_stiffness")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness_rate")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "damping_factors")),
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_apply_variable_unified(const ParameterSetView& parameters,
                                     const std::size_t num_envs,
                                     const bool use_shaping_function,
                                     MutableStatusView parameter_status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianApplyVariableUnified, Scalar>(num_envs);
  apply_variable_unified_parameters_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "initial_stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness")),
          as_cuda_optional_vector_parameter<Scalar>(
              find_parameter(parameters, "minimum_stiffness")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness_rate")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "damping_factors")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "proportional_gain")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "differentiator_gain")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "integral_gain")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "u_0")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "upper_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "lower_limit")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "maximum_distance")),
          as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "shaping_distance")),
          use_shaping_function,
          num_envs,
          parameter_status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_shaping(const TensorView& error,
                           const ParameterView& maximum,
                           const ParameterView& shaping,
                           StatusView parameter_status,
                           const MutableTensorView& output,
                           MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepShaping, Scalar>(
          static_cast<std::size_t>(error.shape[0]));
  step_shaping_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(error),
      as_cuda_vector_parameter<Scalar>(maximum),
      as_cuda_vector_parameter<Scalar>(shaping),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(output),
      static_cast<std::size_t>(error.shape[0]),
      static_cast<std::size_t>(error.shape[1]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_command_observer(const TensorView& command,
                                    StatusView parameter_status,
                                    const MutableTensorView& command_is_valid,
                                    MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepCommandObserver, Scalar>(
          static_cast<std::size_t>(command.shape[0]));
  step_command_observer_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(command),
      parameter_status.flags,
      static_cast<std::uint8_t*>(command_is_valid.data),
      command_is_valid.strides[0],
      static_cast<std::size_t>(command.shape[0]),
      static_cast<std::size_t>(command.shape[1]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_flow(const TensorView& cartesian_pose,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        const MutableTensorView& cartesian_flow,
                        MutableStatusView status) {
  const CudaLaunchConfig config = launch_config<CudaTensorLaunchFamily::kCartesianStepFlow, Scalar>(
      static_cast<std::size_t>(cartesian_pose.shape[0]));
  step_flow_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_tensor3<Scalar>(cartesian_pose),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "funnel_radius")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "max_distance")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "funnel_velocity")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "funnel_start_point")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "funnel_end_point")),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(cartesian_flow),
      static_cast<std::size_t>(cartesian_pose.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_inverse_apf(const TensorView& current_pose,
                               const TensorView& goal_pose,
                               const TensorView& cartesian_twist,
                               const ParameterSetView& parameters,
                               StatusView parameter_status,
                               const MutableTensorView& cartesian_wrench,
                               MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepInverseApf, Scalar>(
          static_cast<std::size_t>(cartesian_twist.shape[0]));
  step_inverse_apf_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_tensor3<Scalar>(current_pose),
      as_cuda_tensor3<Scalar>(goal_pose),
      as_cuda_matrix<Scalar>(cartesian_twist),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "alpha")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "gamma")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "d_min")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "d_max")),
      as_cuda_scalar_parameter<Scalar>(*find_parameter(parameters, "deadzone")),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(cartesian_wrench),
      static_cast<std::size_t>(cartesian_twist.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_damping_design(const bool factorization_design,
                                  const TensorView& cartesian_velocity_error,
                                  const TensorView& cartesian_mass_matrix,
                                  const TensorView& stiffness_matrix,
                                  const ParameterView& damping_factors,
                                  StatusView parameter_status,
                                  const MutableTensorView& cartesian_wrench,
                                  MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepDampingDesign, Scalar>(
          static_cast<std::size_t>(cartesian_velocity_error.shape[0]));
  step_damping_design_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      factorization_design,
      as_cuda_matrix<Scalar>(cartesian_velocity_error),
      as_cuda_tensor3<Scalar>(cartesian_mass_matrix),
      as_cuda_tensor3<Scalar>(stiffness_matrix),
      as_cuda_vector_parameter<Scalar>(damping_factors),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(cartesian_wrench),
      static_cast<std::size_t>(cartesian_velocity_error.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_matrix6(const TensorView& input,
                           const ParameterView& matrix,
                           StatusView parameter_status,
                           const MutableTensorView& output,
                           MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepMatrix6, Scalar>(
          static_cast<std::size_t>(input.shape[0]));
  step_matrix6_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(input),
      as_cuda_matrix_parameter<Scalar>(matrix),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(output),
      static_cast<std::size_t>(input.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_virtual_mass(const TensorView& force,
                                const ParameterView& matrix,
                                StatusView parameter_status,
                                const MutableTensorView& output,
                                MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepVirtualMass, Scalar>(
          static_cast<std::size_t>(force.shape[0]));
  step_virtual_mass_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(force),
      as_cuda_matrix_parameter<Scalar>(matrix),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(output),
      static_cast<std::size_t>(force.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_inertial(const std::size_t num_envs,
                             const std::size_t joint_dim,
                             StateView& state,
                             const TensorView& reset_mask,
                             MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianResetInertial, Scalar>(num_envs);
  reset_inertial_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_mass_matrix")->values),
      as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_jacobian")->values),
      as_cuda_bool_vector(find_state(state, "initialized")->values),
      num_envs,
      joint_dim,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_admittance(const std::size_t num_envs,
                               StateView& state,
                               const TensorView& reset_mask,
                               MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianResetAdmittance, Scalar>(num_envs);
  reset_admittance_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      as_cuda_mutable_matrix<Scalar>(find_state(state, "admittance_pose")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "admittance_twist")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "pose_previous_signal")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "twist_previous_signal")->values),
      as_cuda_bool_vector(find_state(state, "pose_integrator_initialized")->values),
      as_cuda_bool_vector(find_state(state, "twist_integrator_initialized")->values),
      num_envs,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_force(const std::size_t num_envs,
                          StateView& state,
                          const TensorView& reset_mask,
                          MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianResetForce, Scalar>(num_envs);
  reset_force_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      as_cuda_mutable_matrix<Scalar>(find_state(state, "differentiator_previous_error")->values),
      as_cuda_bool_vector(find_state(state, "differentiator_initialized")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "integrator_previous_error")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "integrated_error")->values),
      as_cuda_bool_vector(find_state(state, "integrator_initialized")->values),
      num_envs,
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_reset_variable_stiffness(const ParameterSetView& parameters,
                                       StatusView parameter_status,
                                       const std::size_t num_envs,
                                       StateView& state,
                                       const TensorView& reset_mask,
                                       MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianResetVariableStiffness, Scalar>(num_envs);
  reset_variable_stiffness_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          static_cast<const std::uint8_t*>(reset_mask.data),
          reset_mask.strides[0],
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "initial_stiffness_matrix")),
          parameter_status.flags,
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "current_stiffness_matrix")->values),
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_stiffness_matrix")->values),
          as_cuda_bool_vector(find_state(state, "initialized")->values),
          num_envs,
          status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_operational_space(const TensorView& cartesian_pose_error,
                                     const TensorView& cartesian_velocity_error,
                                     const TensorView& desired_acceleration,
                                     const TensorView& jacobian_dot_joint_velocity,
                                     const TensorView& cartesian_mass_matrix,
                                     const TensorView& jacobian,
                                     const TensorView& joint_bias_torque,
                                     const ParameterSetView& parameters,
                                     StatusView parameter_status,
                                     const MutableTensorView& joint_torque,
                                     MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepOperationalSpace, Scalar>(
          static_cast<std::size_t>(cartesian_pose_error.shape[0]));
  step_operational_space_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(cartesian_pose_error),
      as_cuda_matrix<Scalar>(cartesian_velocity_error),
      as_cuda_matrix<Scalar>(desired_acceleration),
      as_cuda_matrix<Scalar>(jacobian_dot_joint_velocity),
      as_cuda_tensor3<Scalar>(cartesian_mass_matrix),
      as_cuda_tensor3<Scalar>(jacobian),
      as_cuda_matrix<Scalar>(joint_bias_torque),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "position_gain")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "velocity_gain")),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(joint_torque),
      static_cast<std::size_t>(cartesian_pose_error.shape[0]),
      static_cast<std::size_t>(joint_bias_torque.shape[1]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_inertial(const TensorView& cartesian_velocity_error,
                            const TensorView& joint_velocity,
                            const TensorView& cartesian_mass_matrix,
                            const TensorView& jacobian,
                            const TensorView& dt,
                            StatusView parameter_status,
                            StateView& state,
                            const MutableTensorView& cartesian_wrench,
                            MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepInertial, Scalar>(
          static_cast<std::size_t>(cartesian_velocity_error.shape[0]));
  step_inertial_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(cartesian_velocity_error),
      as_cuda_matrix<Scalar>(joint_velocity),
      as_cuda_tensor3<Scalar>(cartesian_mass_matrix),
      as_cuda_tensor3<Scalar>(jacobian),
      as_cuda_scalar_batch<Scalar>(dt),
      parameter_status.flags,
      as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_mass_matrix")->values),
      as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_jacobian")->values),
      as_cuda_bool_vector(find_state(state, "initialized")->values),
      as_cuda_mutable_matrix<Scalar>(cartesian_wrench),
      static_cast<std::size_t>(cartesian_velocity_error.shape[0]),
      static_cast<std::size_t>(joint_velocity.shape[1]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_cartesian_impedance(const TensorView& cartesian_pose_error,
                                       const TensorView& cartesian_velocity_error,
                                       const TensorView& desired_acceleration,
                                       const TensorView& joint_velocity,
                                       const TensorView& cartesian_mass_matrix,
                                       const TensorView& jacobian,
                                       const TensorView& dt,
                                       const ParameterSetView& parameters,
                                       StatusView parameter_status,
                                       StateView& state,
                                       const MutableTensorView& cartesian_wrench,
                                       MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepImpedance, Scalar>(
          static_cast<std::size_t>(cartesian_pose_error.shape[0]));
  step_cartesian_impedance_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix<Scalar>(cartesian_pose_error),
          as_cuda_matrix<Scalar>(cartesian_velocity_error),
          as_cuda_matrix<Scalar>(desired_acceleration),
          as_cuda_matrix<Scalar>(joint_velocity),
          as_cuda_tensor3<Scalar>(cartesian_mass_matrix),
          as_cuda_tensor3<Scalar>(jacobian),
          as_cuda_scalar_batch<Scalar>(dt),
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "damping_factors")),
          parameter_status.flags,
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_mass_matrix")->values),
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_jacobian")->values),
          as_cuda_bool_vector(find_state(state, "initialized")->values),
          as_cuda_mutable_matrix<Scalar>(cartesian_wrench),
          static_cast<std::size_t>(cartesian_pose_error.shape[0]),
          static_cast<std::size_t>(joint_velocity.shape[1]),
          status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_admittance(const TensorView& external_wrench,
                              const TensorView& feedforward_wrench,
                              const TensorView& dt,
                              const ParameterSetView& parameters,
                              StatusView parameter_status,
                              StateView& state,
                              const MutableTensorView& commanded_acceleration,
                              MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepAdmittance, Scalar>(
          static_cast<std::size_t>(external_wrench.shape[0]));
  step_admittance_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(external_wrench),
      as_cuda_matrix<Scalar>(feedforward_wrench),
      as_cuda_scalar_batch<Scalar>(dt),
      as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "stiffness_matrix")),
      as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "damping_matrix")),
      as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "virtual_mass_matrix")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "force_filter_threshold")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "acceleration_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "twist_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "pose_limit")),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(find_state(state, "admittance_pose")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "admittance_twist")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "pose_previous_signal")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "twist_previous_signal")->values),
      as_cuda_bool_vector(find_state(state, "pose_integrator_initialized")->values),
      as_cuda_bool_vector(find_state(state, "twist_integrator_initialized")->values),
      as_cuda_mutable_matrix<Scalar>(commanded_acceleration),
      static_cast<std::size_t>(external_wrench.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_variable_stiffness(const TensorView& cartesian_pose_error,
                                      const TensorView& stiffness_matrix,
                                      const TensorView& dt,
                                      const ParameterSetView& parameters,
                                      StatusView parameter_status,
                                      StateView& state,
                                      const MutableTensorView& cartesian_wrench,
                                      MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepVariableStiffness, Scalar>(
          static_cast<std::size_t>(cartesian_pose_error.shape[0]));
  step_variable_stiffness_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix<Scalar>(cartesian_pose_error),
          as_cuda_tensor3<Scalar>(stiffness_matrix),
          as_cuda_scalar_batch<Scalar>(dt),
          as_cuda_matrix_parameter<Scalar>(*find_parameter(parameters, "initial_stiffness_matrix")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness")),
          as_cuda_optional_vector_parameter<Scalar>(
              find_parameter(parameters, "minimum_stiffness")),
          as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "maximum_stiffness_rate")),
          parameter_status.flags,
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "current_stiffness_matrix")->values),
          as_cuda_mutable_tensor3<Scalar>(find_state(state, "previous_stiffness_matrix")->values),
          as_cuda_bool_vector(find_state(state, "initialized")->values),
          as_cuda_mutable_matrix<Scalar>(cartesian_wrench),
          static_cast<std::size_t>(cartesian_pose_error.shape[0]),
          status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_step_force(const TensorView& external_wrench,
                         const TensorView& desired_wrench,
                         const TensorView& feedforward_wrench,
                         const TensorView& pose_error,
                         const TensorView& dt,
                         const bool use_shaping_function,
                         const ParameterSetView& parameters,
                         StatusView parameter_status,
                         StateView& state,
                         const MutableTensorView& commanded_wrench,
                         MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianStepForce, Scalar>(
          static_cast<std::size_t>(external_wrench.shape[0]));
  step_force_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(external_wrench),
      as_cuda_matrix<Scalar>(desired_wrench),
      as_cuda_matrix<Scalar>(feedforward_wrench),
      as_cuda_matrix<Scalar>(pose_error),
      as_cuda_scalar_batch<Scalar>(dt),
      use_shaping_function,
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "proportional_gain")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "differentiator_gain")),
      as_cuda_vector_parameter<Scalar>(*find_parameter(parameters, "integral_gain")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "u_0")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "upper_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "lower_limit")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "maximum_distance")),
      as_cuda_optional_vector_parameter<Scalar>(find_parameter(parameters, "shaping_distance")),
      parameter_status.flags,
      as_cuda_mutable_matrix<Scalar>(find_state(state, "differentiator_previous_error")->values),
      as_cuda_bool_vector(find_state(state, "differentiator_initialized")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "integrator_previous_error")->values),
      as_cuda_mutable_matrix<Scalar>(find_state(state, "integrated_error")->values),
      as_cuda_bool_vector(find_state(state, "integrator_initialized")->values),
      as_cuda_mutable_matrix<Scalar>(commanded_wrench),
      static_cast<std::size_t>(external_wrench.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_combine_sum(const TensorView& left_wrench,
                          StatusView left_status,
                          const TensorView& right_wrench,
                          StatusView right_status,
                          const MutableTensorView& output,
                          MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianCombineSum, Scalar>(
          static_cast<std::size_t>(left_wrench.shape[0]));
  combine_sum_kernel<Scalar><<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      as_cuda_matrix<Scalar>(left_wrench),
      left_status.flags,
      as_cuda_matrix<Scalar>(right_wrench),
      right_status.flags,
      as_cuda_mutable_matrix<Scalar>(output),
      static_cast<std::size_t>(left_wrench.shape[0]),
      status.flags);
  return cuda_launch_result();
}

template <typename Scalar>
Result launch_combine_variable_impedance(const TensorView& stiffness_wrench,
                                         StatusView stiffness_status,
                                         const TensorView& damping_wrench,
                                         StatusView damping_status,
                                         const TensorView& inertial_wrench,
                                         StatusView inertial_status,
                                         const TensorView& desired_acceleration,
                                         const TensorView& cartesian_mass_matrix,
                                         const MutableTensorView& output,
                                         MutableStatusView status) {
  const CudaLaunchConfig config =
      launch_config<CudaTensorLaunchFamily::kCartesianCombineVariableImpedance, Scalar>(
          static_cast<std::size_t>(stiffness_wrench.shape[0]));
  combine_variable_impedance_kernel<Scalar>
      <<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
          as_cuda_matrix<Scalar>(stiffness_wrench),
          stiffness_status.flags,
          as_cuda_matrix<Scalar>(damping_wrench),
          damping_status.flags,
          as_cuda_matrix<Scalar>(inertial_wrench),
          inertial_status.flags,
          as_cuda_matrix<Scalar>(desired_acceleration),
          as_cuda_tensor3<Scalar>(cartesian_mass_matrix),
          as_cuda_mutable_matrix<Scalar>(output),
          static_cast<std::size_t>(stiffness_wrench.shape[0]),
          status.flags);
  return cuda_launch_result();
}

}  // namespace

Result apply_cartesian_matrix6_parameters_cuda(const ParameterSetView& parameters,
                                               const char* parameter_name,
                                               const std::size_t num_envs,
                                               MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  const DType dtype = matrix == nullptr ? DType::kUnknown : matrix->values.dtype;
  result = validate_cuda_matrix_parameter(matrix, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_matrix6<float>(*matrix, num_envs, parameter_status)
             : launch_apply_matrix6<double>(*matrix, num_envs, parameter_status);
}

Result reset_cartesian_stateless_cuda(const std::size_t num_envs,
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
  const CudaLaunchConfig config =
      launch_untyped_config<CudaTensorLaunchFamily::kCartesianResetStateless>(num_envs);
  reset_stateless_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      static_cast<const std::uint8_t*>(reset_mask.data),
      reset_mask.strides[0],
      num_envs,
      status.flags);
  return cuda_launch_result();
}

Result apply_cartesian_damping_design_parameters_cuda(const ParameterSetView& parameters,
                                                      const std::size_t num_envs,
                                                      MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, "damping_factors");
  const DType dtype = damping_factors == nullptr ? DType::kUnknown : damping_factors->values.dtype;
  result = validate_cuda_vector_parameter(
      damping_factors, "damping_factors parameter is missing", dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_damping_design<float>(*damping_factors, num_envs, parameter_status)
             : launch_apply_damping_design<double>(*damping_factors, num_envs, parameter_status);
}

Result step_cartesian_damping_design_cuda(const bool factorization_design,
                                          const TensorView& cartesian_velocity_error,
                                          const TensorView& cartesian_mass_matrix,
                                          const TensorView& stiffness_matrix,
                                          const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          const MutableTensorView& cartesian_wrench,
                                          MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(cartesian_velocity_error)) {
    return Result{ResultCode::kInvalidView,
                  "cartesian_velocity_error must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_velocity_error.shape[0]);
  Result result = validate_cuda_tensor3_matrix6(cartesian_mass_matrix,
                                                "cartesian_mass_matrix tensor data is null",
                                                cartesian_velocity_error.dtype,
                                                num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix6(stiffness_matrix,
                                         "stiffness_matrix tensor data is null",
                                         cartesian_velocity_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
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
  result = validate_cuda_vector_parameter(damping_factors,
                                          "damping_factors parameter is missing",
                                          cartesian_velocity_error.dtype,
                                          num_envs,
                                          kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return cartesian_velocity_error.dtype == DType::kFloat32
             ? launch_step_damping_design<float>(factorization_design,
                                                 cartesian_velocity_error,
                                                 cartesian_mass_matrix,
                                                 stiffness_matrix,
                                                 *damping_factors,
                                                 parameter_status,
                                                 cartesian_wrench,
                                                 status)
             : launch_step_damping_design<double>(factorization_design,
                                                  cartesian_velocity_error,
                                                  cartesian_mass_matrix,
                                                  stiffness_matrix,
                                                  *damping_factors,
                                                  parameter_status,
                                                  cartesian_wrench,
                                                  status);
}

Result apply_shaping_function_parameters_cuda(const ParameterSetView& parameters,
                                              const std::size_t num_envs,
                                              const std::size_t signal_dim,
                                              MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* maximum = find_parameter(parameters, "maximum_distance");
  const ParameterView* shaping = find_parameter(parameters, "shaping_distance");
  const DType dtype = maximum == nullptr ? DType::kUnknown : maximum->values.dtype;
  result = validate_cuda_vector_parameter(
      maximum, "maximum_distance parameter is missing", dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(
      shaping, "shaping_distance parameter is missing", dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32 ? launch_apply_shaping<float>(
                                        *maximum, *shaping, num_envs, signal_dim, parameter_status)
                                  : launch_apply_shaping<double>(
                                        *maximum, *shaping, num_envs, signal_dim, parameter_status);
}

Result step_shaping_function_cuda(const TensorView& error,
                                  const ParameterSetView& parameters,
                                  StatusView parameter_status,
                                  const MutableTensorView& shaping_factor,
                                  MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result =
      validate_cuda_matrix_signal(error, "error tensor data is null", &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_matrix_output(shaping_factor, error.dtype, num_envs, signal_dim);
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
  const ParameterView* maximum = find_parameter(parameters, "maximum_distance");
  const ParameterView* shaping = find_parameter(parameters, "shaping_distance");
  result = validate_cuda_vector_parameter(
      maximum, "maximum_distance parameter is missing", error.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(
      shaping, "shaping_distance parameter is missing", error.dtype, num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  return error.dtype == DType::kFloat32
             ? launch_step_shaping<float>(
                   error, *maximum, *shaping, parameter_status, shaping_factor, status)
             : launch_step_shaping<double>(
                   error, *maximum, *shaping, parameter_status, shaping_factor, status);
}

Result apply_command_observer_block_parameters_cuda(const std::size_t num_envs,
                                                    MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const CudaLaunchConfig config =
      launch_untyped_config<CudaTensorLaunchFamily::kCartesianClearCommandObserver>(num_envs);
  clear_status_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      num_envs, parameter_status.flags);
  return cuda_launch_result();
}

Result step_command_observer_block_cuda(const TensorView& command,
                                        StatusView parameter_status,
                                        const MutableTensorView& command_is_valid,
                                        MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result =
      validate_cuda_matrix_signal(command, "command tensor data is null", &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_bool_output(command_is_valid, num_envs);
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
  return command.dtype == DType::kFloat32
             ? launch_step_command_observer<float>(
                   command, parameter_status, command_is_valid, status)
             : launch_step_command_observer<double>(
                   command, parameter_status, command_is_valid, status);
}

Result apply_flow_funnel_parameters_cuda(const ParameterSetView& parameters,
                                         const std::size_t num_envs,
                                         MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* radius = find_parameter(parameters, "funnel_radius");
  const DType dtype = radius == nullptr ? DType::kUnknown : radius->values.dtype;
  result = validate_cuda_flow_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_flow<float>(parameters, num_envs, parameter_status)
             : launch_apply_flow<double>(parameters, num_envs, parameter_status);
}

Result step_flow_funnel_cuda(const TensorView& cartesian_pose,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& cartesian_flow,
                             MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cuda_pose_with_num_envs(
      cartesian_pose, "cartesian_pose tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_flow, cartesian_pose.dtype, num_envs);
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
  result = validate_cuda_flow_parameters(parameters, cartesian_pose.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_pose.dtype == DType::kFloat32
             ? launch_step_flow<float>(
                   cartesian_pose, parameters, parameter_status, cartesian_flow, status)
             : launch_step_flow<double>(
                   cartesian_pose, parameters, parameter_status, cartesian_flow, status);
}

Result apply_inverse_apf_parameters_cuda(const ParameterSetView& parameters,
                                         const std::size_t num_envs,
                                         MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* alpha = find_parameter(parameters, "alpha");
  const DType dtype = alpha == nullptr ? DType::kUnknown : alpha->values.dtype;
  result = validate_cuda_inverse_apf_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_inverse_apf<float>(parameters, num_envs, parameter_status)
             : launch_apply_inverse_apf<double>(parameters, num_envs, parameter_status);
}

Result step_inverse_apf_cuda(const TensorView& current_pose,
                             const TensorView& goal_pose,
                             const TensorView& cartesian_twist,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& cartesian_wrench,
                             MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = validate_cuda_matrix_signal(
      cartesian_twist, "cartesian_twist tensor data is null", &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  if (signal_dim != kCartesianDim) {
    return Result{ResultCode::kInvalidView, "cartesian_twist must have shape [num_envs, 6]"};
  }
  result = validate_cuda_pose(
      current_pose, "current_pose tensor data is null", cartesian_twist.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_pose(
      goal_pose, "goal_pose tensor data is null", cartesian_twist.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, cartesian_twist.dtype, num_envs);
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
  result = validate_cuda_inverse_apf_parameters(parameters, cartesian_twist.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_twist.dtype == DType::kFloat32
             ? launch_step_inverse_apf<float>(current_pose,
                                              goal_pose,
                                              cartesian_twist,
                                              parameters,
                                              parameter_status,
                                              cartesian_wrench,
                                              status)
             : launch_step_inverse_apf<double>(current_pose,
                                               goal_pose,
                                               cartesian_twist,
                                               parameters,
                                               parameter_status,
                                               cartesian_wrench,
                                               status);
}

Result step_cartesian_matrix6_cuda(const TensorView& input,
                                   const ParameterSetView& parameters,
                                   const char* parameter_name,
                                   StatusView parameter_status,
                                   const MutableTensorView& output,
                                   MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(input)) {
    return Result{ResultCode::kInvalidView, "input tensor must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(input.shape[0]);
  Result result = validate_cuda_output(output, input.dtype, num_envs);
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
  result = validate_cuda_matrix_parameter(matrix, input.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return input.dtype == DType::kFloat32
             ? launch_step_matrix6<float>(input, *matrix, parameter_status, output, status)
             : launch_step_matrix6<double>(input, *matrix, parameter_status, output, status);
}

Result step_cartesian_virtual_mass_cuda(const TensorView& force,
                                        const ParameterSetView& parameters,
                                        StatusView parameter_status,
                                        const MutableTensorView& cartesian_acceleration,
                                        MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(force)) {
    return Result{ResultCode::kInvalidView, "force tensor must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(force.shape[0]);
  Result result = validate_cuda_output(cartesian_acceleration, force.dtype, num_envs);
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
  const ParameterView* matrix = find_parameter(parameters, "virtual_mass_matrix");
  result = validate_cuda_matrix_parameter(matrix, force.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return force.dtype == DType::kFloat32
             ? launch_step_virtual_mass<float>(
                   force, *matrix, parameter_status, cartesian_acceleration, status)
             : launch_step_virtual_mass<double>(
                   force, *matrix, parameter_status, cartesian_acceleration, status);
}

Result apply_inertial_coupling_parameters_cuda(const std::size_t num_envs,
                                               MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const CudaLaunchConfig config =
      launch_untyped_config<CudaTensorLaunchFamily::kCartesianClearInertial>(num_envs);
  clear_status_kernel<<<config.blocks, config.threads, 0, active_cuda_stream()>>>(
      num_envs, parameter_status.flags);
  return cuda_launch_result();
}

Result reset_inertial_coupling_cuda(const std::size_t num_envs,
                                    const std::size_t joint_dim,
                                    StateView& state,
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
  StateEntryView* previous_mass = find_state(state, "previous_mass_matrix");
  StateEntryView* previous_jacobian = find_state(state, "previous_jacobian");
  StateEntryView* initialized = find_state(state, "initialized");
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "inertial state entries are incomplete"};
  }
  const DType dtype = previous_mass->values.dtype;
  result = validate_cuda_tensor3_matrix6(
      previous_mass->values, "previous_mass_matrix state is missing", dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(previous_jacobian->values,
                                        "previous_jacobian state is missing",
                                        dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(initialized->values, "initialized state is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_inertial<float>(num_envs, joint_dim, state, reset_mask, status)
             : launch_reset_inertial<double>(num_envs, joint_dim, state, reset_mask, status);
}

Result step_inertial_coupling_cuda(const TensorView& cartesian_velocity_error,
                                   const TensorView& joint_velocity,
                                   const TensorView& cartesian_mass_matrix,
                                   const TensorView& jacobian,
                                   const TensorView& dt,
                                   StatusView parameter_status,
                                   StateView& state,
                                   const MutableTensorView& cartesian_wrench,
                                   MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(cartesian_velocity_error)) {
    return Result{ResultCode::kInvalidView,
                  "cartesian_velocity_error must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_velocity_error.shape[0]);
  if (joint_velocity.data == nullptr || joint_velocity.device != Device::kCuda ||
      joint_velocity.dtype != cartesian_velocity_error.dtype || joint_velocity.rank != 2 ||
      joint_velocity.layout != Layout::kRowMajor ||
      joint_velocity.shape[0] != static_cast<std::int64_t>(num_envs) ||
      joint_velocity.shape[1] <= 0 || joint_velocity.strides[1] != 1 ||
      joint_velocity.strides[0] < joint_velocity.shape[1]) {
    return Result{ResultCode::kInvalidView, "joint_velocity must be CUDA [num_envs, joint_dim]"};
  }
  const std::size_t joint_dim = static_cast<std::size_t>(joint_velocity.shape[1]);
  Result result = validate_cuda_tensor3_matrix6(cartesian_mass_matrix,
                                                "cartesian_mass_matrix data is null",
                                                cartesian_velocity_error.dtype,
                                                num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(jacobian,
                                        "jacobian tensor data is null",
                                        cartesian_velocity_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch(
      dt, "dt tensor data is null", cartesian_velocity_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
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
  StateEntryView* previous_mass = find_state(state, "previous_mass_matrix");
  StateEntryView* previous_jacobian = find_state(state, "previous_jacobian");
  StateEntryView* initialized = find_state(state, "initialized");
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "inertial state entries are incomplete"};
  }
  result = validate_cuda_tensor3_matrix6(previous_mass->values,
                                         "previous_mass_matrix state is missing",
                                         cartesian_velocity_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(previous_jacobian->values,
                                        "previous_jacobian state is missing",
                                        cartesian_velocity_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(initialized->values, "initialized state is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_velocity_error.dtype == DType::kFloat32
             ? launch_step_inertial<float>(cartesian_velocity_error,
                                           joint_velocity,
                                           cartesian_mass_matrix,
                                           jacobian,
                                           dt,
                                           parameter_status,
                                           state,
                                           cartesian_wrench,
                                           status)
             : launch_step_inertial<double>(cartesian_velocity_error,
                                            joint_velocity,
                                            cartesian_mass_matrix,
                                            jacobian,
                                            dt,
                                            parameter_status,
                                            state,
                                            cartesian_wrench,
                                            status);
}

Result apply_cartesian_impedance_controller_parameters_cuda(const ParameterSetView& parameters,
                                                            const std::size_t num_envs,
                                                            MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* stiffness = find_parameter(parameters, "stiffness_matrix");
  const DType dtype = stiffness == nullptr ? DType::kUnknown : stiffness->values.dtype;
  result = validate_cuda_impedance_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_impedance<float>(parameters, num_envs, parameter_status)
             : launch_apply_impedance<double>(parameters, num_envs, parameter_status);
}

Result step_cartesian_impedance_controller_cuda(const TensorView& cartesian_pose_error,
                                                const TensorView& cartesian_velocity_error,
                                                const TensorView& desired_acceleration,
                                                const TensorView& joint_velocity,
                                                const TensorView& cartesian_mass_matrix,
                                                const TensorView& jacobian,
                                                const TensorView& dt,
                                                const ParameterSetView& parameters,
                                                StatusView parameter_status,
                                                StateView& state,
                                                const MutableTensorView& cartesian_wrench,
                                                MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(cartesian_pose_error)) {
    return Result{ResultCode::kInvalidView,
                  "cartesian_pose_error must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
  Result result = validate_cuda_matrix_output(
      MutableTensorView{const_cast<void*>(cartesian_velocity_error.data),
                        cartesian_velocity_error.dtype,
                        cartesian_velocity_error.device,
                        cartesian_velocity_error.layout,
                        cartesian_velocity_error.rank,
                        cartesian_velocity_error.shape,
                        cartesian_velocity_error.strides},
      cartesian_pose_error.dtype,
      num_envs,
      kCartesianDim);
  if (!result.ok()) {
    return Result{ResultCode::kInvalidView,
                  "cartesian_velocity_error must be CUDA row-major [num_envs, 6]"};
  }
  result =
      validate_cuda_matrix_output(MutableTensorView{const_cast<void*>(desired_acceleration.data),
                                                    desired_acceleration.dtype,
                                                    desired_acceleration.device,
                                                    desired_acceleration.layout,
                                                    desired_acceleration.rank,
                                                    desired_acceleration.shape,
                                                    desired_acceleration.strides},
                                  cartesian_pose_error.dtype,
                                  num_envs,
                                  kCartesianDim);
  if (!result.ok()) {
    return Result{ResultCode::kInvalidView,
                  "desired_acceleration must be CUDA row-major [num_envs, 6]"};
  }
  if (joint_velocity.data == nullptr || joint_velocity.device != Device::kCuda ||
      joint_velocity.dtype != cartesian_pose_error.dtype || joint_velocity.rank != 2 ||
      joint_velocity.layout != Layout::kRowMajor ||
      joint_velocity.shape[0] != static_cast<std::int64_t>(num_envs) ||
      joint_velocity.shape[1] <= 0 || joint_velocity.strides[1] != 1 ||
      joint_velocity.strides[0] < joint_velocity.shape[1]) {
    return Result{ResultCode::kInvalidView, "joint_velocity must be CUDA [num_envs, joint_dim]"};
  }
  const std::size_t joint_dim = static_cast<std::size_t>(joint_velocity.shape[1]);
  result = validate_cuda_tensor3_matrix6(cartesian_mass_matrix,
                                         "cartesian_mass_matrix data is null",
                                         cartesian_pose_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(jacobian,
                                        "jacobian tensor data is null",
                                        cartesian_pose_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
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
  result = validate_cuda_impedance_parameters(parameters, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* previous_mass = find_state(state, "previous_mass_matrix");
  StateEntryView* previous_jacobian = find_state(state, "previous_jacobian");
  StateEntryView* initialized = find_state(state, "initialized");
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "impedance state entries are incomplete"};
  }
  result = validate_cuda_tensor3_matrix6(previous_mass->values,
                                         "previous_mass_matrix state is missing",
                                         cartesian_pose_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix(previous_jacobian->values,
                                        "previous_jacobian state is missing",
                                        cartesian_pose_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(initialized->values, "initialized state is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_pose_error.dtype == DType::kFloat32
             ? launch_step_cartesian_impedance<float>(cartesian_pose_error,
                                                      cartesian_velocity_error,
                                                      desired_acceleration,
                                                      joint_velocity,
                                                      cartesian_mass_matrix,
                                                      jacobian,
                                                      dt,
                                                      parameters,
                                                      parameter_status,
                                                      state,
                                                      cartesian_wrench,
                                                      status)
             : launch_step_cartesian_impedance<double>(cartesian_pose_error,
                                                       cartesian_velocity_error,
                                                       desired_acceleration,
                                                       joint_velocity,
                                                       cartesian_mass_matrix,
                                                       jacobian,
                                                       dt,
                                                       parameters,
                                                       parameter_status,
                                                       state,
                                                       cartesian_wrench,
                                                       status);
}

Result apply_operational_space_controller_parameters_cuda(const ParameterSetView& parameters,
                                                          const std::size_t num_envs,
                                                          MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* position_gain = find_parameter(parameters, "position_gain");
  const DType dtype = position_gain == nullptr ? DType::kUnknown : position_gain->values.dtype;
  result = validate_cuda_operational_space_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_operational_space<float>(parameters, num_envs, parameter_status)
             : launch_apply_operational_space<double>(parameters, num_envs, parameter_status);
}

Result step_operational_space_controller_cuda(const TensorView& cartesian_pose_error,
                                              const TensorView& cartesian_velocity_error,
                                              const TensorView& desired_acceleration,
                                              const TensorView& jacobian_dot_joint_velocity,
                                              const TensorView& cartesian_mass_matrix,
                                              const TensorView& jacobian,
                                              const TensorView& joint_bias_torque,
                                              const ParameterSetView& parameters,
                                              StatusView parameter_status,
                                              const MutableTensorView& joint_torque,
                                              MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(cartesian_pose_error) ||
      cartesian_velocity_error.device != Device::kCuda ||
      desired_acceleration.device != Device::kCuda ||
      jacobian_dot_joint_velocity.device != Device::kCuda ||
      cartesian_velocity_error.dtype != cartesian_pose_error.dtype ||
      desired_acceleration.dtype != cartesian_pose_error.dtype ||
      jacobian_dot_joint_velocity.dtype != cartesian_pose_error.dtype ||
      cartesian_velocity_error.rank != 2 || desired_acceleration.rank != 2 ||
      jacobian_dot_joint_velocity.rank != 2 ||
      cartesian_velocity_error.shape[0] != cartesian_pose_error.shape[0] ||
      desired_acceleration.shape[0] != cartesian_pose_error.shape[0] ||
      jacobian_dot_joint_velocity.shape[0] != cartesian_pose_error.shape[0] ||
      cartesian_velocity_error.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      desired_acceleration.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      jacobian_dot_joint_velocity.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      cartesian_velocity_error.strides[1] != 1 || desired_acceleration.strides[1] != 1 ||
      jacobian_dot_joint_velocity.strides[1] != 1) {
    return Result{ResultCode::kInvalidView,
                  "operational-space signals must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
  Result result = validate_cuda_tensor3_matrix6(cartesian_mass_matrix,
                                                "cartesian_mass_matrix tensor data is null",
                                                cartesian_pose_error.dtype,
                                                num_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  result = validate_cuda_matrix_signal(
      joint_bias_torque, "joint_bias_torque tensor data is null", &joint_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_envs != num_envs || joint_bias_torque.dtype != cartesian_pose_error.dtype) {
    return Result{ResultCode::kInvalidView,
                  "joint_bias_torque must match Cartesian dtype and num_envs"};
  }
  result = validate_cuda_tensor3_matrix(jacobian,
                                        "jacobian tensor data is null",
                                        cartesian_pose_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        joint_dim);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_cuda_matrix_output(joint_torque, cartesian_pose_error.dtype, num_envs, joint_dim);
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
  result =
      validate_cuda_operational_space_parameters(parameters, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_pose_error.dtype == DType::kFloat32
             ? launch_step_operational_space<float>(cartesian_pose_error,
                                                    cartesian_velocity_error,
                                                    desired_acceleration,
                                                    jacobian_dot_joint_velocity,
                                                    cartesian_mass_matrix,
                                                    jacobian,
                                                    joint_bias_torque,
                                                    parameters,
                                                    parameter_status,
                                                    joint_torque,
                                                    status)
             : launch_step_operational_space<double>(cartesian_pose_error,
                                                     cartesian_velocity_error,
                                                     desired_acceleration,
                                                     jacobian_dot_joint_velocity,
                                                     cartesian_mass_matrix,
                                                     jacobian,
                                                     joint_bias_torque,
                                                     parameters,
                                                     parameter_status,
                                                     joint_torque,
                                                     status);
}

Result apply_cartesian_admittance_controller_parameters_cuda(const ParameterSetView& parameters,
                                                             const std::size_t num_envs,
                                                             MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* stiffness = find_parameter(parameters, "stiffness_matrix");
  const DType dtype = stiffness == nullptr ? DType::kUnknown : stiffness->values.dtype;
  result = validate_cuda_admittance_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_admittance<float>(parameters, num_envs, parameter_status)
             : launch_apply_admittance<double>(parameters, num_envs, parameter_status);
}

Result reset_cartesian_admittance_controller_cuda(const std::size_t num_envs,
                                                  StateView& state,
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
  StateEntryView* pose = find_state(state, "admittance_pose");
  StateEntryView* twist = find_state(state, "admittance_twist");
  StateEntryView* pose_previous = find_state(state, "pose_previous_signal");
  StateEntryView* twist_previous = find_state(state, "twist_previous_signal");
  StateEntryView* pose_initialized = find_state(state, "pose_integrator_initialized");
  StateEntryView* twist_initialized = find_state(state, "twist_integrator_initialized");
  if (pose == nullptr || twist == nullptr || pose_previous == nullptr ||
      twist_previous == nullptr || pose_initialized == nullptr || twist_initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "admittance state entries are incomplete"};
  }
  const DType dtype = pose->values.dtype;
  MutableTensorView* matrices[] = {
      &pose->values, &twist->values, &pose_previous->values, &twist_previous->values};
  for (MutableTensorView* matrix : matrices) {
    result =
        validate_cuda_state_matrix(*matrix, "admittance state matrix is invalid", dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_cuda_state_bool(
      pose_initialized->values, "pose_integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(
      twist_initialized->values, "twist_integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_admittance<float>(num_envs, state, reset_mask, status)
             : launch_reset_admittance<double>(num_envs, state, reset_mask, status);
}

Result step_cartesian_admittance_controller_cuda(const TensorView& external_wrench,
                                                 const TensorView& feedforward_wrench,
                                                 const TensorView& dt,
                                                 const ParameterSetView& parameters,
                                                 StatusView parameter_status,
                                                 StateView& state,
                                                 const MutableTensorView& commanded_acceleration,
                                                 MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(external_wrench) || feedforward_wrench.device != Device::kCuda ||
      feedforward_wrench.dtype != external_wrench.dtype || feedforward_wrench.rank != 2 ||
      feedforward_wrench.shape[0] != external_wrench.shape[0] ||
      feedforward_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      feedforward_wrench.strides[1] != 1) {
    return Result{ResultCode::kInvalidView,
                  "admittance wrench signals must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(external_wrench.shape[0]);
  Result result =
      validate_cuda_scalar_batch(dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(commanded_acceleration, external_wrench.dtype, num_envs);
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
  result = validate_cuda_admittance_parameters(parameters, external_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* pose = find_state(state, "admittance_pose");
  StateEntryView* twist = find_state(state, "admittance_twist");
  StateEntryView* pose_previous = find_state(state, "pose_previous_signal");
  StateEntryView* twist_previous = find_state(state, "twist_previous_signal");
  StateEntryView* pose_initialized = find_state(state, "pose_integrator_initialized");
  StateEntryView* twist_initialized = find_state(state, "twist_integrator_initialized");
  if (pose == nullptr || twist == nullptr || pose_previous == nullptr ||
      twist_previous == nullptr || pose_initialized == nullptr || twist_initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "admittance state entries are incomplete"};
  }
  MutableTensorView* matrices[] = {
      &pose->values, &twist->values, &pose_previous->values, &twist_previous->values};
  for (MutableTensorView* matrix : matrices) {
    result = validate_cuda_state_matrix(
        *matrix, "admittance state matrix is invalid", external_wrench.dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_cuda_state_bool(
      pose_initialized->values, "pose_integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(
      twist_initialized->values, "twist_integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return external_wrench.dtype == DType::kFloat32
             ? launch_step_admittance<float>(external_wrench,
                                             feedforward_wrench,
                                             dt,
                                             parameters,
                                             parameter_status,
                                             state,
                                             commanded_acceleration,
                                             status)
             : launch_step_admittance<double>(external_wrench,
                                              feedforward_wrench,
                                              dt,
                                              parameters,
                                              parameter_status,
                                              state,
                                              commanded_acceleration,
                                              status);
}

Result apply_variable_cartesian_stiffness_parameters_cuda(const ParameterSetView& parameters,
                                                          const std::size_t num_envs,
                                                          MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* initial = find_parameter(parameters, "initial_stiffness_matrix");
  const DType dtype = initial == nullptr ? DType::kUnknown : initial->values.dtype;
  result = validate_cuda_variable_stiffness_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_variable_stiffness<float>(parameters, num_envs, parameter_status)
             : launch_apply_variable_stiffness<double>(parameters, num_envs, parameter_status);
}

Result reset_variable_cartesian_stiffness_cuda(const ParameterSetView& parameters,
                                               StatusView parameter_status,
                                               StateView& state,
                                               const TensorView& reset_mask,
                                               MutableStatusView status) {
  StateEntryView* current = find_state(state, "current_stiffness_matrix");
  StateEntryView* previous = find_state(state, "previous_stiffness_matrix");
  StateEntryView* initialized = find_state(state, "initialized");
  if (current == nullptr || previous == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "variable stiffness state entries are incomplete"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(current->values.shape[0]);
  const DType dtype = current->values.dtype;
  Result result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix6(
      current->values, "current_stiffness_matrix state is missing", dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix6(
      previous->values, "previous_stiffness_matrix state is missing", dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(initialized->values, "initialized state is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_variable_stiffness_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_reset_variable_stiffness<float>(
                   parameters, parameter_status, num_envs, state, reset_mask, status)
             : launch_reset_variable_stiffness<double>(
                   parameters, parameter_status, num_envs, state, reset_mask, status);
}

Result step_variable_cartesian_stiffness_cuda(const TensorView& cartesian_pose_error,
                                              const TensorView& stiffness_matrix,
                                              const TensorView& dt,
                                              const ParameterSetView& parameters,
                                              StatusView parameter_status,
                                              StateView& state,
                                              const MutableTensorView& cartesian_wrench,
                                              MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(cartesian_pose_error)) {
    return Result{ResultCode::kInvalidView,
                  "cartesian_pose_error must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
  Result result = validate_cuda_tensor3_matrix6(stiffness_matrix,
                                                "stiffness_matrix tensor data is null",
                                                cartesian_pose_error.dtype,
                                                num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
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
  result =
      validate_cuda_variable_stiffness_parameters(parameters, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* current = find_state(state, "current_stiffness_matrix");
  StateEntryView* previous = find_state(state, "previous_stiffness_matrix");
  StateEntryView* initialized = find_state(state, "initialized");
  if (current == nullptr || previous == nullptr || initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "variable stiffness state entries are incomplete"};
  }
  result = validate_cuda_tensor3_matrix6(current->values,
                                         "current_stiffness_matrix state is missing",
                                         cartesian_pose_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_tensor3_matrix6(previous->values,
                                         "previous_stiffness_matrix state is missing",
                                         cartesian_pose_error.dtype,
                                         num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(initialized->values, "initialized state is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return cartesian_pose_error.dtype == DType::kFloat32
             ? launch_step_variable_stiffness<float>(cartesian_pose_error,
                                                     stiffness_matrix,
                                                     dt,
                                                     parameters,
                                                     parameter_status,
                                                     state,
                                                     cartesian_wrench,
                                                     status)
             : launch_step_variable_stiffness<double>(cartesian_pose_error,
                                                      stiffness_matrix,
                                                      dt,
                                                      parameters,
                                                      parameter_status,
                                                      state,
                                                      cartesian_wrench,
                                                      status);
}

Result apply_cartesian_force_controller_parameters_cuda(const ParameterSetView& parameters,
                                                        const std::size_t num_envs,
                                                        const bool use_shaping_function,
                                                        MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* proportional = find_parameter(parameters, "proportional_gain");
  const DType dtype = proportional == nullptr ? DType::kUnknown : proportional->values.dtype;
  result = validate_cuda_force_parameters(parameters, dtype, num_envs, use_shaping_function);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_force<float>(
                   parameters, num_envs, use_shaping_function, parameter_status)
             : launch_apply_force<double>(
                   parameters, num_envs, use_shaping_function, parameter_status);
}

Result apply_cartesian_unified_force_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    const bool use_shaping_function,
    MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* stiffness = find_parameter(parameters, "stiffness_matrix");
  const DType dtype = stiffness == nullptr ? DType::kUnknown : stiffness->values.dtype;
  result = validate_cuda_impedance_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_force_parameters(parameters, dtype, num_envs, use_shaping_function);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_unified_force_impedance<float>(
                   parameters, num_envs, use_shaping_function, parameter_status)
             : launch_apply_unified_force_impedance<double>(
                   parameters, num_envs, use_shaping_function, parameter_status);
}

Result apply_variable_cartesian_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* initial = find_parameter(parameters, "initial_stiffness_matrix");
  const DType dtype = initial == nullptr ? DType::kUnknown : initial->values.dtype;
  result = validate_cuda_variable_stiffness_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, "damping_factors"),
                                          "damping_factors parameter is missing",
                                          dtype,
                                          num_envs,
                                          kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_variable_impedance<float>(parameters, num_envs, parameter_status)
             : launch_apply_variable_impedance<double>(parameters, num_envs, parameter_status);
}

Result apply_variable_cartesian_unified_force_impedance_controller_parameters_cuda(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    const bool use_shaping_function,
    MutableStatusView parameter_status) {
  Result result = validate_cuda_status(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* initial = find_parameter(parameters, "initial_stiffness_matrix");
  const DType dtype = initial == nullptr ? DType::kUnknown : initial->values.dtype;
  result = validate_cuda_variable_stiffness_parameters(parameters, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter(find_parameter(parameters, "damping_factors"),
                                          "damping_factors parameter is missing",
                                          dtype,
                                          num_envs,
                                          kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_force_parameters(parameters, dtype, num_envs, use_shaping_function);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32
             ? launch_apply_variable_unified<float>(
                   parameters, num_envs, use_shaping_function, parameter_status)
             : launch_apply_variable_unified<double>(
                   parameters, num_envs, use_shaping_function, parameter_status);
}

Result reset_cartesian_force_controller_cuda(const std::size_t num_envs,
                                             StateView& state,
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
  StateEntryView* differentiator_previous = find_state(state, "differentiator_previous_error");
  StateEntryView* differentiator_initialized = find_state(state, "differentiator_initialized");
  StateEntryView* integrator_previous = find_state(state, "integrator_previous_error");
  StateEntryView* integrated_error = find_state(state, "integrated_error");
  StateEntryView* integrator_initialized = find_state(state, "integrator_initialized");
  if (differentiator_previous == nullptr || differentiator_initialized == nullptr ||
      integrator_previous == nullptr || integrated_error == nullptr ||
      integrator_initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "force controller state entries are incomplete"};
  }
  const DType dtype = integrated_error->values.dtype;
  MutableTensorView* matrices[] = {
      &differentiator_previous->values, &integrator_previous->values, &integrated_error->values};
  for (MutableTensorView* matrix : matrices) {
    result = validate_cuda_state_matrix(
        *matrix, "force controller state matrix is invalid", dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_cuda_state_bool(
      differentiator_initialized->values, "differentiator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(
      integrator_initialized->values, "integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return dtype == DType::kFloat32 ? launch_reset_force<float>(num_envs, state, reset_mask, status)
                                  : launch_reset_force<double>(num_envs, state, reset_mask, status);
}

Result step_cartesian_force_controller_cuda(const TensorView& external_wrench,
                                            const TensorView& desired_wrench,
                                            const TensorView& feedforward_wrench,
                                            const TensorView& pose_error,
                                            const TensorView& dt,
                                            const bool use_shaping_function,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            StateView& state,
                                            const MutableTensorView& commanded_wrench,
                                            MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(external_wrench) || desired_wrench.device != Device::kCuda ||
      feedforward_wrench.device != Device::kCuda || pose_error.device != Device::kCuda ||
      desired_wrench.dtype != external_wrench.dtype ||
      feedforward_wrench.dtype != external_wrench.dtype ||
      pose_error.dtype != external_wrench.dtype || desired_wrench.rank != 2 ||
      feedforward_wrench.rank != 2 || pose_error.rank != 2 ||
      desired_wrench.shape[0] != external_wrench.shape[0] ||
      feedforward_wrench.shape[0] != external_wrench.shape[0] ||
      pose_error.shape[0] != external_wrench.shape[0] ||
      desired_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      feedforward_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      pose_error.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      desired_wrench.strides[1] != 1 || feedforward_wrench.strides[1] != 1 ||
      pose_error.strides[1] != 1) {
    return Result{ResultCode::kInvalidView,
                  "force controller signals must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(external_wrench.shape[0]);
  Result result =
      validate_cuda_scalar_batch(dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(commanded_wrench, external_wrench.dtype, num_envs);
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
  result = validate_cuda_force_parameters(
      parameters, external_wrench.dtype, num_envs, use_shaping_function);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* differentiator_previous = find_state(state, "differentiator_previous_error");
  StateEntryView* differentiator_initialized = find_state(state, "differentiator_initialized");
  StateEntryView* integrator_previous = find_state(state, "integrator_previous_error");
  StateEntryView* integrated_error = find_state(state, "integrated_error");
  StateEntryView* integrator_initialized = find_state(state, "integrator_initialized");
  if (differentiator_previous == nullptr || differentiator_initialized == nullptr ||
      integrator_previous == nullptr || integrated_error == nullptr ||
      integrator_initialized == nullptr) {
    return Result{ResultCode::kInvalidView, "force controller state entries are incomplete"};
  }
  MutableTensorView* matrices[] = {
      &differentiator_previous->values, &integrator_previous->values, &integrated_error->values};
  for (MutableTensorView* matrix : matrices) {
    result = validate_cuda_state_matrix(
        *matrix, "force controller state matrix is invalid", external_wrench.dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_cuda_state_bool(
      differentiator_initialized->values, "differentiator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_state_bool(
      integrator_initialized->values, "integrator_initialized is missing", num_envs);
  if (!result.ok()) {
    return result;
  }
  return external_wrench.dtype == DType::kFloat32 ? launch_step_force<float>(external_wrench,
                                                                             desired_wrench,
                                                                             feedforward_wrench,
                                                                             pose_error,
                                                                             dt,
                                                                             use_shaping_function,
                                                                             parameters,
                                                                             parameter_status,
                                                                             state,
                                                                             commanded_wrench,
                                                                             status)
                                                  : launch_step_force<double>(external_wrench,
                                                                              desired_wrench,
                                                                              feedforward_wrench,
                                                                              pose_error,
                                                                              dt,
                                                                              use_shaping_function,
                                                                              parameters,
                                                                              parameter_status,
                                                                              state,
                                                                              commanded_wrench,
                                                                              status);
}

Result combine_cartesian_wrench_sum_cuda(const TensorView& left_wrench,
                                         StatusView left_status,
                                         const TensorView& right_wrench,
                                         StatusView right_status,
                                         const MutableTensorView& cartesian_wrench,
                                         MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(left_wrench) || right_wrench.device != Device::kCuda ||
      right_wrench.dtype != left_wrench.dtype || right_wrench.rank != 2 ||
      right_wrench.layout != Layout::kRowMajor || right_wrench.shape[0] != left_wrench.shape[0] ||
      right_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      right_wrench.strides[1] != 1) {
    return Result{ResultCode::kInvalidView, "wrench inputs must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(left_wrench.shape[0]);
  Result result = validate_cuda_output(cartesian_wrench, left_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(left_status, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(right_status, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  return left_wrench.dtype == DType::kFloat32
             ? launch_combine_sum<float>(
                   left_wrench, left_status, right_wrench, right_status, cartesian_wrench, status)
             : launch_combine_sum<double>(
                   left_wrench, left_status, right_wrench, right_status, cartesian_wrench, status);
}

Result combine_variable_cartesian_impedance_cuda(const TensorView& stiffness_wrench,
                                                 StatusView stiffness_status,
                                                 const TensorView& damping_wrench,
                                                 StatusView damping_status,
                                                 const TensorView& inertial_wrench,
                                                 StatusView inertial_status,
                                                 const TensorView& desired_acceleration,
                                                 const TensorView& cartesian_mass_matrix,
                                                 const MutableTensorView& cartesian_wrench,
                                                 MutableStatusView status) {
  if (!is_cuda_cartesian_matrix(stiffness_wrench) || damping_wrench.device != Device::kCuda ||
      inertial_wrench.device != Device::kCuda || desired_acceleration.device != Device::kCuda ||
      damping_wrench.dtype != stiffness_wrench.dtype ||
      inertial_wrench.dtype != stiffness_wrench.dtype ||
      desired_acceleration.dtype != stiffness_wrench.dtype || damping_wrench.rank != 2 ||
      inertial_wrench.rank != 2 || desired_acceleration.rank != 2 ||
      damping_wrench.shape[0] != stiffness_wrench.shape[0] ||
      inertial_wrench.shape[0] != stiffness_wrench.shape[0] ||
      desired_acceleration.shape[0] != stiffness_wrench.shape[0] ||
      damping_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      inertial_wrench.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      desired_acceleration.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      damping_wrench.strides[1] != 1 || inertial_wrench.strides[1] != 1 ||
      desired_acceleration.strides[1] != 1) {
    return Result{ResultCode::kInvalidView,
                  "variable impedance inputs must be CUDA row-major [num_envs, 6]"};
  }
  const std::size_t num_envs = static_cast<std::size_t>(stiffness_wrench.shape[0]);
  Result result = validate_cuda_tensor3_matrix6(cartesian_mass_matrix,
                                                "cartesian_mass_matrix tensor data is null",
                                                stiffness_wrench.dtype,
                                                num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_output(cartesian_wrench, stiffness_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(stiffness_status, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(damping_status, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(inertial_status, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_status(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  return stiffness_wrench.dtype == DType::kFloat32
             ? launch_combine_variable_impedance<float>(stiffness_wrench,
                                                        stiffness_status,
                                                        damping_wrench,
                                                        damping_status,
                                                        inertial_wrench,
                                                        inertial_status,
                                                        desired_acceleration,
                                                        cartesian_mass_matrix,
                                                        cartesian_wrench,
                                                        status)
             : launch_combine_variable_impedance<double>(stiffness_wrench,
                                                         stiffness_status,
                                                         damping_wrench,
                                                         damping_status,
                                                         inertial_wrench,
                                                         inertial_status,
                                                         desired_acceleration,
                                                         cartesian_mass_matrix,
                                                         cartesian_wrench,
                                                         status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
