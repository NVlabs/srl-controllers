/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file joint_blocks.cpp
 * @brief CPU tensor kernels for diagonal joint stiffness and damping.
 */

#include "nvidia/srl/controllers/tensor/joint_blocks.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#include "tensor/symmetric_matrix_policy.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/joint_blocks_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kStiffnessDiag = "stiffness_diag";
constexpr const char* kDampingDiag = "damping_diag";
constexpr const char* kStiffnessMatrix = "stiffness_matrix";
constexpr const char* kDampingMatrix = "damping_matrix";
constexpr const char* kDampingFactors = "damping_factors";
constexpr std::size_t kMaxJointDim = 32;
constexpr std::size_t kMaxTaskDim = 12;

bool parameter_set_has_cuda(const ParameterSetView& parameters) {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

bool joint_lifecycle_uses_cuda(const TensorView& joint_error,
                               const ParameterSetView& parameters,
                               const MutableTensorView& joint_torque) {
  return joint_error.device == Device::kCuda || joint_torque.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const ParameterSetView& parameters,
                               const MutableStatusView& status) {
  return status.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const TensorView& reset_mask, const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda;
}

bool joint_lifecycle_uses_cuda(const TensorView& joint_error,
                               const ParameterSetView& parameters,
                               const StatusView& parameter_status,
                               const MutableTensorView& joint_torque,
                               const MutableStatusView& status) {
  return joint_error.device == Device::kCuda || joint_torque.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const TensorView& first_input,
                               const TensorView& second_input,
                               const TensorView& third_input,
                               const ParameterSetView& parameters,
                               const MutableTensorView& output) {
  return first_input.device == Device::kCuda || second_input.device == Device::kCuda ||
         third_input.device == Device::kCuda || output.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const TensorView& first_input,
                               const TensorView& second_input,
                               const TensorView& third_input,
                               const TensorView& fourth_input,
                               const ParameterSetView& parameters,
                               const MutableTensorView& output) {
  return first_input.device == Device::kCuda || second_input.device == Device::kCuda ||
         third_input.device == Device::kCuda || fourth_input.device == Device::kCuda ||
         output.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const TensorView& first_input,
                               const TensorView& second_input,
                               const TensorView& third_input,
                               const ParameterSetView& parameters,
                               const StatusView& parameter_status,
                               const MutableTensorView& output,
                               const MutableStatusView& status) {
  return first_input.device == Device::kCuda || second_input.device == Device::kCuda ||
         third_input.device == Device::kCuda || output.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

bool joint_lifecycle_uses_cuda(const TensorView& first_input,
                               const TensorView& second_input,
                               const TensorView& third_input,
                               const TensorView& fourth_input,
                               const ParameterSetView& parameters,
                               const StatusView& parameter_status,
                               const MutableTensorView& output,
                               const MutableStatusView& status) {
  return first_input.device == Device::kCuda || second_input.device == Device::kCuda ||
         third_input.device == Device::kCuda || fourth_input.device == Device::kCuda ||
         output.device == Device::kCuda || status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
Result validate_cuda_joint_diagonal_prepare(const TensorView& joint_error,
                                            const ParameterSetView& parameters,
                                            const char* diagonal_parameter_name,
                                            const MutableTensorView& joint_torque,
                                            std::size_t* num_envs,
                                            std::size_t* joint_dim) {
  if (joint_error.data == nullptr || joint_error.device != Device::kCuda ||
      !internal::is_float_dtype(joint_error.dtype) || !internal::is_row_major_matrix(joint_error) ||
      joint_error.shape[0] <= 0 || joint_error.shape[1] <= 0) {
    return internal::invalid_view("joint error must be CUDA row-major [num_envs, joint_dim]");
  }
  *num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  *joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  if (joint_torque.data == nullptr || joint_torque.device != Device::kCuda ||
      joint_torque.dtype != joint_error.dtype || !internal::is_row_major_matrix(joint_torque) ||
      joint_torque.shape[0] != joint_error.shape[0] ||
      joint_torque.shape[1] != joint_error.shape[1]) {
    return internal::invalid_view("joint torque must be CUDA row-major [num_envs, joint_dim]");
  }
  const ParameterView* diagonal = find_parameter(parameters, diagonal_parameter_name);
  if (diagonal == nullptr || diagonal->values.data == nullptr ||
      diagonal->values.device != Device::kCuda || diagonal->values.dtype != joint_error.dtype ||
      diagonal->values.layout != Layout::kRowMajor) {
    return internal::invalid_view("diagonal joint parameter must be CUDA and match input dtype");
  }
  if (diagonal->shared) {
    if (diagonal->values.rank == 1 &&
        diagonal->values.shape[0] == static_cast<std::int64_t>(*joint_dim) &&
        diagonal->values.strides[0] >= 1) {
      return internal::ok();
    }
    return internal::invalid_view("shared diagonal joint parameter must be [joint_dim]");
  }
  if (diagonal->values.rank == 2 &&
      diagonal->values.shape[0] == static_cast<std::int64_t>(*num_envs) &&
      diagonal->values.shape[1] == static_cast<std::int64_t>(*joint_dim) &&
      diagonal->values.strides[1] == 1 &&
      diagonal->values.strides[0] >= diagonal->values.shape[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched diagonal joint parameter must be [num_envs, joint_dim]");
}

Result validate_cuda_joint_dense_prepare(const TensorView& joint_error,
                                         const ParameterSetView& parameters,
                                         const char* matrix_parameter_name,
                                         const MutableTensorView& joint_torque,
                                         std::size_t* num_envs,
                                         std::size_t* joint_dim) {
  if (joint_error.data == nullptr || joint_error.device != Device::kCuda ||
      !internal::is_float_dtype(joint_error.dtype) || !internal::is_row_major_matrix(joint_error) ||
      joint_error.shape[0] <= 0 || joint_error.shape[1] <= 0) {
    return internal::invalid_view("joint error must be CUDA row-major [num_envs, joint_dim]");
  }
  *num_envs = static_cast<std::size_t>(joint_error.shape[0]);
  *joint_dim = static_cast<std::size_t>(joint_error.shape[1]);
  if (joint_torque.data == nullptr || joint_torque.device != Device::kCuda ||
      joint_torque.dtype != joint_error.dtype || !internal::is_row_major_matrix(joint_torque) ||
      joint_torque.shape[0] != joint_error.shape[0] ||
      joint_torque.shape[1] != joint_error.shape[1]) {
    return internal::invalid_view("joint torque must be CUDA row-major [num_envs, joint_dim]");
  }
  const ParameterView* matrix = find_parameter(parameters, matrix_parameter_name);
  if (matrix == nullptr || matrix->values.data == nullptr ||
      matrix->values.device != Device::kCuda || matrix->values.dtype != joint_error.dtype ||
      matrix->values.layout != Layout::kRowMajor) {
    return internal::invalid_view("dense joint parameter must be CUDA and match input dtype");
  }
  if (matrix->shared) {
    if (matrix->values.rank == 2 &&
        matrix->values.shape[0] == static_cast<std::int64_t>(*joint_dim) &&
        matrix->values.shape[1] == static_cast<std::int64_t>(*joint_dim) &&
        matrix->values.strides[1] == 1 && matrix->values.strides[0] >= matrix->values.shape[1]) {
      return internal::ok();
    }
    return internal::invalid_view("shared dense joint parameter must be [joint_dim, joint_dim]");
  }
  if (matrix->values.rank == 3 && matrix->values.shape[0] == static_cast<std::int64_t>(*num_envs) &&
      matrix->values.shape[1] == static_cast<std::int64_t>(*joint_dim) &&
      matrix->values.shape[2] == static_cast<std::int64_t>(*joint_dim) &&
      matrix->values.strides[2] == 1 && matrix->values.strides[1] >= matrix->values.shape[2] &&
      matrix->values.strides[0] >= matrix->values.shape[1] * matrix->values.strides[1]) {
    return internal::ok();
  }
  return internal::invalid_view(
      "batched dense joint parameter must be [num_envs, joint_dim, joint_dim]");
}

Result validate_cuda_joint_tensor3_square_metadata(const TensorView& tensor,
                                                   const char* name,
                                                   const DType dtype,
                                                   const std::size_t num_envs,
                                                   const std::size_t joint_dim) {
  if (tensor.data == nullptr || tensor.device != Device::kCuda || tensor.dtype != dtype ||
      !internal::is_row_major_tensor3(tensor) ||
      tensor.shape[0] != static_cast<std::int64_t>(num_envs) ||
      tensor.shape[1] != static_cast<std::int64_t>(joint_dim) ||
      tensor.shape[2] != static_cast<std::int64_t>(joint_dim)) {
    return internal::invalid_view(name);
  }
  return internal::ok();
}

Result validate_cuda_joint_tensor3_matrix_metadata(const TensorView& tensor,
                                                   const char* name,
                                                   const DType dtype,
                                                   const std::size_t num_envs,
                                                   const std::size_t rows,
                                                   const std::size_t cols) {
  if (tensor.data == nullptr || tensor.device != Device::kCuda || tensor.dtype != dtype ||
      !internal::is_row_major_tensor3(tensor) ||
      tensor.shape[0] != static_cast<std::int64_t>(num_envs) ||
      tensor.shape[1] != static_cast<std::int64_t>(rows) ||
      tensor.shape[2] != static_cast<std::int64_t>(cols)) {
    return internal::invalid_view(name);
  }
  return internal::ok();
}

Result validate_cuda_joint_dense_parameter_metadata(const ParameterView* parameter,
                                                    const char* missing_message,
                                                    const DType dtype,
                                                    const std::size_t num_envs,
                                                    const std::size_t joint_dim) {
  if (parameter == nullptr || parameter->values.data == nullptr) {
    return internal::invalid_view(missing_message);
  }
  const TensorView& values = parameter->values;
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view("dense joint parameter must be CUDA and match input dtype");
  }
  if (parameter->shared) {
    if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(joint_dim) &&
        values.shape[1] == static_cast<std::int64_t>(joint_dim) && values.strides[1] == 1 &&
        values.strides[0] >= values.shape[1]) {
      return internal::ok();
    }
    return internal::invalid_view("shared dense joint parameter must be [joint_dim, joint_dim]");
  }
  if (values.rank == 3 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(joint_dim) &&
      values.shape[2] == static_cast<std::int64_t>(joint_dim) && values.strides[2] == 1 &&
      values.strides[1] >= values.shape[2] &&
      values.strides[0] >= values.shape[1] * values.strides[1]) {
    return internal::ok();
  }
  return internal::invalid_view(
      "batched dense joint parameter must be [num_envs, joint_dim, joint_dim]");
}

Result validate_cuda_joint_double_diagonalization_prepare(const TensorView& joint_velocity_error,
                                                          const TensorView& joint_mass_matrix,
                                                          const TensorView& stiffness_matrix,
                                                          const ParameterSetView& parameters,
                                                          const MutableTensorView& joint_torque,
                                                          std::size_t* num_envs,
                                                          std::size_t* joint_dim) {
  if (joint_velocity_error.data == nullptr || joint_velocity_error.device != Device::kCuda ||
      !internal::is_float_dtype(joint_velocity_error.dtype) ||
      !internal::is_row_major_matrix(joint_velocity_error) || joint_velocity_error.shape[0] <= 0 ||
      joint_velocity_error.shape[1] <= 0) {
    return internal::invalid_view(
        "joint_velocity_error must be CUDA row-major [num_envs, joint_dim]");
  }
  *num_envs = static_cast<std::size_t>(joint_velocity_error.shape[0]);
  *joint_dim = static_cast<std::size_t>(joint_velocity_error.shape[1]);
  if (*joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor joint double-diagonalization bound");
  }
  Result result = validate_cuda_joint_tensor3_square_metadata(joint_mass_matrix,
                                                              "joint_mass_matrix must be CUDA "
                                                              "[num_envs, joint_dim, joint_dim]",
                                                              joint_velocity_error.dtype,
                                                              *num_envs,
                                                              *joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_joint_tensor3_square_metadata(stiffness_matrix,
                                                       "stiffness_matrix must be CUDA "
                                                       "[num_envs, joint_dim, joint_dim]",
                                                       joint_velocity_error.dtype,
                                                       *num_envs,
                                                       *joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_torque.data == nullptr || joint_torque.device != Device::kCuda ||
      joint_torque.dtype != joint_velocity_error.dtype ||
      !internal::is_row_major_matrix(joint_torque) ||
      joint_torque.shape[0] != joint_velocity_error.shape[0] ||
      joint_torque.shape[1] != joint_velocity_error.shape[1]) {
    return internal::invalid_view("joint_torque must be CUDA row-major [num_envs, joint_dim]");
  }
  const ParameterView* damping_factors = find_parameter(parameters, kDampingFactors);
  if (damping_factors == nullptr || damping_factors->values.data == nullptr ||
      damping_factors->values.device != Device::kCuda ||
      damping_factors->values.dtype != joint_velocity_error.dtype ||
      damping_factors->values.layout != Layout::kRowMajor) {
    return internal::invalid_view("damping_factors must be CUDA and match input dtype");
  }
  if (damping_factors->shared) {
    if (damping_factors->values.rank == 1 &&
        damping_factors->values.shape[0] == static_cast<std::int64_t>(*joint_dim) &&
        damping_factors->values.strides[0] >= 1) {
      return internal::ok();
    }
    return internal::invalid_view("shared damping_factors must be [joint_dim]");
  }
  if (damping_factors->values.rank == 2 &&
      damping_factors->values.shape[0] == static_cast<std::int64_t>(*num_envs) &&
      damping_factors->values.shape[1] == static_cast<std::int64_t>(*joint_dim) &&
      damping_factors->values.strides[1] == 1 &&
      damping_factors->values.strides[0] >= damping_factors->values.shape[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched damping_factors must be [num_envs, joint_dim]");
}

Result validate_cuda_joint_impedance_prepare(const TensorView& joint_position_error,
                                             const TensorView& joint_velocity_error,
                                             const TensorView& desired_acceleration,
                                             const TensorView& joint_mass_matrix,
                                             const ParameterSetView& parameters,
                                             const MutableTensorView& joint_torque,
                                             std::size_t* num_envs,
                                             std::size_t* joint_dim) {
  Result result = validate_cuda_joint_dense_prepare(
      joint_position_error, parameters, kStiffnessMatrix, joint_torque, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (*joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor joint impedance bound");
  }
  if (joint_velocity_error.data == nullptr || joint_velocity_error.device != Device::kCuda ||
      joint_velocity_error.dtype != joint_position_error.dtype ||
      !internal::is_row_major_matrix(joint_velocity_error) ||
      joint_velocity_error.shape[0] != joint_position_error.shape[0] ||
      joint_velocity_error.shape[1] != joint_position_error.shape[1]) {
    return internal::invalid_view(
        "joint_velocity_error must match joint_position_error CUDA shape and dtype");
  }
  if (desired_acceleration.data == nullptr || desired_acceleration.device != Device::kCuda ||
      desired_acceleration.dtype != joint_position_error.dtype ||
      !internal::is_row_major_matrix(desired_acceleration) ||
      desired_acceleration.shape[0] != joint_position_error.shape[0] ||
      desired_acceleration.shape[1] != joint_position_error.shape[1]) {
    return internal::invalid_view(
        "desired_acceleration must match joint_position_error CUDA shape and dtype");
  }
  result = validate_cuda_joint_tensor3_square_metadata(joint_mass_matrix,
                                                       "joint_mass_matrix must be CUDA "
                                                       "[num_envs, joint_dim, joint_dim]",
                                                       joint_position_error.dtype,
                                                       *num_envs,
                                                       *joint_dim);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, kDampingFactors);
  if (damping_factors == nullptr || damping_factors->values.data == nullptr ||
      damping_factors->values.device != Device::kCuda ||
      damping_factors->values.dtype != joint_position_error.dtype ||
      damping_factors->values.layout != Layout::kRowMajor) {
    return internal::invalid_view("damping_factors must be CUDA and match input dtype");
  }
  if (damping_factors->shared) {
    if (damping_factors->values.rank == 1 &&
        damping_factors->values.shape[0] == static_cast<std::int64_t>(*joint_dim) &&
        damping_factors->values.strides[0] >= 1) {
      return internal::ok();
    }
    return internal::invalid_view("shared damping_factors must be [joint_dim]");
  }
  if (damping_factors->values.rank == 2 &&
      damping_factors->values.shape[0] == static_cast<std::int64_t>(*num_envs) &&
      damping_factors->values.shape[1] == static_cast<std::int64_t>(*joint_dim) &&
      damping_factors->values.strides[1] == 1 &&
      damping_factors->values.strides[0] >= damping_factors->values.shape[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched damping_factors must be [num_envs, joint_dim]");
}

Result validate_cuda_joint_nullspace_prepare(const TensorView& joint_position_error,
                                             const TensorView& joint_velocity,
                                             const TensorView& joint_mass_matrix,
                                             const TensorView& jacobian,
                                             const ParameterSetView& parameters,
                                             const MutableTensorView& joint_torque,
                                             std::size_t* num_envs,
                                             std::size_t* joint_dim) {
  Result result = validate_cuda_joint_dense_prepare(
      joint_position_error, parameters, kStiffnessMatrix, joint_torque, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (*joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor nullspace bound");
  }
  if (joint_velocity.data == nullptr || joint_velocity.device != Device::kCuda ||
      joint_velocity.dtype != joint_position_error.dtype ||
      !internal::is_row_major_matrix(joint_velocity) ||
      joint_velocity.shape[0] != joint_position_error.shape[0] ||
      joint_velocity.shape[1] != joint_position_error.shape[1]) {
    return internal::invalid_view(
        "joint_velocity must match joint_position_error CUDA shape and dtype");
  }
  result = validate_cuda_joint_tensor3_square_metadata(joint_mass_matrix,
                                                       "joint_mass_matrix must be CUDA "
                                                       "[num_envs, joint_dim, joint_dim]",
                                                       joint_position_error.dtype,
                                                       *num_envs,
                                                       *joint_dim);
  if (!result.ok()) {
    return result;
  }
  const std::size_t task_dim =
      jacobian.shape[1] > 0 ? static_cast<std::size_t>(jacobian.shape[1]) : 0;
  if (task_dim == 0 || task_dim > kMaxTaskDim || task_dim > *joint_dim) {
    return internal::invalid_view("nullspace task_dim must be in [1, min(12, joint_dim)]");
  }
  result = validate_cuda_joint_tensor3_matrix_metadata(jacobian,
                                                       "jacobian must be CUDA "
                                                       "[num_envs, task_dim, joint_dim]",
                                                       joint_position_error.dtype,
                                                       *num_envs,
                                                       task_dim,
                                                       *joint_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_joint_dense_parameter_metadata(find_parameter(parameters, kDampingMatrix),
                                                      "damping_matrix parameter is missing",
                                                      joint_position_error.dtype,
                                                      *num_envs,
                                                      *joint_dim);
}
#endif

template <typename Scalar>
void set_identity_n(Scalar matrix[kMaxJointDim][kMaxJointDim], const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      matrix[row][col] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
void zero_matrix_n(Scalar matrix[kMaxJointDim][kMaxJointDim],
                   const std::size_t rows,
                   const std::size_t cols) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      matrix[row][col] = static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
void transpose_n(const Scalar input[kMaxJointDim][kMaxJointDim],
                 Scalar output[kMaxJointDim][kMaxJointDim],
                 const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      output[row][col] = input[col][row];
    }
  }
}

template <typename Scalar>
void multiply_n(const Scalar left[kMaxJointDim][kMaxJointDim],
                const Scalar right[kMaxJointDim][kMaxJointDim],
                Scalar output[kMaxJointDim][kMaxJointDim],
                const std::size_t rows,
                const std::size_t inner_dim,
                const std::size_t cols) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < inner_dim; ++inner) {
        value += left[row][inner] * right[inner][col];
      }
      output[row][col] = value;
    }
  }
}

template <typename Scalar>
void matvec_n(const Scalar matrix[kMaxJointDim][kMaxJointDim],
              const Scalar vector[kMaxJointDim],
              Scalar output[kMaxJointDim],
              const std::size_t rows,
              const std::size_t cols) {
  const detail::ConstMatrixView<Scalar> matrix_view{&matrix[0][0], rows, cols, kMaxJointDim, 1};
  const detail::ConstVectorView<Scalar> vector_view{vector, cols, 1};
  for (std::size_t row = 0; row < rows; ++row) {
    output[row] = detail::dense_feedback_row(matrix_view, vector_view, row);
  }
}

template <typename Scalar>
void scale_rows_n(const Scalar input[kMaxJointDim][kMaxJointDim],
                  const Scalar row_scale[kMaxJointDim],
                  Scalar output[kMaxJointDim][kMaxJointDim],
                  const std::size_t rows,
                  const std::size_t cols) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      output[row][col] = row_scale[row] * input[row][col];
    }
  }
}

template <typename Scalar>
bool validate_and_symmetrize_external_n(Scalar matrix[kMaxJointDim][kMaxJointDim],
                                        const std::size_t dim) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!std::isfinite(matrix[row][col])) {
        return false;
      }
      scale = std::max(scale, std::fabs(matrix[row][col]));
    }
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(dim) * scale;
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = row + 1; col < dim; ++col) {
      if (std::fabs(matrix[row][col] - matrix[col][row]) > tolerance) {
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
bool symmetrize_derived_n(Scalar matrix[kMaxJointDim][kMaxJointDim], const std::size_t dim) {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!std::isfinite(matrix[row][col])) {
        return false;
      }
    }
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar value =
          static_cast<Scalar>(0.5) * matrix[row][col] + static_cast<Scalar>(0.5) * matrix[col][row];
      matrix[row][col] = value;
      matrix[col][row] = value;
    }
  }
  return true;
}

template <typename Scalar>
bool symmetric_eigen_n(const Scalar input[kMaxJointDim][kMaxJointDim],
                       Scalar eigenvectors[kMaxJointDim][kMaxJointDim],
                       Scalar eigenvalues[kMaxJointDim],
                       const std::size_t dim,
                       Scalar* matrix_scale) {
  Scalar matrix[kMaxJointDim][kMaxJointDim]{};
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!std::isfinite(input[row][col])) {
        return false;
      }
      scale = std::max(scale, std::fabs(input[row][col]));
    }
  }
  if (!std::isfinite(scale)) {
    return false;
  }
  if (scale == static_cast<Scalar>(0)) {
    scale = static_cast<Scalar>(1);
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      matrix[row][col] = input[row][col] / scale;
    }
  }

  set_identity_n(eigenvectors, dim);
  const std::size_t max_iterations = std::max<std::size_t>(80, 10 * dim * dim);
  bool converged = dim <= 1;
  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = dim > 1 ? 1 : 0;
    Scalar max_offdiag = dim > 1 ? std::fabs(matrix[pivot_row][pivot_col]) : 0;
    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = row + 1; col < dim; ++col) {
        const Scalar candidate = std::fabs(matrix[row][col]);
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

    const Scalar app = matrix[pivot_row][pivot_row];
    const Scalar aqq = matrix[pivot_col][pivot_col];
    const Scalar apq = matrix[pivot_row][pivot_col];
    const Scalar angle =
        static_cast<Scalar>(0.5) * std::atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar c = std::cos(angle);
    const Scalar s = std::sin(angle);

    for (std::size_t idx = 0; idx < dim; ++idx) {
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

    for (std::size_t row = 0; row < dim; ++row) {
      const Scalar vip = eigenvectors[row][pivot_row];
      const Scalar viq = eigenvectors[row][pivot_col];
      eigenvectors[row][pivot_row] = c * vip - s * viq;
      eigenvectors[row][pivot_col] = s * vip + c * viq;
    }
  }

  if (!converged) {
    return false;
  }

  for (std::size_t i = 0; i < dim; ++i) {
    eigenvalues[i] = matrix[i][i];
    if (!std::isfinite(eigenvalues[i])) {
      return false;
    }
    for (std::size_t row = 0; row < dim; ++row) {
      if (!std::isfinite(eigenvectors[row][i])) {
        return false;
      }
    }
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
bool matrix_roots_n(const Scalar input[kMaxJointDim][kMaxJointDim],
                    Scalar output[kMaxJointDim][kMaxJointDim],
                    Scalar* inverse_output,
                    const std::size_t dim,
                    const bool strict_positive) {
  Scalar eigenvectors[kMaxJointDim][kMaxJointDim]{};
  Scalar eigenvalues[kMaxJointDim]{};
  Scalar scale{};
  if (!symmetric_eigen_n(input, eigenvectors, eigenvalues, dim, &scale)) {
    return false;
  }
  Scalar max_eigenvalue = static_cast<Scalar>(0);
  for (std::size_t col = 0; col < dim; ++col) {
    max_eigenvalue = std::max(max_eigenvalue, std::fabs(eigenvalues[col]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(dim) * max_eigenvalue;
  Scalar scaled_eigenvectors[kMaxJointDim][kMaxJointDim]{};
  Scalar inverse_scaled_eigenvectors[kMaxJointDim][kMaxJointDim]{};
  for (std::size_t col = 0; col < dim; ++col) {
    if (strict_positive) {
      if (eigenvalues[col] <= threshold) {
        return false;
      }
    } else if (eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar normalized = std::max(eigenvalues[col], static_cast<Scalar>(0));
    const Scalar sqrt_value = std::sqrt(scale) * std::sqrt(normalized);
    for (std::size_t row = 0; row < dim; ++row) {
      scaled_eigenvectors[row][col] = eigenvectors[row][col] * sqrt_value;
      if (inverse_output != nullptr) {
        inverse_scaled_eigenvectors[row][col] =
            eigenvectors[row][col] / (std::sqrt(scale) * std::sqrt(normalized));
      }
    }
  }
  Scalar eigenvectors_transpose[kMaxJointDim][kMaxJointDim]{};
  transpose_n(eigenvectors, eigenvectors_transpose, dim);
  multiply_n(scaled_eigenvectors, eigenvectors_transpose, output, dim, dim, dim);
  if (inverse_output != nullptr) {
    Scalar(*inverse_matrix)[kMaxJointDim] =
        reinterpret_cast<Scalar(*)[kMaxJointDim]>(inverse_output);
    multiply_n(inverse_scaled_eigenvectors, eigenvectors_transpose, inverse_matrix, dim, dim, dim);
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      if (!std::isfinite(output[row][col]) ||
          (inverse_output != nullptr &&
           !std::isfinite(reinterpret_cast<Scalar(*)[kMaxJointDim]>(inverse_output)[row][col]))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
bool inverse_n(const Scalar input[kMaxJointDim][kMaxJointDim],
               Scalar output[kMaxJointDim][kMaxJointDim],
               const std::size_t dim) {
  Scalar augmented[kMaxJointDim][kMaxJointDim * 2]{};
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      augmented[row][col] = input[row][col];
      augmented[row][col + dim] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
  for (std::size_t pivot = 0; pivot < dim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = std::fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < dim; ++row) {
      const Scalar candidate = std::fabs(augmented[row][pivot]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot_row = row;
      }
    }
    if (pivot_abs <= static_cast<Scalar>(1e-12)) {
      return false;
    }
    if (pivot_row != pivot) {
      for (std::size_t col = 0; col < dim * 2; ++col) {
        std::swap(augmented[pivot][col], augmented[pivot_row][col]);
      }
    }
    const Scalar pivot_value = augmented[pivot][pivot];
    for (std::size_t col = 0; col < dim * 2; ++col) {
      augmented[pivot][col] /= pivot_value;
    }
    for (std::size_t row = 0; row < dim; ++row) {
      if (row == pivot) {
        continue;
      }
      const Scalar factor = augmented[row][pivot];
      for (std::size_t col = 0; col < dim * 2; ++col) {
        augmented[row][col] -= factor * augmented[pivot][col];
      }
    }
  }
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      output[row][col] = augmented[row][col + dim];
    }
  }
  return true;
}

template <typename Scalar>
bool compute_double_diagonalization_damping_matrix(
    const Scalar mass_matrix[kMaxJointDim][kMaxJointDim],
    const Scalar stiffness_matrix[kMaxJointDim][kMaxJointDim],
    const Scalar damping_factors[kMaxJointDim],
    Scalar damping_matrix[kMaxJointDim][kMaxJointDim],
    const std::size_t joint_dim) {
  Scalar mass_sqrt[kMaxJointDim][kMaxJointDim]{};
  Scalar mass_sqrt_inv[kMaxJointDim][kMaxJointDim]{};
  if (!matrix_roots_n(mass_matrix, mass_sqrt, &mass_sqrt_inv[0][0], joint_dim, true)) {
    return false;
  }
  Scalar mass_sqrt_inv_transpose[kMaxJointDim][kMaxJointDim]{};
  transpose_n(mass_sqrt_inv, mass_sqrt_inv_transpose, joint_dim);
  Scalar temp[kMaxJointDim][kMaxJointDim]{};
  Scalar transformed_stiffness[kMaxJointDim][kMaxJointDim]{};
  multiply_n(mass_sqrt_inv, stiffness_matrix, temp, joint_dim, joint_dim, joint_dim);
  multiply_n(temp, mass_sqrt_inv_transpose, transformed_stiffness, joint_dim, joint_dim, joint_dim);
  if (!symmetrize_derived_n(transformed_stiffness, joint_dim)) {
    return false;
  }
  Scalar transformed_stiffness_sqrt[kMaxJointDim][kMaxJointDim]{};
  if (!matrix_roots_n(transformed_stiffness,
                      transformed_stiffness_sqrt,
                      static_cast<Scalar*>(nullptr),
                      joint_dim,
                      true)) {
    return false;
  }
  Scalar d_stiffness[kMaxJointDim][kMaxJointDim]{};
  Scalar left[kMaxJointDim][kMaxJointDim]{};
  Scalar mass_sqrt_transpose[kMaxJointDim][kMaxJointDim]{};
  scale_rows_n(transformed_stiffness_sqrt, damping_factors, d_stiffness, joint_dim, joint_dim);
  multiply_n(mass_sqrt, d_stiffness, left, joint_dim, joint_dim, joint_dim);
  transpose_n(mass_sqrt, mass_sqrt_transpose, joint_dim);
  multiply_n(left, mass_sqrt_transpose, damping_matrix, joint_dim, joint_dim, joint_dim);
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      damping_matrix[row][col] =
          detail::damping_double_diagonalization_value(damping_matrix[row][col]);
    }
  }
  return true;
}

template <typename Scalar>
void read_tensor3_square(const TensorView& view,
                         const std::size_t env,
                         Scalar matrix[kMaxJointDim][kMaxJointDim],
                         const std::size_t joint_dim) {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      matrix[row][col] = internal::read_tensor3_value<Scalar>(view, env, row, col);
    }
  }
}

template <typename Scalar>
void read_parameter_square(const ParameterView* parameter,
                           const std::size_t env,
                           Scalar matrix[kMaxJointDim][kMaxJointDim],
                           const std::size_t joint_dim) {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      matrix[row][col] =
          internal::read_matrix_parameter<Scalar>(parameter, env, row, col, static_cast<Scalar>(0));
    }
  }
}

template <typename Scalar>
bool diagonal_parameter_is_valid(const ParameterSetView& parameters,
                                 const char* parameter_name,
                                 const std::size_t env,
                                 const std::size_t joint_dim) {
  const ParameterView* diagonal = find_parameter(parameters, parameter_name);
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    const Scalar value =
        internal::read_vector_parameter<Scalar>(diagonal, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_diagonal_parameters(const ParameterSetView& parameters,
                                 const char* parameter_name,
                                 const std::size_t num_envs,
                                 const std::size_t joint_dim,
                                 MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const ParameterView* diagonal = find_parameter(parameters, parameter_name);
  result = internal::validate_vector_parameter_metadata(diagonal,
                                                        "diagonal joint parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        joint_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!diagonal_parameter_is_valid<Scalar>(parameters, parameter_name, env, joint_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool dense_parameter_is_valid(const ParameterSetView& parameters,
                              const char* parameter_name,
                              const std::size_t env,
                              const std::size_t joint_dim) {
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col < joint_dim; ++col) {
      const Scalar value =
          internal::read_matrix_parameter<Scalar>(matrix, env, row, col, static_cast<Scalar>(0));
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
Result apply_dense_parameters(const ParameterSetView& parameters,
                              const char* parameter_name,
                              const std::size_t num_envs,
                              const std::size_t joint_dim,
                              MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  result = internal::validate_matrix_parameter_metadata(matrix,
                                                        "dense joint parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        joint_dim,
                                                        joint_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!dense_parameter_is_valid<Scalar>(parameters, parameter_name, env, joint_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool damping_factors_are_valid(const ParameterSetView& parameters,
                               const std::size_t env,
                               const std::size_t joint_dim) {
  const ParameterView* damping_factors = find_parameter(parameters, kDampingFactors);
  for (std::size_t dim = 0; dim < joint_dim; ++dim) {
    const Scalar value =
        internal::read_vector_parameter<Scalar>(damping_factors, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_damping_factors_parameters(const ParameterSetView& parameters,
                                        const std::size_t num_envs,
                                        const std::size_t joint_dim,
                                        MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  if (joint_dim == 0 || joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor joint double-diagonalization bound");
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                        "damping_factors parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        joint_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!damping_factors_are_valid<Scalar>(parameters, env, joint_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result apply_joint_impedance_parameters_impl(const ParameterSetView& parameters,
                                             const std::size_t num_envs,
                                             const std::size_t joint_dim,
                                             MutableStatusView parameter_status) {
  Result result =
      apply_damping_factors_parameters<Scalar>(parameters, num_envs, joint_dim, parameter_status);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   joint_dim,
                                                   joint_dim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!dense_parameter_is_valid<Scalar>(parameters, kStiffnessMatrix, env, joint_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result apply_nullspace_parameters_impl(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       const std::size_t joint_dim,
                                       MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  if (joint_dim == 0 || joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor nullspace bound");
  }
  result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   joint_dim,
                                                   joint_dim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matrix_parameter_metadata(find_parameter(parameters, kDampingMatrix),
                                                        "damping_matrix parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        joint_dim,
                                                        joint_dim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!dense_parameter_is_valid<Scalar>(parameters, kStiffnessMatrix, env, joint_dim) ||
        !dense_parameter_is_valid<Scalar>(parameters, kDampingMatrix, env, joint_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_diagonal(const TensorView& joint_error,
                     const ParameterSetView& parameters,
                     const char* parameter_name,
                     StatusView parameter_status,
                     const MutableTensorView& joint_torque,
                     MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_error, "joint error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result =
      apply_diagonal_parameters<Scalar>(parameters, parameter_name, num_envs, joint_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* diagonal = find_parameter(parameters, parameter_name);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(joint_error, env, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kInvalidParameter) |
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      internal::write_matrix_value(
          joint_torque,
          env,
          dim,
          detail::joint_diagonal_feedback_value(
              internal::read_vector_parameter<Scalar>(diagonal, env, dim, static_cast<Scalar>(0)),
              internal::read_matrix_value<Scalar>(joint_error, env, dim)));
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_dense(const TensorView& joint_error,
                  const ParameterSetView& parameters,
                  const char* parameter_name,
                  StatusView parameter_status,
                  const MutableTensorView& joint_torque,
                  MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_error, "joint error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = apply_dense_parameters<Scalar>(parameters, parameter_name, num_envs, joint_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* matrix = find_parameter(parameters, parameter_name);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(joint_error, env, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kInvalidParameter) |
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    Scalar dense_matrix[kMaxJointDim][kMaxJointDim]{};
    Scalar error[kMaxJointDim]{};
    read_parameter_square(matrix, env, dense_matrix, joint_dim);
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      error[dim] = internal::read_matrix_value<Scalar>(joint_error, env, dim);
    }
    const detail::ConstMatrixView<Scalar> matrix_view{
        &dense_matrix[0][0], joint_dim, joint_dim, kMaxJointDim, 1};
    const detail::ConstVectorView<Scalar> error_view{error, joint_dim, 1};
    const bool stiffness = std::strcmp(parameter_name, kStiffnessMatrix) == 0;
    for (std::size_t row = 0; row < joint_dim; ++row) {
      const Scalar value = stiffness ? detail::joint_stiffness_value(matrix_view, error_view, row)
                                     : detail::joint_damping_value(matrix_view, error_view, row);
      internal::write_matrix_value(joint_torque, env, row, value);
    }
  }
  return internal::ok();
}

PrepareResult prepare_joint(ComponentMode mode,
                            const TensorView& joint_error,
                            const ParameterSetView& parameters,
                            const char* diagonal_parameter_name,
                            const char* matrix_parameter_name,
                            const MutableTensorView& joint_torque,
                            const char* units,
                            const char* semantics) {
  PrepareResult prepare_result;
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    prepare_result.result = internal::unsupported("tensor joint mode is not supported");
    return prepare_result;
  }
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  if (joint_lifecycle_uses_cuda(joint_error, parameters, joint_torque)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = mode == ComponentMode::kDiagonal
                                ? validate_cuda_joint_diagonal_prepare(joint_error,
                                                                       parameters,
                                                                       diagonal_parameter_name,
                                                                       joint_torque,
                                                                       &num_envs,
                                                                       &joint_dim)
                                : validate_cuda_joint_dense_prepare(joint_error,
                                                                    parameters,
                                                                    matrix_parameter_name,
                                                                    joint_torque,
                                                                    &num_envs,
                                                                    &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = joint_dim;
    prepare_result.outputs.push_back(internal::matrix_descriptor(
        "joint_torque", joint_error.dtype, num_envs, joint_dim, units, "joint_space", semantics));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      joint_error, "joint error tensor data is null", &num_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matching_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (mode == ComponentMode::kDiagonal) {
    prepare_result.result = internal::validate_vector_parameter_metadata(
        find_parameter(parameters, diagonal_parameter_name),
        "diagonal joint parameter is missing",
        joint_error.dtype,
        num_envs,
        joint_dim,
        true);
  } else {
    prepare_result.result = internal::validate_matrix_parameter_metadata(
        find_parameter(parameters, matrix_parameter_name),
        "dense joint parameter is missing",
        joint_error.dtype,
        num_envs,
        joint_dim,
        joint_dim,
        true);
  }
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = joint_dim;
  prepare_result.outputs.push_back(internal::matrix_descriptor(
      "joint_torque", joint_error.dtype, num_envs, joint_dim, units, "joint_space", semantics));
  return prepare_result;
}

Result reset_stateless_joint(ComponentMode mode,
                             const std::size_t num_envs,
                             const TensorView& reset_mask,
                             MutableStatusView status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    return mark_unsupported(num_envs, status);
  }
  if (joint_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_stateless_joint_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (internal::reset_mask_value(reset_mask, env)) {
      status.flags[env] |= status_bit(StatusFlag::kResetApplied);
    }
  }
  return internal::ok();
}

Result step_unsupported_joint(const TensorView& joint_error,
                              const MutableTensorView& joint_torque,
                              MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_error, "joint error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(joint_torque, joint_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  if (joint_error.dtype == DType::kFloat32) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      internal::zero_output_env<float>(joint_torque, env, joint_dim);
    }
  } else {
    for (std::size_t env = 0; env < num_envs; ++env) {
      internal::zero_output_env<double>(joint_torque, env, joint_dim);
    }
  }
  internal::set_all_status(
      status,
      num_envs,
      status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kSafeMode));
  return internal::unsupported("tensor joint mode is staged unsupported");
}

}  // namespace

Result mark_unsupported(const std::size_t num_envs, MutableStatusView status) {
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  internal::set_all_status(
      status,
      num_envs,
      status_bit(StatusFlag::kUnsupportedMode) | status_bit(StatusFlag::kSafeMode));
  return internal::unsupported("tensor component or mode is staged unsupported");
}

PrepareResult prepare_joint_stiffness(ComponentMode mode,
                                      const TensorView& joint_position_error,
                                      const ParameterSetView& parameters,
                                      const MutableTensorView& joint_torque) {
  return prepare_joint(mode,
                       joint_position_error,
                       parameters,
                       kStiffnessDiag,
                       kStiffnessMatrix,
                       joint_torque,
                       "newton_meter",
                       "stiffness torque from joint position error");
}

Result apply_joint_stiffness_parameters(ComponentMode mode,
                                        const ParameterSetView& parameters,
                                        const std::size_t num_envs,
                                        const std::size_t joint_dim,
                                        MutableStatusView parameter_status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    return mark_unsupported(num_envs, parameter_status);
  }
  if (joint_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return mode == ComponentMode::kDiagonal
               ? apply_joint_diagonal_parameters_cuda(
                     parameters, kStiffnessDiag, num_envs, joint_dim, parameter_status)
               : apply_joint_dense_parameters_cuda(
                     parameters, kStiffnessMatrix, num_envs, joint_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const ParameterView* parameter = find_parameter(
      parameters, mode == ComponentMode::kDiagonal ? kStiffnessDiag : kStiffnessMatrix);
  if (parameter == nullptr) {
    return internal::invalid_view("stiffness parameter is missing");
  }
  if (mode == ComponentMode::kDiagonal) {
    return parameter->values.dtype == DType::kFloat32
               ? apply_diagonal_parameters<float>(
                     parameters, kStiffnessDiag, num_envs, joint_dim, parameter_status)
               : apply_diagonal_parameters<double>(
                     parameters, kStiffnessDiag, num_envs, joint_dim, parameter_status);
  }
  return parameter->values.dtype == DType::kFloat32
             ? apply_dense_parameters<float>(
                   parameters, kStiffnessMatrix, num_envs, joint_dim, parameter_status)
             : apply_dense_parameters<double>(
                   parameters, kStiffnessMatrix, num_envs, joint_dim, parameter_status);
}

Result reset_joint_stiffness(ComponentMode mode,
                             const std::size_t num_envs,
                             const TensorView& reset_mask,
                             MutableStatusView status) {
  return reset_stateless_joint(mode, num_envs, reset_mask, status);
}

Result step_joint_stiffness(ComponentMode mode,
                            const TensorView& joint_position_error,
                            const ParameterSetView& parameters,
                            StatusView parameter_status,
                            const MutableTensorView& joint_torque,
                            MutableStatusView status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    return step_unsupported_joint(joint_position_error, joint_torque, status);
  }
  if (joint_lifecycle_uses_cuda(
          joint_position_error, parameters, parameter_status, joint_torque, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return mode == ComponentMode::kDiagonal ? step_joint_diagonal_cuda(joint_position_error,
                                                                       parameters,
                                                                       kStiffnessDiag,
                                                                       parameter_status,
                                                                       joint_torque,
                                                                       status)
                                            : step_joint_dense_cuda(joint_position_error,
                                                                    parameters,
                                                                    kStiffnessMatrix,
                                                                    parameter_status,
                                                                    joint_torque,
                                                                    status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (joint_position_error.dtype == DType::kFloat32) {
    if (mode == ComponentMode::kDense) {
      return step_dense<float>(joint_position_error,
                               parameters,
                               kStiffnessMatrix,
                               parameter_status,
                               joint_torque,
                               status);
    }
    return step_diagonal<float>(
        joint_position_error, parameters, kStiffnessDiag, parameter_status, joint_torque, status);
  }
  if (joint_position_error.dtype == DType::kFloat64) {
    if (mode == ComponentMode::kDense) {
      return step_dense<double>(joint_position_error,
                                parameters,
                                kStiffnessMatrix,
                                parameter_status,
                                joint_torque,
                                status);
    }
    return step_diagonal<double>(
        joint_position_error, parameters, kStiffnessDiag, parameter_status, joint_torque, status);
  }
  return internal::invalid_view("joint error dtype must be float32 or float64");
}

PrepareResult prepare_joint_damping(ComponentMode mode,
                                    const TensorView& joint_velocity_error,
                                    const ParameterSetView& parameters,
                                    const MutableTensorView& joint_torque) {
  return prepare_joint(mode,
                       joint_velocity_error,
                       parameters,
                       kDampingDiag,
                       kDampingMatrix,
                       joint_torque,
                       "newton_meter",
                       "damping torque from joint velocity error");
}

Result apply_joint_damping_parameters(ComponentMode mode,
                                      const ParameterSetView& parameters,
                                      const std::size_t num_envs,
                                      const std::size_t joint_dim,
                                      MutableStatusView parameter_status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    return mark_unsupported(num_envs, parameter_status);
  }
  if (joint_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return mode == ComponentMode::kDiagonal
               ? apply_joint_diagonal_parameters_cuda(
                     parameters, kDampingDiag, num_envs, joint_dim, parameter_status)
               : apply_joint_dense_parameters_cuda(
                     parameters, kDampingMatrix, num_envs, joint_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const ParameterView* parameter =
      find_parameter(parameters, mode == ComponentMode::kDiagonal ? kDampingDiag : kDampingMatrix);
  if (parameter == nullptr) {
    return internal::invalid_view("damping parameter is missing");
  }
  if (mode == ComponentMode::kDiagonal) {
    return parameter->values.dtype == DType::kFloat32
               ? apply_diagonal_parameters<float>(
                     parameters, kDampingDiag, num_envs, joint_dim, parameter_status)
               : apply_diagonal_parameters<double>(
                     parameters, kDampingDiag, num_envs, joint_dim, parameter_status);
  }
  return parameter->values.dtype == DType::kFloat32
             ? apply_dense_parameters<float>(
                   parameters, kDampingMatrix, num_envs, joint_dim, parameter_status)
             : apply_dense_parameters<double>(
                   parameters, kDampingMatrix, num_envs, joint_dim, parameter_status);
}

Result reset_joint_damping(ComponentMode mode,
                           const std::size_t num_envs,
                           const TensorView& reset_mask,
                           MutableStatusView status) {
  return reset_stateless_joint(mode, num_envs, reset_mask, status);
}

Result step_joint_damping(ComponentMode mode,
                          const TensorView& joint_velocity_error,
                          const ParameterSetView& parameters,
                          StatusView parameter_status,
                          const MutableTensorView& joint_torque,
                          MutableStatusView status) {
  if (mode != ComponentMode::kDiagonal && mode != ComponentMode::kDense) {
    return step_unsupported_joint(joint_velocity_error, joint_torque, status);
  }
  if (joint_lifecycle_uses_cuda(
          joint_velocity_error, parameters, parameter_status, joint_torque, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return mode == ComponentMode::kDiagonal ? step_joint_diagonal_cuda(joint_velocity_error,
                                                                       parameters,
                                                                       kDampingDiag,
                                                                       parameter_status,
                                                                       joint_torque,
                                                                       status)
                                            : step_joint_dense_cuda(joint_velocity_error,
                                                                    parameters,
                                                                    kDampingMatrix,
                                                                    parameter_status,
                                                                    joint_torque,
                                                                    status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (joint_velocity_error.dtype == DType::kFloat32) {
    if (mode == ComponentMode::kDense) {
      return step_dense<float>(
          joint_velocity_error, parameters, kDampingMatrix, parameter_status, joint_torque, status);
    }
    return step_diagonal<float>(
        joint_velocity_error, parameters, kDampingDiag, parameter_status, joint_torque, status);
  }
  if (joint_velocity_error.dtype == DType::kFloat64) {
    if (mode == ComponentMode::kDense) {
      return step_dense<double>(
          joint_velocity_error, parameters, kDampingMatrix, parameter_status, joint_torque, status);
    }
    return step_diagonal<double>(
        joint_velocity_error, parameters, kDampingDiag, parameter_status, joint_torque, status);
  }
  return internal::invalid_view("joint error dtype must be float32 or float64");
}

PrepareResult prepare_joint_damping_double_diagonalization(const TensorView& joint_velocity_error,
                                                           const TensorView& joint_mass_matrix,
                                                           const TensorView& stiffness_matrix,
                                                           const ParameterSetView& parameters,
                                                           const MutableTensorView& joint_torque) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  if (joint_lifecycle_uses_cuda(
          joint_velocity_error, joint_mass_matrix, stiffness_matrix, parameters, joint_torque)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_joint_double_diagonalization_prepare(joint_velocity_error,
                                                                               joint_mass_matrix,
                                                                               stiffness_matrix,
                                                                               parameters,
                                                                               joint_torque,
                                                                               &num_envs,
                                                                               &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = joint_dim;
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("joint_damping_torque",
                                    joint_velocity_error.dtype,
                                    num_envs,
                                    joint_dim,
                                    "newton_meter",
                                    "joint_space",
                                    "double-diagonalization damping torque"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity_error, "joint_velocity_error tensor data is null", &num_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_dim > kMaxJointDim) {
    prepare_result.result =
        internal::invalid_view("joint_dim exceeds tensor joint double-diagonalization bound");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(joint_mass_matrix,
                                                            "joint_mass_matrix tensor data is null",
                                                            joint_velocity_error.dtype,
                                                            num_envs,
                                                            joint_dim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(stiffness_matrix,
                                                            "stiffness_matrix tensor data is null",
                                                            joint_velocity_error.dtype,
                                                            num_envs,
                                                            joint_dim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matching_output(
      joint_torque, joint_velocity_error.dtype, num_envs, joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                   "damping_factors parameter is missing",
                                                   joint_velocity_error.dtype,
                                                   num_envs,
                                                   joint_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = joint_dim;
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("joint_damping_torque",
                                  joint_velocity_error.dtype,
                                  num_envs,
                                  joint_dim,
                                  "newton_meter",
                                  "joint_space",
                                  "double-diagonalization damping torque"));
  return prepare_result;
}

Result apply_joint_damping_double_diagonalization_parameters(const ParameterSetView& parameters,
                                                             const std::size_t num_envs,
                                                             const std::size_t joint_dim,
                                                             MutableStatusView parameter_status) {
  if (joint_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_joint_damping_factors_parameters_cuda(
        parameters, num_envs, joint_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const ParameterView* parameter = find_parameter(parameters, kDampingFactors);
  if (parameter == nullptr) {
    return internal::invalid_view("damping_factors parameter is missing");
  }
  if (parameter->values.dtype == DType::kFloat32) {
    return apply_damping_factors_parameters<float>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  if (parameter->values.dtype == DType::kFloat64) {
    return apply_damping_factors_parameters<double>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  return internal::invalid_view("damping_factors dtype must be float32 or float64");
}

Result reset_joint_damping_double_diagonalization(const std::size_t num_envs,
                                                  const TensorView& reset_mask,
                                                  MutableStatusView status) {
  return reset_stateless_joint(ComponentMode::kDense, num_envs, reset_mask, status);
}

template <typename Scalar>
Result step_joint_damping_double_diagonalization_impl(const TensorView& joint_velocity_error,
                                                      const TensorView& joint_mass_matrix,
                                                      const TensorView& stiffness_matrix,
                                                      const ParameterSetView& parameters,
                                                      StatusView parameter_status,
                                                      const MutableTensorView& joint_torque,
                                                      MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_velocity_error, "joint_velocity_error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor joint double-diagonalization bound");
  }
  result = internal::validate_batched_matrix(joint_mass_matrix,
                                             "joint_mass_matrix tensor data is null",
                                             joint_velocity_error.dtype,
                                             num_envs,
                                             joint_dim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(stiffness_matrix,
                                             "stiffness_matrix tensor data is null",
                                             joint_velocity_error.dtype,
                                             num_envs,
                                             joint_dim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(
      joint_torque, joint_velocity_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = apply_damping_factors_parameters<Scalar>(parameters, num_envs, joint_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* damping_factors_parameter = find_parameter(parameters, kDampingFactors);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(joint_velocity_error, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(joint_mass_matrix, env, joint_dim, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(stiffness_matrix, env, joint_dim, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    Scalar mass[kMaxJointDim][kMaxJointDim]{};
    Scalar stiffness[kMaxJointDim][kMaxJointDim]{};
    Scalar damping_matrix[kMaxJointDim][kMaxJointDim]{};
    Scalar damping_factors[kMaxJointDim]{};
    Scalar velocity[kMaxJointDim]{};
    Scalar torque[kMaxJointDim]{};
    read_tensor3_square(joint_mass_matrix, env, mass, joint_dim);
    read_tensor3_square(stiffness_matrix, env, stiffness, joint_dim);
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      damping_factors[dim] = internal::read_vector_parameter<Scalar>(
          damping_factors_parameter, env, dim, static_cast<Scalar>(0));
      velocity[dim] = internal::read_matrix_value<Scalar>(joint_velocity_error, env, dim);
    }
    if (!validate_and_symmetrize_external_n(mass, joint_dim) ||
        !validate_and_symmetrize_external_n(stiffness, joint_dim) ||
        !compute_double_diagonalization_damping_matrix(
            mass, stiffness, damping_factors, damping_matrix, joint_dim)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    matvec_n(damping_matrix, velocity, torque, joint_dim, joint_dim);
    bool torque_is_finite = true;
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      torque_is_finite = torque_is_finite && std::isfinite(torque[dim]);
    }
    if (!torque_is_finite) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      internal::write_matrix_value(joint_torque, env, dim, torque[dim]);
    }
  }
  return internal::ok();
}

Result step_joint_damping_double_diagonalization(const TensorView& joint_velocity_error,
                                                 const TensorView& joint_mass_matrix,
                                                 const TensorView& stiffness_matrix,
                                                 const ParameterSetView& parameters,
                                                 StatusView parameter_status,
                                                 const MutableTensorView& joint_torque,
                                                 MutableStatusView status) {
  if (joint_lifecycle_uses_cuda(joint_velocity_error,
                                joint_mass_matrix,
                                stiffness_matrix,
                                parameters,
                                parameter_status,
                                joint_torque,
                                status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_joint_damping_double_diagonalization_cuda(joint_velocity_error,
                                                          joint_mass_matrix,
                                                          stiffness_matrix,
                                                          parameters,
                                                          parameter_status,
                                                          joint_torque,
                                                          status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (joint_velocity_error.dtype == DType::kFloat32) {
    return step_joint_damping_double_diagonalization_impl<float>(joint_velocity_error,
                                                                 joint_mass_matrix,
                                                                 stiffness_matrix,
                                                                 parameters,
                                                                 parameter_status,
                                                                 joint_torque,
                                                                 status);
  }
  if (joint_velocity_error.dtype == DType::kFloat64) {
    return step_joint_damping_double_diagonalization_impl<double>(joint_velocity_error,
                                                                  joint_mass_matrix,
                                                                  stiffness_matrix,
                                                                  parameters,
                                                                  parameter_status,
                                                                  joint_torque,
                                                                  status);
  }
  return internal::invalid_view("joint_velocity_error dtype must be float32 or float64");
}

PrepareResult prepare_joint_impedance_controller(const TensorView& joint_position_error,
                                                 const TensorView& joint_velocity_error,
                                                 const TensorView& desired_acceleration,
                                                 const TensorView& joint_mass_matrix,
                                                 const ParameterSetView& parameters,
                                                 const MutableTensorView& joint_torque) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  if (joint_lifecycle_uses_cuda(joint_position_error,
                                joint_velocity_error,
                                desired_acceleration,
                                joint_mass_matrix,
                                parameters,
                                joint_torque)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_joint_impedance_prepare(joint_position_error,
                                                                  joint_velocity_error,
                                                                  desired_acceleration,
                                                                  joint_mass_matrix,
                                                                  parameters,
                                                                  joint_torque,
                                                                  &num_envs,
                                                                  &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = joint_dim;
    prepare_result.outputs.push_back(internal::matrix_descriptor("joint_torque",
                                                                 joint_position_error.dtype,
                                                                 num_envs,
                                                                 joint_dim,
                                                                 "newton_meter",
                                                                 "joint_space",
                                                                 "joint impedance torque"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      joint_position_error, "joint_position_error tensor data is null", &num_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_dim > kMaxJointDim) {
    prepare_result.result =
        internal::invalid_view("joint_dim exceeds tensor joint impedance bound");
    return prepare_result;
  }
  std::size_t other_envs = 0;
  std::size_t other_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity_error, "joint_velocity_error tensor data is null", &other_envs, &other_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      joint_velocity_error.dtype != joint_position_error.dtype) {
    prepare_result.result = internal::invalid_view(
        "joint_velocity_error must match joint_position_error dtype and shape");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &other_envs, &other_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      desired_acceleration.dtype != joint_position_error.dtype) {
    prepare_result.result = internal::invalid_view(
        "desired_acceleration must match joint_position_error dtype and shape");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(joint_mass_matrix,
                                                            "joint_mass_matrix tensor data is null",
                                                            joint_position_error.dtype,
                                                            num_envs,
                                                            joint_dim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matching_output(
      joint_torque, joint_position_error.dtype, num_envs, joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   joint_position_error.dtype,
                                                   num_envs,
                                                   joint_dim,
                                                   joint_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                   "damping_factors parameter is missing",
                                                   joint_position_error.dtype,
                                                   num_envs,
                                                   joint_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = joint_dim;
  prepare_result.outputs.push_back(internal::matrix_descriptor("joint_torque",
                                                               joint_position_error.dtype,
                                                               num_envs,
                                                               joint_dim,
                                                               "newton_meter",
                                                               "joint_space",
                                                               "joint impedance torque"));
  return prepare_result;
}

Result apply_joint_impedance_controller_parameters(const ParameterSetView& parameters,
                                                   const std::size_t num_envs,
                                                   const std::size_t joint_dim,
                                                   MutableStatusView parameter_status) {
  if (joint_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_joint_impedance_parameters_cuda(parameters, num_envs, joint_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const ParameterView* parameter = find_parameter(parameters, kStiffnessMatrix);
  if (parameter == nullptr) {
    return internal::invalid_view("stiffness_matrix parameter is missing");
  }
  if (parameter->values.dtype == DType::kFloat32) {
    return apply_joint_impedance_parameters_impl<float>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  if (parameter->values.dtype == DType::kFloat64) {
    return apply_joint_impedance_parameters_impl<double>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  return internal::invalid_view("stiffness_matrix dtype must be float32 or float64");
}

Result reset_joint_impedance_controller(const std::size_t num_envs,
                                        const TensorView& reset_mask,
                                        MutableStatusView status) {
  return reset_stateless_joint(ComponentMode::kDense, num_envs, reset_mask, status);
}

template <typename Scalar>
Result step_joint_impedance_controller_impl(const TensorView& joint_position_error,
                                            const TensorView& joint_velocity_error,
                                            const TensorView& desired_acceleration,
                                            const TensorView& joint_mass_matrix,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            const MutableTensorView& joint_torque,
                                            MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_position_error, "joint_position_error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor joint impedance bound");
  }
  std::size_t other_envs = 0;
  std::size_t other_dim = 0;
  result = internal::validate_batched_signal(
      joint_velocity_error, "joint_velocity_error tensor data is null", &other_envs, &other_dim);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      joint_velocity_error.dtype != joint_position_error.dtype) {
    return internal::invalid_view("joint_velocity_error must match joint_position_error");
  }
  result = internal::validate_batched_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &other_envs, &other_dim);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      desired_acceleration.dtype != joint_position_error.dtype) {
    return internal::invalid_view("desired_acceleration must match joint_position_error");
  }
  result = internal::validate_batched_matrix(joint_mass_matrix,
                                             "joint_mass_matrix tensor data is null",
                                             joint_position_error.dtype,
                                             num_envs,
                                             joint_dim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(
      joint_torque, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = apply_joint_impedance_parameters_impl<Scalar>(parameters, num_envs, joint_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* stiffness_parameter = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping_factors_parameter = find_parameter(parameters, kDampingFactors);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(joint_position_error, env, joint_dim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity_error, env, joint_dim) ||
        !internal::matrix_env_is_finite<Scalar>(desired_acceleration, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(joint_mass_matrix, env, joint_dim, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    Scalar mass[kMaxJointDim][kMaxJointDim]{};
    Scalar stiffness[kMaxJointDim][kMaxJointDim]{};
    Scalar damping_matrix[kMaxJointDim][kMaxJointDim]{};
    Scalar damping_factors[kMaxJointDim]{};
    Scalar position_error[kMaxJointDim]{};
    Scalar velocity_error[kMaxJointDim]{};
    Scalar acceleration[kMaxJointDim]{};
    Scalar stiffness_torque[kMaxJointDim]{};
    Scalar damping_torque[kMaxJointDim]{};
    Scalar acceleration_torque[kMaxJointDim]{};
    Scalar torque[kMaxJointDim]{};
    read_tensor3_square(joint_mass_matrix, env, mass, joint_dim);
    read_parameter_square(stiffness_parameter, env, stiffness, joint_dim);
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      damping_factors[dim] = internal::read_vector_parameter<Scalar>(
          damping_factors_parameter, env, dim, static_cast<Scalar>(0));
      position_error[dim] = internal::read_matrix_value<Scalar>(joint_position_error, env, dim);
      velocity_error[dim] = internal::read_matrix_value<Scalar>(joint_velocity_error, env, dim);
      acceleration[dim] = internal::read_matrix_value<Scalar>(desired_acceleration, env, dim);
    }
    if (!validate_and_symmetrize_external_n(mass, joint_dim) ||
        !validate_and_symmetrize_external_n(stiffness, joint_dim) ||
        !compute_double_diagonalization_damping_matrix(
            mass, stiffness, damping_factors, damping_matrix, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidInput) |
                           status_bit(StatusFlag::kInvalidParameter) |
                           status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    matvec_n(stiffness, position_error, stiffness_torque, joint_dim, joint_dim);
    matvec_n(damping_matrix, velocity_error, damping_torque, joint_dim, joint_dim);
    matvec_n(mass, acceleration, acceleration_torque, joint_dim, joint_dim);
    bool torque_is_finite = true;
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      torque[dim] = detail::joint_impedance_value(
          stiffness_torque[dim], damping_torque[dim], acceleration_torque[dim]);
      torque_is_finite = torque_is_finite && std::isfinite(torque[dim]);
    }
    if (!torque_is_finite) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidInput) |
                           status_bit(StatusFlag::kInvalidParameter) |
                           status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      internal::write_matrix_value(joint_torque, env, dim, torque[dim]);
    }
  }
  return internal::ok();
}

Result step_joint_impedance_controller(const TensorView& joint_position_error,
                                       const TensorView& joint_velocity_error,
                                       const TensorView& desired_acceleration,
                                       const TensorView& joint_mass_matrix,
                                       const ParameterSetView& parameters,
                                       StatusView parameter_status,
                                       const MutableTensorView& joint_torque,
                                       MutableStatusView status) {
  if (joint_lifecycle_uses_cuda(joint_position_error,
                                joint_velocity_error,
                                desired_acceleration,
                                joint_mass_matrix,
                                parameters,
                                parameter_status,
                                joint_torque,
                                status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_joint_impedance_controller_cuda(joint_position_error,
                                                joint_velocity_error,
                                                desired_acceleration,
                                                joint_mass_matrix,
                                                parameters,
                                                parameter_status,
                                                joint_torque,
                                                status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (joint_position_error.dtype == DType::kFloat32) {
    return step_joint_impedance_controller_impl<float>(joint_position_error,
                                                       joint_velocity_error,
                                                       desired_acceleration,
                                                       joint_mass_matrix,
                                                       parameters,
                                                       parameter_status,
                                                       joint_torque,
                                                       status);
  }
  if (joint_position_error.dtype == DType::kFloat64) {
    return step_joint_impedance_controller_impl<double>(joint_position_error,
                                                        joint_velocity_error,
                                                        desired_acceleration,
                                                        joint_mass_matrix,
                                                        parameters,
                                                        parameter_status,
                                                        joint_torque,
                                                        status);
  }
  return internal::invalid_view("joint_position_error dtype must be float32 or float64");
}

PrepareResult prepare_nullspace_controller(const TensorView& joint_position_error,
                                           const TensorView& joint_velocity,
                                           const TensorView& joint_mass_matrix,
                                           const TensorView& jacobian,
                                           const ParameterSetView& parameters,
                                           const MutableTensorView& joint_torque) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  if (joint_lifecycle_uses_cuda(joint_position_error,
                                joint_velocity,
                                joint_mass_matrix,
                                jacobian,
                                parameters,
                                joint_torque)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_joint_nullspace_prepare(joint_position_error,
                                                                  joint_velocity,
                                                                  joint_mass_matrix,
                                                                  jacobian,
                                                                  parameters,
                                                                  joint_torque,
                                                                  &num_envs,
                                                                  &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = joint_dim;
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("joint_torque",
                                    joint_position_error.dtype,
                                    num_envs,
                                    joint_dim,
                                    "newton_meter",
                                    "joint_space",
                                    "mass-weighted nullspace projected joint torque"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      joint_position_error, "joint_position_error tensor data is null", &num_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_dim > kMaxJointDim) {
    prepare_result.result = internal::invalid_view("joint_dim exceeds tensor nullspace bound");
    return prepare_result;
  }
  std::size_t other_envs = 0;
  std::size_t other_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &other_envs, &other_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      joint_velocity.dtype != joint_position_error.dtype) {
    prepare_result.result =
        internal::invalid_view("joint_velocity must match joint_position_error dtype and shape");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(joint_mass_matrix,
                                                            "joint_mass_matrix tensor data is null",
                                                            joint_position_error.dtype,
                                                            num_envs,
                                                            joint_dim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  const std::size_t task_dim =
      jacobian.shape[1] > 0 ? static_cast<std::size_t>(jacobian.shape[1]) : 0;
  if (task_dim == 0 || task_dim > kMaxTaskDim || task_dim > joint_dim) {
    prepare_result.result =
        internal::invalid_view("nullspace task_dim must be in [1, min(12, joint_dim)]");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(jacobian,
                                                            "jacobian tensor data is null",
                                                            joint_position_error.dtype,
                                                            num_envs,
                                                            task_dim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matching_output(
      joint_torque, joint_position_error.dtype, num_envs, joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   joint_position_error.dtype,
                                                   num_envs,
                                                   joint_dim,
                                                   joint_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kDampingMatrix),
                                                   "damping_matrix parameter is missing",
                                                   joint_position_error.dtype,
                                                   num_envs,
                                                   joint_dim,
                                                   joint_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = joint_dim;
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("joint_torque",
                                  joint_position_error.dtype,
                                  num_envs,
                                  joint_dim,
                                  "newton_meter",
                                  "joint_space",
                                  "mass-weighted nullspace projected joint torque"));
  return prepare_result;
}

Result apply_nullspace_controller_parameters(const ParameterSetView& parameters,
                                             const std::size_t num_envs,
                                             const std::size_t joint_dim,
                                             MutableStatusView parameter_status) {
  if (joint_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_nullspace_parameters_cuda(parameters, num_envs, joint_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const ParameterView* parameter = find_parameter(parameters, kStiffnessMatrix);
  if (parameter == nullptr) {
    return internal::invalid_view("stiffness_matrix parameter is missing");
  }
  if (parameter->values.dtype == DType::kFloat32) {
    return apply_nullspace_parameters_impl<float>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  if (parameter->values.dtype == DType::kFloat64) {
    return apply_nullspace_parameters_impl<double>(
        parameters, num_envs, joint_dim, parameter_status);
  }
  return internal::invalid_view("stiffness_matrix dtype must be float32 or float64");
}

Result reset_nullspace_controller(const std::size_t num_envs,
                                  const TensorView& reset_mask,
                                  MutableStatusView status) {
  return reset_stateless_joint(ComponentMode::kDense, num_envs, reset_mask, status);
}

template <typename Scalar>
Result step_nullspace_controller_impl(const TensorView& joint_position_error,
                                      const TensorView& joint_velocity,
                                      const TensorView& joint_mass_matrix,
                                      const TensorView& jacobian,
                                      const ParameterSetView& parameters,
                                      StatusView parameter_status,
                                      const MutableTensorView& joint_torque,
                                      MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  Result result = internal::validate_batched_signal(
      joint_position_error, "joint_position_error tensor data is null", &num_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_dim > kMaxJointDim) {
    return internal::invalid_view("joint_dim exceeds tensor nullspace bound");
  }
  std::size_t other_envs = 0;
  std::size_t other_dim = 0;
  result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &other_envs, &other_dim);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || other_dim != joint_dim ||
      joint_velocity.dtype != joint_position_error.dtype) {
    return internal::invalid_view("joint_velocity must match joint_position_error");
  }
  result = internal::validate_batched_matrix(joint_mass_matrix,
                                             "joint_mass_matrix tensor data is null",
                                             joint_position_error.dtype,
                                             num_envs,
                                             joint_dim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  const std::size_t task_dim =
      jacobian.shape[1] > 0 ? static_cast<std::size_t>(jacobian.shape[1]) : 0;
  if (task_dim == 0 || task_dim > kMaxTaskDim || task_dim > joint_dim) {
    return internal::invalid_view("nullspace task_dim must be in [1, min(12, joint_dim)]");
  }
  result = internal::validate_batched_matrix(jacobian,
                                             "jacobian tensor data is null",
                                             joint_position_error.dtype,
                                             num_envs,
                                             task_dim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(
      joint_torque, joint_position_error.dtype, num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = apply_nullspace_parameters_impl<Scalar>(parameters, num_envs, joint_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* stiffness_parameter = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping_parameter = find_parameter(parameters, kDampingMatrix);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(joint_position_error, env, joint_dim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(joint_mass_matrix, env, joint_dim, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, task_dim, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }

    Scalar mass[kMaxJointDim][kMaxJointDim]{};
    Scalar mass_inv[kMaxJointDim][kMaxJointDim]{};
    Scalar stiffness[kMaxJointDim][kMaxJointDim]{};
    Scalar damping[kMaxJointDim][kMaxJointDim]{};
    Scalar jac[kMaxJointDim][kMaxJointDim]{};
    Scalar jac_transpose[kMaxJointDim][kMaxJointDim]{};
    Scalar jm_inv[kMaxJointDim][kMaxJointDim]{};
    Scalar task_matrix[kMaxJointDim][kMaxJointDim]{};
    Scalar task_matrix_inv[kMaxJointDim][kMaxJointDim]{};
    Scalar position_error[kMaxJointDim]{};
    Scalar neg_velocity[kMaxJointDim]{};
    Scalar stiffness_torque[kMaxJointDim]{};
    Scalar damping_torque[kMaxJointDim]{};
    Scalar desired_torque[kMaxJointDim]{};
    Scalar mass_inverse_torque[kMaxJointDim]{};
    Scalar task_torque[kMaxJointDim]{};
    Scalar task_correction[kMaxJointDim]{};
    Scalar output[kMaxJointDim]{};
    read_tensor3_square(joint_mass_matrix, env, mass, joint_dim);
    read_parameter_square(stiffness_parameter, env, stiffness, joint_dim);
    read_parameter_square(damping_parameter, env, damping, joint_dim);
    zero_matrix_n(jac, task_dim, joint_dim);
    for (std::size_t row = 0; row < task_dim; ++row) {
      for (std::size_t col = 0; col < joint_dim; ++col) {
        jac[row][col] = internal::read_tensor3_value<Scalar>(jacobian, env, row, col);
      }
    }
    if (!inverse_n(mass, mass_inv, joint_dim)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      position_error[dim] = internal::read_matrix_value<Scalar>(joint_position_error, env, dim);
      neg_velocity[dim] = -internal::read_matrix_value<Scalar>(joint_velocity, env, dim);
    }
    matvec_n(stiffness, position_error, stiffness_torque, joint_dim, joint_dim);
    matvec_n(damping, neg_velocity, damping_torque, joint_dim, joint_dim);
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      desired_torque[dim] =
          detail::nullspace_feedback_value(stiffness_torque[dim], damping_torque[dim]);
    }

    for (std::size_t row = 0; row < joint_dim; ++row) {
      for (std::size_t col = 0; col < task_dim; ++col) {
        jac_transpose[row][col] = jac[col][row];
      }
    }
    multiply_n(jac, mass_inv, jm_inv, task_dim, joint_dim, joint_dim);
    multiply_n(jm_inv, jac_transpose, task_matrix, task_dim, joint_dim, task_dim);
    if (!inverse_n(task_matrix, task_matrix_inv, task_dim)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    // Torque-space projection N^T tau_0 = tau_0 - J^T Lambda J M^-1 tau_0.
    matvec_n(mass_inv, desired_torque, mass_inverse_torque, joint_dim, joint_dim);
    matvec_n(jac, mass_inverse_torque, task_torque, task_dim, joint_dim);
    matvec_n(task_matrix_inv, task_torque, task_correction, task_dim, task_dim);
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      Scalar correction = static_cast<Scalar>(0);
      for (std::size_t task_row = 0; task_row < task_dim; ++task_row) {
        correction += jac[task_row][joint] * task_correction[task_row];
      }
      output[joint] = detail::nullspace_projected_torque_value(desired_torque[joint], correction);
    }
    for (std::size_t dim = 0; dim < joint_dim; ++dim) {
      internal::write_matrix_value(joint_torque, env, dim, output[dim]);
    }
  }
  return internal::ok();
}

Result step_nullspace_controller(const TensorView& joint_position_error,
                                 const TensorView& joint_velocity,
                                 const TensorView& joint_mass_matrix,
                                 const TensorView& jacobian,
                                 const ParameterSetView& parameters,
                                 StatusView parameter_status,
                                 const MutableTensorView& joint_torque,
                                 MutableStatusView status) {
  if (joint_lifecycle_uses_cuda(joint_position_error,
                                joint_velocity,
                                joint_mass_matrix,
                                jacobian,
                                parameters,
                                parameter_status,
                                joint_torque,
                                status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_nullspace_controller_cuda(joint_position_error,
                                          joint_velocity,
                                          joint_mass_matrix,
                                          jacobian,
                                          parameters,
                                          parameter_status,
                                          joint_torque,
                                          status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (joint_position_error.dtype == DType::kFloat32) {
    return step_nullspace_controller_impl<float>(joint_position_error,
                                                 joint_velocity,
                                                 joint_mass_matrix,
                                                 jacobian,
                                                 parameters,
                                                 parameter_status,
                                                 joint_torque,
                                                 status);
  }
  if (joint_position_error.dtype == DType::kFloat64) {
    return step_nullspace_controller_impl<double>(joint_position_error,
                                                  joint_velocity,
                                                  joint_mass_matrix,
                                                  jacobian,
                                                  parameters,
                                                  parameter_status,
                                                  joint_torque,
                                                  status);
  }
  return internal::invalid_view("joint_position_error dtype must be float32 or float64");
}

template <typename Scalar>
Result step_bimanual_independent_nullspace_impl(const TensorView& joint_position_error,
                                                const TensorView& joint_velocity,
                                                const TensorView& joint_mass_matrix,
                                                const TensorView& arm1_jacobian,
                                                const TensorView& arm2_jacobian,
                                                const ParameterSetView& parameters,
                                                const StatusView parameter_status,
                                                const MutableTensorView& joint_torque,
                                                const MutableStatusView status) {
  constexpr std::size_t kArmJoints = 7;
  constexpr std::size_t kArmTasks = 6;
  constexpr std::size_t kJoints = 14;
  const std::size_t num_envs = static_cast<std::size_t>(joint_position_error.shape[0]);
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping = find_parameter(parameters, kDampingMatrix);
  internal::clear_status(status, num_envs);
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags == nullptr ? 0U : parameter_status.flags[env];
    for (std::size_t row = 0; row < kArmJoints; ++row) {
      for (std::size_t col = 0; col < kArmJoints; ++col) {
        if (internal::read_matrix_parameter<Scalar>(
                stiffness, env, row, kArmJoints + col, static_cast<Scalar>(0)) !=
                static_cast<Scalar>(0) ||
            internal::read_matrix_parameter<Scalar>(
                stiffness, env, kArmJoints + row, col, static_cast<Scalar>(0)) !=
                static_cast<Scalar>(0) ||
            internal::read_matrix_parameter<Scalar>(
                damping, env, row, kArmJoints + col, static_cast<Scalar>(0)) !=
                static_cast<Scalar>(0) ||
            internal::read_matrix_parameter<Scalar>(
                damping, env, kArmJoints + row, col, static_cast<Scalar>(0)) !=
                static_cast<Scalar>(0)) {
          flags |= status_bit(StatusFlag::kInvalidParameter);
        }
      }
    }
    if (!internal::matrix_env_is_finite<Scalar>(joint_position_error, env, kJoints) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity, env, kJoints) ||
        !internal::tensor3_env_is_finite<Scalar>(joint_mass_matrix, env, kJoints, kJoints) ||
        !internal::tensor3_env_is_finite<Scalar>(arm1_jacobian, env, kArmTasks, kArmJoints) ||
        !internal::tensor3_env_is_finite<Scalar>(arm2_jacobian, env, kArmTasks, kArmJoints)) {
      flags |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if (flags != 0U) {
      status.flags[env] = flags | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, kJoints);
      continue;
    }
    for (std::size_t arm = 0; arm < 2; ++arm) {
      const std::size_t offset = arm * kArmJoints;
      const TensorView& jacobian = arm == 0 ? arm1_jacobian : arm2_jacobian;
      Scalar mass[kMaxJointDim][kMaxJointDim]{};
      Scalar mass_inverse[kMaxJointDim][kMaxJointDim]{};
      Scalar task_matrix[kMaxJointDim][kMaxJointDim]{};
      Scalar task_inverse[kMaxJointDim][kMaxJointDim]{};
      Scalar desired_torque[kMaxJointDim]{};
      Scalar mass_inverse_torque[kMaxJointDim]{};
      Scalar task_torque[kMaxJointDim]{};
      Scalar task_correction[kMaxJointDim]{};
      for (std::size_t row = 0; row < kArmJoints; ++row) {
        for (std::size_t col = 0; col < kArmJoints; ++col) {
          mass[row][col] = internal::read_tensor3_value<Scalar>(
              joint_mass_matrix, env, offset + row, offset + col);
          desired_torque[row] +=
              internal::read_matrix_parameter<Scalar>(
                  stiffness, env, offset + row, offset + col, static_cast<Scalar>(0)) *
                  internal::read_matrix_value<Scalar>(joint_position_error, env, offset + col) -
              internal::read_matrix_parameter<Scalar>(
                  damping, env, offset + row, offset + col, static_cast<Scalar>(0)) *
                  internal::read_matrix_value<Scalar>(joint_velocity, env, offset + col);
        }
      }
      if (!inverse_n(mass, mass_inverse, kArmJoints)) {
        flags |= status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
        break;
      }
      for (std::size_t row = 0; row < kArmTasks; ++row) {
        for (std::size_t col = 0; col < kArmTasks; ++col) {
          for (std::size_t left = 0; left < kArmJoints; ++left) {
            const Scalar jacobian_left =
                internal::read_tensor3_value<Scalar>(jacobian, env, row, left);
            for (std::size_t right = 0; right < kArmJoints; ++right) {
              task_matrix[row][col] +=
                  jacobian_left * mass_inverse[left][right] *
                  internal::read_tensor3_value<Scalar>(jacobian, env, col, right);
            }
          }
        }
      }
      if (!inverse_n(task_matrix, task_inverse, kArmTasks)) {
        flags |= status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
        break;
      }
      matvec_n(mass_inverse, desired_torque, mass_inverse_torque, kArmJoints, kArmJoints);
      for (std::size_t row = 0; row < kArmTasks; ++row) {
        for (std::size_t col = 0; col < kArmJoints; ++col) {
          task_torque[row] += internal::read_tensor3_value<Scalar>(jacobian, env, row, col) *
                              mass_inverse_torque[col];
        }
      }
      matvec_n(task_inverse, task_torque, task_correction, kArmTasks, kArmTasks);
      for (std::size_t joint = 0; joint < kArmJoints; ++joint) {
        Scalar correction = static_cast<Scalar>(0);
        for (std::size_t row = 0; row < kArmTasks; ++row) {
          correction += internal::read_tensor3_value<Scalar>(jacobian, env, row, joint) *
                        task_correction[row];
        }
        internal::write_matrix_value(
            joint_torque, env, offset + joint, desired_torque[joint] - correction);
      }
    }
    if ((flags & status_bit(StatusFlag::kSafeMode)) != 0U) {
      internal::zero_output_env<Scalar>(joint_torque, env, kJoints);
    }
    status.flags[env] = flags;
  }
  return internal::ok();
}

Result step_bimanual_independent_nullspace_controller(const TensorView& joint_position_error,
                                                      const TensorView& joint_velocity,
                                                      const TensorView& joint_mass_matrix,
                                                      const TensorView& arm1_jacobian,
                                                      const TensorView& arm2_jacobian,
                                                      const ParameterSetView& parameters,
                                                      const StatusView parameter_status,
                                                      const MutableTensorView& joint_torque,
                                                      const MutableStatusView status) {
  if (joint_position_error.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_bimanual_independent_nullspace_controller_cuda(joint_position_error,
                                                               joint_velocity,
                                                               joint_mass_matrix,
                                                               arm1_jacobian,
                                                               arm2_jacobian,
                                                               parameters,
                                                               parameter_status,
                                                               joint_torque,
                                                               status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return joint_position_error.dtype == DType::kFloat32
             ? step_bimanual_independent_nullspace_impl<float>(joint_position_error,
                                                               joint_velocity,
                                                               joint_mass_matrix,
                                                               arm1_jacobian,
                                                               arm2_jacobian,
                                                               parameters,
                                                               parameter_status,
                                                               joint_torque,
                                                               status)
             : step_bimanual_independent_nullspace_impl<double>(joint_position_error,
                                                                joint_velocity,
                                                                joint_mass_matrix,
                                                                arm1_jacobian,
                                                                arm2_jacobian,
                                                                parameters,
                                                                parameter_status,
                                                                joint_torque,
                                                                status);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
