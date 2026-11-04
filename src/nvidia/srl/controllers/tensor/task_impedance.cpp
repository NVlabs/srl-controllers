/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file task_impedance.cpp
 * @brief CPU implementation and dispatch for bounded variable-dimension impedance.
 */

#include "tensor/task_impedance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#include "tensor/symmetric_matrix_policy.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/task_impedance_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr std::size_t kMaxTaskDim = 12;
constexpr std::size_t kMaxJointDim = 64;
constexpr const char* kStiffnessMatrix = "stiffness_matrix";
constexpr const char* kDampingFactors = "damping_factors";
constexpr const char* kPreviousMassMatrix = "previous_mass_matrix";
constexpr const char* kPreviousJacobian = "previous_jacobian";
constexpr const char* kInitialized = "initialized";

bool parameters_have_cuda(const ParameterSetView& parameters) noexcept {
  for (std::size_t index = 0; index < parameters.parameter_count; ++index) {
    if (parameters.parameters[index].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

Result validate_signal(const TensorView& signal,
                       const DType dtype,
                       const Device device,
                       const std::size_t num_envs,
                       const std::size_t task_dim,
                       const char* message) noexcept {
  return signal.data != nullptr && signal.dtype == dtype && signal.device == device &&
                 internal::is_row_major_matrix(signal) &&
                 signal.shape[0] == static_cast<std::int64_t>(num_envs) &&
                 signal.shape[1] == static_cast<std::int64_t>(task_dim)
             ? internal::ok()
             : internal::invalid_view(message);
}

Result validate_problem(const TensorView& task_pose_error,
                        const TensorView& task_velocity_error,
                        const TensorView& desired_acceleration,
                        const TensorView& joint_velocity,
                        const TensorView& task_mass_matrix,
                        const TensorView& jacobian,
                        const TensorView& dt,
                        const MutableTensorView& task_wrench,
                        std::size_t* num_envs,
                        std::size_t* task_dim,
                        std::size_t* joint_dim) noexcept {
  if (task_pose_error.data == nullptr || !internal::is_float_dtype(task_pose_error.dtype) ||
      !internal::is_row_major_matrix(task_pose_error) || task_pose_error.shape[0] <= 0 ||
      task_pose_error.shape[1] <= 0) {
    return internal::invalid_view("task_pose_error must be [num_envs,task_dim]");
  }
  *num_envs = static_cast<std::size_t>(task_pose_error.shape[0]);
  *task_dim = static_cast<std::size_t>(task_pose_error.shape[1]);
  if (*task_dim == 0 || *task_dim > kMaxTaskDim) {
    return internal::invalid_argument("task_dim must be in [1,12]");
  }
  const Device device = task_pose_error.device;
  if (device != Device::kCpu && device != Device::kCuda) {
    return internal::invalid_view("task impedance device is unsupported");
  }
  Result result = validate_signal(task_velocity_error,
                                  task_pose_error.dtype,
                                  device,
                                  *num_envs,
                                  *task_dim,
                                  "task_velocity_error must match task_pose_error");
  if (!result.ok()) {
    return result;
  }
  result = validate_signal(desired_acceleration,
                           task_pose_error.dtype,
                           device,
                           *num_envs,
                           *task_dim,
                           "desired_acceleration must match task_pose_error");
  if (!result.ok()) {
    return result;
  }
  if (joint_velocity.data == nullptr || joint_velocity.dtype != task_pose_error.dtype ||
      joint_velocity.device != device || !internal::is_row_major_matrix(joint_velocity) ||
      joint_velocity.shape[0] != static_cast<std::int64_t>(*num_envs) ||
      joint_velocity.shape[1] <= 0 ||
      joint_velocity.shape[1] > static_cast<std::int64_t>(kMaxJointDim)) {
    return internal::invalid_view("joint_velocity must be [num_envs,joint_dim] on the task device");
  }
  *joint_dim = static_cast<std::size_t>(joint_velocity.shape[1]);
  if (task_mass_matrix.data == nullptr || task_mass_matrix.dtype != task_pose_error.dtype ||
      task_mass_matrix.device != device || !internal::is_row_major_tensor3(task_mass_matrix) ||
      task_mass_matrix.shape[0] != static_cast<std::int64_t>(*num_envs) ||
      task_mass_matrix.shape[1] != static_cast<std::int64_t>(*task_dim) ||
      task_mass_matrix.shape[2] != static_cast<std::int64_t>(*task_dim)) {
    return internal::invalid_view("task_mass_matrix must be [num_envs,task_dim,task_dim]");
  }
  if (jacobian.data == nullptr || jacobian.dtype != task_pose_error.dtype ||
      jacobian.device != device || !internal::is_row_major_tensor3(jacobian) ||
      jacobian.shape[0] != static_cast<std::int64_t>(*num_envs) ||
      jacobian.shape[1] != static_cast<std::int64_t>(*task_dim) ||
      jacobian.shape[2] != static_cast<std::int64_t>(*joint_dim)) {
    return internal::invalid_view("jacobian must be [num_envs,task_dim,joint_dim]");
  }
  const bool dt_shape_valid =
      dt.rank == 0 ||
      (dt.rank == 1 && dt.shape[0] >= static_cast<std::int64_t>(*num_envs) && dt.strides[0] >= 1);
  if (dt.data == nullptr || dt.dtype != task_pose_error.dtype || dt.device != device ||
      !dt_shape_valid) {
    return internal::invalid_view("dt must be a scalar batch on the task device");
  }
  if (task_wrench.data == nullptr || task_wrench.dtype != task_pose_error.dtype ||
      task_wrench.device != device || !internal::is_row_major_matrix(task_wrench) ||
      task_wrench.shape[0] != static_cast<std::int64_t>(*num_envs) ||
      task_wrench.shape[1] != static_cast<std::int64_t>(*task_dim)) {
    return internal::invalid_view("task_wrench must be [num_envs,task_dim]");
  }
  return internal::ok();
}

Result validate_parameters(const ParameterSetView& parameters,
                           const DType dtype,
                           const Device device,
                           const std::size_t num_envs,
                           const std::size_t task_dim) noexcept {
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping = find_parameter(parameters, kDampingFactors);
  if (stiffness == nullptr) {
    return internal::invalid_view("stiffness_matrix is missing");
  }
  const TensorView& stiffness_values = stiffness->values;
  const bool stiffness_shape_valid =
      stiffness->shared
          ? stiffness_values.rank == 2 && stiffness_values.layout == Layout::kRowMajor &&
                stiffness_values.shape[0] == static_cast<std::int64_t>(task_dim) &&
                stiffness_values.shape[1] == static_cast<std::int64_t>(task_dim) &&
                stiffness_values.strides[1] == 1 &&
                stiffness_values.strides[0] >= stiffness_values.shape[1]
          : internal::is_row_major_tensor3(stiffness_values) &&
                stiffness_values.shape[0] == static_cast<std::int64_t>(num_envs) &&
                stiffness_values.shape[1] == static_cast<std::int64_t>(task_dim) &&
                stiffness_values.shape[2] == static_cast<std::int64_t>(task_dim);
  if (stiffness_values.data == nullptr || stiffness_values.dtype != dtype ||
      stiffness_values.device != device || !stiffness_shape_valid) {
    return internal::invalid_view("stiffness_matrix must match task dtype, device, and dimension");
  }
  if (damping == nullptr) {
    return internal::invalid_view("damping_factors is missing");
  }
  const TensorView& damping_values = damping->values;
  const bool damping_shape_valid =
      damping->shared ? damping_values.rank == 1 &&
                            damping_values.shape[0] == static_cast<std::int64_t>(task_dim) &&
                            damping_values.strides[0] >= 1
                      : damping_values.rank == 2 &&
                            damping_values.shape[0] == static_cast<std::int64_t>(num_envs) &&
                            damping_values.shape[1] == static_cast<std::int64_t>(task_dim) &&
                            damping_values.strides[1] >= 1 &&
                            damping_values.strides[0] >= damping_values.shape[1];
  if (damping_values.data == nullptr || damping_values.dtype != dtype ||
      damping_values.device != device || !damping_shape_valid) {
    return internal::invalid_view("damping_factors must match task dtype, device, and dimension");
  }
  return internal::ok();
}

Result validate_state(StateView& state,
                      const DType dtype,
                      const Device device,
                      const std::size_t num_envs,
                      const std::size_t task_dim,
                      const std::size_t joint_dim) noexcept {
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return internal::invalid_view("task impedance state entries are missing");
  }
  const MutableTensorView& mass = previous_mass->values;
  const MutableTensorView& jacobian = previous_jacobian->values;
  const MutableTensorView& mask = initialized->values;
  if (mass.data == nullptr || mass.dtype != dtype || mass.device != device ||
      !internal::is_row_major_tensor3(mass) ||
      mass.shape[0] != static_cast<std::int64_t>(num_envs) ||
      mass.shape[1] != static_cast<std::int64_t>(task_dim) ||
      mass.shape[2] != static_cast<std::int64_t>(task_dim)) {
    return internal::invalid_view("previous_mass_matrix state has an invalid shape");
  }
  if (jacobian.data == nullptr || jacobian.dtype != dtype || jacobian.device != device ||
      !internal::is_row_major_tensor3(jacobian) ||
      jacobian.shape[0] != static_cast<std::int64_t>(num_envs) ||
      jacobian.shape[1] != static_cast<std::int64_t>(task_dim) ||
      jacobian.shape[2] != static_cast<std::int64_t>(joint_dim)) {
    return internal::invalid_view("previous_jacobian state has an invalid shape");
  }
  return mask.data != nullptr && mask.dtype == DType::kBool && mask.device == device &&
                 mask.rank == 1 && mask.shape[0] >= static_cast<std::int64_t>(num_envs) &&
                 mask.strides[0] >= 1
             ? internal::ok()
             : internal::invalid_view("initialized state must be bool [num_envs]");
}

TensorDescriptor tensor3_descriptor(const char* name,
                                    const DType dtype,
                                    const Device device,
                                    const std::size_t num_envs,
                                    const std::size_t rows,
                                    const std::size_t cols,
                                    const char* units,
                                    const char* semantics) {
  TensorDescriptor descriptor;
  descriptor.name = name;
  descriptor.dtype = dtype;
  descriptor.device = device;
  descriptor.layout = Layout::kRowMajor;
  descriptor.rank = 3;
  descriptor.shape = {static_cast<std::int64_t>(num_envs),
                      static_cast<std::int64_t>(rows),
                      static_cast<std::int64_t>(cols),
                      0};
  descriptor.strides = {
      static_cast<std::int64_t>(rows * cols), static_cast<std::int64_t>(cols), 1, 0};
  descriptor.units = units;
  descriptor.frame = "task_space";
  descriptor.semantics = semantics;
  return descriptor;
}

template <typename Scalar>
bool parameter_values_valid(const ParameterSetView& parameters,
                            const std::size_t env,
                            const std::size_t task_dim) noexcept {
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping = find_parameter(parameters, kDampingFactors);
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < task_dim; ++col) {
      const Scalar value =
          internal::read_matrix_parameter<Scalar>(stiffness, env, row, col, static_cast<Scalar>(0));
      if (!std::isfinite(value)) {
        return false;
      }
    }
    const Scalar damping_value =
        internal::read_vector_parameter<Scalar>(damping, env, row, static_cast<Scalar>(-1));
    if (!std::isfinite(damping_value) || damping_value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void zero_output(const MutableTensorView& output,
                 const std::size_t env,
                 const std::size_t task_dim) noexcept {
  for (std::size_t dim = 0; dim < task_dim; ++dim) {
    internal::write_matrix_value(output, env, dim, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
void identity(Scalar matrix[kMaxTaskDim][kMaxTaskDim], const std::size_t dim) noexcept {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      matrix[row][col] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
void transpose(const Scalar input[kMaxTaskDim][kMaxTaskDim],
               Scalar output[kMaxTaskDim][kMaxTaskDim],
               const std::size_t dim) noexcept {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      output[row][col] = input[col][row];
    }
  }
}

template <typename Scalar>
void multiply(const Scalar left[kMaxTaskDim][kMaxTaskDim],
              const Scalar right[kMaxTaskDim][kMaxTaskDim],
              Scalar output[kMaxTaskDim][kMaxTaskDim],
              const std::size_t dim) noexcept {
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < dim; ++inner) {
        value += left[row][inner] * right[inner][col];
      }
      output[row][col] = value;
    }
  }
}

template <typename Scalar>
bool validate_and_symmetrize(Scalar matrix[kMaxTaskDim][kMaxTaskDim],
                             const std::size_t dim) noexcept {
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
      const Scalar value = static_cast<Scalar>(0.5) * (matrix[row][col] + matrix[col][row]);
      matrix[row][col] = value;
      matrix[col][row] = value;
    }
  }
  return true;
}

template <typename Scalar>
bool symmetric_eigen(const Scalar input[kMaxTaskDim][kMaxTaskDim],
                     Scalar eigenvectors[kMaxTaskDim][kMaxTaskDim],
                     Scalar eigenvalues[kMaxTaskDim],
                     Scalar* matrix_scale,
                     const std::size_t dim) noexcept {
  Scalar matrix[kMaxTaskDim][kMaxTaskDim]{};
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
  scale = scale == static_cast<Scalar>(0) ? static_cast<Scalar>(1) : scale;
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      matrix[row][col] = input[row][col] / scale;
    }
  }
  identity(eigenvectors, dim);
  bool converged = false;
  const std::size_t max_iterations = 16 * dim * dim;
  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = dim > 1 ? 1 : 0;
    Scalar maximum = dim > 1 ? std::fabs(matrix[0][1]) : static_cast<Scalar>(0);
    for (std::size_t row = 0; row < dim; ++row) {
      for (std::size_t col = row + 1; col < dim; ++col) {
        const Scalar candidate = std::fabs(matrix[row][col]);
        if (candidate > maximum) {
          maximum = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }
    if (maximum <= internal::SymmetricMatrixPolicy<Scalar>::jacobi_relative_tolerance(dim)) {
      converged = true;
      break;
    }
    const Scalar app = matrix[pivot_row][pivot_row];
    const Scalar aqq = matrix[pivot_col][pivot_col];
    const Scalar apq = matrix[pivot_row][pivot_col];
    const Scalar angle =
        static_cast<Scalar>(0.5) * std::atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    for (std::size_t index = 0; index < dim; ++index) {
      if (index == pivot_row || index == pivot_col) {
        continue;
      }
      const Scalar aip = matrix[index][pivot_row];
      const Scalar aiq = matrix[index][pivot_col];
      matrix[index][pivot_row] = cosine * aip - sine * aiq;
      matrix[pivot_row][index] = matrix[index][pivot_row];
      matrix[index][pivot_col] = sine * aip + cosine * aiq;
      matrix[pivot_col][index] = matrix[index][pivot_col];
    }
    matrix[pivot_row][pivot_row] =
        cosine * cosine * app - static_cast<Scalar>(2) * sine * cosine * apq + sine * sine * aqq;
    matrix[pivot_col][pivot_col] =
        sine * sine * app + static_cast<Scalar>(2) * sine * cosine * apq + cosine * cosine * aqq;
    matrix[pivot_row][pivot_col] = static_cast<Scalar>(0);
    matrix[pivot_col][pivot_row] = static_cast<Scalar>(0);
    for (std::size_t row = 0; row < dim; ++row) {
      const Scalar vip = eigenvectors[row][pivot_row];
      const Scalar viq = eigenvectors[row][pivot_col];
      eigenvectors[row][pivot_row] = cosine * vip - sine * viq;
      eigenvectors[row][pivot_col] = sine * vip + cosine * viq;
    }
  }
  if (!converged) {
    return false;
  }
  for (std::size_t index = 0; index < dim; ++index) {
    eigenvalues[index] = matrix[index][index];
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
bool matrix_roots(const Scalar input[kMaxTaskDim][kMaxTaskDim],
                  Scalar output[kMaxTaskDim][kMaxTaskDim],
                  Scalar inverse_output[kMaxTaskDim][kMaxTaskDim],
                  const bool compute_inverse,
                  const bool strict_positive,
                  const std::size_t dim) noexcept {
  Scalar eigenvectors[kMaxTaskDim][kMaxTaskDim]{};
  Scalar eigenvalues[kMaxTaskDim]{};
  Scalar scale{};
  if (!symmetric_eigen(input, eigenvectors, eigenvalues, &scale, dim)) {
    return false;
  }
  Scalar scaled[kMaxTaskDim][kMaxTaskDim]{};
  Scalar inverse_scaled[kMaxTaskDim][kMaxTaskDim]{};
  Scalar maximum = static_cast<Scalar>(0);
  for (std::size_t index = 0; index < dim; ++index) {
    maximum = std::max(maximum, std::fabs(eigenvalues[index]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(dim) * maximum;
  for (std::size_t col = 0; col < dim; ++col) {
    if ((strict_positive && eigenvalues[col] <= threshold) || eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar root =
        std::sqrt(scale) * std::sqrt(std::max(eigenvalues[col], static_cast<Scalar>(0)));
    if (compute_inverse && (!std::isfinite(root) || root <= static_cast<Scalar>(0))) {
      return false;
    }
    for (std::size_t row = 0; row < dim; ++row) {
      scaled[row][col] = eigenvectors[row][col] * root;
      if (compute_inverse) {
        inverse_scaled[row][col] = eigenvectors[row][col] / root;
      }
    }
  }
  Scalar eigenvectors_transpose[kMaxTaskDim][kMaxTaskDim]{};
  transpose(eigenvectors, eigenvectors_transpose, dim);
  multiply(scaled, eigenvectors_transpose, output, dim);
  if (compute_inverse) {
    multiply(inverse_scaled, eigenvectors_transpose, inverse_output, dim);
  }
  return true;
}

template <typename Scalar>
bool damping_matrix(Scalar mass[kMaxTaskDim][kMaxTaskDim],
                    Scalar stiffness[kMaxTaskDim][kMaxTaskDim],
                    const Scalar factors[kMaxTaskDim],
                    Scalar damping[kMaxTaskDim][kMaxTaskDim],
                    const std::size_t dim) noexcept {
  if (!validate_and_symmetrize(mass, dim) || !validate_and_symmetrize(stiffness, dim)) {
    return false;
  }
  Scalar mass_root[kMaxTaskDim][kMaxTaskDim]{};
  Scalar mass_inverse_root[kMaxTaskDim][kMaxTaskDim]{};
  if (!matrix_roots(mass, mass_root, mass_inverse_root, true, true, dim)) {
    return false;
  }
  Scalar inverse_root_transpose[kMaxTaskDim][kMaxTaskDim]{};
  Scalar temporary[kMaxTaskDim][kMaxTaskDim]{};
  Scalar transformed_stiffness[kMaxTaskDim][kMaxTaskDim]{};
  transpose(mass_inverse_root, inverse_root_transpose, dim);
  multiply(mass_inverse_root, stiffness, temporary, dim);
  multiply(temporary, inverse_root_transpose, transformed_stiffness, dim);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = row + 1; col < dim; ++col) {
      const Scalar value = static_cast<Scalar>(0.5) *
                           (transformed_stiffness[row][col] + transformed_stiffness[col][row]);
      transformed_stiffness[row][col] = value;
      transformed_stiffness[col][row] = value;
    }
  }
  Scalar transformed_root[kMaxTaskDim][kMaxTaskDim]{};
  Scalar unused[kMaxTaskDim][kMaxTaskDim]{};
  if (!matrix_roots(transformed_stiffness, transformed_root, unused, false, false, dim)) {
    return false;
  }
  Scalar scaled_root[kMaxTaskDim][kMaxTaskDim]{};
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      scaled_root[row][col] = factors[row] * transformed_root[row][col];
    }
  }
  Scalar left[kMaxTaskDim][kMaxTaskDim]{};
  Scalar mass_root_transpose[kMaxTaskDim][kMaxTaskDim]{};
  multiply(mass_root, scaled_root, left, dim);
  transpose(mass_root, mass_root_transpose, dim);
  multiply(left, mass_root_transpose, damping, dim);
  for (std::size_t row = 0; row < dim; ++row) {
    for (std::size_t col = 0; col < dim; ++col) {
      damping[row][col] = detail::damping_double_diagonalization_value(damping[row][col]);
      if (!std::isfinite(damping[row][col])) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
Result apply_cpu(const ParameterSetView& parameters,
                 const std::size_t num_envs,
                 const std::size_t task_dim,
                 const MutableStatusView status) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] = parameter_values_valid<Scalar>(parameters, env, task_dim)
                            ? 0U
                            : status_bit(StatusFlag::kInvalidParameter);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_cpu(const TensorView& pose_error,
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
                const MutableStatusView status,
                const std::size_t num_envs,
                const std::size_t task_dim,
                const std::size_t joint_dim) noexcept {
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_data = static_cast<std::uint8_t*>(initialized->values.data);
  const ParameterView* stiffness_parameter = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping_parameter = find_parameter(parameters, kDampingFactors);
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags == nullptr ? 0U : parameter_status.flags[env];
    if (!parameter_values_valid<Scalar>(parameters, env, task_dim)) {
      flags |= status_bit(StatusFlag::kInvalidParameter);
    }
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      flags |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(pose_error, env, task_dim) ||
        !internal::matrix_env_is_finite<Scalar>(velocity_error, env, task_dim) ||
        !internal::matrix_env_is_finite<Scalar>(desired_acceleration, env, task_dim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(mass_matrix, env, task_dim, task_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, task_dim, joint_dim)) {
      flags |= status_bit(StatusFlag::kNonFiniteInput);
    }
    Scalar mass[kMaxTaskDim][kMaxTaskDim]{};
    Scalar stiffness[kMaxTaskDim][kMaxTaskDim]{};
    Scalar factors[kMaxTaskDim]{};
    for (std::size_t row = 0; row < task_dim; ++row) {
      factors[row] = internal::read_vector_parameter<Scalar>(
          damping_parameter, env, row, static_cast<Scalar>(-1));
      for (std::size_t col = 0; col < task_dim; ++col) {
        mass[row][col] = internal::read_tensor3_value<Scalar>(mass_matrix, env, row, col);
        stiffness[row][col] = internal::read_matrix_parameter<Scalar>(
            stiffness_parameter, env, row, col, static_cast<Scalar>(0));
      }
      if (mass[row][row] <= static_cast<Scalar>(0) ||
          stiffness[row][row] < static_cast<Scalar>(0)) {
        flags |= status_bit(StatusFlag::kInvalidInput);
      }
    }
    if ((flags &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] = flags | status_bit(StatusFlag::kSafeMode);
      zero_output<Scalar>(wrench, env, task_dim);
      continue;
    }
    Scalar damping[kMaxTaskDim][kMaxTaskDim]{};
    if (!damping_matrix(mass, stiffness, factors, damping, task_dim)) {
      status.flags[env] =
          flags | status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_output<Scalar>(wrench, env, task_dim);
      continue;
    }
    Scalar stiffness_wrench[kMaxTaskDim]{};
    Scalar damping_wrench[kMaxTaskDim]{};
    Scalar acceleration_wrench[kMaxTaskDim]{};
    Scalar inertial_wrench[kMaxTaskDim]{};
    for (std::size_t row = 0; row < task_dim; ++row) {
      for (std::size_t col = 0; col < task_dim; ++col) {
        stiffness_wrench[row] +=
            stiffness[row][col] * internal::read_matrix_value<Scalar>(pose_error, env, col);
        damping_wrench[row] +=
            damping[row][col] * internal::read_matrix_value<Scalar>(velocity_error, env, col);
        acceleration_wrench[row] +=
            mass[row][col] * internal::read_matrix_value<Scalar>(desired_acceleration, env, col);
      }
    }
    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    if (initialized_data[initialized_offset] == 0U) {
      initialized_data[initialized_offset] = 1U;
    } else {
      Scalar jacobian_derivative_dq[kMaxTaskDim]{};
      for (std::size_t row = 0; row < task_dim; ++row) {
        for (std::size_t joint = 0; joint < joint_dim; ++joint) {
          jacobian_derivative_dq[row] +=
              detail::matrix_derivative_value(
                  internal::read_tensor3_value<Scalar>(jacobian, env, row, joint),
                  internal::read_tensor3_value<Scalar>(previous_jacobian->values, env, row, joint),
                  dt_value) *
              internal::read_matrix_value<Scalar>(joint_velocity, env, joint);
        }
      }
      for (std::size_t row = 0; row < task_dim; ++row) {
        Scalar mass_derivative_term = static_cast<Scalar>(0);
        Scalar jacobian_term = static_cast<Scalar>(0);
        for (std::size_t col = 0; col < task_dim; ++col) {
          mass_derivative_term +=
              detail::matrix_derivative_value(
                  internal::read_tensor3_value<Scalar>(mass_matrix, env, row, col),
                  internal::read_tensor3_value<Scalar>(previous_mass->values, env, row, col),
                  dt_value) *
              internal::read_matrix_value<Scalar>(velocity_error, env, col);
          jacobian_term += mass[row][col] * jacobian_derivative_dq[col];
        }
        inertial_wrench[row] = detail::inertial_coupling_value(mass_derivative_term, jacobian_term);
      }
    }
    for (std::size_t row = 0; row < task_dim; ++row) {
      for (std::size_t col = 0; col < task_dim; ++col) {
        internal::write_tensor3_value(
            previous_mass->values,
            env,
            row,
            col,
            internal::read_tensor3_value<Scalar>(mass_matrix, env, row, col));
      }
      for (std::size_t joint = 0; joint < joint_dim; ++joint) {
        internal::write_tensor3_value(
            previous_jacobian->values,
            env,
            row,
            joint,
            internal::read_tensor3_value<Scalar>(jacobian, env, row, joint));
      }
    }
    bool finite = true;
    for (std::size_t row = 0; row < task_dim; ++row) {
      const Scalar value = detail::cartesian_impedance_value(stiffness_wrench[row],
                                                             damping_wrench[row],
                                                             inertial_wrench[row],
                                                             acceleration_wrench[row]);
      finite = finite && std::isfinite(value);
      internal::write_matrix_value(wrench, env, row, value);
    }
    if (!finite) {
      flags |= status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_output<Scalar>(wrench, env, task_dim);
    }
    status.flags[env] = flags;
  }
  return internal::ok();
}

}  // namespace

PrepareResult prepare_task_impedance_controller(const TensorView& task_pose_error,
                                                const TensorView& task_velocity_error,
                                                const TensorView& desired_acceleration,
                                                const TensorView& joint_velocity,
                                                const TensorView& task_mass_matrix,
                                                const TensorView& jacobian,
                                                const TensorView& dt,
                                                const ParameterSetView& parameters,
                                                StateView& state,
                                                const MutableTensorView& task_wrench) {
  PrepareResult prepared;
  std::size_t num_envs = 0;
  std::size_t task_dim = 0;
  std::size_t joint_dim = 0;
  prepared.result = validate_problem(task_pose_error,
                                     task_velocity_error,
                                     desired_acceleration,
                                     joint_velocity,
                                     task_mass_matrix,
                                     jacobian,
                                     dt,
                                     task_wrench,
                                     &num_envs,
                                     &task_dim,
                                     &joint_dim);
  if (!prepared.result.ok()) {
    return prepared;
  }
  prepared.result = validate_parameters(
      parameters, task_pose_error.dtype, task_pose_error.device, num_envs, task_dim);
  if (!prepared.result.ok()) {
    return prepared;
  }
  prepared.result = validate_state(
      state, task_pose_error.dtype, task_pose_error.device, num_envs, task_dim, joint_dim);
  if (!prepared.result.ok()) {
    return prepared;
  }
  prepared.num_envs = num_envs;
  prepared.signal_dim = task_dim;
  prepared.state.push_back(tensor3_descriptor(kPreviousMassMatrix,
                                              task_pose_error.dtype,
                                              task_pose_error.device,
                                              num_envs,
                                              task_dim,
                                              task_dim,
                                              "operational_space_mass_units",
                                              "previous task mass matrix"));
  prepared.state.push_back(tensor3_descriptor(kPreviousJacobian,
                                              task_pose_error.dtype,
                                              task_pose_error.device,
                                              num_envs,
                                              task_dim,
                                              joint_dim,
                                              "task_velocity_per_joint_velocity",
                                              "previous task Jacobian"));
  TensorDescriptor initialized;
  initialized.name = kInitialized;
  initialized.dtype = DType::kBool;
  initialized.device = task_pose_error.device;
  initialized.layout = Layout::kRowMajor;
  initialized.rank = 1;
  initialized.shape = {static_cast<std::int64_t>(num_envs), 0, 0, 0};
  initialized.strides = {1, 0, 0, 0};
  initialized.units = "unitless";
  initialized.frame = "none";
  initialized.semantics = "per-environment initialized flag";
  prepared.state.push_back(initialized);
  prepared.outputs.push_back(internal::matrix_descriptor("task_wrench",
                                                         task_pose_error.dtype,
                                                         num_envs,
                                                         task_dim,
                                                         "newton_and_newton_meter",
                                                         "task_space",
                                                         "coupled task impedance wrench"));
  prepared.outputs.back().device = task_pose_error.device;
  return prepared;
}

Result apply_task_impedance_parameters(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       const std::size_t task_dim,
                                       const MutableStatusView parameter_status) noexcept {
  if (num_envs == 0 || task_dim == 0 || task_dim > kMaxTaskDim ||
      parameter_status.flags == nullptr || parameter_status.size < num_envs) {
    return internal::invalid_argument("invalid task impedance parameter dimensions or status");
  }
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  const DType dtype = stiffness == nullptr ? DType::kUnknown : stiffness->values.dtype;
  Result result =
      validate_parameters(parameters, dtype, parameter_status.device, num_envs, task_dim);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.device == Device::kCuda || parameters_have_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_task_impedance_parameters_cuda(parameters, num_envs, task_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return dtype == DType::kFloat32
             ? apply_cpu<float>(parameters, num_envs, task_dim, parameter_status)
             : apply_cpu<double>(parameters, num_envs, task_dim, parameter_status);
}

Result reset_task_impedance_controller(const std::size_t num_envs,
                                       const std::size_t task_dim,
                                       const std::size_t joint_dim,
                                       StateView& state,
                                       const TensorView& reset_mask,
                                       const MutableStatusView status) noexcept {
  if (num_envs == 0 || task_dim == 0 || task_dim > kMaxTaskDim || joint_dim == 0 ||
      status.flags == nullptr || status.size < num_envs) {
    return internal::invalid_argument("invalid task impedance reset dimensions or status");
  }
  StateEntryView* initialized = find_state(state, kInitialized);
  if (initialized == nullptr) {
    return internal::invalid_view("initialized state is missing");
  }
  if (status.device == Device::kCuda || reset_mask.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_task_impedance_controller_cuda(
        num_envs, task_dim, joint_dim, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  auto* initialized_data = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    const bool selected = internal::reset_mask_value(reset_mask, env);
    if (selected) {
      initialized_data[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    }
    status.flags[env] = selected ? status_bit(StatusFlag::kResetApplied) : 0U;
  }
  return internal::ok();
}

Result step_task_impedance_controller(const TensorView& task_pose_error,
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
  std::size_t num_envs = 0;
  std::size_t task_dim = 0;
  std::size_t joint_dim = 0;
  Result result = validate_problem(task_pose_error,
                                   task_velocity_error,
                                   desired_acceleration,
                                   joint_velocity,
                                   task_mass_matrix,
                                   jacobian,
                                   dt,
                                   task_wrench,
                                   &num_envs,
                                   &task_dim,
                                   &joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_parameters(
      parameters, task_pose_error.dtype, task_pose_error.device, num_envs, task_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_state(
      state, task_pose_error.dtype, task_pose_error.device, num_envs, task_dim, joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (status.flags == nullptr || status.device != task_pose_error.device ||
      status.size < num_envs ||
      (parameter_status.flags != nullptr &&
       (parameter_status.device != task_pose_error.device || parameter_status.size < num_envs))) {
    return internal::invalid_view("task impedance status views do not match the input batch");
  }
  if (task_pose_error.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_task_impedance_controller_cuda(task_pose_error,
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
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return task_pose_error.dtype == DType::kFloat32 ? step_cpu<float>(task_pose_error,
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
                                                                    status,
                                                                    num_envs,
                                                                    task_dim,
                                                                    joint_dim)
                                                  : step_cpu<double>(task_pose_error,
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
                                                                     status,
                                                                     num_envs,
                                                                     task_dim,
                                                                     joint_dim);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
