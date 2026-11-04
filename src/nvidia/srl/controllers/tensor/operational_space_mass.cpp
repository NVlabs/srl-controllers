/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file operational_space_mass.cpp
 * @brief CPU tensor kernel and dispatch for operational-space mass calculation.
 */

#include "nvidia/srl/controllers/tensor/operational_space_mass.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/operational_space_mass_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr const char* kRegularization = "regularization";
constexpr const char* kMinimumPivot = "minimum_pivot";
constexpr const char* kJointMassFactor = "joint_mass_factor";
constexpr const char* kMassInverseJacobianTranspose = "mass_inverse_jacobian_transpose";
constexpr const char* kCartesianInverseMassFactor = "cartesian_inverse_mass_factor";

bool parameters_have_device(const ParameterSetView& parameters, const Device device) noexcept {
  for (std::size_t index = 0; index < parameters.parameter_count; ++index) {
    if (parameters.parameters[index].values.device == device) {
      return true;
    }
  }
  return false;
}

Result validate_scalar_parameter(const ParameterView* parameter,
                                 const DType dtype,
                                 const Device device,
                                 const std::size_t num_envs,
                                 const bool required) noexcept {
  if (parameter == nullptr) {
    return required ? internal::invalid_view("regularization parameter is required")
                    : internal::ok();
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr || values.dtype != dtype || values.device != device ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view("scalar parameters must match the input dtype and device");
  }
  if (parameter->shared) {
    return values.rank == 0 || (values.rank == 1 && values.shape[0] == 1 && values.strides[0] >= 1)
               ? internal::ok()
               : internal::invalid_view("shared scalar parameters must be scalar or shape [1]");
  }
  return values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
                 values.strides[0] >= 1
             ? internal::ok()
             : internal::invalid_view("batched scalar parameters must have shape [num_envs]");
}

Result validate_problem_views(const TensorView& joint_mass_matrix,
                              const TensorView& jacobian,
                              const MutableTensorView& cartesian_mass_matrix,
                              std::size_t* num_envs,
                              std::size_t* joint_dim,
                              std::size_t* task_dim) noexcept {
  if (joint_mass_matrix.data == nullptr || !internal::is_float_dtype(joint_mass_matrix.dtype) ||
      !internal::is_row_major_tensor3(joint_mass_matrix) || joint_mass_matrix.shape[0] <= 0 ||
      joint_mass_matrix.shape[1] <= 0 || joint_mass_matrix.shape[1] != joint_mass_matrix.shape[2]) {
    return internal::invalid_view(
        "joint_mass_matrix must be row-major [num_envs, joint_dim, joint_dim]");
  }
  *num_envs = static_cast<std::size_t>(joint_mass_matrix.shape[0]);
  *joint_dim = static_cast<std::size_t>(joint_mass_matrix.shape[1]);
  if (*joint_dim > kMaxOperationalSpaceJointDimension) {
    return internal::invalid_argument("joint_dim exceeds the bounded operational-space limit");
  }
  const Device device = joint_mass_matrix.device;
  if (device != Device::kCpu && device != Device::kCuda) {
    return internal::invalid_view("operational-space mass device is unsupported");
  }
  if (jacobian.data == nullptr || jacobian.dtype != joint_mass_matrix.dtype ||
      jacobian.device != device || !internal::is_row_major_tensor3(jacobian) ||
      jacobian.shape[0] != joint_mass_matrix.shape[0] || jacobian.shape[1] <= 0 ||
      jacobian.shape[2] != static_cast<std::int64_t>(*joint_dim)) {
    return internal::invalid_view("jacobian must be row-major [num_envs, task_dim, joint_dim]");
  }
  *task_dim = static_cast<std::size_t>(jacobian.shape[1]);
  if (*task_dim > kMaxOperationalSpaceDimension) {
    return internal::invalid_argument("task_dim exceeds the bounded operational-space limit");
  }
  if (cartesian_mass_matrix.data == nullptr ||
      cartesian_mass_matrix.dtype != joint_mass_matrix.dtype ||
      cartesian_mass_matrix.device != device ||
      !internal::is_row_major_tensor3(cartesian_mass_matrix) ||
      cartesian_mass_matrix.shape[0] != joint_mass_matrix.shape[0] ||
      cartesian_mass_matrix.shape[1] != static_cast<std::int64_t>(*task_dim) ||
      cartesian_mass_matrix.shape[2] != static_cast<std::int64_t>(*task_dim)) {
    return internal::invalid_view(
        "cartesian_mass_matrix must be row-major [num_envs, task_dim, task_dim]");
  }
  return internal::ok();
}

Result validate_status(const MutableStatusView status,
                       const Device device,
                       const std::size_t num_envs) noexcept {
  return status.flags != nullptr && status.device == device && status.size >= num_envs
             ? internal::ok()
             : internal::invalid_view("status must match the component device and num_envs");
}

Result validate_parameter_status(const StatusView status,
                                 const Device device,
                                 const std::size_t num_envs) noexcept {
  if (status.flags == nullptr) {
    return internal::ok();
  }
  return status.device == device && status.size >= num_envs
             ? internal::ok()
             : internal::invalid_view(
                   "parameter_status must match the component device and num_envs");
}

Result validate_scratch_tensor(const MutableTensorView& view,
                               const DType dtype,
                               const Device device,
                               const std::size_t num_envs,
                               const std::size_t rows,
                               const std::size_t cols) noexcept {
  return view.data != nullptr && view.dtype == dtype && view.device == device &&
                 internal::is_row_major_tensor3(view) &&
                 view.shape[0] == static_cast<std::int64_t>(num_envs) &&
                 view.shape[1] == static_cast<std::int64_t>(rows) &&
                 view.shape[2] == static_cast<std::int64_t>(cols)
             ? internal::ok()
             : internal::invalid_view("operational-space scratch view has an invalid shape");
}

Result find_and_validate_scratch(ScratchView& scratch,
                                 const DType dtype,
                                 const Device device,
                                 const std::size_t num_envs,
                                 const std::size_t joint_dim,
                                 const std::size_t task_dim,
                                 MutableTensorView** joint_factor,
                                 MutableTensorView** inverse_jacobian,
                                 MutableTensorView** cartesian_factor) noexcept {
  ScratchEntryView* joint_entry = find_scratch(scratch, kJointMassFactor);
  ScratchEntryView* inverse_entry = find_scratch(scratch, kMassInverseJacobianTranspose);
  ScratchEntryView* cartesian_entry = find_scratch(scratch, kCartesianInverseMassFactor);
  if (joint_entry == nullptr || inverse_entry == nullptr || cartesian_entry == nullptr) {
    return internal::invalid_view("operational-space mass scratch entries are missing");
  }
  Result result =
      validate_scratch_tensor(joint_entry->values, dtype, device, num_envs, joint_dim, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_scratch_tensor(inverse_entry->values, dtype, device, num_envs, joint_dim, task_dim);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_scratch_tensor(cartesian_entry->values, dtype, device, num_envs, task_dim, task_dim);
  if (!result.ok()) {
    return result;
  }
  *joint_factor = &joint_entry->values;
  *inverse_jacobian = &inverse_entry->values;
  *cartesian_factor = &cartesian_entry->values;
  return internal::ok();
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
  descriptor.frame = "controller contract";
  descriptor.semantics = semantics;
  return descriptor;
}

template <typename Scalar>
Scalar default_minimum_pivot() noexcept;

template <>
float default_minimum_pivot<float>() noexcept {
  return 1.0e-6F;
}

template <>
double default_minimum_pivot<double>() noexcept {
  return 1.0e-12;
}

template <typename Scalar>
bool parameters_are_valid(const ParameterSetView& parameters, const std::size_t env) noexcept {
  const Scalar regularization = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kRegularization), env, static_cast<Scalar>(-1));
  const Scalar minimum_pivot = internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, kMinimumPivot), env, default_minimum_pivot<Scalar>());
  return detail::operational_space_mass_parameters_valid(regularization, minimum_pivot);
}

template <typename Scalar>
void zero_cartesian_matrix(const MutableTensorView& output,
                           const std::size_t env,
                           const std::size_t task_dim) noexcept {
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < task_dim; ++col) {
      internal::write_tensor3_value(output, env, row, col, static_cast<Scalar>(0));
    }
  }
}

template <typename Scalar>
bool factor_joint_mass(const TensorView& mass,
                       const std::size_t env,
                       const std::size_t joint_dim,
                       const Scalar minimum_pivot,
                       const MutableTensorView& factor) noexcept {
  for (std::size_t row = 0; row < joint_dim; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      const Scalar forward = internal::read_tensor3_value<Scalar>(mass, env, row, col);
      const Scalar transpose = internal::read_tensor3_value<Scalar>(mass, env, col, row);
      if (!detail::symmetric_pair_is_valid(forward, transpose, minimum_pivot)) {
        return false;
      }
      Scalar value = detail::symmetric_average(forward, transpose);
      for (std::size_t inner = 0; inner < col; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, row, inner) *
                 internal::read_tensor3_value<Scalar>(factor, env, col, inner);
      }
      if (row == col) {
        if (!std::isfinite(value) || value <= minimum_pivot) {
          return false;
        }
        internal::write_tensor3_value(factor, env, row, col, std::sqrt(value));
      } else {
        const Scalar diagonal = internal::read_tensor3_value<Scalar>(factor, env, col, col);
        internal::write_tensor3_value(factor, env, row, col, value / diagonal);
      }
    }
    for (std::size_t col = row + 1; col < joint_dim; ++col) {
      internal::write_tensor3_value(factor, env, row, col, static_cast<Scalar>(0));
    }
  }
  return true;
}

template <typename Scalar>
void solve_mass_inverse_jacobian(const TensorView& jacobian,
                                 const MutableTensorView& factor,
                                 const std::size_t env,
                                 const std::size_t joint_dim,
                                 const std::size_t task_dim,
                                 const MutableTensorView& solution) noexcept {
  for (std::size_t cartesian_col = 0; cartesian_col < task_dim; ++cartesian_col) {
    for (std::size_t row = 0; row < joint_dim; ++row) {
      Scalar value = internal::read_tensor3_value<Scalar>(jacobian, env, cartesian_col, row);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, row, inner) *
                 internal::read_tensor3_value<Scalar>(solution, env, inner, cartesian_col);
      }
      value /= internal::read_tensor3_value<Scalar>(factor, env, row, row);
      internal::write_tensor3_value(solution, env, row, cartesian_col, value);
    }
    for (std::size_t reverse = joint_dim; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      Scalar value = internal::read_tensor3_value<Scalar>(solution, env, row, cartesian_col);
      for (std::size_t inner = row + 1; inner < joint_dim; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, inner, row) *
                 internal::read_tensor3_value<Scalar>(solution, env, inner, cartesian_col);
      }
      value /= internal::read_tensor3_value<Scalar>(factor, env, row, row);
      internal::write_tensor3_value(solution, env, row, cartesian_col, value);
    }
  }
}

template <typename Scalar>
bool factor_cartesian_inverse_mass(const TensorView& jacobian,
                                   const MutableTensorView& solution,
                                   const std::size_t env,
                                   const std::size_t joint_dim,
                                   const std::size_t task_dim,
                                   const Scalar regularization,
                                   const Scalar minimum_pivot,
                                   const MutableTensorView& factor) noexcept {
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      Scalar forward = static_cast<Scalar>(0);
      Scalar transpose = static_cast<Scalar>(0);
      for (std::size_t joint = 0; joint < joint_dim; ++joint) {
        forward += internal::read_tensor3_value<Scalar>(jacobian, env, row, joint) *
                   internal::read_tensor3_value<Scalar>(solution, env, joint, col);
        transpose += internal::read_tensor3_value<Scalar>(jacobian, env, col, joint) *
                     internal::read_tensor3_value<Scalar>(solution, env, joint, row);
      }
      Scalar value = detail::operational_space_inverse_mass_value(
          forward, transpose, regularization, row == col);
      for (std::size_t inner = 0; inner < col; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, row, inner) *
                 internal::read_tensor3_value<Scalar>(factor, env, col, inner);
      }
      if (row == col) {
        if (!std::isfinite(value) || value <= minimum_pivot) {
          return false;
        }
        internal::write_tensor3_value(factor, env, row, col, std::sqrt(value));
      } else {
        const Scalar diagonal = internal::read_tensor3_value<Scalar>(factor, env, col, col);
        internal::write_tensor3_value(factor, env, row, col, value / diagonal);
      }
    }
    for (std::size_t col = row + 1; col < task_dim; ++col) {
      internal::write_tensor3_value(factor, env, row, col, static_cast<Scalar>(0));
    }
  }
  return true;
}

template <typename Scalar>
bool invert_cartesian_factor(const MutableTensorView& factor,
                             const std::size_t env,
                             const std::size_t task_dim,
                             const MutableTensorView& output) noexcept {
  for (std::size_t col = 0; col < task_dim; ++col) {
    for (std::size_t row = 0; row < task_dim; ++row) {
      Scalar value = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, row, inner) *
                 internal::read_tensor3_value<Scalar>(output, env, inner, col);
      }
      value /= internal::read_tensor3_value<Scalar>(factor, env, row, row);
      internal::write_tensor3_value(output, env, row, col, value);
    }
    for (std::size_t reverse = task_dim; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      Scalar value = internal::read_tensor3_value<Scalar>(output, env, row, col);
      for (std::size_t inner = row + 1; inner < task_dim; ++inner) {
        value -= internal::read_tensor3_value<Scalar>(factor, env, inner, row) *
                 internal::read_tensor3_value<Scalar>(output, env, inner, col);
      }
      value /= internal::read_tensor3_value<Scalar>(factor, env, row, row);
      if (!std::isfinite(value)) {
        return false;
      }
      internal::write_tensor3_value(output, env, row, col, value);
    }
  }
  for (std::size_t row = 0; row < task_dim; ++row) {
    for (std::size_t col = 0; col < row; ++col) {
      const Scalar symmetric =
          detail::symmetric_average(internal::read_tensor3_value<Scalar>(output, env, row, col),
                                    internal::read_tensor3_value<Scalar>(output, env, col, row));
      internal::write_tensor3_value(output, env, row, col, symmetric);
      internal::write_tensor3_value(output, env, col, row, symmetric);
    }
  }
  return true;
}

template <typename Scalar>
Result apply_parameters_cpu(const ParameterSetView& parameters,
                            const std::size_t num_envs,
                            const MutableStatusView status) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] = parameters_are_valid<Scalar>(parameters, env)
                            ? 0U
                            : status_bit(StatusFlag::kInvalidParameter);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_cpu(const TensorView& joint_mass_matrix,
                const TensorView& jacobian,
                const ParameterSetView& parameters,
                const StatusView parameter_status,
                const MutableTensorView& joint_factor,
                const MutableTensorView& inverse_jacobian,
                const MutableTensorView& cartesian_factor,
                const MutableTensorView& output,
                const MutableStatusView status,
                const std::size_t num_envs,
                const std::size_t joint_dim,
                const std::size_t task_dim) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    std::uint64_t flags = parameter_status.flags == nullptr ? 0U : parameter_status.flags[env];
    if (!parameters_are_valid<Scalar>(parameters, env)) {
      flags |= status_bit(StatusFlag::kInvalidParameter);
    }
    if (!internal::tensor3_env_is_finite<Scalar>(joint_mass_matrix, env, joint_dim, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, task_dim, joint_dim)) {
      flags |= status_bit(StatusFlag::kNonFiniteInput);
    }
    const Scalar regularization = internal::read_scalar_parameter<Scalar>(
        find_parameter(parameters, kRegularization), env, static_cast<Scalar>(-1));
    const Scalar minimum_pivot = internal::read_scalar_parameter<Scalar>(
        find_parameter(parameters, kMinimumPivot), env, default_minimum_pivot<Scalar>());
    if (flags == 0U && !factor_joint_mass<Scalar>(
                           joint_mass_matrix, env, joint_dim, minimum_pivot, joint_factor)) {
      flags |= status_bit(StatusFlag::kInvalidInput);
    }
    if (flags == 0U) {
      solve_mass_inverse_jacobian<Scalar>(
          jacobian, joint_factor, env, joint_dim, task_dim, inverse_jacobian);
      if (!factor_cartesian_inverse_mass<Scalar>(jacobian,
                                                 inverse_jacobian,
                                                 env,
                                                 joint_dim,
                                                 task_dim,
                                                 regularization,
                                                 minimum_pivot,
                                                 cartesian_factor) ||
          !invert_cartesian_factor<Scalar>(cartesian_factor, env, task_dim, output)) {
        flags |= status_bit(StatusFlag::kInvalidInput);
      }
    }
    if (flags != 0U) {
      flags |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_matrix<Scalar>(output, env, task_dim);
    }
    status.flags[env] = flags;
  }
  return internal::ok();
}

}  // namespace

PrepareResult prepare_operational_space_mass(const TensorView& joint_mass_matrix,
                                             const TensorView& jacobian,
                                             const ParameterSetView& parameters,
                                             ScratchView& scratch,
                                             const MutableTensorView& cartesian_mass_matrix) {
  (void)scratch;
  PrepareResult result;
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  std::size_t task_dim = 0;
  result.result = validate_problem_views(
      joint_mass_matrix, jacobian, cartesian_mass_matrix, &num_envs, &joint_dim, &task_dim);
  if (!result.result.ok()) {
    return result;
  }
  result.result = validate_scalar_parameter(find_parameter(parameters, kRegularization),
                                            joint_mass_matrix.dtype,
                                            joint_mass_matrix.device,
                                            num_envs,
                                            true);
  if (!result.result.ok()) {
    return result;
  }
  result.result = validate_scalar_parameter(find_parameter(parameters, kMinimumPivot),
                                            joint_mass_matrix.dtype,
                                            joint_mass_matrix.device,
                                            num_envs,
                                            false);
  if (!result.result.ok()) {
    return result;
  }
  result.num_envs = num_envs;
  result.signal_dim = task_dim;
  result.scratch.push_back(tensor3_descriptor(kJointMassFactor,
                                              joint_mass_matrix.dtype,
                                              joint_mass_matrix.device,
                                              num_envs,
                                              joint_dim,
                                              joint_dim,
                                              "sqrt(kilogram)",
                                              "lower Cholesky factor of joint mass"));
  result.scratch.push_back(tensor3_descriptor(kMassInverseJacobianTranspose,
                                              joint_mass_matrix.dtype,
                                              joint_mass_matrix.device,
                                              num_envs,
                                              joint_dim,
                                              task_dim,
                                              "inverse mass times Jacobian transpose",
                                              "M^-1 J^T solve workspace"));
  result.scratch.push_back(
      tensor3_descriptor(kCartesianInverseMassFactor,
                         joint_mass_matrix.dtype,
                         joint_mass_matrix.device,
                         num_envs,
                         task_dim,
                         task_dim,
                         "sqrt(inverse Cartesian mass)",
                         "regularized Cartesian inverse-mass Cholesky factor"));
  result.outputs.push_back(tensor3_descriptor("cartesian_mass_matrix",
                                              joint_mass_matrix.dtype,
                                              joint_mass_matrix.device,
                                              num_envs,
                                              task_dim,
                                              task_dim,
                                              "Cartesian mass",
                                              "Lambda=(J M^-1 J^T + regularization I)^-1"));
  return result;
}

Result apply_operational_space_mass_parameters(const ParameterSetView& parameters,
                                               const std::size_t num_envs,
                                               const MutableStatusView parameter_status) noexcept {
  if (num_envs == 0) {
    return internal::invalid_argument("num_envs must be positive");
  }
  if (parameter_status.flags == nullptr || parameter_status.size < num_envs ||
      (parameter_status.device != Device::kCpu && parameter_status.device != Device::kCuda)) {
    return internal::invalid_view("parameter_status must contain num_envs entries");
  }
  const ParameterView* regularization = find_parameter(parameters, kRegularization);
  const DType dtype = regularization == nullptr ? DType::kUnknown : regularization->values.dtype;
  if (!internal::is_float_dtype(dtype)) {
    return internal::invalid_view("regularization must use float32 or float64");
  }
  Result result =
      validate_scalar_parameter(regularization, dtype, parameter_status.device, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_scalar_parameter(
      find_parameter(parameters, kMinimumPivot), dtype, parameter_status.device, num_envs, false);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.device == Device::kCuda ||
      parameters_have_device(parameters, Device::kCuda)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_operational_space_mass_parameters_cuda(parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return dtype == DType::kFloat32
             ? apply_parameters_cpu<float>(parameters, num_envs, parameter_status)
             : apply_parameters_cpu<double>(parameters, num_envs, parameter_status);
}

Result reset_operational_space_mass(const std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    const MutableStatusView status) noexcept {
  if (num_envs == 0) {
    return internal::invalid_argument("num_envs must be positive");
  }
  if (status.device != Device::kCpu && status.device != Device::kCuda) {
    return internal::invalid_view("status device is unsupported");
  }
  Result result = validate_status(status, status.device, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (reset_mask.data != nullptr &&
      (reset_mask.dtype != DType::kBool || reset_mask.device != status.device ||
       reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs) ||
       reset_mask.strides[0] < 1)) {
    return internal::invalid_view("reset_mask must be bool [num_envs] on the status device");
  }
  if (status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_operational_space_mass_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] =
        internal::reset_mask_value(reset_mask, env) ? status_bit(StatusFlag::kResetApplied) : 0U;
  }
  return internal::ok();
}

Result step_operational_space_mass(const TensorView& joint_mass_matrix,
                                   const TensorView& jacobian,
                                   const ParameterSetView& parameters,
                                   const StatusView parameter_status,
                                   ScratchView& scratch,
                                   const MutableTensorView& cartesian_mass_matrix,
                                   const MutableStatusView status) noexcept {
  std::size_t num_envs = 0;
  std::size_t joint_dim = 0;
  std::size_t task_dim = 0;
  Result result = validate_problem_views(
      joint_mass_matrix, jacobian, cartesian_mass_matrix, &num_envs, &joint_dim, &task_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_status(status, joint_mass_matrix.device, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_parameter_status(parameter_status, joint_mass_matrix.device, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_scalar_parameter(find_parameter(parameters, kRegularization),
                                     joint_mass_matrix.dtype,
                                     joint_mass_matrix.device,
                                     num_envs,
                                     true);
  if (!result.ok()) {
    return result;
  }
  result = validate_scalar_parameter(find_parameter(parameters, kMinimumPivot),
                                     joint_mass_matrix.dtype,
                                     joint_mass_matrix.device,
                                     num_envs,
                                     false);
  if (!result.ok()) {
    return result;
  }
  MutableTensorView* joint_factor = nullptr;
  MutableTensorView* inverse_jacobian = nullptr;
  MutableTensorView* cartesian_factor = nullptr;
  result = find_and_validate_scratch(scratch,
                                     joint_mass_matrix.dtype,
                                     joint_mass_matrix.device,
                                     num_envs,
                                     joint_dim,
                                     task_dim,
                                     &joint_factor,
                                     &inverse_jacobian,
                                     &cartesian_factor);
  if (!result.ok()) {
    return result;
  }
  const TensorView scratch_views[] = {
      joint_factor->as_const(), inverse_jacobian->as_const(), cartesian_factor->as_const()};
  const TensorView protected_views[] = {joint_mass_matrix,
                                        jacobian,
                                        cartesian_mass_matrix.as_const(),
                                        joint_factor->as_const(),
                                        inverse_jacobian->as_const(),
                                        cartesian_factor->as_const()};
  for (std::size_t index = 0; index < 3; ++index) {
    for (std::size_t other = 0; other < 3 + index; ++other) {
      result =
          internal::validate_disjoint(scratch_views[index],
                                      protected_views[other],
                                      "operational-space scratch must not alias inputs or output");
      if (!result.ok()) {
        return result;
      }
    }
  }
  if (joint_mass_matrix.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_operational_space_mass_cuda(joint_mass_matrix,
                                            jacobian,
                                            parameters,
                                            parameter_status,
                                            scratch,
                                            cartesian_mass_matrix,
                                            status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return joint_mass_matrix.dtype == DType::kFloat32 ? step_cpu<float>(joint_mass_matrix,
                                                                      jacobian,
                                                                      parameters,
                                                                      parameter_status,
                                                                      *joint_factor,
                                                                      *inverse_jacobian,
                                                                      *cartesian_factor,
                                                                      cartesian_mass_matrix,
                                                                      status,
                                                                      num_envs,
                                                                      joint_dim,
                                                                      task_dim)
                                                    : step_cpu<double>(joint_mass_matrix,
                                                                       jacobian,
                                                                       parameters,
                                                                       parameter_status,
                                                                       *joint_factor,
                                                                       *inverse_jacobian,
                                                                       *cartesian_factor,
                                                                       cartesian_mass_matrix,
                                                                       status,
                                                                       num_envs,
                                                                       joint_dim,
                                                                       task_dim);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
