/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cartesian_blocks.cpp
 * @brief CPU tensor kernels for Cartesian and utility controller blocks.
 */

#include "nvidia/srl/controllers/tensor/cartesian_blocks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#include "tensor/symmetric_matrix_policy.hpp"
#include "tensor/task_impedance.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/cartesian_blocks_cuda.hpp"
#endif

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

constexpr const char* kStiffnessMatrix = "stiffness_matrix";
constexpr const char* kInitialStiffnessMatrix = "initial_stiffness_matrix";
constexpr const char* kCurrentStiffnessMatrix = "current_stiffness_matrix";
constexpr const char* kPreviousStiffnessMatrix = "previous_stiffness_matrix";
constexpr const char* kMaximumStiffness = "maximum_stiffness";
constexpr const char* kMinimumStiffness = "minimum_stiffness";
constexpr const char* kMaximumStiffnessRate = "maximum_stiffness_rate";
constexpr const char* kDampingMatrix = "damping_matrix";
constexpr const char* kDampingFactors = "damping_factors";
constexpr const char* kVirtualMassMatrix = "virtual_mass_matrix";
constexpr const char* kForceFilterThreshold = "force_filter_threshold";
constexpr const char* kAccelerationLimit = "acceleration_limit";
constexpr const char* kTwistLimit = "twist_limit";
constexpr const char* kPoseLimit = "pose_limit";
constexpr const char* kMaximumDistance = "maximum_distance";
constexpr const char* kShapingDistance = "shaping_distance";
constexpr const char* kFunnelRadius = "funnel_radius";
constexpr const char* kMaxDistance = "max_distance";
constexpr const char* kFunnelVelocity = "funnel_velocity";
constexpr const char* kFunnelStartPoint = "funnel_start_point";
constexpr const char* kFunnelEndPoint = "funnel_end_point";
constexpr const char* kAlpha = "alpha";
constexpr const char* kGamma = "gamma";
constexpr const char* kDMin = "d_min";
constexpr const char* kDMax = "d_max";
constexpr const char* kDeadzone = "deadzone";
constexpr const char* kPreviousMassMatrix = "previous_mass_matrix";
constexpr const char* kPreviousJacobian = "previous_jacobian";
constexpr const char* kInitialized = "initialized";
constexpr const char* kProportionalGain = "proportional_gain";
constexpr const char* kPositionGain = "position_gain";
constexpr const char* kVelocityGain = "velocity_gain";
constexpr const char* kDifferentiatorGain = "differentiator_gain";
constexpr const char* kIntegralGain = "integral_gain";
constexpr const char* kU0 = "u_0";
constexpr const char* kUpperLimit = "upper_limit";
constexpr const char* kLowerLimit = "lower_limit";
constexpr const char* kDifferentiatorPreviousError = "differentiator_previous_error";
constexpr const char* kDifferentiatorInitialized = "differentiator_initialized";
constexpr const char* kIntegratorPreviousError = "integrator_previous_error";
constexpr const char* kIntegratedError = "integrated_error";
constexpr const char* kIntegratorInitialized = "integrator_initialized";
constexpr const char* kAdmittancePose = "admittance_pose";
constexpr const char* kAdmittanceTwist = "admittance_twist";
constexpr const char* kPosePreviousSignal = "pose_previous_signal";
constexpr const char* kTwistPreviousSignal = "twist_previous_signal";
constexpr const char* kPoseIntegratorInitialized = "pose_integrator_initialized";
constexpr const char* kTwistIntegratorInitialized = "twist_integrator_initialized";
constexpr const char* kImpedanceWrenchScratch = "impedance_wrench";
constexpr const char* kForceWrenchScratch = "force_wrench";
constexpr const char* kImpedanceStatusScratch = "impedance_status";
constexpr const char* kForceStatusScratch = "force_status";
constexpr const char* kStiffnessInitialized = "stiffness_initialized";
constexpr const char* kInertialInitialized = "inertial_initialized";
constexpr const char* kStiffnessWrenchScratch = "stiffness_wrench";
constexpr const char* kDampingWrenchScratch = "damping_wrench";
constexpr const char* kInertialWrenchScratch = "inertial_wrench";
constexpr const char* kVariableImpedanceWrenchScratch = "variable_impedance_wrench";
constexpr const char* kStiffnessStatusScratch = "stiffness_status";
constexpr const char* kDampingStatusScratch = "damping_status";
constexpr const char* kInertialStatusScratch = "inertial_status";
constexpr const char* kVariableImpedanceStatusScratch = "variable_impedance_status";

struct Vec3 {
  double values[3]{};
};

template <typename Scalar>
struct Vec6 {
  Scalar values[6]{};
};

template <typename Scalar>
detail::ConstVectorView<Scalar> core_vector_env(const TensorView& view,
                                                const std::size_t env) noexcept {
  const auto* data =
      static_cast<const Scalar*>(view.data) + static_cast<std::ptrdiff_t>(env) * view.strides[0];
  return detail::ConstVectorView<Scalar>{
      data, static_cast<std::size_t>(view.shape[1]), view.strides[1]};
}

template <typename Scalar>
detail::ConstVectorView<Scalar> core_vector_env(const MutableTensorView& view,
                                                const std::size_t env) noexcept {
  return core_vector_env<Scalar>(view.as_const(), env);
}

template <typename Scalar>
detail::ConstMatrixView<Scalar> core_matrix_parameter(const ParameterView* parameter,
                                                      const std::size_t env) noexcept {
  const TensorView& values = parameter->values;
  const auto* data = static_cast<const Scalar*>(values.data);
  if (parameter->shared) {
    return detail::ConstMatrixView<Scalar>{data,
                                           static_cast<std::size_t>(values.shape[0]),
                                           static_cast<std::size_t>(values.shape[1]),
                                           values.strides[0],
                                           values.strides[1]};
  }
  data += static_cast<std::ptrdiff_t>(env) * values.strides[0];
  return detail::ConstMatrixView<Scalar>{data,
                                         static_cast<std::size_t>(values.shape[1]),
                                         static_cast<std::size_t>(values.shape[2]),
                                         values.strides[1],
                                         values.strides[2]};
}

template <typename Scalar>
detail::ConstMatrixView<Scalar> core_tensor3_env(const TensorView& view,
                                                 const std::size_t env) noexcept {
  const auto* data =
      static_cast<const Scalar*>(view.data) + static_cast<std::ptrdiff_t>(env) * view.strides[0];
  return detail::ConstMatrixView<Scalar>{data,
                                         static_cast<std::size_t>(view.shape[1]),
                                         static_cast<std::size_t>(view.shape[2]),
                                         view.strides[1],
                                         view.strides[2]};
}

template <typename Scalar>
detail::ConstMatrixView<Scalar> core_tensor3_env(const MutableTensorView& view,
                                                 const std::size_t env) noexcept {
  return core_tensor3_env<Scalar>(view.as_const(), env);
}

bool parameter_set_has_cuda(const ParameterSetView& parameters) {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    if (parameters.parameters[i].values.device == Device::kCuda) {
      return true;
    }
  }
  return false;
}

bool matrix6_lifecycle_uses_cuda(const TensorView& input,
                                 const ParameterSetView& parameters,
                                 const MutableTensorView& output) {
  return input.device == Device::kCuda || output.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const TensorView& input,
                                 const ParameterSetView& parameters,
                                 const MutableTensorView& output) {
  return input.device == Device::kCuda || output.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const TensorView& input, const MutableTensorView& output) {
  return input.device == Device::kCuda || output.device == Device::kCuda;
}

bool matrix6_lifecycle_uses_cuda(const ParameterSetView& parameters,
                                 const MutableStatusView& status) {
  return status.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const ParameterSetView& parameters,
                                 const MutableStatusView& status) {
  return status.device == Device::kCuda || parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const MutableStatusView& status) {
  return status.device == Device::kCuda;
}

bool matrix6_lifecycle_uses_cuda(const TensorView& reset_mask, const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda;
}

bool utility_lifecycle_uses_cuda(const TensorView& reset_mask, const MutableStatusView& status) {
  return reset_mask.device == Device::kCuda || status.device == Device::kCuda;
}

bool matrix6_lifecycle_uses_cuda(const TensorView& input,
                                 const ParameterSetView& parameters,
                                 const StatusView& parameter_status,
                                 const MutableTensorView& output,
                                 const MutableStatusView& status) {
  return input.device == Device::kCuda || output.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const TensorView& input,
                                 const ParameterSetView& parameters,
                                 const StatusView& parameter_status,
                                 const MutableTensorView& output,
                                 const MutableStatusView& status) {
  return input.device == Device::kCuda || output.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
         parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const TensorView& input,
                                 const StatusView& parameter_status,
                                 const MutableTensorView& output,
                                 const MutableStatusView& status) {
  return input.device == Device::kCuda || output.device == Device::kCuda ||
         status.device == Device::kCuda ||
         (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda);
}

bool utility_lifecycle_uses_cuda(const TensorView& first_input,
                                 const TensorView& second_input,
                                 const TensorView& third_input,
                                 const ParameterSetView& parameters,
                                 const MutableTensorView& output) {
  return first_input.device == Device::kCuda || second_input.device == Device::kCuda ||
         third_input.device == Device::kCuda || output.device == Device::kCuda ||
         parameter_set_has_cuda(parameters);
}

bool utility_lifecycle_uses_cuda(const TensorView& first_input,
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

#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
Result validate_cuda_matrix6_parameter_metadata(const ParameterView* parameter,
                                                const char* missing_message,
                                                const DType dtype,
                                                const std::size_t num_envs) {
  if (parameter == nullptr) {
    return internal::invalid_view(missing_message);
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr) {
    return internal::invalid_view("Cartesian matrix parameter data is null");
  }
  if (values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view(
        "Cartesian matrix parameter must be a CUDA tensor matching input dtype");
  }
  if (parameter->shared) {
    if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(kCartesianDim) &&
        values.shape[1] == static_cast<std::int64_t>(kCartesianDim) && values.strides[1] == 1 &&
        values.strides[0] >= values.shape[1]) {
      return internal::ok();
    }
    return internal::invalid_view("shared Cartesian matrix parameter must have [6, 6]");
  }
  if (values.rank == 3 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(kCartesianDim) &&
      values.shape[2] == static_cast<std::int64_t>(kCartesianDim) && values.strides[2] == 1 &&
      values.strides[1] >= values.shape[2] &&
      values.strides[0] >= values.shape[1] * values.strides[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched Cartesian matrix parameter must have [num_envs, 6, 6]");
}

Result validate_cuda_matrix6_prepare(const TensorView& input,
                                     const ParameterSetView& parameters,
                                     const char* parameter_name,
                                     const MutableTensorView& output,
                                     std::size_t* num_envs) {
  if (input.data == nullptr || input.device != Device::kCuda ||
      !internal::is_float_dtype(input.dtype) || !internal::is_row_major_matrix(input) ||
      input.shape[0] <= 0 || input.shape[1] != static_cast<std::int64_t>(kCartesianDim)) {
    return internal::invalid_view("Cartesian input must be CUDA row-major [num_envs, 6]");
  }
  *num_envs = static_cast<std::size_t>(input.shape[0]);
  if (output.data == nullptr || output.device != Device::kCuda || output.dtype != input.dtype ||
      !internal::is_row_major_matrix(output) || output.shape[0] != input.shape[0] ||
      output.shape[1] != input.shape[1]) {
    return internal::invalid_view("Cartesian output must be CUDA row-major [num_envs, 6]");
  }
  return validate_cuda_matrix6_parameter_metadata(find_parameter(parameters, parameter_name),
                                                  "Cartesian matrix parameter is missing",
                                                  input.dtype,
                                                  *num_envs);
}

Result validate_cuda_signal_metadata(const TensorView& input,
                                     const char* name,
                                     std::size_t* num_envs,
                                     std::size_t* signal_dim) {
  if (input.data == nullptr || input.device != Device::kCuda ||
      !internal::is_float_dtype(input.dtype) || !internal::is_row_major_matrix(input) ||
      input.shape[0] <= 0 || input.shape[1] <= 0) {
    return internal::invalid_view(name);
  }
  *num_envs = static_cast<std::size_t>(input.shape[0]);
  *signal_dim = static_cast<std::size_t>(input.shape[1]);
  return internal::ok();
}

Result validate_cuda_pose_metadata(const TensorView& pose,
                                   const char* name,
                                   const DType dtype,
                                   const std::size_t num_envs) {
  if (pose.data == nullptr || pose.device != Device::kCuda || pose.dtype != dtype ||
      !internal::is_row_major_tensor3(pose) ||
      pose.shape[0] != static_cast<std::int64_t>(num_envs) ||
      pose.shape[1] != static_cast<std::int64_t>(kPoseRows) ||
      pose.shape[2] != static_cast<std::int64_t>(kPoseCols)) {
    return internal::invalid_view(name);
  }
  return internal::ok();
}

Result validate_cuda_pose_metadata(const TensorView& pose,
                                   const char* name,
                                   std::size_t* num_envs) {
  if (pose.data == nullptr || pose.device != Device::kCuda ||
      !internal::is_float_dtype(pose.dtype) || !internal::is_row_major_tensor3(pose) ||
      pose.shape[0] <= 0 || pose.shape[1] != static_cast<std::int64_t>(kPoseRows) ||
      pose.shape[2] != static_cast<std::int64_t>(kPoseCols)) {
    return internal::invalid_view(name);
  }
  *num_envs = static_cast<std::size_t>(pose.shape[0]);
  return internal::ok();
}

Result validate_cuda_matrix6_tensor3_metadata(const TensorView& matrix,
                                              const char* name,
                                              const DType dtype,
                                              const std::size_t num_envs) {
  if (matrix.data == nullptr || matrix.device != Device::kCuda || matrix.dtype != dtype ||
      !internal::is_row_major_tensor3(matrix) ||
      matrix.shape[0] != static_cast<std::int64_t>(num_envs) ||
      matrix.shape[1] != static_cast<std::int64_t>(kCartesianDim) ||
      matrix.shape[2] != static_cast<std::int64_t>(kCartesianDim)) {
    return internal::invalid_view(name);
  }
  return internal::ok();
}

Result validate_cuda_signal_output_metadata(const MutableTensorView& output,
                                            const DType dtype,
                                            const std::size_t num_envs,
                                            const std::size_t signal_dim) {
  if (output.data == nullptr || output.device != Device::kCuda || output.dtype != dtype ||
      !internal::is_row_major_matrix(output) ||
      output.shape[0] != static_cast<std::int64_t>(num_envs) ||
      output.shape[1] != static_cast<std::int64_t>(signal_dim)) {
    return internal::invalid_view("output must be CUDA row-major [num_envs, signal_dim]");
  }
  return internal::ok();
}

Result validate_cuda_bool_output_metadata(const MutableTensorView& output,
                                          const std::size_t num_envs) {
  if (output.data == nullptr || output.device != Device::kCuda || output.dtype != DType::kBool ||
      output.rank != 1 || output.shape[0] < static_cast<std::int64_t>(num_envs) ||
      output.strides[0] < 1) {
    return internal::invalid_view("bool output must be CUDA [num_envs]");
  }
  return internal::ok();
}

Result validate_cuda_vector_parameter_metadata(const ParameterView* parameter,
                                               const char* missing_message,
                                               const DType dtype,
                                               const std::size_t num_envs,
                                               const std::size_t signal_dim) {
  if (parameter == nullptr) {
    return internal::invalid_view(missing_message);
  }
  const TensorView& values = parameter->values;
  if (values.data == nullptr || values.device != Device::kCuda || values.dtype != dtype ||
      values.layout != Layout::kRowMajor) {
    return internal::invalid_view("vector parameter must be CUDA and match input dtype");
  }
  if (parameter->shared) {
    if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
        values.strides[0] >= 1) {
      return internal::ok();
    }
    return internal::invalid_view("shared vector parameter must have [signal_dim]");
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(signal_dim) && values.strides[1] == 1 &&
      values.strides[0] >= values.shape[1]) {
    return internal::ok();
  }
  return internal::invalid_view("batched vector parameter must have [num_envs, signal_dim]");
}

Result validate_cuda_optional_vector_parameter_metadata(const ParameterView* parameter,
                                                        const DType dtype,
                                                        const std::size_t num_envs,
                                                        const std::size_t signal_dim) {
  if (parameter == nullptr) {
    return internal::ok();
  }
  return validate_cuda_vector_parameter_metadata(
      parameter, "optional vector parameter is invalid", dtype, num_envs, signal_dim);
}

Result validate_cuda_scalar_batch_metadata(const TensorView& input,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs) {
  if (input.data == nullptr || input.device != Device::kCuda || input.dtype != dtype ||
      input.layout != Layout::kRowMajor) {
    return internal::invalid_view(name);
  }
  if (input.rank == 0) {
    return internal::ok();
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return internal::ok();
  }
  return internal::invalid_view("scalar batch must be CUDA scalar or [num_envs]");
}

Result validate_cuda_joint_signal_metadata(const TensorView& input,
                                           const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           std::size_t* joint_dim) {
  if (input.data == nullptr || input.device != Device::kCuda || input.dtype != dtype ||
      !internal::is_row_major_matrix(input) ||
      input.shape[0] != static_cast<std::int64_t>(num_envs) || input.shape[1] <= 0) {
    return internal::invalid_view(name);
  }
  *joint_dim = static_cast<std::size_t>(input.shape[1]);
  return internal::ok();
}

Result validate_cuda_tensor3_metadata(const TensorView& input,
                                      const char* name,
                                      const DType dtype,
                                      const std::size_t num_envs,
                                      const std::size_t rows,
                                      const std::size_t cols) {
  if (input.data == nullptr || input.device != Device::kCuda || input.dtype != dtype ||
      !internal::is_row_major_tensor3(input) ||
      input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(rows) ||
      input.shape[2] != static_cast<std::int64_t>(cols)) {
    return internal::invalid_view(name);
  }
  return internal::ok();
}

Result validate_cuda_cartesian_output_metadata(const MutableTensorView& output,
                                               const DType dtype,
                                               const std::size_t num_envs) {
  return validate_cuda_signal_output_metadata(output, dtype, num_envs, kCartesianDim);
}

void mark_descriptors_cuda(PrepareResult* prepare_result) {
  for (TensorDescriptor& descriptor : prepare_result->state) {
    descriptor.device = Device::kCuda;
  }
  for (TensorDescriptor& descriptor : prepare_result->scratch) {
    descriptor.device = Device::kCuda;
  }
  for (TensorDescriptor& descriptor : prepare_result->outputs) {
    descriptor.device = Device::kCuda;
  }
}

Result validate_cuda_shaping_prepare(const TensorView& error,
                                     const ParameterSetView& parameters,
                                     const MutableTensorView& shaping_factor,
                                     std::size_t* num_envs,
                                     std::size_t* signal_dim) {
  Result result = validate_cuda_signal_metadata(
      error, "error must be CUDA [num_envs, signal_dim]", num_envs, signal_dim);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_cuda_signal_output_metadata(shaping_factor, error.dtype, *num_envs, *signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                   "maximum_distance parameter is missing",
                                                   error.dtype,
                                                   *num_envs,
                                                   *signal_dim);
  if (!result.ok()) {
    return result;
  }
  return validate_cuda_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                 "shaping_distance parameter is missing",
                                                 error.dtype,
                                                 *num_envs,
                                                 *signal_dim);
}
#endif

DType first_parameter_dtype(const ParameterSetView& parameters) {
  for (std::size_t i = 0; i < parameters.parameter_count; ++i) {
    const DType dtype = parameters.parameters[i].values.dtype;
    if (dtype == DType::kFloat32 || dtype == DType::kFloat64) {
      return dtype;
    }
  }
  return DType::kFloat64;
}

Result validate_cartesian_signal(const TensorView& input, const char* name, std::size_t* num_envs) {
  std::size_t signal_dim = 0;
  Result result = internal::validate_batched_signal(input, name, num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  if (signal_dim != kCartesianDim) {
    return internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
  }
  return internal::ok();
}

Result validate_cartesian_output(const MutableTensorView& output,
                                 const DType dtype,
                                 const std::size_t num_envs) {
  return internal::validate_matching_output(output, dtype, num_envs, kCartesianDim);
}

Result validate_pose_batch(const TensorView& pose,
                           const char* name,
                           const DType dtype,
                           const std::size_t num_envs) {
  return internal::validate_batched_matrix(pose, name, dtype, num_envs, kPoseRows, kPoseCols);
}

Result validate_bool_output(const MutableTensorView& output, const std::size_t num_envs) {
  if (output.data == nullptr) {
    return internal::invalid_view("bool output data is null");
  }
  if (internal::is_cuda(output)) {
    return internal::unsupported("CUDA tensor execution is schema-visible but not implemented");
  }
  if (!internal::is_cpu(output)) {
    return internal::invalid_view("bool output device is unsupported");
  }
  if (output.dtype != DType::kBool || output.rank != 1 ||
      output.shape[0] != static_cast<std::int64_t>(num_envs) || output.strides[0] < 1) {
    return internal::invalid_view("bool output must have shape [num_envs]");
  }
  return internal::ok();
}

Result validate_tensor3_state(const MutableTensorView& state,
                              const DType dtype,
                              const std::size_t num_envs,
                              const std::size_t rows,
                              const std::size_t cols,
                              const char* message) {
  if (state.data == nullptr) {
    return internal::invalid_view(message);
  }
  if (internal::is_cuda(state)) {
    return internal::unsupported("CUDA state tensors require CUDA tensor kernels");
  }
  if (!internal::is_cpu(state)) {
    return internal::invalid_view("state tensor device is unsupported");
  }
  if (state.dtype != dtype) {
    return internal::invalid_view("state tensor dtype must match component dtype");
  }
  if (!internal::is_row_major_tensor3(state) ||
      state.shape[0] != static_cast<std::int64_t>(num_envs) ||
      state.shape[1] != static_cast<std::int64_t>(rows) ||
      state.shape[2] != static_cast<std::int64_t>(cols)) {
    return internal::invalid_view("state tensor has an incompatible shape");
  }
  return internal::ok();
}

Result validate_initialized_state(const MutableTensorView& initialized,
                                  const std::size_t num_envs) {
  if (initialized.data == nullptr) {
    return internal::invalid_view("initialized state is missing");
  }
  if (internal::is_cuda(initialized)) {
    return internal::unsupported("CUDA initialized state requires CUDA tensor kernels");
  }
  if (!internal::is_cpu(initialized)) {
    return internal::invalid_view("initialized state device is unsupported");
  }
  if (initialized.dtype != DType::kBool || initialized.rank != 1 ||
      initialized.shape[0] != static_cast<std::int64_t>(num_envs) || initialized.strides[0] < 1) {
    return internal::invalid_view("initialized state must be bool [num_envs]");
  }
  return internal::ok();
}

Result validate_inertial_state(StateView& state,
                               const DType dtype,
                               const std::size_t num_envs,
                               const std::size_t joint_dim) {
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return internal::invalid_view(
        "inertial coupling state requires previous_mass_matrix, previous_jacobian, initialized");
  }
  Result result = validate_tensor3_state(previous_mass->values,
                                         dtype,
                                         num_envs,
                                         kCartesianDim,
                                         kCartesianDim,
                                         "previous_mass_matrix state is missing");
  if (!result.ok()) {
    return result;
  }
  result = validate_tensor3_state(previous_jacobian->values,
                                  dtype,
                                  num_envs,
                                  kCartesianDim,
                                  joint_dim,
                                  "previous_jacobian state is missing");
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(initialized->values, num_envs);
}

Result reset_stateless(const std::size_t num_envs,
                       const TensorView& reset_mask,
                       MutableStatusView status) {
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

template <typename Scalar>
bool matrix_parameter_is_valid(const ParameterSetView& parameters,
                               const char* parameter_name,
                               const std::size_t env,
                               const std::size_t rows,
                               const std::size_t cols) {
  return internal::matrix_parameter_env_is_finite<Scalar>(
      find_parameter(parameters, parameter_name), env, rows, cols);
}

template <typename Scalar>
Result apply_matrix6_parameters(const ParameterSetView& parameters,
                                const char* parameter_name,
                                const std::size_t num_envs,
                                MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  result = internal::validate_matrix_parameter_metadata(find_parameter(parameters, parameter_name),
                                                        "Cartesian matrix parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!matrix_parameter_is_valid<Scalar>(
            parameters, parameter_name, env, kCartesianDim, kCartesianDim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool matrix_parameter_is_symmetric_psd(const ParameterView* parameter, const std::size_t env) {
  if (parameter == nullptr) {
    return false;
  }
  double matrix[6][6]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      const double value = static_cast<double>(internal::read_matrix_parameter<Scalar>(
          parameter, env, row, col, static_cast<Scalar>(0)));
      if (!std::isfinite(value)) {
        return false;
      }
      matrix[row][col] = value;
    }
  }
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
      const double scale =
          std::max({1.0, std::fabs(matrix[row][col]), std::fabs(matrix[col][row])});
      if (std::fabs(matrix[row][col] - matrix[col][row]) > 1e-9 * scale) {
        return false;
      }
    }
  }

  for (std::size_t iter = 0; iter < 64; ++iter) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = 1;
    double max_offdiag = std::fabs(matrix[pivot_row][pivot_col]);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
        const double candidate = std::fabs(matrix[row][col]);
        if (candidate > max_offdiag) {
          max_offdiag = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }
    if (max_offdiag < 1e-10) {
      break;
    }

    const double app = matrix[pivot_row][pivot_row];
    const double aqq = matrix[pivot_col][pivot_col];
    const double apq = matrix[pivot_row][pivot_col];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    for (std::size_t idx = 0; idx < kCartesianDim; ++idx) {
      if (idx == pivot_row || idx == pivot_col) {
        continue;
      }
      const double aip = matrix[idx][pivot_row];
      const double aiq = matrix[idx][pivot_col];
      matrix[idx][pivot_row] = c * aip - s * aiq;
      matrix[pivot_row][idx] = matrix[idx][pivot_row];
      matrix[idx][pivot_col] = s * aip + c * aiq;
      matrix[pivot_col][idx] = matrix[idx][pivot_col];
    }
    matrix[pivot_row][pivot_row] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[pivot_col][pivot_col] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[pivot_row][pivot_col] = 0.0;
    matrix[pivot_col][pivot_row] = 0.0;
  }

  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (matrix[dim][dim] < -1e-8) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void set_identity_6x6(Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
    }
  }
}

template <typename Scalar>
void read_tensor3_matrix6(const TensorView& view, const std::size_t env, Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = internal::read_tensor3_value<Scalar>(view, env, row, col);
    }
  }
}

template <typename Scalar>
bool matrix6_has_positive_diagonal(const Scalar matrix[6][6]) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!std::isfinite(matrix[dim][dim]) || matrix[dim][dim] <= static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
bool matrix6_has_non_negative_diagonal(const Scalar matrix[6][6]) {
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    if (!std::isfinite(matrix[dim][dim]) || matrix[dim][dim] < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
void transpose_6x6(const Scalar input[6][6], Scalar output[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      output[row][col] = input[col][row];
    }
  }
}

template <typename Scalar>
void multiply_6x6(const Scalar left[6][6], const Scalar right[6][6], Scalar output[6][6]) {
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
void scale_rows_6x6(const Scalar input[6][6], const Scalar row_scale[6], Scalar output[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      output[row][col] = row_scale[row] * input[row][col];
    }
  }
}

template <typename Scalar>
bool symmetric_eigen_6x6(const Scalar input[6][6],
                         Scalar eigenvectors[6][6],
                         Scalar eigenvalues[6],
                         Scalar* matrix_scale) {
  Scalar matrix[6][6]{};
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
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
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      matrix[row][col] = input[row][col] / scale;
    }
  }

  set_identity_6x6(eigenvectors);
  bool converged = false;
  for (std::size_t iter = 0; iter < 80; ++iter) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = 1;
    Scalar max_offdiag = std::fabs(matrix[pivot_row][pivot_col]);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
        const Scalar candidate = std::fabs(matrix[row][col]);
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
    const Scalar angle =
        static_cast<Scalar>(0.5) * std::atan2(static_cast<Scalar>(2) * apq, aqq - app);
    const Scalar c = std::cos(angle);
    const Scalar s = std::sin(angle);

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
    if (!std::isfinite(eigenvalues[dim])) {
      return false;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      if (!std::isfinite(eigenvectors[row][dim])) {
        return false;
      }
    }
  }
  *matrix_scale = scale;
  return true;
}

template <typename Scalar>
bool matrix_roots_6x6(const Scalar input[6][6],
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
    max_eigenvalue = std::max(max_eigenvalue, std::fabs(eigenvalues[col]));
  }
  const Scalar threshold =
      internal::SymmetricMatrixPolicy<Scalar>::spectral_relative_tolerance(kCartesianDim) *
      max_eigenvalue;
  for (std::size_t col = 0; col < kCartesianDim; ++col) {
    if ((strict_positive && eigenvalues[col] <= threshold) || eigenvalues[col] < -threshold) {
      return false;
    }
    const Scalar normalized = std::max(eigenvalues[col], static_cast<Scalar>(0));
    const Scalar sqrt_value = std::sqrt(scale) * std::sqrt(normalized);
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
      if (!std::isfinite(output[row][col]) ||
          (inverse_output != nullptr && !std::isfinite(inverse_output[row][col]))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
bool validate_and_symmetrize_external_6x6(Scalar matrix[6][6]) {
  Scalar scale = static_cast<Scalar>(0);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!std::isfinite(matrix[row][col])) {
        return false;
      }
      scale = std::max(scale, std::fabs(matrix[row][col]));
    }
  }
  const Scalar tolerance =
      internal::SymmetricMatrixPolicy<Scalar>::symmetry_relative_tolerance(kCartesianDim) * scale;
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = row + 1; col < kCartesianDim; ++col) {
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
bool symmetrize_derived_6x6(Scalar matrix[6][6]) {
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      if (!std::isfinite(matrix[row][col])) {
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
Result apply_damping_design_parameters(const ParameterSetView& parameters,
                                       const std::size_t num_envs,
                                       MutableStatusView parameter_status,
                                       const bool clear_status = true) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (clear_status) {
    internal::clear_status(parameter_status, num_envs);
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                        "damping_factors parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, kDampingFactors);
  for (std::size_t env = 0; env < num_envs; ++env) {
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar value = internal::read_vector_parameter<Scalar>(
          damping_factors, env, dim, static_cast<Scalar>(0));
      if (!std::isfinite(value) || value < static_cast<Scalar>(0)) {
        parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
        break;
      }
    }
  }
  return internal::ok();
}

enum class DampingDesignKind { kDoubleDiagonalization, kFactorizationDesign };

template <typename Scalar>
bool compute_damping_design_matrix(const DampingDesignKind kind,
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
                        kind == DampingDesignKind::kDoubleDiagonalization ? mass_sqrt_inv : nullptr,
                        kind == DampingDesignKind::kDoubleDiagonalization)) {
    return false;
  }

  if (kind == DampingDesignKind::kFactorizationDesign) {
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
Result step_damping_design(const DampingDesignKind kind,
                           const TensorView& cartesian_velocity_error,
                           const TensorView& cartesian_mass_matrix,
                           const TensorView& stiffness_matrix,
                           const ParameterSetView& parameters,
                           StatusView parameter_status,
                           const MutableTensorView& cartesian_wrench,
                           MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                             "cartesian_mass_matrix tensor data is null",
                                             cartesian_velocity_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(stiffness_matrix,
                                             "stiffness_matrix tensor data is null",
                                             cartesian_velocity_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
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
  result = apply_damping_design_parameters<Scalar>(parameters, num_envs, status);
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
    if (!internal::matrix_env_is_finite<Scalar>(cartesian_velocity_error, env, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            stiffness_matrix, env, kCartesianDim, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    Scalar mass[6][6]{};
    Scalar stiffness[6][6]{};
    read_tensor3_matrix6(cartesian_mass_matrix, env, mass);
    read_tensor3_matrix6(stiffness_matrix, env, stiffness);
    if (!matrix6_has_positive_diagonal(mass) || !matrix6_has_non_negative_diagonal(stiffness)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(cartesian_wrench, env, kCartesianDim);
      continue;
    }

    Scalar damping_factors[6]{};
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      damping_factors[dim] = internal::read_vector_parameter<Scalar>(
          damping_factors_parameter, env, dim, static_cast<Scalar>(0));
    }
    Scalar damping_matrix[6][6]{};
    if (!compute_damping_design_matrix(kind, mass, stiffness, damping_factors, damping_matrix)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(cartesian_wrench, env, kCartesianDim);
      continue;
    }
    Scalar wrench[6]{};
    bool wrench_is_finite = true;
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const detail::ConstMatrixView<Scalar> damping_view{
          &damping_matrix[0][0], kCartesianDim, kCartesianDim, kCartesianDim, 1};
      const Scalar value = detail::cartesian_damping_value(
          damping_view, core_vector_env<Scalar>(cartesian_velocity_error, env), row);
      wrench[row] = value;
      wrench_is_finite = wrench_is_finite && std::isfinite(value);
    }
    if (!wrench_is_finite) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(cartesian_wrench, env, kCartesianDim);
      continue;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      internal::write_matrix_value(cartesian_wrench, env, row, wrench[row]);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool variable_stiffness_parameters_are_valid(const ParameterSetView& parameters,
                                             const std::size_t env) {
  const ParameterView* initial = find_parameter(parameters, kInitialStiffnessMatrix);
  const ParameterView* maximum = find_parameter(parameters, kMaximumStiffness);
  const ParameterView* minimum = find_parameter(parameters, kMinimumStiffness);
  const ParameterView* maximum_rate = find_parameter(parameters, kMaximumStiffnessRate);
  if (!matrix_parameter_is_symmetric_psd<Scalar>(initial, env)) {
    return false;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar max_value =
        internal::read_vector_parameter<Scalar>(maximum, env, dim, static_cast<Scalar>(0));
    const Scalar min_value =
        internal::read_vector_parameter<Scalar>(minimum, env, dim, static_cast<Scalar>(0));
    const Scalar rate_value =
        internal::read_vector_parameter<Scalar>(maximum_rate, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(max_value) || !std::isfinite(min_value) || !std::isfinite(rate_value) ||
        max_value < static_cast<Scalar>(0) || min_value < static_cast<Scalar>(0) ||
        rate_value < static_cast<Scalar>(0) || max_value < min_value) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_variable_stiffness_parameters_impl(const ParameterSetView& parameters,
                                                const std::size_t num_envs,
                                                MutableStatusView parameter_status,
                                                const bool clear_status = true) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (clear_status) {
    internal::clear_status(parameter_status, num_envs);
  }
  result = internal::validate_matrix_parameter_metadata(
      find_parameter(parameters, kInitialStiffnessMatrix),
      "initial_stiffness_matrix parameter is missing",
      dtype_for<Scalar>(),
      num_envs,
      kCartesianDim,
      kCartesianDim,
      true);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffness),
                                                   "maximum_stiffness parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMinimumStiffness),
                                                   "minimum_stiffness parameter is invalid",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kCartesianDim,
                                                   false);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(
      find_parameter(parameters, kMaximumStiffnessRate),
      "maximum_stiffness_rate parameter is missing",
      dtype_for<Scalar>(),
      num_envs,
      kCartesianDim,
      true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!variable_stiffness_parameters_are_valid<Scalar>(parameters, env)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

Result validate_variable_stiffness_state(StateView& state,
                                         const DType dtype,
                                         const std::size_t num_envs) {
  StateEntryView* current = find_state(state, kCurrentStiffnessMatrix);
  StateEntryView* previous = find_state(state, kPreviousStiffnessMatrix);
  StateEntryView* initialized = find_state(state, kInitialized);
  if (current == nullptr || previous == nullptr || initialized == nullptr) {
    return internal::invalid_view(
        "variable stiffness state requires current_stiffness_matrix, "
        "previous_stiffness_matrix, initialized");
  }
  Result result = validate_tensor3_state(current->values,
                                         dtype,
                                         num_envs,
                                         kCartesianDim,
                                         kCartesianDim,
                                         "current_stiffness_matrix state is missing");
  if (!result.ok()) {
    return result;
  }
  result = validate_tensor3_state(previous->values,
                                  dtype,
                                  num_envs,
                                  kCartesianDim,
                                  kCartesianDim,
                                  "previous_stiffness_matrix state is missing");
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(initialized->values, num_envs);
}

template <typename Scalar>
void write_initial_variable_stiffness_state(const ParameterSetView& parameters,
                                            const std::size_t env,
                                            MutableTensorView current,
                                            MutableTensorView previous) {
  const ParameterView* initial = find_parameter(parameters, kInitialStiffnessMatrix);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      const Scalar value =
          internal::read_matrix_parameter<Scalar>(initial, env, row, col, static_cast<Scalar>(0));
      internal::write_tensor3_value(current, env, row, col, value);
      internal::write_tensor3_value(previous, env, row, col, value);
    }
  }
}

template <typename Scalar>
void multiply_stiffness_state(const MutableTensorView& stiffness,
                              const TensorView& pose_error,
                              const std::size_t env,
                              const MutableTensorView& output) {
  const auto stiffness_view = core_tensor3_env<Scalar>(stiffness, env);
  const auto error_view = core_vector_env<Scalar>(pose_error, env);
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    internal::write_matrix_value(
        output, env, row, detail::cartesian_stiffness_value(stiffness_view, error_view, row));
  }
}

template <typename Scalar>
Result reset_variable_stiffness_impl(const ParameterSetView& parameters,
                                     StatusView parameter_status,
                                     StateView& state,
                                     const TensorView& reset_mask,
                                     MutableStatusView status) {
  StateEntryView* initialized = find_state(state, kInitialized);
  if (initialized == nullptr || initialized->values.rank != 1 ||
      initialized->values.shape[0] <= 0) {
    return internal::invalid_view("initialized state must be bool [num_envs]");
  }
  const std::size_t num_envs = static_cast<std::size_t>(initialized->values.shape[0]);
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_status_view(parameter_status, num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_variable_stiffness_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_variable_stiffness_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* current = find_state(state, kCurrentStiffnessMatrix);
  StateEntryView* previous = find_state(state, kPreviousStiffnessMatrix);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    if ((status.flags[env] & status_bit(StatusFlag::kInvalidParameter)) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      continue;
    }
    write_initial_variable_stiffness_state<Scalar>(
        parameters, env, current->values, previous->values);
    initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

template <typename Scalar>
Result step_variable_stiffness_impl(const TensorView& cartesian_pose_error,
                                    const TensorView& stiffness_matrix,
                                    const TensorView& dt,
                                    const ParameterSetView& parameters,
                                    StatusView parameter_status,
                                    StateView& state,
                                    const MutableTensorView& cartesian_wrench,
                                    MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(stiffness_matrix,
                                             "stiffness_matrix tensor data is null",
                                             cartesian_pose_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
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
  result = validate_variable_stiffness_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_variable_stiffness_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* current = find_state(state, kCurrentStiffnessMatrix);
  StateEntryView* previous = find_state(state, kPreviousStiffnessMatrix);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  const ParameterView* maximum = find_parameter(parameters, kMaximumStiffness);
  const ParameterView* minimum = find_parameter(parameters, kMinimumStiffness);
  const ParameterView* maximum_rate = find_parameter(parameters, kMaximumStiffnessRate);

  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!internal::matrix_env_is_finite<Scalar>(cartesian_pose_error, env, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            stiffness_matrix, env, kCartesianDim, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(cartesian_wrench, env, kCartesianDim);
      continue;
    }

    const bool initialized_env =
        initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] != 0U;
    if (!initialized_env) {
      write_initial_variable_stiffness_state<Scalar>(
          parameters, env, current->values, previous->values);
      initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 1U;
      multiply_stiffness_state<Scalar>(
          current->values, cartesian_pose_error, env, cartesian_wrench);
      continue;
    }

    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const Scalar max_stiffness =
          internal::read_vector_parameter<Scalar>(maximum, env, row, static_cast<Scalar>(0));
      const Scalar min_stiffness =
          internal::read_vector_parameter<Scalar>(minimum, env, row, static_cast<Scalar>(0));
      const Scalar max_rate =
          internal::read_vector_parameter<Scalar>(maximum_rate, env, row, static_cast<Scalar>(0));
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        const Scalar old_value =
            internal::read_tensor3_value<Scalar>(current->values, env, row, col);
        const Scalar target_value =
            internal::read_tensor3_value<Scalar>(stiffness_matrix, env, row, col);
        const Scalar upper_limit = row == col ? max_stiffness : static_cast<Scalar>(0);
        const Scalar lower_limit = row == col ? min_stiffness : static_cast<Scalar>(0);
        const Scalar rate_limit = row == col ? max_rate : static_cast<Scalar>(0);
        const Scalar new_value = detail::variable_stiffness_value(
            old_value, target_value, lower_limit, upper_limit, rate_limit, dt_value);
        if (new_value != target_value) {
          status.flags[env] |= status_bit(StatusFlag::kSaturation);
        }
        internal::write_tensor3_value(previous->values, env, row, col, new_value);
        internal::write_tensor3_value(current->values, env, row, col, new_value);
      }
    }
    multiply_stiffness_state<Scalar>(current->values, cartesian_pose_error, env, cartesian_wrench);
  }
  return internal::ok();
}

template <typename Scalar>
void zero_cartesian_output(const MutableTensorView& output, const std::size_t env) {
  internal::zero_output_env<Scalar>(output, env, kCartesianDim);
}

template <typename Scalar>
Result step_matrix6_block(const TensorView& input,
                          const ParameterSetView& parameters,
                          const char* parameter_name,
                          StatusView parameter_status,
                          const MutableTensorView& output,
                          MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(input, "Cartesian input data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(output, input.dtype, num_envs);
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
  result = apply_matrix6_parameters<Scalar>(parameters, parameter_name, num_envs, status);
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
    if (!internal::matrix_env_is_finite<Scalar>(input, env, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(output, env);
      continue;
    }
    const auto matrix_view = core_matrix_parameter<Scalar>(matrix, env);
    const auto input_view = core_vector_env<Scalar>(input, env);
    const bool stiffness = std::strcmp(parameter_name, kStiffnessMatrix) == 0;
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const Scalar value = stiffness
                               ? detail::cartesian_stiffness_value(matrix_view, input_view, row)
                               : detail::cartesian_damping_value(matrix_view, input_view, row);
      internal::write_matrix_value(output, env, row, value);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool solve_6x6(const ParameterView* matrix,
               const std::size_t env,
               const TensorView& rhs,
               Scalar solution[6]) {
  Scalar augmented[6][7]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      augmented[row][col] =
          internal::read_matrix_parameter<Scalar>(matrix, env, row, col, static_cast<Scalar>(0));
    }
    augmented[row][6] = internal::read_matrix_value<Scalar>(rhs, env, row);
  }

  for (std::size_t pivot = 0; pivot < kCartesianDim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = std::fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < kCartesianDim; ++row) {
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
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        std::swap(augmented[pivot][col], augmented[pivot_row][col]);
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
Result step_virtual_mass(const TensorView& force,
                         const ParameterSetView& parameters,
                         StatusView parameter_status,
                         const MutableTensorView& cartesian_acceleration,
                         MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(force, "force tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_acceleration, force.dtype, num_envs);
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
  result = apply_matrix6_parameters<Scalar>(parameters, kVirtualMassMatrix, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  const ParameterView* matrix = find_parameter(parameters, kVirtualMassMatrix);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(force, env, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    Scalar solution[6]{};
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) == 0U &&
        !solve_6x6(matrix, env, force, solution)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_acceleration, env);
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      internal::write_matrix_value(cartesian_acceleration, env, dim, solution[dim]);
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool shaping_parameters_are_valid(const ParameterSetView& parameters,
                                  const std::size_t env,
                                  const std::size_t signal_dim) {
  const ParameterView* maximum = find_parameter(parameters, kMaximumDistance);
  const ParameterView* shaping = find_parameter(parameters, kShapingDistance);
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    const Scalar maximum_value =
        internal::read_vector_parameter<Scalar>(maximum, env, dim, static_cast<Scalar>(0));
    const Scalar shaping_value =
        internal::read_vector_parameter<Scalar>(shaping, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(maximum_value) || !std::isfinite(shaping_value) ||
        maximum_value <= shaping_value) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result apply_shaping_parameters(const ParameterSetView& parameters,
                                const std::size_t num_envs,
                                const std::size_t signal_dim,
                                MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                   "maximum_distance parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   signal_dim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                   "shaping_distance parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   signal_dim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!shaping_parameters_are_valid<Scalar>(parameters, env, signal_dim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_shaping(const TensorView& error,
                    const ParameterSetView& parameters,
                    StatusView parameter_status,
                    const MutableTensorView& shaping_factor,
                    MutableStatusView status) {
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result =
      internal::validate_batched_signal(error, "error tensor data is null", &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(shaping_factor, error.dtype, num_envs, signal_dim);
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
  result = apply_shaping_parameters<Scalar>(parameters, num_envs, signal_dim, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  const ParameterView* maximum = find_parameter(parameters, kMaximumDistance);
  const ParameterView* shaping = find_parameter(parameters, kShapingDistance);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(error, env, signal_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(shaping_factor, env, signal_dim);
      continue;
    }
    for (std::size_t dim = 0; dim < signal_dim; ++dim) {
      const Scalar maximum_value =
          internal::read_vector_parameter<Scalar>(maximum, env, dim, static_cast<Scalar>(0));
      const Scalar shaping_value =
          internal::read_vector_parameter<Scalar>(shaping, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(
          shaping_factor,
          env,
          dim,
          detail::shaping_factor_value(
              internal::read_matrix_value<Scalar>(error, env, dim), maximum_value, shaping_value));
    }
  }
  return internal::ok();
}

template <typename Scalar>
bool scalar_parameter_set_is_valid(const ParameterSetView& parameters,
                                   const std::size_t env,
                                   const char* first,
                                   const char* second,
                                   const char* third,
                                   const char* fourth,
                                   const char* fifth) {
  const char* names[] = {first, second, third, fourth, fifth};
  for (const char* name : names) {
    if (name == nullptr) {
      continue;
    }
    const Scalar value = internal::read_scalar_parameter<Scalar>(
        find_parameter(parameters, name), env, static_cast<Scalar>(0));
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Result validate_scalar_parameters(const ParameterSetView& parameters,
                                  const std::size_t num_envs,
                                  MutableStatusView parameter_status,
                                  const char* first,
                                  const char* second,
                                  const char* third,
                                  const char* fourth,
                                  const char* fifth,
                                  const bool inverse_apf = false) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const char* names[] = {first, second, third, fourth, fifth};
  for (const char* name : names) {
    if (name == nullptr) {
      continue;
    }
    result = internal::validate_scalar_parameter_metadata(find_parameter(parameters, name),
                                                          "scalar parameter is missing",
                                                          dtype_for<Scalar>(),
                                                          num_envs,
                                                          true);
    if (!result.ok()) {
      return result;
    }
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!scalar_parameter_set_is_valid<Scalar>(
            parameters, env, first, second, third, fourth, fifth)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    if (inverse_apf &&
        !equations::validate_inverse_apf_parameters(
            internal::read_scalar_parameter<Scalar>(
                find_parameter(parameters, kAlpha), env, static_cast<Scalar>(0)),
            internal::read_scalar_parameter<Scalar>(
                find_parameter(parameters, kGamma), env, static_cast<Scalar>(0)),
            internal::read_scalar_parameter<Scalar>(
                find_parameter(parameters, kDMin), env, static_cast<Scalar>(0)),
            internal::read_scalar_parameter<Scalar>(
                find_parameter(parameters, kDMax), env, static_cast<Scalar>(0)),
            internal::read_scalar_parameter<Scalar>(
                find_parameter(parameters, kDeadzone), env, static_cast<Scalar>(0)))) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Scalar read_scalar_param(const ParameterSetView& parameters,
                         const char* name,
                         const std::size_t env) {
  return internal::read_scalar_parameter<Scalar>(
      find_parameter(parameters, name), env, static_cast<Scalar>(0));
}

template <typename Scalar>
void read_vector3_param(const ParameterSetView& parameters,
                        const char* name,
                        const std::size_t env,
                        Scalar values[3]) {
  const ParameterView* parameter = find_parameter(parameters, name);
  for (std::size_t i = 0; i < kPointDim; ++i) {
    values[i] = internal::read_vector_parameter<Scalar>(parameter, env, i, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
Scalar norm3(const Scalar values[3]) {
  return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

template <typename Scalar>
Scalar norm6(const Scalar values[6]) {
  Scalar sum = static_cast<Scalar>(0);
  for (std::size_t i = 0; i < kCartesianDim; ++i) {
    sum += values[i] * values[i];
  }
  return std::sqrt(sum);
}

template <typename Scalar>
bool so3_log_error(const TensorView& current_pose,
                   const TensorView& goal_pose,
                   const std::size_t env,
                   Scalar rotation_error[3]) noexcept {
  return detail::so3_log_error(core_tensor3_env<Scalar>(current_pose, env),
                               core_tensor3_env<Scalar>(goal_pose, env),
                               detail::MutableVectorView<Scalar>{rotation_error, kPointDim, 1});
}

template <typename Scalar>
Result apply_flow_parameters_impl(const ParameterSetView& parameters,
                                  const std::size_t num_envs,
                                  MutableStatusView parameter_status) {
  Result result = validate_scalar_parameters<Scalar>(parameters,
                                                     num_envs,
                                                     parameter_status,
                                                     kFunnelRadius,
                                                     kMaxDistance,
                                                     kFunnelVelocity,
                                                     nullptr,
                                                     nullptr);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kFunnelStartPoint),
                                                   "funnel_start_point parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kPointDim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kFunnelEndPoint),
                                                        "funnel_end_point parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kPointDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    Scalar start[3]{};
    Scalar end[3]{};
    read_vector3_param(parameters, kFunnelStartPoint, env, start);
    read_vector3_param(parameters, kFunnelEndPoint, env, end);
    Scalar direction[3]{};
    for (std::size_t i = 0; i < kPointDim; ++i) {
      direction[i] = end[i] - start[i];
    }
    const Scalar radius = read_scalar_param<Scalar>(parameters, kFunnelRadius, env);
    const Scalar max_distance = read_scalar_param<Scalar>(parameters, kMaxDistance, env);
    const Scalar velocity = read_scalar_param<Scalar>(parameters, kFunnelVelocity, env);
    if (!internal::vector_parameters_are_finite<Scalar>(
            parameters, kFunnelStartPoint, kFunnelEndPoint, nullptr, env, kPointDim) ||
        radius < static_cast<Scalar>(0) || max_distance < static_cast<Scalar>(0) ||
        velocity < static_cast<Scalar>(0) || radius > max_distance ||
        norm3(direction) < static_cast<Scalar>(1e-6)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_flow(const TensorView& cartesian_pose,
                 const ParameterSetView& parameters,
                 StatusView parameter_status,
                 const MutableTensorView& cartesian_flow,
                 MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = internal::validate_batched_matrix(
      cartesian_pose,
      "cartesian_pose tensor data is null",
      dtype_for<Scalar>(),
      cartesian_pose.shape[0] > 0 ? static_cast<std::size_t>(cartesian_pose.shape[0]) : 0,
      kPoseRows,
      kPoseCols);
  if (!result.ok()) {
    return result;
  }
  num_envs = static_cast<std::size_t>(cartesian_pose.shape[0]);
  result = validate_cartesian_output(cartesian_flow, cartesian_pose.dtype, num_envs);
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
  result = apply_flow_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::tensor3_env_is_finite<Scalar>(cartesian_pose, env, kPoseRows, kPoseCols)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_flow, env);
      continue;
    }

    Scalar start[3]{};
    Scalar end[3]{};
    read_vector3_param(parameters, kFunnelStartPoint, env, start);
    read_vector3_param(parameters, kFunnelEndPoint, env, end);
    Scalar position[3]{};
    for (std::size_t i = 0; i < kPointDim; ++i) {
      position[i] = internal::read_tensor3_value<Scalar>(cartesian_pose, env, i, 3);
    }
    Scalar direction[3]{};
    Scalar direction_to_projection[3]{};
    const detail::FlowFunnelGeometry<Scalar> geometry = detail::flow_funnel_geometry(
        detail::ConstVectorView<Scalar>{start, kPointDim, 1},
        detail::ConstVectorView<Scalar>{end, kPointDim, 1},
        detail::ConstVectorView<Scalar>{position, kPointDim, 1},
        detail::MutableVectorView<Scalar>{direction, kPointDim, 1},
        detail::MutableVectorView<Scalar>{direction_to_projection, kPointDim, 1});
    const Scalar funnel_radius = read_scalar_param<Scalar>(parameters, kFunnelRadius, env);
    const Scalar max_distance = read_scalar_param<Scalar>(parameters, kMaxDistance, env);
    const Scalar weight = detail::flow_funnel_weight(geometry.radius, funnel_radius, max_distance);
    Scalar unit_flow[3]{};
    for (std::size_t i = 0; i < kPointDim; ++i) {
      unit_flow[i] = detail::flow_funnel_direction_value(geometry.radius,
                                                         funnel_radius,
                                                         max_distance,
                                                         weight,
                                                         direction[i],
                                                         direction_to_projection[i]);
    }
    const Scalar velocity = read_scalar_param<Scalar>(parameters, kFunnelVelocity, env);
    for (std::size_t i = 0; i < kPointDim; ++i) {
      internal::write_matrix_value(cartesian_flow, env, i, velocity * unit_flow[i]);
      internal::write_matrix_value(cartesian_flow, env, i + kPointDim, static_cast<Scalar>(0));
    }
  }
  return internal::ok();
}

template <typename Scalar>
Result step_inverse_apf_impl(const TensorView& current_pose,
                             const TensorView& goal_pose,
                             const TensorView& cartesian_twist,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& cartesian_wrench,
                             MutableStatusView status) {
  std::size_t checked_num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_twist, "cartesian_twist tensor data is null", &checked_num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_pose_batch(
      current_pose, "current_pose tensor data is null", cartesian_twist.dtype, checked_num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_pose_batch(
      goal_pose, "goal_pose tensor data is null", cartesian_twist.dtype, checked_num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_twist.dtype, checked_num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, checked_num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, checked_num_envs);
  result = internal::validate_status_view(parameter_status, checked_num_envs, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_scalar_parameters<Scalar>(
      parameters, checked_num_envs, status, kAlpha, kGamma, kDMin, kDMax, kDeadzone, true);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < checked_num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  for (std::size_t env = 0; env < checked_num_envs; ++env) {
    if (!internal::tensor3_env_is_finite<Scalar>(current_pose, env, kPoseRows, kPoseCols) ||
        !internal::tensor3_env_is_finite<Scalar>(goal_pose, env, kPoseRows, kPoseCols) ||
        !internal::matrix_env_is_finite<Scalar>(cartesian_twist, env, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    const Scalar alpha = read_scalar_param<Scalar>(parameters, kAlpha, env);
    const Scalar gamma = read_scalar_param<Scalar>(parameters, kGamma, env);
    const Scalar d_min = read_scalar_param<Scalar>(parameters, kDMin, env);
    const Scalar d_max = read_scalar_param<Scalar>(parameters, kDMax, env);
    const Scalar deadzone = read_scalar_param<Scalar>(parameters, kDeadzone, env);
    if (alpha < static_cast<Scalar>(0) || gamma < static_cast<Scalar>(0) ||
        d_min < static_cast<Scalar>(0) || d_max < d_min || deadzone < static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }

    Scalar position_error[3]{};
    for (std::size_t i = 0; i < kPointDim; ++i) {
      position_error[i] = internal::read_tensor3_value<Scalar>(goal_pose, env, i, 3) -
                          internal::read_tensor3_value<Scalar>(current_pose, env, i, 3);
    }
    Scalar rotation_error[3]{};
    if (!so3_log_error<Scalar>(current_pose, goal_pose, env, rotation_error)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }
    Scalar error[6]{};
    for (std::size_t i = 0; i < kPointDim; ++i) {
      error[i] = position_error[i];
      error[i + kPointDim] = rotation_error[i];
    }
    const Scalar error_norm = norm6(error);
    const Scalar error_norm_squared = error_norm * error_norm;
    const Scalar damping = detail::inverse_apf_damping(error_norm_squared, d_min, d_max, gamma);
    Scalar wrench[6]{};
    bool wrench_is_finite = std::isfinite(error_norm) && std::isfinite(damping);
    for (std::size_t i = 0; i < kCartesianDim; ++i) {
      wrench[i] = detail::inverse_apf_wrench_value(
          error[i],
          internal::read_matrix_value<Scalar>(cartesian_twist, env, i),
          error_norm,
          error_norm_squared,
          alpha,
          deadzone,
          damping);
      wrench_is_finite = wrench_is_finite && std::isfinite(wrench[i]);
    }
    if (!wrench_is_finite) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }
    for (std::size_t i = 0; i < kCartesianDim; ++i) {
      internal::write_matrix_value(cartesian_wrench, env, i, wrench[i]);
    }
  }
  return internal::ok();
}

PrepareResult prepare_matrix6(const TensorView& input,
                              const ParameterSetView& parameters,
                              const char* parameter_name,
                              const MutableTensorView& output,
                              const char* output_name,
                              const char* units,
                              const char* semantics) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  if (matrix6_lifecycle_uses_cuda(input, parameters, output)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result =
        validate_cuda_matrix6_prepare(input, parameters, parameter_name, output, &num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.outputs.push_back(internal::matrix_descriptor(
        output_name, input.dtype, num_envs, kCartesianDim, units, "task_space", semantics));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result =
      validate_cartesian_signal(input, "Cartesian input data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_cartesian_output(output, input.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, parameter_name),
                                                   "Cartesian matrix parameter is missing",
                                                   input.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.outputs.push_back(internal::matrix_descriptor(
      output_name, input.dtype, num_envs, kCartesianDim, units, "task_space", semantics));
  return prepare_result;
}

}  // namespace

PrepareResult prepare_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                          const ParameterSetView& parameters,
                                          const MutableTensorView& cartesian_wrench) {
  return prepare_matrix6(cartesian_pose_error,
                         parameters,
                         kStiffnessMatrix,
                         cartesian_wrench,
                         "cartesian_wrench",
                         "newton_and_newton_meter",
                         "Cartesian stiffness wrench from pose error");
}

Result apply_cartesian_stiffness_parameters(const ParameterSetView& parameters,
                                            const std::size_t num_envs,
                                            MutableStatusView parameter_status) {
  if (matrix6_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_matrix6_parameters_cuda(
        parameters, kStiffnessMatrix, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_matrix6_parameters<float>(
                                        parameters, kStiffnessMatrix, num_envs, parameter_status)
                                  : apply_matrix6_parameters<double>(
                                        parameters, kStiffnessMatrix, num_envs, parameter_status);
}

Result reset_cartesian_stiffness(const std::size_t num_envs,
                                 const TensorView& reset_mask,
                                 MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                const ParameterSetView& parameters,
                                StatusView parameter_status,
                                const MutableTensorView& cartesian_wrench,
                                MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(
          cartesian_pose_error, parameters, parameter_status, cartesian_wrench, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_matrix6_cuda(cartesian_pose_error,
                                       parameters,
                                       kStiffnessMatrix,
                                       parameter_status,
                                       cartesian_wrench,
                                       status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    return step_matrix6_block<float>(cartesian_pose_error,
                                     parameters,
                                     kStiffnessMatrix,
                                     parameter_status,
                                     cartesian_wrench,
                                     status);
  }
  if (cartesian_pose_error.dtype == DType::kFloat64) {
    return step_matrix6_block<double>(cartesian_pose_error,
                                      parameters,
                                      kStiffnessMatrix,
                                      parameter_status,
                                      cartesian_wrench,
                                      status);
  }
  return internal::invalid_view("Cartesian input dtype must be float32 or float64");
}

PrepareResult prepare_variable_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                                   const TensorView& stiffness_matrix,
                                                   const TensorView& dt,
                                                   const ParameterSetView& parameters,
                                                   StateView& state,
                                                   const MutableTensorView& cartesian_wrench) {
  (void)state;
  PrepareResult prepare_result;
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(cartesian_pose_error,
                                      "cartesian_pose_error must be CUDA [num_envs, 6]",
                                      &num_envs,
                                      &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(stiffness_matrix,
                                                           "stiffness_matrix tensor data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_cartesian_output_metadata(
        cartesian_wrench, cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_matrix6_parameter_metadata(
        find_parameter(parameters, kInitialStiffnessMatrix),
        "initial_stiffness_matrix parameter is missing",
        cartesian_pose_error.dtype,
        num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffness),
                                                "maximum_stiffness parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_optional_vector_parameter_metadata(
        find_parameter(parameters, kMinimumStiffness),
        cartesian_pose_error.dtype,
        num_envs,
        kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffnessRate),
                                                "maximum_stiffness_rate parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.state.push_back(internal::tensor3_descriptor("current_stiffness_matrix",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                kCartesianDim,
                                                                "wrench_per_pose_error",
                                                                "task_space",
                                                                "current limited stiffness "
                                                                "matrix"));
    prepare_result.state.push_back(internal::tensor3_descriptor("previous_stiffness_matrix",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                kCartesianDim,
                                                                "wrench_per_pose_error",
                                                                "task_space",
                                                                "previous limited stiffness "
                                                                "matrix"));
    prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "per-environment initialized flag"));
    prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                                 cartesian_pose_error.dtype,
                                                                 num_envs,
                                                                 kCartesianDim,
                                                                 "newton_and_newton_meter",
                                                                 "task_space",
                                                                 "variable stiffness wrench"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(stiffness_matrix,
                                                            "stiffness_matrix tensor data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matrix_parameter_metadata(
      find_parameter(parameters, kInitialStiffnessMatrix),
      "initial_stiffness_matrix parameter is missing",
      cartesian_pose_error.dtype,
      num_envs,
      kCartesianDim,
      kCartesianDim,
      true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffness),
                                                   "maximum_stiffness parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_vector_parameter_metadata(
      find_parameter(parameters, kMaximumStiffnessRate),
      "maximum_stiffness_rate parameter is missing",
      cartesian_pose_error.dtype,
      num_envs,
      kCartesianDim,
      true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMinimumStiffness),
                                                   "minimum_stiffness parameter is invalid",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   false);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.state.push_back(internal::tensor3_descriptor("current_stiffness_matrix",
                                                              cartesian_pose_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              kCartesianDim,
                                                              "wrench_per_pose_error",
                                                              "task_space",
                                                              "current limited stiffness matrix"));
  prepare_result.state.push_back(internal::tensor3_descriptor("previous_stiffness_matrix",
                                                              cartesian_pose_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              kCartesianDim,
                                                              "wrench_per_pose_error",
                                                              "task_space",
                                                              "previous limited stiffness matrix"));
  prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "per-environment initialized flag"));
  prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                               cartesian_pose_error.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "variable stiffness wrench"));
  return prepare_result;
}

Result apply_variable_cartesian_stiffness_parameters(const ParameterSetView& parameters,
                                                     const std::size_t num_envs,
                                                     MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_variable_cartesian_stiffness_parameters_cuda(
        parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_variable_stiffness_parameters_impl<float>(
                                        parameters, num_envs, parameter_status)
                                  : apply_variable_stiffness_parameters_impl<double>(
                                        parameters, num_envs, parameter_status);
}

Result reset_variable_cartesian_stiffness(const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          StateView& state,
                                          const TensorView& reset_mask,
                                          MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(parameters, status) ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      reset_mask.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_variable_cartesian_stiffness_cuda(
        parameters, parameter_status, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* current = find_state(state, kCurrentStiffnessMatrix);
  if (current == nullptr) {
    return internal::invalid_view("current_stiffness_matrix state is missing");
  }
  if (current->values.dtype == DType::kFloat32) {
    return reset_variable_stiffness_impl<float>(
        parameters, parameter_status, state, reset_mask, status);
  }
  if (current->values.dtype == DType::kFloat64) {
    return reset_variable_stiffness_impl<double>(
        parameters, parameter_status, state, reset_mask, status);
  }
  return internal::invalid_view("variable stiffness state dtype must be float32 or float64");
}

Result step_variable_cartesian_stiffness(const TensorView& cartesian_pose_error,
                                         const TensorView& stiffness_matrix,
                                         const TensorView& dt,
                                         const ParameterSetView& parameters,
                                         StatusView parameter_status,
                                         StateView& state,
                                         const MutableTensorView& cartesian_wrench,
                                         MutableStatusView status) {
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_variable_cartesian_stiffness_cuda(cartesian_pose_error,
                                                  stiffness_matrix,
                                                  dt,
                                                  parameters,
                                                  parameter_status,
                                                  state,
                                                  cartesian_wrench,
                                                  status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    return step_variable_stiffness_impl<float>(cartesian_pose_error,
                                               stiffness_matrix,
                                               dt,
                                               parameters,
                                               parameter_status,
                                               state,
                                               cartesian_wrench,
                                               status);
  }
  if (cartesian_pose_error.dtype == DType::kFloat64) {
    return step_variable_stiffness_impl<double>(cartesian_pose_error,
                                                stiffness_matrix,
                                                dt,
                                                parameters,
                                                parameter_status,
                                                state,
                                                cartesian_wrench,
                                                status);
  }
  return internal::invalid_view("cartesian_pose_error dtype must be float32 or float64");
}

PrepareResult prepare_cartesian_damping(const TensorView& cartesian_velocity_error,
                                        const ParameterSetView& parameters,
                                        const MutableTensorView& cartesian_wrench) {
  return prepare_matrix6(cartesian_velocity_error,
                         parameters,
                         kDampingMatrix,
                         cartesian_wrench,
                         "cartesian_wrench",
                         "newton_and_newton_meter",
                         "Cartesian damping wrench from velocity error");
}

Result apply_cartesian_damping_parameters(const ParameterSetView& parameters,
                                          const std::size_t num_envs,
                                          MutableStatusView parameter_status) {
  if (matrix6_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_matrix6_parameters_cuda(
        parameters, kDampingMatrix, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_matrix6_parameters<float>(
                                        parameters, kDampingMatrix, num_envs, parameter_status)
                                  : apply_matrix6_parameters<double>(
                                        parameters, kDampingMatrix, num_envs, parameter_status);
}

Result reset_cartesian_damping(const std::size_t num_envs,
                               const TensorView& reset_mask,
                               MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_cartesian_damping(const TensorView& cartesian_velocity_error,
                              const ParameterSetView& parameters,
                              StatusView parameter_status,
                              const MutableTensorView& cartesian_wrench,
                              MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(
          cartesian_velocity_error, parameters, parameter_status, cartesian_wrench, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_matrix6_cuda(cartesian_velocity_error,
                                       parameters,
                                       kDampingMatrix,
                                       parameter_status,
                                       cartesian_wrench,
                                       status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_velocity_error.dtype == DType::kFloat32) {
    return step_matrix6_block<float>(cartesian_velocity_error,
                                     parameters,
                                     kDampingMatrix,
                                     parameter_status,
                                     cartesian_wrench,
                                     status);
  }
  if (cartesian_velocity_error.dtype == DType::kFloat64) {
    return step_matrix6_block<double>(cartesian_velocity_error,
                                      parameters,
                                      kDampingMatrix,
                                      parameter_status,
                                      cartesian_wrench,
                                      status);
  }
  return internal::invalid_view("Cartesian input dtype must be float32 or float64");
}

PrepareResult prepare_cartesian_damping_double_diagonalization(
    const TensorView& cartesian_velocity_error,
    const TensorView& cartesian_mass_matrix,
    const TensorView& stiffness_matrix,
    const ParameterSetView& parameters,
    const MutableTensorView& cartesian_wrench) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  if (utility_lifecycle_uses_cuda(cartesian_velocity_error,
                                  cartesian_mass_matrix,
                                  stiffness_matrix,
                                  parameters,
                                  cartesian_wrench)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t signal_dim = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(cartesian_velocity_error,
                                      "cartesian_velocity_error must be CUDA [num_envs, 6]",
                                      &num_envs,
                                      &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("cartesian_velocity_error must have shape [num_envs, 6]");
      return prepare_result;
    }
    prepare_result.result = validate_cuda_matrix6_tensor3_metadata(
        cartesian_mass_matrix,
        "cartesian_mass_matrix must be CUDA [num_envs, 6, 6]",
        cartesian_velocity_error.dtype,
        num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_matrix6_tensor3_metadata(stiffness_matrix,
                                               "stiffness_matrix must be CUDA [num_envs, 6, 6]",
                                               cartesian_velocity_error.dtype,
                                               num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_signal_output_metadata(
        cartesian_wrench, cartesian_velocity_error.dtype, num_envs, kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                "damping_factors parameter is missing",
                                                cartesian_velocity_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("cartesian_wrench",
                                    cartesian_velocity_error.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "newton_and_newton_meter",
                                    "task_space",
                                    "Cartesian damping wrench from double diagonalization"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_batched_matrix(cartesian_mass_matrix,
                                        "cartesian_mass_matrix tensor data is null",
                                        cartesian_velocity_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(stiffness_matrix,
                                                            "stiffness_matrix tensor data is null",
                                                            cartesian_velocity_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                   "damping_factors parameter is missing",
                                                   cartesian_velocity_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("cartesian_wrench",
                                  cartesian_velocity_error.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "newton_and_newton_meter",
                                  "task_space",
                                  "Cartesian damping wrench from double diagonalization"));
  return prepare_result;
}

Result apply_cartesian_damping_double_diagonalization_parameters(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_damping_design_parameters_cuda(parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32
             ? apply_damping_design_parameters<float>(parameters, num_envs, parameter_status)
             : apply_damping_design_parameters<double>(parameters, num_envs, parameter_status);
}

Result reset_cartesian_damping_double_diagonalization(const std::size_t num_envs,
                                                      const TensorView& reset_mask,
                                                      MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_cartesian_damping_double_diagonalization(const TensorView& cartesian_velocity_error,
                                                     const TensorView& cartesian_mass_matrix,
                                                     const TensorView& stiffness_matrix,
                                                     const ParameterSetView& parameters,
                                                     StatusView parameter_status,
                                                     const MutableTensorView& cartesian_wrench,
                                                     MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(cartesian_velocity_error,
                                  cartesian_mass_matrix,
                                  stiffness_matrix,
                                  parameters,
                                  parameter_status,
                                  cartesian_wrench,
                                  status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_damping_design_cuda(false,
                                              cartesian_velocity_error,
                                              cartesian_mass_matrix,
                                              stiffness_matrix,
                                              parameters,
                                              parameter_status,
                                              cartesian_wrench,
                                              status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_velocity_error.dtype == DType::kFloat32) {
    return step_damping_design<float>(DampingDesignKind::kDoubleDiagonalization,
                                      cartesian_velocity_error,
                                      cartesian_mass_matrix,
                                      stiffness_matrix,
                                      parameters,
                                      parameter_status,
                                      cartesian_wrench,
                                      status);
  }
  if (cartesian_velocity_error.dtype == DType::kFloat64) {
    return step_damping_design<double>(DampingDesignKind::kDoubleDiagonalization,
                                       cartesian_velocity_error,
                                       cartesian_mass_matrix,
                                       stiffness_matrix,
                                       parameters,
                                       parameter_status,
                                       cartesian_wrench,
                                       status);
  }
  return internal::invalid_view("cartesian_velocity_error dtype must be float32 or float64");
}

PrepareResult prepare_cartesian_damping_factorization_design(
    const TensorView& cartesian_velocity_error,
    const TensorView& cartesian_mass_matrix,
    const TensorView& stiffness_matrix,
    const ParameterSetView& parameters,
    const MutableTensorView& cartesian_wrench) {
  return prepare_cartesian_damping_double_diagonalization(cartesian_velocity_error,
                                                          cartesian_mass_matrix,
                                                          stiffness_matrix,
                                                          parameters,
                                                          cartesian_wrench);
}

Result apply_cartesian_damping_factorization_design_parameters(const ParameterSetView& parameters,
                                                               const std::size_t num_envs,
                                                               MutableStatusView parameter_status) {
  return apply_cartesian_damping_double_diagonalization_parameters(
      parameters, num_envs, parameter_status);
}

Result reset_cartesian_damping_factorization_design(const std::size_t num_envs,
                                                    const TensorView& reset_mask,
                                                    MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_cartesian_damping_factorization_design(const TensorView& cartesian_velocity_error,
                                                   const TensorView& cartesian_mass_matrix,
                                                   const TensorView& stiffness_matrix,
                                                   const ParameterSetView& parameters,
                                                   StatusView parameter_status,
                                                   const MutableTensorView& cartesian_wrench,
                                                   MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(cartesian_velocity_error,
                                  cartesian_mass_matrix,
                                  stiffness_matrix,
                                  parameters,
                                  parameter_status,
                                  cartesian_wrench,
                                  status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_damping_design_cuda(true,
                                              cartesian_velocity_error,
                                              cartesian_mass_matrix,
                                              stiffness_matrix,
                                              parameters,
                                              parameter_status,
                                              cartesian_wrench,
                                              status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_velocity_error.dtype == DType::kFloat32) {
    return step_damping_design<float>(DampingDesignKind::kFactorizationDesign,
                                      cartesian_velocity_error,
                                      cartesian_mass_matrix,
                                      stiffness_matrix,
                                      parameters,
                                      parameter_status,
                                      cartesian_wrench,
                                      status);
  }
  if (cartesian_velocity_error.dtype == DType::kFloat64) {
    return step_damping_design<double>(DampingDesignKind::kFactorizationDesign,
                                       cartesian_velocity_error,
                                       cartesian_mass_matrix,
                                       stiffness_matrix,
                                       parameters,
                                       parameter_status,
                                       cartesian_wrench,
                                       status);
  }
  return internal::invalid_view("cartesian_velocity_error dtype must be float32 or float64");
}

PrepareResult prepare_cartesian_virtual_mass(const TensorView& force,
                                             const ParameterSetView& parameters,
                                             const MutableTensorView& cartesian_acceleration) {
  return prepare_matrix6(force,
                         parameters,
                         kVirtualMassMatrix,
                         cartesian_acceleration,
                         "cartesian_acceleration",
                         "meter_per_second_squared_and_radian_per_second_squared",
                         "Cartesian acceleration from inverse virtual mass and force");
}

Result apply_cartesian_virtual_mass_parameters(const ParameterSetView& parameters,
                                               const std::size_t num_envs,
                                               MutableStatusView parameter_status) {
  if (matrix6_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_matrix6_parameters_cuda(
        parameters, kVirtualMassMatrix, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_matrix6_parameters<float>(
                                        parameters, kVirtualMassMatrix, num_envs, parameter_status)
                                  : apply_matrix6_parameters<double>(
                                        parameters, kVirtualMassMatrix, num_envs, parameter_status);
}

Result reset_cartesian_virtual_mass(const std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_cartesian_virtual_mass(const TensorView& force,
                                   const ParameterSetView& parameters,
                                   StatusView parameter_status,
                                   const MutableTensorView& cartesian_acceleration,
                                   MutableStatusView status) {
  if (matrix6_lifecycle_uses_cuda(
          force, parameters, parameter_status, cartesian_acceleration, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_virtual_mass_cuda(
        force, parameters, parameter_status, cartesian_acceleration, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (force.dtype == DType::kFloat32) {
    return step_virtual_mass<float>(
        force, parameters, parameter_status, cartesian_acceleration, status);
  }
  if (force.dtype == DType::kFloat64) {
    return step_virtual_mass<double>(
        force, parameters, parameter_status, cartesian_acceleration, status);
  }
  return internal::invalid_view("force dtype must be float32 or float64");
}

PrepareResult prepare_shaping_function(const TensorView& error,
                                       const ParameterSetView& parameters,
                                       const MutableTensorView& shaping_factor) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  if (utility_lifecycle_uses_cuda(error, parameters, shaping_factor)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result =
        validate_cuda_shaping_prepare(error, parameters, shaping_factor, &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = signal_dim;
    prepare_result.outputs.push_back(internal::matrix_descriptor("shaping_factor",
                                                                 error.dtype,
                                                                 num_envs,
                                                                 signal_dim,
                                                                 "unitless",
                                                                 "caller_defined",
                                                                 "elementwise shaping factor"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result =
      internal::validate_batched_signal(error, "error tensor data is null", &num_envs, &signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matching_output(shaping_factor, error.dtype, num_envs, signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                   "maximum_distance parameter is missing",
                                                   error.dtype,
                                                   num_envs,
                                                   signal_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                   "shaping_distance parameter is missing",
                                                   error.dtype,
                                                   num_envs,
                                                   signal_dim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = signal_dim;
  prepare_result.outputs.push_back(internal::matrix_descriptor("shaping_factor",
                                                               error.dtype,
                                                               num_envs,
                                                               signal_dim,
                                                               "unitless",
                                                               "caller_defined",
                                                               "elementwise shaping factor"));
  return prepare_result;
}

Result apply_shaping_function_parameters(const ParameterSetView& parameters,
                                         const std::size_t num_envs,
                                         const std::size_t signal_dim,
                                         MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_shaping_function_parameters_cuda(
        parameters, num_envs, signal_dim, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32
             ? apply_shaping_parameters<float>(parameters, num_envs, signal_dim, parameter_status)
             : apply_shaping_parameters<double>(parameters, num_envs, signal_dim, parameter_status);
}

Result reset_shaping_function(const std::size_t num_envs,
                              const TensorView& reset_mask,
                              MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_shaping_function(const TensorView& error,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& shaping_factor,
                             MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(error, parameters, parameter_status, shaping_factor, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_shaping_function_cuda(error, parameters, parameter_status, shaping_factor, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (error.dtype == DType::kFloat32) {
    return step_shaping<float>(error, parameters, parameter_status, shaping_factor, status);
  }
  if (error.dtype == DType::kFloat64) {
    return step_shaping<double>(error, parameters, parameter_status, shaping_factor, status);
  }
  return internal::invalid_view("error dtype must be float32 or float64");
}

PrepareResult prepare_command_observer_block(const TensorView& command,
                                             const MutableTensorView& command_is_valid) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  if (utility_lifecycle_uses_cuda(command, command_is_valid)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    prepare_result.result = validate_cuda_signal_metadata(
        command, "command must be CUDA [num_envs, signal_dim]", &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_bool_output_metadata(command_is_valid, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = signal_dim;
    prepare_result.outputs.push_back(
        internal::vector_descriptor("command_is_valid",
                                    DType::kBool,
                                    num_envs,
                                    "unitless",
                                    "none",
                                    "per-environment command validity"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result = internal::validate_batched_signal(
      command, "command tensor data is null", &num_envs, &signal_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_bool_output(command_is_valid, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = signal_dim;
  prepare_result.outputs.push_back(internal::vector_descriptor("command_is_valid",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "per-environment command validity"));
  return prepare_result;
}

Result apply_command_observer_block_parameters(const std::size_t num_envs,
                                               MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_command_observer_block_parameters_cuda(num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  return internal::ok();
}

Result reset_command_observer_block(const std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_command_observer_block(const TensorView& command,
                                   StatusView parameter_status,
                                   const MutableTensorView& command_is_valid,
                                   MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(command, parameter_status, command_is_valid, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_command_observer_block_cuda(command, parameter_status, command_is_valid, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  std::size_t num_envs = 0;
  std::size_t signal_dim = 0;
  Result result = internal::validate_batched_signal(
      command, "command tensor data is null", &num_envs, &signal_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_bool_output(command_is_valid, num_envs);
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
  auto* output = static_cast<std::uint8_t*>(command_is_valid.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
    bool valid = true;
    if (command.dtype == DType::kFloat32) {
      for (std::size_t dim = 0; dim < signal_dim; ++dim) {
        const float value = internal::read_matrix_value<float>(command, env, dim);
        valid = valid && detail::command_observer_accepts(value, static_cast<float>(1e30));
      }
    } else {
      for (std::size_t dim = 0; dim < signal_dim; ++dim) {
        const double value = internal::read_matrix_value<double>(command, env, dim);
        valid = valid && detail::command_observer_accepts(value, 1e100);
      }
    }
    if (!valid) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
    }
    output[env * static_cast<std::size_t>(command_is_valid.strides[0])] = valid ? 1U : 0U;
  }
  return internal::ok();
}

PrepareResult prepare_flow_funnel(const TensorView& cartesian_pose,
                                  const ParameterSetView& parameters,
                                  const MutableTensorView& cartesian_flow) {
  PrepareResult prepare_result;
  if (utility_lifecycle_uses_cuda(cartesian_pose, parameters, cartesian_flow)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    prepare_result.result = validate_cuda_pose_metadata(
        cartesian_pose, "cartesian_pose must be CUDA [num_envs, 4, 4]", &num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_signal_output_metadata(
        cartesian_flow, cartesian_pose.dtype, num_envs, kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("cartesian_flow",
                                    cartesian_pose.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "meter_per_second_and_radian_per_second",
                                    "task_space",
                                    "flow funnel twist"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  if (cartesian_pose.rank != 3 || cartesian_pose.shape[0] <= 0) {
    prepare_result.result =
        internal::invalid_view("cartesian_pose must have shape [num_envs, 4, 4]");
    return prepare_result;
  }
  const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose.shape[0]);
  prepare_result.result = validate_pose_batch(
      cartesian_pose, "cartesian_pose tensor data is null", cartesian_pose.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_cartesian_output(cartesian_flow, cartesian_pose.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("cartesian_flow",
                                  cartesian_pose.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "meter_per_second_and_radian_per_second",
                                  "task_space",
                                  "flow funnel twist"));
  (void)parameters;
  return prepare_result;
}

Result apply_flow_funnel_parameters(const ParameterSetView& parameters,
                                    const std::size_t num_envs,
                                    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_flow_funnel_parameters_cuda(parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32
             ? apply_flow_parameters_impl<float>(parameters, num_envs, parameter_status)
             : apply_flow_parameters_impl<double>(parameters, num_envs, parameter_status);
}

Result reset_flow_funnel(const std::size_t num_envs,
                         const TensorView& reset_mask,
                         MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_flow_funnel(const TensorView& cartesian_pose,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        const MutableTensorView& cartesian_flow,
                        MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(
          cartesian_pose, parameters, parameter_status, cartesian_flow, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_flow_funnel_cuda(
        cartesian_pose, parameters, parameter_status, cartesian_flow, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_pose.dtype == DType::kFloat32) {
    return step_flow<float>(cartesian_pose, parameters, parameter_status, cartesian_flow, status);
  }
  if (cartesian_pose.dtype == DType::kFloat64) {
    return step_flow<double>(cartesian_pose, parameters, parameter_status, cartesian_flow, status);
  }
  return internal::invalid_view("cartesian_pose dtype must be float32 or float64");
}

PrepareResult prepare_inverse_apf(const TensorView& current_pose,
                                  const TensorView& goal_pose,
                                  const TensorView& cartesian_twist,
                                  const ParameterSetView& parameters,
                                  const MutableTensorView& cartesian_wrench) {
  PrepareResult prepare_result;
  std::size_t num_envs = 0;
  if (utility_lifecycle_uses_cuda(
          current_pose, goal_pose, cartesian_twist, parameters, cartesian_wrench)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t signal_dim = 0;
    prepare_result.result = validate_cuda_signal_metadata(
        cartesian_twist, "cartesian_twist must be CUDA [num_envs, 6]", &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("cartesian_twist must have shape [num_envs, 6]");
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_pose_metadata(current_pose,
                                    "current_pose must be CUDA [num_envs, 4, 4]",
                                    cartesian_twist.dtype,
                                    num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_pose_metadata(
        goal_pose, "goal_pose must be CUDA [num_envs, 4, 4]", cartesian_twist.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_signal_output_metadata(
        cartesian_wrench, cartesian_twist.dtype, num_envs, kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                                 cartesian_twist.dtype,
                                                                 num_envs,
                                                                 kCartesianDim,
                                                                 "newton_and_newton_meter",
                                                                 "task_space",
                                                                 "inverse APF wrench"));
    prepare_result.outputs.back().device = Device::kCuda;
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  prepare_result.result =
      validate_cartesian_signal(cartesian_twist, "cartesian_twist tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_pose_batch(
      current_pose, "current_pose tensor data is null", cartesian_twist.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = validate_pose_batch(
      goal_pose, "goal_pose tensor data is null", cartesian_twist.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_twist.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                               cartesian_twist.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "inverse APF wrench"));
  (void)parameters;
  return prepare_result;
}

Result apply_inverse_apf_parameters(const ParameterSetView& parameters,
                                    const std::size_t num_envs,
                                    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_inverse_apf_parameters_cuda(parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? validate_scalar_parameters<float>(parameters,
                                                                      num_envs,
                                                                      parameter_status,
                                                                      kAlpha,
                                                                      kGamma,
                                                                      kDMin,
                                                                      kDMax,
                                                                      kDeadzone,
                                                                      true)
                                  : validate_scalar_parameters<double>(parameters,
                                                                       num_envs,
                                                                       parameter_status,
                                                                       kAlpha,
                                                                       kGamma,
                                                                       kDMin,
                                                                       kDMax,
                                                                       kDeadzone,
                                                                       true);
}

Result reset_inverse_apf(const std::size_t num_envs,
                         const TensorView& reset_mask,
                         MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_stateless(num_envs, reset_mask, status);
}

Result step_inverse_apf(const TensorView& current_pose,
                        const TensorView& goal_pose,
                        const TensorView& cartesian_twist,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        const MutableTensorView& cartesian_wrench,
                        MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(current_pose,
                                  goal_pose,
                                  cartesian_twist,
                                  parameters,
                                  parameter_status,
                                  cartesian_wrench,
                                  status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_inverse_apf_cuda(current_pose,
                                 goal_pose,
                                 cartesian_twist,
                                 parameters,
                                 parameter_status,
                                 cartesian_wrench,
                                 status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_twist.dtype == DType::kFloat32) {
    return step_inverse_apf_impl<float>(current_pose,
                                        goal_pose,
                                        cartesian_twist,
                                        parameters,
                                        parameter_status,
                                        cartesian_wrench,
                                        status);
  }
  if (cartesian_twist.dtype == DType::kFloat64) {
    return step_inverse_apf_impl<double>(current_pose,
                                         goal_pose,
                                         cartesian_twist,
                                         parameters,
                                         parameter_status,
                                         cartesian_wrench,
                                         status);
  }
  return internal::invalid_view("cartesian_twist dtype must be float32 or float64");
}

PrepareResult prepare_inertial_coupling(const TensorView& cartesian_velocity_error,
                                        const TensorView& joint_velocity,
                                        const TensorView& cartesian_mass_matrix,
                                        const TensorView& jacobian,
                                        const TensorView& dt,
                                        StateView& state,
                                        const MutableTensorView& cartesian_wrench) {
  (void)state;
  PrepareResult prepare_result;
  if (cartesian_velocity_error.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result = validate_cuda_signal_metadata(
        cartesian_velocity_error, "cartesian_velocity_error must be CUDA", &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    std::size_t joint_dim = 0;
    prepare_result.result = validate_cuda_joint_signal_metadata(joint_velocity,
                                                                "joint_velocity must be CUDA",
                                                                cartesian_velocity_error.dtype,
                                                                num_envs,
                                                                &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(cartesian_mass_matrix,
                                                           "cartesian_mass_matrix data is null",
                                                           cartesian_velocity_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(jacobian,
                                                           "jacobian tensor data is null",
                                                           cartesian_velocity_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", cartesian_velocity_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_cartesian_output_metadata(
        cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.state.push_back(internal::tensor3_descriptor("previous_mass_matrix",
                                                                cartesian_velocity_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                kCartesianDim,
                                                                "operational_space_mass_units",
                                                                "task_space",
                                                                "previous Cartesian mass matrix"));
    prepare_result.state.push_back(internal::tensor3_descriptor("previous_jacobian",
                                                                cartesian_velocity_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                joint_dim,
                                                                "task_velocity_per_joint_velocity",
                                                                "task_space_to_joint_space",
                                                                "previous Jacobian"));
    prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "per-environment initialized flag"));
    prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                                 cartesian_velocity_error.dtype,
                                                                 num_envs,
                                                                 kCartesianDim,
                                                                 "newton_and_newton_meter",
                                                                 "task_space",
                                                                 "inertial coupling wrench"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &joint_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_envs != num_envs || joint_velocity.dtype != cartesian_velocity_error.dtype) {
    prepare_result.result =
        internal::invalid_view("joint_velocity must match Cartesian dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                                            "cartesian_mass_matrix data is null",
                                                            cartesian_velocity_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(jacobian,
                                                            "jacobian tensor data is null",
                                                            cartesian_velocity_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_velocity_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.state.push_back(internal::tensor3_descriptor("previous_mass_matrix",
                                                              cartesian_velocity_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              kCartesianDim,
                                                              "operational_space_mass_units",
                                                              "task_space",
                                                              "previous Cartesian mass matrix"));
  prepare_result.state.push_back(internal::tensor3_descriptor("previous_jacobian",
                                                              cartesian_velocity_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              joint_dim,
                                                              "task_velocity_per_joint_velocity",
                                                              "task_space_to_joint_space",
                                                              "previous Jacobian"));
  prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "per-environment initialized flag"));
  prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                               cartesian_velocity_error.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "inertial coupling wrench"));
  return prepare_result;
}

Result apply_inertial_coupling_parameters_impl(const std::size_t num_envs,
                                               MutableStatusView parameter_status,
                                               const bool clear_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (clear_status) {
    internal::clear_status(parameter_status, num_envs);
  }
  return internal::ok();
}

Result apply_inertial_coupling_parameters(const std::size_t num_envs,
                                          MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_inertial_coupling_parameters_cuda(num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return apply_inertial_coupling_parameters_impl(num_envs, parameter_status, true);
}

template <typename Scalar>
Result reset_inertial_impl(const std::size_t num_envs,
                           const std::size_t joint_dim,
                           StateView& state,
                           const TensorView& reset_mask,
                           MutableStatusView status) {
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_inertial_state(state, dtype_for<Scalar>(), num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        internal::write_tensor3_value(previous_mass->values, env, row, col, static_cast<Scalar>(0));
      }
      for (std::size_t col = 0; col < joint_dim; ++col) {
        internal::write_tensor3_value(
            previous_jacobian->values, env, row, col, static_cast<Scalar>(0));
      }
    }
    initialized_values[env * static_cast<std::size_t>(initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

Result reset_inertial_coupling(const std::size_t num_envs,
                               const std::size_t joint_dim,
                               StateView& state,
                               const TensorView& reset_mask,
                               MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_inertial_coupling_cuda(num_envs, joint_dim, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  if (previous_mass == nullptr) {
    return internal::invalid_view("previous_mass_matrix state is missing");
  }
  if (previous_mass->values.dtype == DType::kFloat32) {
    return reset_inertial_impl<float>(num_envs, joint_dim, state, reset_mask, status);
  }
  return reset_inertial_impl<double>(num_envs, joint_dim, state, reset_mask, status);
}

template <typename Scalar>
Result step_inertial_impl(const TensorView& cartesian_velocity_error,
                          const TensorView& joint_velocity,
                          const TensorView& cartesian_mass_matrix,
                          const TensorView& jacobian,
                          const TensorView& dt,
                          StatusView parameter_status,
                          StateView& state,
                          const MutableTensorView& cartesian_wrench,
                          MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &joint_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_envs != num_envs || joint_velocity.dtype != cartesian_velocity_error.dtype) {
    return internal::invalid_view("joint_velocity must match Cartesian dtype and num_envs");
  }
  result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                             "cartesian_mass_matrix data is null",
                                             cartesian_velocity_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(jacobian,
                                             "jacobian tensor data is null",
                                             cartesian_velocity_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_velocity_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_velocity_error.dtype, num_envs);
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
  result = validate_inertial_state(state, dtype_for<Scalar>(), num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }

  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);

  for (std::size_t env = 0; env < num_envs; ++env) {
    if (parameter_status.flags != nullptr) {
      status.flags[env] |= parameter_status.flags[env];
    }
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(cartesian_velocity_error, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, kCartesianDim, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      if (internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, dim, dim) <
          static_cast<Scalar>(0)) {
        status.flags[env] |= status_bit(StatusFlag::kInvalidInput);
      }
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }

    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    if (initialized_values[initialized_offset] == 0U) {
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          internal::write_tensor3_value(
              previous_mass->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col));
        }
        for (std::size_t col = 0; col < joint_dim; ++col) {
          internal::write_tensor3_value(
              previous_jacobian->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(jacobian, env, row, col));
        }
        internal::write_matrix_value(cartesian_wrench, env, row, static_cast<Scalar>(0));
      }
      initialized_values[initialized_offset] = 1U;
      continue;
    }

    Scalar jacobian_derivative_times_dq[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t joint = 0; joint < joint_dim; ++joint) {
        const Scalar d_jacobian = detail::matrix_derivative_value(
            internal::read_tensor3_value<Scalar>(jacobian, env, row, joint),
            internal::read_tensor3_value<Scalar>(previous_jacobian->values, env, row, joint),
            dt_value);
        jacobian_derivative_times_dq[row] +=
            d_jacobian * internal::read_matrix_value<Scalar>(joint_velocity, env, joint);
      }
    }

    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      Scalar mass_derivative_term = static_cast<Scalar>(0);
      Scalar jacobian_term = static_cast<Scalar>(0);
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        const Scalar d_mass = detail::matrix_derivative_value(
            internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col),
            internal::read_tensor3_value<Scalar>(previous_mass->values, env, row, col),
            dt_value);
        mass_derivative_term +=
            d_mass * internal::read_matrix_value<Scalar>(cartesian_velocity_error, env, col);
        jacobian_term +=
            internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col) *
            jacobian_derivative_times_dq[col];
      }
      internal::write_matrix_value(
          cartesian_wrench,
          env,
          row,
          detail::inertial_coupling_value(mass_derivative_term, jacobian_term));
    }

    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        internal::write_tensor3_value(
            previous_mass->values,
            env,
            row,
            col,
            internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col));
      }
      for (std::size_t col = 0; col < joint_dim; ++col) {
        internal::write_tensor3_value(
            previous_jacobian->values,
            env,
            row,
            col,
            internal::read_tensor3_value<Scalar>(jacobian, env, row, col));
      }
    }
  }
  return internal::ok();
}

Result step_inertial_coupling(const TensorView& cartesian_velocity_error,
                              const TensorView& joint_velocity,
                              const TensorView& cartesian_mass_matrix,
                              const TensorView& jacobian,
                              const TensorView& dt,
                              StatusView parameter_status,
                              StateView& state,
                              const MutableTensorView& cartesian_wrench,
                              MutableStatusView status) {
  if (cartesian_velocity_error.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_inertial_coupling_cuda(cartesian_velocity_error,
                                       joint_velocity,
                                       cartesian_mass_matrix,
                                       jacobian,
                                       dt,
                                       parameter_status,
                                       state,
                                       cartesian_wrench,
                                       status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_velocity_error.dtype == DType::kFloat32) {
    return step_inertial_impl<float>(cartesian_velocity_error,
                                     joint_velocity,
                                     cartesian_mass_matrix,
                                     jacobian,
                                     dt,
                                     parameter_status,
                                     state,
                                     cartesian_wrench,
                                     status);
  }
  if (cartesian_velocity_error.dtype == DType::kFloat64) {
    return step_inertial_impl<double>(cartesian_velocity_error,
                                      joint_velocity,
                                      cartesian_mass_matrix,
                                      jacobian,
                                      dt,
                                      parameter_status,
                                      state,
                                      cartesian_wrench,
                                      status);
  }
  return internal::invalid_view("cartesian_velocity_error dtype must be float32 or float64");
}

template <typename Scalar>
Result apply_cartesian_impedance_parameters_impl(const ParameterSetView& parameters,
                                                 const std::size_t num_envs,
                                                 MutableStatusView parameter_status,
                                                 const bool clear_status = true) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (clear_status) {
    internal::clear_status(parameter_status, num_envs);
  }
  result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                        "damping_factors parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  const ParameterView* damping_factors = find_parameter(parameters, kDampingFactors);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!matrix_parameter_is_valid<Scalar>(
            parameters, kStiffnessMatrix, env, kCartesianDim, kCartesianDim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar value = internal::read_vector_parameter<Scalar>(
          damping_factors, env, dim, static_cast<Scalar>(0));
      if (!std::isfinite(value) || value < static_cast<Scalar>(0)) {
        parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
        break;
      }
    }
  }
  return internal::ok();
}

PrepareResult prepare_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
                                                     const TensorView& cartesian_velocity_error,
                                                     const TensorView& desired_acceleration,
                                                     const TensorView& joint_velocity,
                                                     const TensorView& cartesian_mass_matrix,
                                                     const TensorView& jacobian,
                                                     const TensorView& dt,
                                                     const ParameterSetView& parameters,
                                                     StateView& state,
                                                     const MutableTensorView& cartesian_wrench) {
  if (cartesian_pose_error.rank == 2 && cartesian_pose_error.shape[1] != 6) {
    return prepare_task_impedance_controller(cartesian_pose_error,
                                             cartesian_velocity_error,
                                             desired_acceleration,
                                             joint_velocity,
                                             cartesian_mass_matrix,
                                             jacobian,
                                             dt,
                                             parameters,
                                             state,
                                             cartesian_wrench);
  }
  (void)state;
  PrepareResult prepare_result;
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(cartesian_pose_error,
                                      "cartesian_pose_error must be CUDA [num_envs, 6]",
                                      &num_envs,
                                      &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    const TensorView cartesian_signals[] = {cartesian_velocity_error, desired_acceleration};
    for (const TensorView& signal : cartesian_signals) {
      std::size_t signal_envs = 0;
      std::size_t signal_dims = 0;
      prepare_result.result = validate_cuda_signal_metadata(
          signal, "Cartesian impedance signal is invalid", &signal_envs, &signal_dims);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
      if (signal_envs != num_envs || signal_dims != kCartesianDim ||
          signal.dtype != cartesian_pose_error.dtype) {
        prepare_result.result =
            internal::invalid_view("Cartesian impedance signals must share dtype and num_envs");
        return prepare_result;
      }
    }
    std::size_t joint_dim = 0;
    prepare_result.result = validate_cuda_joint_signal_metadata(joint_velocity,
                                                                "joint_velocity must be CUDA",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(cartesian_mass_matrix,
                                                           "cartesian_mass_matrix data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(jacobian,
                                                           "jacobian tensor data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_cartesian_output_metadata(
        cartesian_wrench, cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_matrix6_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                 "stiffness_matrix parameter is missing",
                                                 cartesian_pose_error.dtype,
                                                 num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                "damping_factors parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.state.push_back(internal::tensor3_descriptor("previous_mass_matrix",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                kCartesianDim,
                                                                "operational_space_mass_units",
                                                                "task_space",
                                                                "previous Cartesian mass matrix"));
    prepare_result.state.push_back(internal::tensor3_descriptor("previous_jacobian",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                joint_dim,
                                                                "task_velocity_per_joint_velocity",
                                                                "task_space_to_joint_space",
                                                                "previous Jacobian"));
    prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "per-environment initialized flag"));
    prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                                 cartesian_pose_error.dtype,
                                                                 num_envs,
                                                                 kCartesianDim,
                                                                 "newton_and_newton_meter",
                                                                 "task_space",
                                                                 "Cartesian impedance wrench"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t velocity_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &velocity_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t acceleration_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &acceleration_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (velocity_envs != num_envs || acceleration_envs != num_envs ||
      cartesian_velocity_error.dtype != cartesian_pose_error.dtype ||
      desired_acceleration.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("Cartesian impedance signals must share dtype and num_envs");
    return prepare_result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &joint_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_envs != num_envs || joint_velocity.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("joint_velocity must match Cartesian dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                                            "cartesian_mass_matrix data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(jacobian,
                                                            "jacobian tensor data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                   "damping_factors parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.state.push_back(internal::tensor3_descriptor("previous_mass_matrix",
                                                              cartesian_pose_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              kCartesianDim,
                                                              "operational_space_mass_units",
                                                              "task_space",
                                                              "previous Cartesian mass matrix"));
  prepare_result.state.push_back(internal::tensor3_descriptor("previous_jacobian",
                                                              cartesian_pose_error.dtype,
                                                              num_envs,
                                                              kCartesianDim,
                                                              joint_dim,
                                                              "task_velocity_per_joint_velocity",
                                                              "task_space_to_joint_space",
                                                              "previous Jacobian"));
  prepare_result.state.push_back(internal::vector_descriptor("initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "per-environment initialized flag"));
  prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                               cartesian_pose_error.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "Cartesian impedance wrench"));
  return prepare_result;
}

Result apply_cartesian_impedance_controller_parameters(const ParameterSetView& parameters,
                                                       const std::size_t num_envs,
                                                       MutableStatusView parameter_status) {
  const ParameterView* stiffness = find_parameter(parameters, kStiffnessMatrix);
  if (stiffness != nullptr) {
    const TensorView& values = stiffness->values;
    const std::int64_t task_dim =
        stiffness->shared && values.rank == 2
            ? values.shape[0]
            : (!stiffness->shared && values.rank == 3 ? values.shape[1] : 0);
    if (task_dim != 0 && task_dim != 6) {
      return apply_task_impedance_parameters(
          parameters, num_envs, static_cast<std::size_t>(task_dim), parameter_status);
    }
  }
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_impedance_controller_parameters_cuda(
        parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_cartesian_impedance_parameters_impl<float>(
                                        parameters, num_envs, parameter_status)
                                  : apply_cartesian_impedance_parameters_impl<double>(
                                        parameters, num_envs, parameter_status);
}

Result reset_cartesian_impedance_controller(const std::size_t num_envs,
                                            const std::size_t joint_dim,
                                            StateView& state,
                                            const TensorView& reset_mask,
                                            MutableStatusView status) {
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  if (previous_mass != nullptr && previous_mass->values.rank == 3 &&
      previous_mass->values.shape[1] != 6) {
    return reset_task_impedance_controller(num_envs,
                                           static_cast<std::size_t>(previous_mass->values.shape[1]),
                                           joint_dim,
                                           state,
                                           reset_mask,
                                           status);
  }
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_inertial_coupling_cuda(num_envs, joint_dim, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  return reset_inertial_coupling(num_envs, joint_dim, state, reset_mask, status);
}

template <typename Scalar>
Result step_cartesian_impedance_impl(const TensorView& cartesian_pose_error,
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
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t velocity_envs = 0;
  result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &velocity_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t acceleration_envs = 0;
  result = validate_cartesian_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &acceleration_envs);
  if (!result.ok()) {
    return result;
  }
  if (velocity_envs != num_envs || acceleration_envs != num_envs ||
      cartesian_velocity_error.dtype != cartesian_pose_error.dtype ||
      desired_acceleration.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("Cartesian impedance signals must share dtype and num_envs");
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &joint_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_envs != num_envs || joint_velocity.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("joint_velocity must match Cartesian dtype and num_envs");
  }
  result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                             "cartesian_mass_matrix data is null",
                                             cartesian_pose_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_batched_matrix(jacobian,
                                             "jacobian tensor data is null",
                                             cartesian_pose_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
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
  result = validate_inertial_state(state, dtype_for<Scalar>(), num_envs, joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = apply_cartesian_impedance_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInitialized);
  auto* initialized_values = static_cast<std::uint8_t*>(initialized->values.data);
  const ParameterView* stiffness_parameter = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping_factors_parameter = find_parameter(parameters, kDampingFactors);

  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(cartesian_pose_error, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(cartesian_velocity_error, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(desired_acceleration, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_velocity, env, joint_dim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, kCartesianDim, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }

    Scalar mass[6][6]{};
    Scalar stiffness[6][6]{};
    read_tensor3_matrix6(cartesian_mass_matrix, env, mass);
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        stiffness[row][col] = internal::read_matrix_parameter<Scalar>(
            stiffness_parameter, env, row, col, static_cast<Scalar>(0));
      }
    }
    if (!matrix6_has_positive_diagonal(mass) || !matrix6_has_non_negative_diagonal(stiffness)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }

    Scalar damping_factors[6]{};
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      damping_factors[dim] = internal::read_vector_parameter<Scalar>(
          damping_factors_parameter, env, dim, static_cast<Scalar>(0));
    }
    Scalar damping_matrix[6][6]{};
    if (!compute_damping_design_matrix(DampingDesignKind::kDoubleDiagonalization,
                                       mass,
                                       stiffness,
                                       damping_factors,
                                       damping_matrix)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }

    Scalar stiffness_wrench[6]{};
    Scalar damping_wrench[6]{};
    Scalar inertial_wrench[6]{};
    Scalar acceleration_wrench[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        stiffness_wrench[row] += stiffness[row][col] * internal::read_matrix_value<Scalar>(
                                                           cartesian_pose_error, env, col);
        damping_wrench[row] += damping_matrix[row][col] * internal::read_matrix_value<Scalar>(
                                                              cartesian_velocity_error, env, col);
        acceleration_wrench[row] +=
            mass[row][col] * internal::read_matrix_value<Scalar>(desired_acceleration, env, col);
      }
    }

    const std::size_t initialized_offset =
        env * static_cast<std::size_t>(initialized->values.strides[0]);
    if (initialized_values[initialized_offset] == 0U) {
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          internal::write_tensor3_value(
              previous_mass->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col));
        }
        for (std::size_t col = 0; col < joint_dim; ++col) {
          internal::write_tensor3_value(
              previous_jacobian->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(jacobian, env, row, col));
        }
      }
      initialized_values[initialized_offset] = 1U;
    } else {
      Scalar jacobian_derivative_times_dq[6]{};
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        for (std::size_t joint = 0; joint < joint_dim; ++joint) {
          const Scalar d_jacobian = detail::matrix_derivative_value(
              internal::read_tensor3_value<Scalar>(jacobian, env, row, joint),
              internal::read_tensor3_value<Scalar>(previous_jacobian->values, env, row, joint),
              dt_value);
          jacobian_derivative_times_dq[row] +=
              d_jacobian * internal::read_matrix_value<Scalar>(joint_velocity, env, joint);
        }
      }
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        Scalar mass_derivative_term = static_cast<Scalar>(0);
        Scalar jacobian_term = static_cast<Scalar>(0);
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          const Scalar d_mass = detail::matrix_derivative_value(
              internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col),
              internal::read_tensor3_value<Scalar>(previous_mass->values, env, row, col),
              dt_value);
          mass_derivative_term +=
              d_mass * internal::read_matrix_value<Scalar>(cartesian_velocity_error, env, col);
          jacobian_term += mass[row][col] * jacobian_derivative_times_dq[col];
        }
        inertial_wrench[row] = detail::inertial_coupling_value(mass_derivative_term, jacobian_term);
      }
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          internal::write_tensor3_value(
              previous_mass->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col));
        }
        for (std::size_t col = 0; col < joint_dim; ++col) {
          internal::write_tensor3_value(
              previous_jacobian->values,
              env,
              row,
              col,
              internal::read_tensor3_value<Scalar>(jacobian, env, row, col));
        }
      }
    }

    Scalar wrench[6]{};
    bool wrench_is_finite = true;
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      wrench[dim] = detail::cartesian_impedance_value(stiffness_wrench[dim],
                                                      damping_wrench[dim],
                                                      inertial_wrench[dim],
                                                      acceleration_wrench[dim]);
      wrench_is_finite = wrench_is_finite && std::isfinite(wrench[dim]);
    }
    if (!wrench_is_finite) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(cartesian_wrench, env);
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      internal::write_matrix_value(cartesian_wrench, env, dim, wrench[dim]);
    }
  }
  return internal::ok();
}

Result step_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
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
  if (cartesian_pose_error.rank == 2 && cartesian_pose_error.shape[1] != 6) {
    return step_task_impedance_controller(cartesian_pose_error,
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
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_impedance_controller_cuda(cartesian_pose_error,
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
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    return step_cartesian_impedance_impl<float>(cartesian_pose_error,
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
  if (cartesian_pose_error.dtype == DType::kFloat64) {
    return step_cartesian_impedance_impl<double>(cartesian_pose_error,
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
  return internal::invalid_view("cartesian_pose_error dtype must be float32 or float64");
}

template <typename Scalar>
bool solve_matrix6_vector(const Scalar matrix[6][6], const Scalar rhs[6], Scalar solution[6]) {
  Scalar augmented[6][7]{};
  for (std::size_t row = 0; row < kCartesianDim; ++row) {
    for (std::size_t col = 0; col < kCartesianDim; ++col) {
      augmented[row][col] = matrix[row][col];
    }
    augmented[row][6] = rhs[row];
  }
  for (std::size_t pivot = 0; pivot < kCartesianDim; ++pivot) {
    std::size_t pivot_row = pivot;
    Scalar pivot_abs = std::fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < kCartesianDim; ++row) {
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
      for (std::size_t col = pivot; col <= kCartesianDim; ++col) {
        std::swap(augmented[pivot][col], augmented[pivot_row][col]);
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

Result validate_admittance_state(StateView& state, const DType dtype, const std::size_t num_envs) {
  StateEntryView* pose = find_state(state, kAdmittancePose);
  StateEntryView* twist = find_state(state, kAdmittanceTwist);
  StateEntryView* pose_previous = find_state(state, kPosePreviousSignal);
  StateEntryView* twist_previous = find_state(state, kTwistPreviousSignal);
  StateEntryView* pose_initialized = find_state(state, kPoseIntegratorInitialized);
  StateEntryView* twist_initialized = find_state(state, kTwistIntegratorInitialized);
  if (pose == nullptr || twist == nullptr || pose_previous == nullptr ||
      twist_previous == nullptr || pose_initialized == nullptr || twist_initialized == nullptr) {
    return internal::invalid_view("admittance controller state entries are incomplete");
  }
  Result result = internal::validate_matching_output(pose->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(twist->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matching_output(pose_previous->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matching_output(twist_previous->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = validate_initialized_state(pose_initialized->values, num_envs);
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(twist_initialized->values, num_envs);
}

template <typename Scalar>
bool limit_vector_parameter_is_valid(const ParameterSetView& parameters,
                                     const char* name,
                                     const std::size_t env) {
  const ParameterView* parameter = find_parameter(parameters, name);
  if (parameter == nullptr) {
    return true;
  }
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar value = internal::read_vector_parameter<Scalar>(
        parameter, env, dim, std::numeric_limits<Scalar>::infinity());
    if (std::isnan(value) || value < static_cast<Scalar>(0)) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
Scalar read_limit_parameter(const ParameterSetView& parameters,
                            const char* name,
                            const std::size_t env,
                            const std::size_t dim) {
  return internal::read_vector_parameter<Scalar>(
      find_parameter(parameters, name), env, dim, std::numeric_limits<Scalar>::infinity());
}

template <typename Scalar>
Result apply_cartesian_admittance_parameters_impl(const ParameterSetView& parameters,
                                                  const std::size_t num_envs,
                                                  MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matrix_parameter_metadata(find_parameter(parameters, kDampingMatrix),
                                                        "damping_matrix parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kVirtualMassMatrix),
                                                   "virtual_mass_matrix parameter is missing",
                                                   dtype_for<Scalar>(),
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(
      find_parameter(parameters, kForceFilterThreshold),
      "force_filter_threshold parameter is missing",
      dtype_for<Scalar>(),
      num_envs,
      kCartesianDim,
      true);
  if (!result.ok()) {
    return result;
  }
  const char* optional_limits[] = {kAccelerationLimit, kTwistLimit, kPoseLimit};
  for (const char* name : optional_limits) {
    result = internal::validate_vector_parameter_metadata(find_parameter(parameters, name),
                                                          "limit parameter is invalid",
                                                          dtype_for<Scalar>(),
                                                          num_envs,
                                                          kCartesianDim,
                                                          false);
    if (!result.ok()) {
      return result;
    }
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!matrix_parameter_is_valid<Scalar>(
            parameters, kStiffnessMatrix, env, kCartesianDim, kCartesianDim) ||
        !matrix_parameter_is_valid<Scalar>(
            parameters, kDampingMatrix, env, kCartesianDim, kCartesianDim) ||
        !matrix_parameter_is_valid<Scalar>(
            parameters, kVirtualMassMatrix, env, kCartesianDim, kCartesianDim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
    const ParameterView* threshold = find_parameter(parameters, kForceFilterThreshold);
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar value =
          internal::read_vector_parameter<Scalar>(threshold, env, dim, static_cast<Scalar>(0));
      if (std::isnan(value)) {
        parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
        break;
      }
    }
    if (!limit_vector_parameter_is_valid<Scalar>(parameters, kAccelerationLimit, env) ||
        !limit_vector_parameter_is_valid<Scalar>(parameters, kTwistLimit, env) ||
        !limit_vector_parameter_is_valid<Scalar>(parameters, kPoseLimit, env)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

PrepareResult prepare_cartesian_admittance_controller(
    const TensorView& external_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    const ParameterSetView& parameters,
    StateView& state,
    const MutableTensorView& commanded_acceleration) {
  (void)state;
  PrepareResult prepare_result;
  if (external_wrench.device == Device::kCuda || feedforward_wrench.device == Device::kCuda ||
      dt.device == Device::kCuda || commanded_acceleration.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result = validate_cuda_signal_metadata(
        external_wrench, "external_wrench must be CUDA [num_envs, 6]", &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    std::size_t feedforward_envs = 0;
    std::size_t feedforward_dims = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(feedforward_wrench,
                                      "feedforward_wrench must be CUDA [num_envs, 6]",
                                      &feedforward_envs,
                                      &feedforward_dims);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (feedforward_envs != num_envs || feedforward_dims != kCartesianDim ||
        feedforward_wrench.dtype != external_wrench.dtype) {
      prepare_result.result =
          internal::invalid_view("feedforward_wrench must match external_wrench");
      return prepare_result;
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", external_wrench.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_cartesian_output_metadata(
        commanded_acceleration, external_wrench.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    const char* matrix_names[] = {kStiffnessMatrix, kDampingMatrix, kVirtualMassMatrix};
    for (const char* name : matrix_names) {
      prepare_result.result =
          validate_cuda_matrix6_parameter_metadata(find_parameter(parameters, name),
                                                   "admittance matrix parameter is missing",
                                                   external_wrench.dtype,
                                                   num_envs);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kForceFilterThreshold),
                                                "force_filter_threshold parameter is missing",
                                                external_wrench.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    const char* optional_limits[] = {kAccelerationLimit, kTwistLimit, kPoseLimit};
    for (const char* name : optional_limits) {
      prepare_result.result = validate_cuda_optional_vector_parameter_metadata(
          find_parameter(parameters, name), external_wrench.dtype, num_envs, kCartesianDim);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.state.push_back(internal::matrix_descriptor("admittance_pose",
                                                               external_wrench.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "meter_and_radian",
                                                               "task_space",
                                                               "commanded admittance pose offset"));
    prepare_result.state.push_back(
        internal::matrix_descriptor("admittance_twist",
                                    external_wrench.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "meter_per_second_and_radian_per_second",
                                    "task_space",
                                    "commanded admittance twist"));
    prepare_result.state.push_back(
        internal::matrix_descriptor("pose_previous_signal",
                                    external_wrench.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "meter_per_second_and_radian_per_second",
                                    "task_space",
                                    "previous pose-integrator input"));
    prepare_result.state.push_back(
        internal::matrix_descriptor("twist_previous_signal",
                                    external_wrench.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "meter_per_second_squared_and_radian_per_second_squared",
                                    "task_space",
                                    "previous twist-integrator input"));
    prepare_result.state.push_back(internal::vector_descriptor("pose_integrator_initialized",
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "pose integrator initialized flag"));
    prepare_result.state.push_back(
        internal::vector_descriptor("twist_integrator_initialized",
                                    DType::kBool,
                                    num_envs,
                                    "unitless",
                                    "none",
                                    "twist integrator initialized flag"));
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("commanded_acceleration",
                                    external_wrench.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "meter_per_second_squared_and_radian_per_second_squared",
                                    "task_space",
                                    "admittance commanded acceleration"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result =
      validate_cartesian_signal(external_wrench, "external_wrench tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t feedforward_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      feedforward_wrench, "feedforward_wrench tensor data is null", &feedforward_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (feedforward_envs != num_envs || feedforward_wrench.dtype != external_wrench.dtype) {
    prepare_result.result =
        internal::invalid_view("feedforward_wrench must match external_wrench dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(commanded_acceleration, external_wrench.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kStiffnessMatrix),
                                                   "stiffness_matrix parameter is missing",
                                                   external_wrench.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kDampingMatrix),
                                                   "damping_matrix parameter is missing",
                                                   external_wrench.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_matrix_parameter_metadata(find_parameter(parameters, kVirtualMassMatrix),
                                                   "virtual_mass_matrix parameter is missing",
                                                   external_wrench.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_vector_parameter_metadata(
      find_parameter(parameters, kForceFilterThreshold),
      "force_filter_threshold parameter is missing",
      external_wrench.dtype,
      num_envs,
      kCartesianDim,
      true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.state.push_back(internal::matrix_descriptor("admittance_pose",
                                                             external_wrench.dtype,
                                                             num_envs,
                                                             kCartesianDim,
                                                             "meter_and_radian",
                                                             "task_space",
                                                             "commanded admittance pose offset"));
  prepare_result.state.push_back(
      internal::matrix_descriptor("admittance_twist",
                                  external_wrench.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "meter_per_second_and_radian_per_second",
                                  "task_space",
                                  "commanded admittance twist"));
  prepare_result.state.push_back(
      internal::matrix_descriptor("pose_previous_signal",
                                  external_wrench.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "meter_per_second_and_radian_per_second",
                                  "task_space",
                                  "previous pose-integrator input"));
  prepare_result.state.push_back(
      internal::matrix_descriptor("twist_previous_signal",
                                  external_wrench.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "meter_per_second_squared_and_radian_per_second_squared",
                                  "task_space",
                                  "previous twist-integrator input"));
  prepare_result.state.push_back(internal::vector_descriptor("pose_integrator_initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "pose integrator initialized flag"));
  prepare_result.state.push_back(internal::vector_descriptor("twist_integrator_initialized",
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "twist integrator initialized flag"));
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("commanded_acceleration",
                                  external_wrench.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "meter_per_second_squared_and_radian_per_second_squared",
                                  "task_space",
                                  "admittance commanded acceleration"));
  return prepare_result;
}

Result apply_cartesian_admittance_controller_parameters(const ParameterSetView& parameters,
                                                        const std::size_t num_envs,
                                                        MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_admittance_controller_parameters_cuda(
        parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32 ? apply_cartesian_admittance_parameters_impl<float>(
                                        parameters, num_envs, parameter_status)
                                  : apply_cartesian_admittance_parameters_impl<double>(
                                        parameters, num_envs, parameter_status);
}

template <typename Scalar>
Result reset_cartesian_admittance_impl(const std::size_t num_envs,
                                       StateView& state,
                                       const TensorView& reset_mask,
                                       MutableStatusView status) {
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_admittance_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* pose = find_state(state, kAdmittancePose);
  StateEntryView* twist = find_state(state, kAdmittanceTwist);
  StateEntryView* pose_previous = find_state(state, kPosePreviousSignal);
  StateEntryView* twist_previous = find_state(state, kTwistPreviousSignal);
  StateEntryView* pose_initialized = find_state(state, kPoseIntegratorInitialized);
  StateEntryView* twist_initialized = find_state(state, kTwistIntegratorInitialized);
  auto* pose_init = static_cast<std::uint8_t*>(pose_initialized->values.data);
  auto* twist_init = static_cast<std::uint8_t*>(twist_initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      internal::write_matrix_value(pose->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(twist->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(pose_previous->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(twist_previous->values, env, dim, static_cast<Scalar>(0));
    }
    pose_init[env * static_cast<std::size_t>(pose_initialized->values.strides[0])] = 0U;
    twist_init[env * static_cast<std::size_t>(twist_initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

Result reset_cartesian_admittance_controller(const std::size_t num_envs,
                                             StateView& state,
                                             const TensorView& reset_mask,
                                             MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_admittance_controller_cuda(num_envs, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* pose = find_state(state, kAdmittancePose);
  if (pose == nullptr) {
    return internal::invalid_view("admittance_pose state is missing");
  }
  if (pose->values.dtype == DType::kFloat32) {
    return reset_cartesian_admittance_impl<float>(num_envs, state, reset_mask, status);
  }
  if (pose->values.dtype == DType::kFloat64) {
    return reset_cartesian_admittance_impl<double>(num_envs, state, reset_mask, status);
  }
  return internal::invalid_view("admittance state dtype must be float32 or float64");
}

template <typename Scalar>
Result step_cartesian_admittance_impl(const TensorView& external_wrench,
                                      const TensorView& feedforward_wrench,
                                      const TensorView& dt,
                                      const ParameterSetView& parameters,
                                      StatusView parameter_status,
                                      StateView& state,
                                      const MutableTensorView& commanded_acceleration,
                                      MutableStatusView status) {
  std::size_t num_envs = 0;
  Result result =
      validate_cartesian_signal(external_wrench, "external_wrench tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t feedforward_envs = 0;
  result = validate_cartesian_signal(
      feedforward_wrench, "feedforward_wrench tensor data is null", &feedforward_envs);
  if (!result.ok()) {
    return result;
  }
  if (feedforward_envs != num_envs || feedforward_wrench.dtype != external_wrench.dtype) {
    return internal::invalid_view(
        "feedforward_wrench must match external_wrench dtype and num_envs");
  }
  result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(commanded_acceleration, external_wrench.dtype, num_envs);
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
  result = validate_admittance_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_cartesian_admittance_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* pose = find_state(state, kAdmittancePose);
  StateEntryView* twist = find_state(state, kAdmittanceTwist);
  StateEntryView* pose_previous = find_state(state, kPosePreviousSignal);
  StateEntryView* twist_previous = find_state(state, kTwistPreviousSignal);
  StateEntryView* pose_initialized = find_state(state, kPoseIntegratorInitialized);
  StateEntryView* twist_initialized = find_state(state, kTwistIntegratorInitialized);
  auto* pose_init = static_cast<std::uint8_t*>(pose_initialized->values.data);
  auto* twist_init = static_cast<std::uint8_t*>(twist_initialized->values.data);
  const ParameterView* stiffness_parameter = find_parameter(parameters, kStiffnessMatrix);
  const ParameterView* damping_parameter = find_parameter(parameters, kDampingMatrix);
  const ParameterView* mass_parameter = find_parameter(parameters, kVirtualMassMatrix);
  const ParameterView* threshold_parameter = find_parameter(parameters, kForceFilterThreshold);

  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(external_wrench, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(feedforward_wrench, env, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(commanded_acceleration, env);
      continue;
    }

    const auto stiffness_view = core_matrix_parameter<Scalar>(stiffness_parameter, env);
    const auto damping_view = core_matrix_parameter<Scalar>(damping_parameter, env);
    const auto pose_view = core_vector_env<Scalar>(pose->values, env);
    const auto twist_view = core_vector_env<Scalar>(twist->values, env);
    Scalar generalized_force[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      const Scalar threshold = internal::read_vector_parameter<Scalar>(
          threshold_parameter, env, row, static_cast<Scalar>(0));
      const Scalar filtered_external = detail::deadband_value(
          internal::read_matrix_value<Scalar>(external_wrench, env, row), threshold);
      generalized_force[row] = detail::admittance_force_balance_value(
          filtered_external,
          internal::read_matrix_value<Scalar>(feedforward_wrench, env, row),
          detail::cartesian_stiffness_value(stiffness_view, pose_view, row),
          detail::cartesian_damping_value(damping_view, twist_view, row));
    }

    Scalar mass[6][6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        mass[row][col] = internal::read_matrix_parameter<Scalar>(
            mass_parameter, env, row, col, static_cast<Scalar>(0));
      }
    }
    Scalar acceleration[6]{};
    if (!solve_matrix6_vector(mass, generalized_force, acceleration)) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(commanded_acceleration, env);
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      const Scalar limit = read_limit_parameter<Scalar>(parameters, kAccelerationLimit, env, dim);
      const auto limited = detail::clamp_value(acceleration[dim], -limit, limit);
      acceleration[dim] = limited.value;
      if (limited.saturated) {
        status.flags[env] |= status_bit(StatusFlag::kSaturation);
      }
      internal::write_matrix_value(commanded_acceleration, env, dim, acceleration[dim]);
    }

    const std::size_t twist_init_offset =
        env * static_cast<std::size_t>(twist_initialized->values.strides[0]);
    Scalar new_twist[6]{};
    if (twist_init[twist_init_offset] == 0U) {
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(twist_previous->values, env, dim, acceleration[dim]);
        internal::write_matrix_value(twist->values, env, dim, static_cast<Scalar>(0));
      }
      twist_init[twist_init_offset] = 1U;
    } else {
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        const Scalar old_value = internal::read_matrix_value<Scalar>(twist->values, env, dim);
        const Scalar previous_input =
            internal::read_matrix_value<Scalar>(twist_previous->values, env, dim);
        const Scalar limit = read_limit_parameter<Scalar>(parameters, kTwistLimit, env, dim);
        const auto update = detail::integrator_trapezoid_value(
            old_value, acceleration[dim], previous_input, dt_value, -limit, limit);
        if (update.saturated) {
          status.flags[env] |= status_bit(StatusFlag::kSaturation);
        }
        new_twist[dim] = update.value;
        internal::write_matrix_value(twist->values, env, dim, update.value);
        internal::write_matrix_value(twist_previous->values, env, dim, acceleration[dim]);
      }
    }

    const std::size_t pose_init_offset =
        env * static_cast<std::size_t>(pose_initialized->values.strides[0]);
    if (pose_init[pose_init_offset] == 0U) {
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(pose_previous->values, env, dim, new_twist[dim]);
        internal::write_matrix_value(pose->values, env, dim, static_cast<Scalar>(0));
      }
      pose_init[pose_init_offset] = 1U;
    } else {
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        const Scalar old_value = internal::read_matrix_value<Scalar>(pose->values, env, dim);
        const Scalar previous_input =
            internal::read_matrix_value<Scalar>(pose_previous->values, env, dim);
        const Scalar limit = read_limit_parameter<Scalar>(parameters, kPoseLimit, env, dim);
        const auto update = detail::integrator_trapezoid_value(
            old_value, new_twist[dim], previous_input, dt_value, -limit, limit);
        if (update.saturated) {
          status.flags[env] |= status_bit(StatusFlag::kSaturation);
        }
        internal::write_matrix_value(pose->values, env, dim, update.value);
        internal::write_matrix_value(pose_previous->values, env, dim, new_twist[dim]);
      }
    }
  }
  return internal::ok();
}

Result step_cartesian_admittance_controller(const TensorView& external_wrench,
                                            const TensorView& feedforward_wrench,
                                            const TensorView& dt,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            StateView& state,
                                            const MutableTensorView& commanded_acceleration,
                                            MutableStatusView status) {
  if (external_wrench.device == Device::kCuda || feedforward_wrench.device == Device::kCuda ||
      dt.device == Device::kCuda || commanded_acceleration.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_admittance_controller_cuda(external_wrench,
                                                     feedforward_wrench,
                                                     dt,
                                                     parameters,
                                                     parameter_status,
                                                     state,
                                                     commanded_acceleration,
                                                     status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (external_wrench.dtype == DType::kFloat32) {
    return step_cartesian_admittance_impl<float>(external_wrench,
                                                 feedforward_wrench,
                                                 dt,
                                                 parameters,
                                                 parameter_status,
                                                 state,
                                                 commanded_acceleration,
                                                 status);
  }
  if (external_wrench.dtype == DType::kFloat64) {
    return step_cartesian_admittance_impl<double>(external_wrench,
                                                  feedforward_wrench,
                                                  dt,
                                                  parameters,
                                                  parameter_status,
                                                  state,
                                                  commanded_acceleration,
                                                  status);
  }
  return internal::invalid_view("external_wrench dtype must be float32 or float64");
}

template <typename Scalar>
Result apply_operational_space_parameters_impl(const ParameterSetView& parameters,
                                               const std::size_t num_envs,
                                               MutableStatusView parameter_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kPositionGain),
                                                        "position_gain parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kVelocityGain),
                                                        "velocity_gain parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        true);
  if (!result.ok()) {
    return result;
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::vector_parameters_are_finite<Scalar>(
            parameters, kPositionGain, kVelocityGain, nullptr, env, kCartesianDim)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      if (internal::read_vector_parameter<Scalar>(
              find_parameter(parameters, kPositionGain), env, dim, static_cast<Scalar>(-1)) <
              static_cast<Scalar>(0) ||
          internal::read_vector_parameter<Scalar>(
              find_parameter(parameters, kVelocityGain), env, dim, static_cast<Scalar>(-1)) <
              static_cast<Scalar>(0)) {
        parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
        break;
      }
    }
  }
  return internal::ok();
}

PrepareResult prepare_operational_space_controller(const TensorView& cartesian_pose_error,
                                                   const TensorView& cartesian_velocity_error,
                                                   const TensorView& desired_acceleration,
                                                   const TensorView& jacobian_dot_joint_velocity,
                                                   const TensorView& cartesian_mass_matrix,
                                                   const TensorView& jacobian,
                                                   const TensorView& joint_bias_torque,
                                                   const ParameterSetView& parameters,
                                                   const MutableTensorView& joint_torque) {
  PrepareResult prepare_result;
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda ||
      jacobian_dot_joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      joint_bias_torque.device == Device::kCuda || joint_torque.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(cartesian_pose_error,
                                      "cartesian_pose_error must be CUDA [num_envs, 6]",
                                      &num_envs,
                                      &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    const TensorView signals[] = {
        cartesian_velocity_error, desired_acceleration, jacobian_dot_joint_velocity};
    for (const TensorView& signal : signals) {
      std::size_t signal_envs = 0;
      std::size_t signal_dims = 0;
      prepare_result.result = validate_cuda_signal_metadata(
          signal, "operational-space signal is invalid", &signal_envs, &signal_dims);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
      if (signal_envs != num_envs || signal_dims != kCartesianDim ||
          signal.dtype != cartesian_pose_error.dtype) {
        prepare_result.result =
            internal::invalid_view("operational-space signals must share dtype and num_envs");
        return prepare_result;
      }
    }
    prepare_result.result =
        validate_cuda_tensor3_metadata(cartesian_mass_matrix,
                                       "cartesian_mass_matrix tensor data is null",
                                       cartesian_pose_error.dtype,
                                       num_envs,
                                       kCartesianDim,
                                       kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    std::size_t joint_dim = 0;
    prepare_result.result = validate_cuda_joint_signal_metadata(joint_bias_torque,
                                                                "joint_bias_torque is invalid",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(jacobian,
                                                           "jacobian is invalid",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_signal_output_metadata(
        joint_torque, cartesian_pose_error.dtype, num_envs, joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kPositionGain),
                                                "position_gain parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kVelocityGain),
                                                "velocity_gain parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = joint_dim;
    prepare_result.outputs.push_back(internal::matrix_descriptor("joint_torque",
                                                                 cartesian_pose_error.dtype,
                                                                 num_envs,
                                                                 joint_dim,
                                                                 "newton_meter",
                                                                 "joint_space",
                                                                 "operational-space joint torque"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t other_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &other_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || cartesian_velocity_error.dtype != cartesian_pose_error.dtype) {
    prepare_result.result = internal::invalid_view(
        "cartesian_velocity_error must match cartesian_pose_error dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = validate_cartesian_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &other_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || desired_acceleration.dtype != cartesian_pose_error.dtype) {
    prepare_result.result = internal::invalid_view(
        "desired_acceleration must match cartesian_pose_error dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = validate_cartesian_signal(
      jacobian_dot_joint_velocity, "jacobian_dot_joint_velocity data is null", &other_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || jacobian_dot_joint_velocity.dtype != cartesian_pose_error.dtype) {
    prepare_result.result = internal::invalid_view(
        "jacobian_dot_joint_velocity must match cartesian_pose_error dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_batched_matrix(cartesian_mass_matrix,
                                        "cartesian_mass_matrix tensor data is null",
                                        cartesian_pose_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_bias_torque, "joint_bias_torque data is null", &joint_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_envs != num_envs || joint_bias_torque.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("joint_bias_torque must match Cartesian dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(jacobian,
                                                            "jacobian data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matching_output(
      joint_torque, cartesian_pose_error.dtype, num_envs, joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kPositionGain),
                                                   "position_gain parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kVelocityGain),
                                                   "velocity_gain parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = joint_dim;
  prepare_result.outputs.push_back(internal::matrix_descriptor("joint_torque",
                                                               cartesian_pose_error.dtype,
                                                               num_envs,
                                                               joint_dim,
                                                               "newton_meter",
                                                               "joint_space",
                                                               "operational-space joint torque"));
  return prepare_result;
}

Result apply_operational_space_controller_parameters(const ParameterSetView& parameters,
                                                     const std::size_t num_envs,
                                                     MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_operational_space_controller_parameters_cuda(
        parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  if (dtype == DType::kFloat32) {
    return apply_operational_space_parameters_impl<float>(parameters, num_envs, parameter_status);
  }
  if (dtype == DType::kFloat64) {
    return apply_operational_space_parameters_impl<double>(parameters, num_envs, parameter_status);
  }
  return internal::invalid_view("operational-space parameters must be float32 or float64");
}

Result reset_operational_space_controller(const std::size_t num_envs,
                                          const TensorView& reset_mask,
                                          MutableStatusView status) {
  if (utility_lifecycle_uses_cuda(reset_mask, status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_stateless_cuda(num_envs, reset_mask, status);
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

template <typename Scalar>
Result step_operational_space_controller_impl(const TensorView& cartesian_pose_error,
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
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  std::size_t other_envs = 0;
  result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &other_envs);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || cartesian_velocity_error.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("cartesian_velocity_error must match cartesian_pose_error");
  }
  result = validate_cartesian_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &other_envs);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || desired_acceleration.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("desired_acceleration must match cartesian_pose_error");
  }
  result = validate_cartesian_signal(
      jacobian_dot_joint_velocity, "jacobian_dot_joint_velocity tensor data is null", &other_envs);
  if (!result.ok()) {
    return result;
  }
  if (other_envs != num_envs || jacobian_dot_joint_velocity.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("jacobian_dot_joint_velocity must match cartesian_pose_error");
  }
  result = internal::validate_batched_matrix(cartesian_mass_matrix,
                                             "cartesian_mass_matrix tensor data is null",
                                             cartesian_pose_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  result = internal::validate_batched_signal(
      joint_bias_torque, "joint_bias_torque tensor data is null", &joint_envs, &joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (joint_envs != num_envs || joint_bias_torque.dtype != cartesian_pose_error.dtype) {
    return internal::invalid_view("joint_bias_torque must match Cartesian dtype and num_envs");
  }
  result = internal::validate_batched_matrix(jacobian,
                                             "jacobian tensor data is null",
                                             cartesian_pose_error.dtype,
                                             num_envs,
                                             kCartesianDim,
                                             joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(
      joint_torque, cartesian_pose_error.dtype, num_envs, joint_dim);
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
  result = apply_operational_space_parameters_impl<Scalar>(parameters, num_envs, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }
  const ParameterView* position_gain = find_parameter(parameters, kPositionGain);
  const ParameterView* velocity_gain = find_parameter(parameters, kVelocityGain);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::matrix_env_is_finite<Scalar>(cartesian_pose_error, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(cartesian_velocity_error, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(desired_acceleration, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(jacobian_dot_joint_velocity, env, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(
            cartesian_mass_matrix, env, kCartesianDim, kCartesianDim) ||
        !internal::tensor3_env_is_finite<Scalar>(jacobian, env, kCartesianDim, joint_dim) ||
        !internal::matrix_env_is_finite<Scalar>(joint_bias_torque, env, joint_dim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      if (internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, dim, dim) <
          static_cast<Scalar>(0)) {
        status.flags[env] |= status_bit(StatusFlag::kInvalidInput);
      }
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
      continue;
    }
    Scalar commanded_acceleration[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      commanded_acceleration[row] = detail::operational_space_acceleration_value(
          internal::read_matrix_value<Scalar>(desired_acceleration, env, row) -
              internal::read_matrix_value<Scalar>(jacobian_dot_joint_velocity, env, row),
          detail::proportional_value(
              internal::read_vector_parameter<Scalar>(
                  position_gain, env, row, static_cast<Scalar>(0)),
              internal::read_matrix_value<Scalar>(cartesian_pose_error, env, row)),
          detail::proportional_value(
              internal::read_vector_parameter<Scalar>(
                  velocity_gain, env, row, static_cast<Scalar>(0)),
              internal::read_matrix_value<Scalar>(cartesian_velocity_error, env, row)));
    }
    Scalar wrench[6]{};
    for (std::size_t row = 0; row < kCartesianDim; ++row) {
      for (std::size_t col = 0; col < kCartesianDim; ++col) {
        wrench[row] += internal::read_tensor3_value<Scalar>(cartesian_mass_matrix, env, row, col) *
                       commanded_acceleration[col];
      }
    }
    bool torque_is_finite = true;
    for (std::size_t joint = 0; joint < joint_dim; ++joint) {
      Scalar torque = internal::read_matrix_value<Scalar>(joint_bias_torque, env, joint);
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        torque += internal::read_tensor3_value<Scalar>(jacobian, env, row, joint) * wrench[row];
      }
      torque_is_finite = torque_is_finite && std::isfinite(torque);
      internal::write_matrix_value(joint_torque, env, joint, torque);
    }
    if (!torque_is_finite) {
      status.flags[env] |=
          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      internal::zero_output_env<Scalar>(joint_torque, env, joint_dim);
    }
  }
  return internal::ok();
}

Result step_operational_space_controller(const TensorView& cartesian_pose_error,
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
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda ||
      jacobian_dot_joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      joint_bias_torque.device == Device::kCuda || joint_torque.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_operational_space_controller_cuda(cartesian_pose_error,
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
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    return step_operational_space_controller_impl<float>(cartesian_pose_error,
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
  if (cartesian_pose_error.dtype == DType::kFloat64) {
    return step_operational_space_controller_impl<double>(cartesian_pose_error,
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
  return internal::invalid_view("cartesian_pose_error dtype must be float32 or float64");
}

template <typename Scalar>
Result apply_force_parameters(const ParameterSetView& parameters,
                              std::size_t num_envs,
                              bool use_shaping_function,
                              MutableStatusView parameter_status,
                              bool clear_status = true);

Result validate_status_scratch(const MutableTensorView& view, const std::size_t num_envs) {
  if (view.data == nullptr) {
    return internal::invalid_view("scratch status data is null");
  }
  if (internal::is_cuda(view)) {
    return internal::unsupported("CUDA scratch status writes require CUDA tensor kernels");
  }
  if (!internal::is_cpu(view)) {
    return internal::invalid_view("scratch status device is unsupported");
  }
  if (view.dtype != DType::kUInt64 || view.rank != 1 ||
      view.shape[0] < static_cast<std::int64_t>(num_envs) || view.strides[0] < 1) {
    return internal::invalid_view("scratch status must be uint64 [num_envs]");
  }
  return internal::ok();
}

Result validate_unified_scratch(ScratchView& scratch,
                                const DType dtype,
                                const std::size_t num_envs) {
  ScratchEntryView* impedance_wrench = find_scratch(scratch, kImpedanceWrenchScratch);
  ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
  ScratchEntryView* impedance_status = find_scratch(scratch, kImpedanceStatusScratch);
  ScratchEntryView* force_status = find_scratch(scratch, kForceStatusScratch);
  if (impedance_wrench == nullptr || force_wrench == nullptr || impedance_status == nullptr ||
      force_status == nullptr) {
    return internal::invalid_view("unified force/impedance scratch entries are incomplete");
  }
  Result result = validate_cartesian_output(impedance_wrench->values, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(force_wrench->values, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_status_scratch(impedance_status->values, num_envs);
  if (!result.ok()) {
    return result;
  }
  return validate_status_scratch(force_status->values, num_envs);
}

MutableStatusView scratch_status_view(ScratchView& scratch,
                                      const char* name,
                                      const std::size_t num_envs) {
  ScratchEntryView* entry = find_scratch(scratch, name);
  return MutableStatusView{
      static_cast<std::uint64_t*>(entry->values.data), num_envs, entry->values.device};
}

PrepareResult prepare_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    const bool use_shaping_function,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench) {
  (void)scratch;
  PrepareResult prepare_result = prepare_cartesian_impedance_controller(cartesian_pose_error,
                                                                        cartesian_velocity_error,
                                                                        desired_acceleration,
                                                                        joint_velocity,
                                                                        cartesian_mass_matrix,
                                                                        jacobian,
                                                                        dt,
                                                                        parameters,
                                                                        state,
                                                                        cartesian_wrench);
  if (!prepare_result.ok()) {
    return prepare_result;
  }
  PrepareResult force_prepare = prepare_cartesian_force_controller(external_wrench,
                                                                   desired_wrench,
                                                                   feedforward_wrench,
                                                                   cartesian_pose_error,
                                                                   dt,
                                                                   use_shaping_function,
                                                                   parameters,
                                                                   state,
                                                                   cartesian_wrench);
  if (!force_prepare.ok()) {
    prepare_result.result = force_prepare.result;
    return prepare_result;
  }
  prepare_result.state.insert(
      prepare_result.state.end(), force_prepare.state.begin(), force_prepare.state.end());
  prepare_result.scratch.push_back(internal::matrix_descriptor(kImpedanceWrenchScratch,
                                                               cartesian_pose_error.dtype,
                                                               prepare_result.num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "impedance sub-controller wrench"));
  prepare_result.scratch.push_back(internal::matrix_descriptor(kForceWrenchScratch,
                                                               cartesian_pose_error.dtype,
                                                               prepare_result.num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "force sub-controller wrench"));
  prepare_result.scratch.push_back(internal::vector_descriptor(kImpedanceStatusScratch,
                                                               DType::kUInt64,
                                                               prepare_result.num_envs,
                                                               "status_flags",
                                                               "none",
                                                               "impedance sub-controller status"));
  prepare_result.scratch.push_back(internal::vector_descriptor(kForceStatusScratch,
                                                               DType::kUInt64,
                                                               prepare_result.num_envs,
                                                               "status_flags",
                                                               "none",
                                                               "force sub-controller status"));
  prepare_result.outputs.clear();
  prepare_result.outputs.push_back(internal::matrix_descriptor("cartesian_wrench",
                                                               cartesian_pose_error.dtype,
                                                               prepare_result.num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "unified force/impedance wrench"));
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || dt.device == Device::kCuda ||
      cartesian_wrench.device == Device::kCuda || parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    mark_descriptors_cuda(&prepare_result);
#endif
  }
  return prepare_result;
}

Result apply_cartesian_unified_force_impedance_controller_parameters(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    const bool use_shaping_function,
    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_unified_force_impedance_controller_parameters_cuda(
        parameters, num_envs, use_shaping_function, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const DType dtype = first_parameter_dtype(parameters);
  result = dtype == DType::kFloat32 ? apply_cartesian_impedance_parameters_impl<float>(
                                          parameters, num_envs, parameter_status, false)
                                    : apply_cartesian_impedance_parameters_impl<double>(
                                          parameters, num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  result = dtype == DType::kFloat32
               ? apply_force_parameters<float>(
                     parameters, num_envs, use_shaping_function, parameter_status, false)
               : apply_force_parameters<double>(
                     parameters, num_envs, use_shaping_function, parameter_status, false);
  return result;
}

Result reset_cartesian_unified_force_impedance_controller(const std::size_t num_envs,
                                                          const std::size_t joint_dim,
                                                          StateView& state,
                                                          const TensorView& reset_mask,
                                                          MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    Result result = reset_cartesian_force_controller(num_envs, state, reset_mask, status);
    if (!result.ok()) {
      return result;
    }
    return reset_cartesian_impedance_controller(num_envs, joint_dim, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = reset_cartesian_impedance_controller(num_envs, joint_dim, state, reset_mask, status);
  if (!result.ok()) {
    return result;
  }
  return reset_cartesian_force_controller(num_envs, state, reset_mask, status);
}

Result step_cartesian_unified_force_impedance_controller(const TensorView& cartesian_pose_error,
                                                         const TensorView& cartesian_velocity_error,
                                                         const TensorView& desired_acceleration,
                                                         const TensorView& joint_velocity,
                                                         const TensorView& cartesian_mass_matrix,
                                                         const TensorView& jacobian,
                                                         const TensorView& external_wrench,
                                                         const TensorView& desired_wrench,
                                                         const TensorView& feedforward_wrench,
                                                         const TensorView& dt,
                                                         const bool use_shaping_function,
                                                         const ParameterSetView& parameters,
                                                         StatusView parameter_status,
                                                         StateView& state,
                                                         ScratchView& scratch,
                                                         const MutableTensorView& cartesian_wrench,
                                                         MutableStatusView status) {
  if (cartesian_pose_error.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || dt.device == Device::kCuda ||
      cartesian_wrench.device == Device::kCuda || status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
    ScratchEntryView* impedance_wrench = find_scratch(scratch, kImpedanceWrenchScratch);
    ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
    ScratchEntryView* impedance_status_entry = find_scratch(scratch, kImpedanceStatusScratch);
    ScratchEntryView* force_status_entry = find_scratch(scratch, kForceStatusScratch);
    if (impedance_wrench == nullptr || force_wrench == nullptr ||
        impedance_status_entry == nullptr || force_status_entry == nullptr) {
      return internal::invalid_view("unified force/impedance scratch entries are incomplete");
    }
    MutableStatusView impedance_status =
        scratch_status_view(scratch, kImpedanceStatusScratch, num_envs);
    MutableStatusView force_status = scratch_status_view(scratch, kForceStatusScratch, num_envs);
    Result result = step_cartesian_impedance_controller(cartesian_pose_error,
                                                        cartesian_velocity_error,
                                                        desired_acceleration,
                                                        joint_velocity,
                                                        cartesian_mass_matrix,
                                                        jacobian,
                                                        dt,
                                                        parameters,
                                                        parameter_status,
                                                        state,
                                                        impedance_wrench->values,
                                                        impedance_status);
    if (!result.ok()) {
      return result;
    }
    result = step_cartesian_force_controller(external_wrench,
                                             desired_wrench,
                                             feedforward_wrench,
                                             cartesian_pose_error,
                                             dt,
                                             use_shaping_function,
                                             parameters,
                                             parameter_status,
                                             state,
                                             force_wrench->values,
                                             force_status);
    if (!result.ok()) {
      return result;
    }
    return combine_cartesian_wrench_sum_cuda(
        impedance_wrench->values.as_const(),
        StatusView{impedance_status.flags, num_envs, impedance_status.device},
        force_wrench->values.as_const(),
        StatusView{force_status.flags, num_envs, force_status.device},
        cartesian_wrench,
        status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_unified_scratch(scratch, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  ScratchEntryView* impedance_wrench = find_scratch(scratch, kImpedanceWrenchScratch);
  ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
  MutableStatusView impedance_status =
      scratch_status_view(scratch, kImpedanceStatusScratch, num_envs);
  MutableStatusView force_status = scratch_status_view(scratch, kForceStatusScratch, num_envs);

  result = step_cartesian_impedance_controller(cartesian_pose_error,
                                               cartesian_velocity_error,
                                               desired_acceleration,
                                               joint_velocity,
                                               cartesian_mass_matrix,
                                               jacobian,
                                               dt,
                                               parameters,
                                               parameter_status,
                                               state,
                                               impedance_wrench->values,
                                               impedance_status);
  if (!result.ok()) {
    return result;
  }
  result = step_cartesian_force_controller(external_wrench,
                                           desired_wrench,
                                           feedforward_wrench,
                                           cartesian_pose_error,
                                           dt,
                                           use_shaping_function,
                                           parameters,
                                           parameter_status,
                                           state,
                                           force_wrench->values,
                                           force_status);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] = impedance_status.flags[env] | force_status.flags[env];
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<float>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            dim,
            detail::unified_force_impedance_value(
                internal::read_matrix_value<float>(impedance_wrench->values, env, dim),
                internal::read_matrix_value<float>(force_wrench->values, env, dim)));
      }
    }
  } else {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] = impedance_status.flags[env] | force_status.flags[env];
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<double>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            dim,
            detail::unified_force_impedance_value(
                internal::read_matrix_value<double>(impedance_wrench->values, env, dim),
                internal::read_matrix_value<double>(force_wrench->values, env, dim)));
      }
    }
  }
  return internal::ok();
}

Result make_variable_stiffness_state(StateView& state,
                                     StateEntryView entries[3],
                                     StateView* aliased_state) {
  StateEntryView* current = find_state(state, kCurrentStiffnessMatrix);
  StateEntryView* previous = find_state(state, kPreviousStiffnessMatrix);
  StateEntryView* initialized = find_state(state, kStiffnessInitialized);
  if (current == nullptr || previous == nullptr || initialized == nullptr) {
    return internal::invalid_view("variable stiffness composite state entries are incomplete");
  }
  entries[0] = StateEntryView{kCurrentStiffnessMatrix, current->values};
  entries[1] = StateEntryView{kPreviousStiffnessMatrix, previous->values};
  entries[2] = StateEntryView{kInitialized, initialized->values};
  *aliased_state = StateView{entries, 3};
  return internal::ok();
}

Result make_inertial_state(StateView& state, StateEntryView entries[3], StateView* aliased_state) {
  StateEntryView* previous_mass = find_state(state, kPreviousMassMatrix);
  StateEntryView* previous_jacobian = find_state(state, kPreviousJacobian);
  StateEntryView* initialized = find_state(state, kInertialInitialized);
  if (previous_mass == nullptr || previous_jacobian == nullptr || initialized == nullptr) {
    return internal::invalid_view("inertial composite state entries are incomplete");
  }
  entries[0] = StateEntryView{kPreviousMassMatrix, previous_mass->values};
  entries[1] = StateEntryView{kPreviousJacobian, previous_jacobian->values};
  entries[2] = StateEntryView{kInitialized, initialized->values};
  *aliased_state = StateView{entries, 3};
  return internal::ok();
}

Result validate_variable_impedance_scratch(ScratchView& scratch,
                                           const DType dtype,
                                           const std::size_t num_envs) {
  const char* wrench_names[] = {
      kStiffnessWrenchScratch, kDampingWrenchScratch, kInertialWrenchScratch};
  for (const char* name : wrench_names) {
    ScratchEntryView* entry = find_scratch(scratch, name);
    if (entry == nullptr) {
      return internal::invalid_view("variable impedance wrench scratch is incomplete");
    }
    Result result = validate_cartesian_output(entry->values, dtype, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  const char* status_names[] = {
      kStiffnessStatusScratch, kDampingStatusScratch, kInertialStatusScratch};
  for (const char* name : status_names) {
    ScratchEntryView* entry = find_scratch(scratch, name);
    if (entry == nullptr) {
      return internal::invalid_view("variable impedance status scratch is incomplete");
    }
    Result result = validate_status_scratch(entry->values, num_envs);
    if (!result.ok()) {
      return result;
    }
  }
  return internal::ok();
}

void append_variable_impedance_state_descriptors(PrepareResult* prepare_result,
                                                 const DType dtype,
                                                 const std::size_t num_envs,
                                                 const std::size_t joint_dim) {
  prepare_result->state.push_back(internal::tensor3_descriptor(kCurrentStiffnessMatrix,
                                                               dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               kCartesianDim,
                                                               "wrench_per_pose_error",
                                                               "task_space",
                                                               "current variable stiffness"));
  prepare_result->state.push_back(internal::tensor3_descriptor(kPreviousStiffnessMatrix,
                                                               dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               kCartesianDim,
                                                               "wrench_per_pose_error",
                                                               "task_space",
                                                               "previous variable stiffness"));
  prepare_result->state.push_back(internal::vector_descriptor(kStiffnessInitialized,
                                                              DType::kBool,
                                                              num_envs,
                                                              "unitless",
                                                              "none",
                                                              "variable stiffness initialized"));
  prepare_result->state.push_back(internal::tensor3_descriptor(kPreviousMassMatrix,
                                                               dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               kCartesianDim,
                                                               "operational_space_mass_units",
                                                               "task_space",
                                                               "previous Cartesian mass"));
  prepare_result->state.push_back(internal::tensor3_descriptor(kPreviousJacobian,
                                                               dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               joint_dim,
                                                               "task_velocity_per_joint_velocity",
                                                               "task_space_to_joint_space",
                                                               "previous Jacobian"));
  prepare_result->state.push_back(internal::vector_descriptor(kInertialInitialized,
                                                              DType::kBool,
                                                              num_envs,
                                                              "unitless",
                                                              "none",
                                                              "inertial coupling initialized"));
}

void append_variable_impedance_scratch_descriptors(PrepareResult* prepare_result,
                                                   const DType dtype,
                                                   const std::size_t num_envs) {
  prepare_result->scratch.push_back(internal::matrix_descriptor(kStiffnessWrenchScratch,
                                                                dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                "newton_and_newton_meter",
                                                                "task_space",
                                                                "variable stiffness wrench"));
  prepare_result->scratch.push_back(internal::matrix_descriptor(kDampingWrenchScratch,
                                                                dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                "newton_and_newton_meter",
                                                                "task_space",
                                                                "variable damping wrench"));
  prepare_result->scratch.push_back(internal::matrix_descriptor(kInertialWrenchScratch,
                                                                dtype,
                                                                num_envs,
                                                                kCartesianDim,
                                                                "newton_and_newton_meter",
                                                                "task_space",
                                                                "inertial coupling wrench"));
  prepare_result->scratch.push_back(internal::vector_descriptor(kStiffnessStatusScratch,
                                                                DType::kUInt64,
                                                                num_envs,
                                                                "status_flags",
                                                                "none",
                                                                "variable stiffness status"));
  prepare_result->scratch.push_back(internal::vector_descriptor(kDampingStatusScratch,
                                                                DType::kUInt64,
                                                                num_envs,
                                                                "status_flags",
                                                                "none",
                                                                "variable damping status"));
  prepare_result->scratch.push_back(internal::vector_descriptor(kInertialStatusScratch,
                                                                DType::kUInt64,
                                                                num_envs,
                                                                "status_flags",
                                                                "none",
                                                                "inertial coupling status"));
}

PrepareResult prepare_variable_cartesian_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& dt,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench) {
  (void)state;
  (void)scratch;
  PrepareResult prepare_result;
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result =
        validate_cuda_signal_metadata(cartesian_pose_error,
                                      "cartesian_pose_error must be CUDA [num_envs, 6]",
                                      &num_envs,
                                      &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    const TensorView signals[] = {cartesian_velocity_error, desired_acceleration};
    for (const TensorView& signal : signals) {
      std::size_t signal_envs = 0;
      std::size_t signal_dims = 0;
      prepare_result.result = validate_cuda_signal_metadata(
          signal, "variable impedance signal is invalid", &signal_envs, &signal_dims);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
      if (signal_envs != num_envs || signal_dims != kCartesianDim ||
          signal.dtype != cartesian_pose_error.dtype) {
        prepare_result.result =
            internal::invalid_view("variable impedance signals must share dtype and num_envs");
        return prepare_result;
      }
    }
    prepare_result.result = validate_cuda_tensor3_metadata(stiffness_matrix,
                                                           "stiffness_matrix tensor data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    std::size_t joint_dim = 0;
    prepare_result.result = validate_cuda_joint_signal_metadata(joint_velocity,
                                                                "joint_velocity must be CUDA",
                                                                cartesian_pose_error.dtype,
                                                                num_envs,
                                                                &joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(cartesian_mass_matrix,
                                                           "cartesian_mass_matrix data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_tensor3_metadata(jacobian,
                                                           "jacobian tensor data is null",
                                                           cartesian_pose_error.dtype,
                                                           num_envs,
                                                           kCartesianDim,
                                                           joint_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_cartesian_output_metadata(
        cartesian_wrench, cartesian_pose_error.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_matrix6_parameter_metadata(
        find_parameter(parameters, kInitialStiffnessMatrix),
        "initial_stiffness_matrix parameter is missing",
        cartesian_pose_error.dtype,
        num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffness),
                                                "maximum_stiffness parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result = validate_cuda_optional_vector_parameter_metadata(
        find_parameter(parameters, kMinimumStiffness),
        cartesian_pose_error.dtype,
        num_envs,
        kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumStiffnessRate),
                                                "maximum_stiffness_rate parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                "damping_factors parameter is missing",
                                                cartesian_pose_error.dtype,
                                                num_envs,
                                                kCartesianDim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    append_variable_impedance_state_descriptors(
        &prepare_result, cartesian_pose_error.dtype, num_envs, joint_dim);
    append_variable_impedance_scratch_descriptors(
        &prepare_result, cartesian_pose_error.dtype, num_envs);
    prepare_result.outputs.push_back(
        internal::matrix_descriptor("cartesian_wrench",
                                    cartesian_pose_error.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "newton_and_newton_meter",
                                    "task_space",
                                    "variable Cartesian impedance wrench"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t other_envs = 0;
  prepare_result.result = validate_cartesian_signal(
      cartesian_velocity_error, "cartesian_velocity_error tensor data is null", &other_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || cartesian_velocity_error.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("cartesian_velocity_error must match cartesian_pose_error");
    return prepare_result;
  }
  prepare_result.result = validate_cartesian_signal(
      desired_acceleration, "desired_acceleration tensor data is null", &other_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (other_envs != num_envs || desired_acceleration.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("desired_acceleration must match cartesian_pose_error");
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(stiffness_matrix,
                                                            "stiffness_matrix tensor data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  std::size_t joint_envs = 0;
  std::size_t joint_dim = 0;
  prepare_result.result = internal::validate_batched_signal(
      joint_velocity, "joint_velocity tensor data is null", &joint_envs, &joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  if (joint_envs != num_envs || joint_velocity.dtype != cartesian_pose_error.dtype) {
    prepare_result.result =
        internal::invalid_view("joint_velocity must match cartesian_pose_error dtype and num_envs");
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_batched_matrix(cartesian_mass_matrix,
                                        "cartesian_mass_matrix tensor data is null",
                                        cartesian_pose_error.dtype,
                                        num_envs,
                                        kCartesianDim,
                                        kCartesianDim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_batched_matrix(jacobian,
                                                            "jacobian tensor data is null",
                                                            cartesian_pose_error.dtype,
                                                            num_envs,
                                                            kCartesianDim,
                                                            joint_dim);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result = internal::validate_matrix_parameter_metadata(
      find_parameter(parameters, kInitialStiffnessMatrix),
      "initial_stiffness_matrix parameter is missing",
      cartesian_pose_error.dtype,
      num_envs,
      kCartesianDim,
      kCartesianDim,
      true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      internal::validate_vector_parameter_metadata(find_parameter(parameters, kDampingFactors),
                                                   "damping_factors parameter is missing",
                                                   cartesian_pose_error.dtype,
                                                   num_envs,
                                                   kCartesianDim,
                                                   true);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  append_variable_impedance_state_descriptors(
      &prepare_result, cartesian_pose_error.dtype, num_envs, joint_dim);
  append_variable_impedance_scratch_descriptors(
      &prepare_result, cartesian_pose_error.dtype, num_envs);
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("cartesian_wrench",
                                  cartesian_pose_error.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "newton_and_newton_meter",
                                  "task_space",
                                  "variable Cartesian impedance wrench"));
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    mark_descriptors_cuda(&prepare_result);
#endif
  }
  return prepare_result;
}

Result apply_variable_cartesian_impedance_controller_parameters(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_variable_cartesian_impedance_controller_parameters_cuda(
        parameters, num_envs, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const DType dtype = first_parameter_dtype(parameters);
  result = dtype == DType::kFloat32 ? apply_variable_stiffness_parameters_impl<float>(
                                          parameters, num_envs, parameter_status, false)
                                    : apply_variable_stiffness_parameters_impl<double>(
                                          parameters, num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  result =
      dtype == DType::kFloat32
          ? apply_damping_design_parameters<float>(parameters, num_envs, parameter_status, false)
          : apply_damping_design_parameters<double>(parameters, num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  return apply_inertial_coupling_parameters_impl(num_envs, parameter_status, false);
}

Result reset_variable_cartesian_impedance_controller(const ParameterSetView& parameters,
                                                     StatusView parameter_status,
                                                     const std::size_t num_envs,
                                                     const std::size_t joint_dim,
                                                     StateView& state,
                                                     const TensorView& reset_mask,
                                                     MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    StateEntryView stiffness_entries[3]{};
    StateView stiffness_state{};
    Result result = make_variable_stiffness_state(state, stiffness_entries, &stiffness_state);
    if (!result.ok()) {
      return result;
    }
    StateEntryView inertial_entries[3]{};
    StateView inertial_state{};
    result = make_inertial_state(state, inertial_entries, &inertial_state);
    if (!result.ok()) {
      return result;
    }
    result = reset_inertial_coupling(num_envs, joint_dim, inertial_state, reset_mask, status);
    if (!result.ok()) {
      return result;
    }
    return reset_variable_cartesian_stiffness(
        parameters, parameter_status, stiffness_state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView stiffness_entries[3]{};
  StateView stiffness_state{};
  Result result = make_variable_stiffness_state(state, stiffness_entries, &stiffness_state);
  if (!result.ok()) {
    return result;
  }
  StateEntryView inertial_entries[3]{};
  StateView inertial_state{};
  result = make_inertial_state(state, inertial_entries, &inertial_state);
  if (!result.ok()) {
    return result;
  }
  result = reset_variable_cartesian_stiffness(
      parameters, parameter_status, stiffness_state, reset_mask, status);
  if (!result.ok()) {
    return result;
  }
  return reset_inertial_coupling(num_envs, joint_dim, inertial_state, reset_mask, status);
}

Result step_variable_cartesian_impedance_controller(const TensorView& cartesian_pose_error,
                                                    const TensorView& stiffness_matrix,
                                                    const TensorView& cartesian_velocity_error,
                                                    const TensorView& desired_acceleration,
                                                    const TensorView& joint_velocity,
                                                    const TensorView& cartesian_mass_matrix,
                                                    const TensorView& jacobian,
                                                    const TensorView& dt,
                                                    const ParameterSetView& parameters,
                                                    StatusView parameter_status,
                                                    StateView& state,
                                                    ScratchView& scratch,
                                                    const MutableTensorView& cartesian_wrench,
                                                    MutableStatusView status) {
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      dt.device == Device::kCuda || cartesian_wrench.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
    ScratchEntryView* stiffness_wrench = find_scratch(scratch, kStiffnessWrenchScratch);
    ScratchEntryView* damping_wrench = find_scratch(scratch, kDampingWrenchScratch);
    ScratchEntryView* inertial_wrench = find_scratch(scratch, kInertialWrenchScratch);
    ScratchEntryView* stiffness_status_entry = find_scratch(scratch, kStiffnessStatusScratch);
    ScratchEntryView* damping_status_entry = find_scratch(scratch, kDampingStatusScratch);
    ScratchEntryView* inertial_status_entry = find_scratch(scratch, kInertialStatusScratch);
    if (stiffness_wrench == nullptr || damping_wrench == nullptr || inertial_wrench == nullptr ||
        stiffness_status_entry == nullptr || damping_status_entry == nullptr ||
        inertial_status_entry == nullptr) {
      return internal::invalid_view("variable impedance scratch entries are incomplete");
    }
    StateEntryView stiffness_entries[3]{};
    StateView stiffness_state{};
    Result result = make_variable_stiffness_state(state, stiffness_entries, &stiffness_state);
    if (!result.ok()) {
      return result;
    }
    StateEntryView inertial_entries[3]{};
    StateView inertial_state{};
    result = make_inertial_state(state, inertial_entries, &inertial_state);
    if (!result.ok()) {
      return result;
    }
    MutableStatusView stiffness_status =
        scratch_status_view(scratch, kStiffnessStatusScratch, num_envs);
    MutableStatusView damping_status =
        scratch_status_view(scratch, kDampingStatusScratch, num_envs);
    MutableStatusView inertial_status =
        scratch_status_view(scratch, kInertialStatusScratch, num_envs);
    result = step_variable_cartesian_stiffness(cartesian_pose_error,
                                               stiffness_matrix,
                                               dt,
                                               parameters,
                                               parameter_status,
                                               stiffness_state,
                                               stiffness_wrench->values,
                                               stiffness_status);
    if (!result.ok()) {
      return result;
    }
    StateEntryView* current_stiffness = find_state(stiffness_state, kCurrentStiffnessMatrix);
    result = step_cartesian_damping_double_diagonalization(cartesian_velocity_error,
                                                           cartesian_mass_matrix,
                                                           current_stiffness->values.as_const(),
                                                           parameters,
                                                           parameter_status,
                                                           damping_wrench->values,
                                                           damping_status);
    if (!result.ok()) {
      return result;
    }
    result = step_inertial_coupling(cartesian_velocity_error,
                                    joint_velocity,
                                    cartesian_mass_matrix,
                                    jacobian,
                                    dt,
                                    parameter_status,
                                    inertial_state,
                                    inertial_wrench->values,
                                    inertial_status);
    if (!result.ok()) {
      return result;
    }
    return combine_variable_cartesian_impedance_cuda(
        stiffness_wrench->values.as_const(),
        StatusView{stiffness_status.flags, num_envs, stiffness_status.device},
        damping_wrench->values.as_const(),
        StatusView{damping_status.flags, num_envs, damping_status.device},
        inertial_wrench->values.as_const(),
        StatusView{inertial_status.flags, num_envs, inertial_status.device},
        desired_acceleration,
        cartesian_mass_matrix,
        cartesian_wrench,
        status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_variable_impedance_scratch(scratch, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView stiffness_entries[3]{};
  StateView stiffness_state{};
  result = make_variable_stiffness_state(state, stiffness_entries, &stiffness_state);
  if (!result.ok()) {
    return result;
  }
  StateEntryView inertial_entries[3]{};
  StateView inertial_state{};
  result = make_inertial_state(state, inertial_entries, &inertial_state);
  if (!result.ok()) {
    return result;
  }
  ScratchEntryView* stiffness_wrench = find_scratch(scratch, kStiffnessWrenchScratch);
  ScratchEntryView* damping_wrench = find_scratch(scratch, kDampingWrenchScratch);
  ScratchEntryView* inertial_wrench = find_scratch(scratch, kInertialWrenchScratch);
  MutableStatusView stiffness_status =
      scratch_status_view(scratch, kStiffnessStatusScratch, num_envs);
  MutableStatusView damping_status = scratch_status_view(scratch, kDampingStatusScratch, num_envs);
  MutableStatusView inertial_status =
      scratch_status_view(scratch, kInertialStatusScratch, num_envs);

  result = step_variable_cartesian_stiffness(cartesian_pose_error,
                                             stiffness_matrix,
                                             dt,
                                             parameters,
                                             parameter_status,
                                             stiffness_state,
                                             stiffness_wrench->values,
                                             stiffness_status);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* current_stiffness = find_state(stiffness_state, kCurrentStiffnessMatrix);
  result = step_cartesian_damping_double_diagonalization(cartesian_velocity_error,
                                                         cartesian_mass_matrix,
                                                         current_stiffness->values.as_const(),
                                                         parameters,
                                                         parameter_status,
                                                         damping_wrench->values,
                                                         damping_status);
  if (!result.ok()) {
    return result;
  }
  result = step_inertial_coupling(cartesian_velocity_error,
                                  joint_velocity,
                                  cartesian_mass_matrix,
                                  jacobian,
                                  dt,
                                  parameter_status,
                                  inertial_state,
                                  inertial_wrench->values,
                                  inertial_status);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] =
          stiffness_status.flags[env] | damping_status.flags[env] | inertial_status.flags[env];
      if (!internal::matrix_env_is_finite<float>(desired_acceleration, env, kCartesianDim)) {
        status.flags[env] |=
            status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kSafeMode);
      }
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<float>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        float acceleration_wrench = 0.0F;
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          acceleration_wrench +=
              internal::read_tensor3_value<float>(cartesian_mass_matrix, env, row, col) *
              internal::read_matrix_value<float>(desired_acceleration, env, col);
        }
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            row,
            internal::read_matrix_value<float>(stiffness_wrench->values, env, row) +
                internal::read_matrix_value<float>(damping_wrench->values, env, row) +
                internal::read_matrix_value<float>(inertial_wrench->values, env, row) +
                acceleration_wrench);
      }
    }
  } else {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] =
          stiffness_status.flags[env] | damping_status.flags[env] | inertial_status.flags[env];
      if (!internal::matrix_env_is_finite<double>(desired_acceleration, env, kCartesianDim)) {
        status.flags[env] |=
            status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kSafeMode);
      }
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<double>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t row = 0; row < kCartesianDim; ++row) {
        double acceleration_wrench = 0.0;
        for (std::size_t col = 0; col < kCartesianDim; ++col) {
          acceleration_wrench +=
              internal::read_tensor3_value<double>(cartesian_mass_matrix, env, row, col) *
              internal::read_matrix_value<double>(desired_acceleration, env, col);
        }
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            row,
            internal::read_matrix_value<double>(stiffness_wrench->values, env, row) +
                internal::read_matrix_value<double>(damping_wrench->values, env, row) +
                internal::read_matrix_value<double>(inertial_wrench->values, env, row) +
                acceleration_wrench);
      }
    }
  }
  return internal::ok();
}

Result validate_variable_unified_scratch(ScratchView& scratch,
                                         const DType dtype,
                                         const std::size_t num_envs) {
  Result result = validate_variable_impedance_scratch(scratch, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  ScratchEntryView* variable_impedance_wrench =
      find_scratch(scratch, kVariableImpedanceWrenchScratch);
  ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
  ScratchEntryView* variable_impedance_status =
      find_scratch(scratch, kVariableImpedanceStatusScratch);
  ScratchEntryView* force_status = find_scratch(scratch, kForceStatusScratch);
  if (variable_impedance_wrench == nullptr || force_wrench == nullptr ||
      variable_impedance_status == nullptr || force_status == nullptr) {
    return internal::invalid_view("variable unified scratch entries are incomplete");
  }
  result = validate_cartesian_output(variable_impedance_wrench->values, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(force_wrench->values, dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_status_scratch(variable_impedance_status->values, num_envs);
  if (!result.ok()) {
    return result;
  }
  return validate_status_scratch(force_status->values, num_envs);
}

PrepareResult prepare_variable_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    const bool use_shaping_function,
    const ParameterSetView& parameters,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench) {
  PrepareResult prepare_result =
      prepare_variable_cartesian_impedance_controller(cartesian_pose_error,
                                                      stiffness_matrix,
                                                      cartesian_velocity_error,
                                                      desired_acceleration,
                                                      joint_velocity,
                                                      cartesian_mass_matrix,
                                                      jacobian,
                                                      dt,
                                                      parameters,
                                                      state,
                                                      scratch,
                                                      cartesian_wrench);
  if (!prepare_result.ok()) {
    return prepare_result;
  }
  PrepareResult force_prepare = prepare_cartesian_force_controller(external_wrench,
                                                                   desired_wrench,
                                                                   feedforward_wrench,
                                                                   cartesian_pose_error,
                                                                   dt,
                                                                   use_shaping_function,
                                                                   parameters,
                                                                   state,
                                                                   cartesian_wrench);
  if (!force_prepare.ok()) {
    prepare_result.result = force_prepare.result;
    return prepare_result;
  }
  prepare_result.state.insert(
      prepare_result.state.end(), force_prepare.state.begin(), force_prepare.state.end());
  prepare_result.scratch.push_back(internal::matrix_descriptor(kVariableImpedanceWrenchScratch,
                                                               cartesian_pose_error.dtype,
                                                               prepare_result.num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "variable impedance wrench"));
  prepare_result.scratch.push_back(internal::matrix_descriptor(kForceWrenchScratch,
                                                               cartesian_pose_error.dtype,
                                                               prepare_result.num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "force wrench"));
  prepare_result.scratch.push_back(internal::vector_descriptor(kVariableImpedanceStatusScratch,
                                                               DType::kUInt64,
                                                               prepare_result.num_envs,
                                                               "status_flags",
                                                               "none",
                                                               "variable impedance status"));
  prepare_result.scratch.push_back(internal::vector_descriptor(kForceStatusScratch,
                                                               DType::kUInt64,
                                                               prepare_result.num_envs,
                                                               "status_flags",
                                                               "none",
                                                               "force status"));
  prepare_result.outputs.clear();
  prepare_result.outputs.push_back(
      internal::matrix_descriptor("cartesian_wrench",
                                  cartesian_pose_error.dtype,
                                  prepare_result.num_envs,
                                  kCartesianDim,
                                  "newton_and_newton_meter",
                                  "task_space",
                                  "variable unified force/impedance wrench"));
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || dt.device == Device::kCuda ||
      cartesian_wrench.device == Device::kCuda || parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    mark_descriptors_cuda(&prepare_result);
#endif
  }
  return prepare_result;
}

Result apply_variable_cartesian_unified_force_impedance_controller_parameters(
    const ParameterSetView& parameters,
    const std::size_t num_envs,
    const bool use_shaping_function,
    MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_variable_cartesian_unified_force_impedance_controller_parameters_cuda(
        parameters, num_envs, use_shaping_function, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(parameter_status, num_envs);
  const DType dtype = first_parameter_dtype(parameters);
  result = dtype == DType::kFloat32 ? apply_variable_stiffness_parameters_impl<float>(
                                          parameters, num_envs, parameter_status, false)
                                    : apply_variable_stiffness_parameters_impl<double>(
                                          parameters, num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  result =
      dtype == DType::kFloat32
          ? apply_damping_design_parameters<float>(parameters, num_envs, parameter_status, false)
          : apply_damping_design_parameters<double>(parameters, num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  result = apply_inertial_coupling_parameters_impl(num_envs, parameter_status, false);
  if (!result.ok()) {
    return result;
  }
  result = dtype == DType::kFloat32
               ? apply_force_parameters<float>(
                     parameters, num_envs, use_shaping_function, parameter_status, false)
               : apply_force_parameters<double>(
                     parameters, num_envs, use_shaping_function, parameter_status, false);
  return result;
}

Result reset_variable_cartesian_unified_force_impedance_controller(
    const ParameterSetView& parameters,
    StatusView parameter_status,
    const std::size_t num_envs,
    const std::size_t joint_dim,
    StateView& state,
    const TensorView& reset_mask,
    MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    Result result = reset_cartesian_force_controller(num_envs, state, reset_mask, status);
    if (!result.ok()) {
      return result;
    }
    return reset_variable_cartesian_impedance_controller(
        parameters, parameter_status, num_envs, joint_dim, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  Result result = reset_variable_cartesian_impedance_controller(
      parameters, parameter_status, num_envs, joint_dim, state, reset_mask, status);
  if (!result.ok()) {
    return result;
  }
  return reset_cartesian_force_controller(num_envs, state, reset_mask, status);
}

Result step_variable_cartesian_unified_force_impedance_controller(
    const TensorView& cartesian_pose_error,
    const TensorView& stiffness_matrix,
    const TensorView& cartesian_velocity_error,
    const TensorView& desired_acceleration,
    const TensorView& joint_velocity,
    const TensorView& cartesian_mass_matrix,
    const TensorView& jacobian,
    const TensorView& external_wrench,
    const TensorView& desired_wrench,
    const TensorView& feedforward_wrench,
    const TensorView& dt,
    const bool use_shaping_function,
    const ParameterSetView& parameters,
    StatusView parameter_status,
    StateView& state,
    ScratchView& scratch,
    const MutableTensorView& cartesian_wrench,
    MutableStatusView status) {
  if (cartesian_pose_error.device == Device::kCuda || stiffness_matrix.device == Device::kCuda ||
      cartesian_velocity_error.device == Device::kCuda ||
      desired_acceleration.device == Device::kCuda || joint_velocity.device == Device::kCuda ||
      cartesian_mass_matrix.device == Device::kCuda || jacobian.device == Device::kCuda ||
      external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || dt.device == Device::kCuda ||
      cartesian_wrench.device == Device::kCuda || status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    const std::size_t num_envs = static_cast<std::size_t>(cartesian_pose_error.shape[0]);
    ScratchEntryView* variable_impedance_wrench =
        find_scratch(scratch, kVariableImpedanceWrenchScratch);
    ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
    ScratchEntryView* variable_impedance_status_entry =
        find_scratch(scratch, kVariableImpedanceStatusScratch);
    ScratchEntryView* force_status_entry = find_scratch(scratch, kForceStatusScratch);
    if (variable_impedance_wrench == nullptr || force_wrench == nullptr ||
        variable_impedance_status_entry == nullptr || force_status_entry == nullptr) {
      return internal::invalid_view("variable unified scratch entries are incomplete");
    }
    MutableStatusView variable_impedance_status =
        scratch_status_view(scratch, kVariableImpedanceStatusScratch, num_envs);
    MutableStatusView force_status = scratch_status_view(scratch, kForceStatusScratch, num_envs);
    Result result = step_variable_cartesian_impedance_controller(cartesian_pose_error,
                                                                 stiffness_matrix,
                                                                 cartesian_velocity_error,
                                                                 desired_acceleration,
                                                                 joint_velocity,
                                                                 cartesian_mass_matrix,
                                                                 jacobian,
                                                                 dt,
                                                                 parameters,
                                                                 parameter_status,
                                                                 state,
                                                                 scratch,
                                                                 variable_impedance_wrench->values,
                                                                 variable_impedance_status);
    if (!result.ok()) {
      return result;
    }
    result = step_cartesian_force_controller(external_wrench,
                                             desired_wrench,
                                             feedforward_wrench,
                                             cartesian_pose_error,
                                             dt,
                                             use_shaping_function,
                                             parameters,
                                             parameter_status,
                                             state,
                                             force_wrench->values,
                                             force_status);
    if (!result.ok()) {
      return result;
    }
    return combine_cartesian_wrench_sum_cuda(
        variable_impedance_wrench->values.as_const(),
        StatusView{variable_impedance_status.flags, num_envs, variable_impedance_status.device},
        force_wrench->values.as_const(),
        StatusView{force_status.flags, num_envs, force_status.device},
        cartesian_wrench,
        status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  std::size_t num_envs = 0;
  Result result = validate_cartesian_signal(
      cartesian_pose_error, "cartesian_pose_error tensor data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(cartesian_wrench, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_variable_unified_scratch(scratch, cartesian_pose_error.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  ScratchEntryView* variable_impedance_wrench =
      find_scratch(scratch, kVariableImpedanceWrenchScratch);
  ScratchEntryView* force_wrench = find_scratch(scratch, kForceWrenchScratch);
  MutableStatusView variable_impedance_status =
      scratch_status_view(scratch, kVariableImpedanceStatusScratch, num_envs);
  MutableStatusView force_status = scratch_status_view(scratch, kForceStatusScratch, num_envs);
  result = step_variable_cartesian_impedance_controller(cartesian_pose_error,
                                                        stiffness_matrix,
                                                        cartesian_velocity_error,
                                                        desired_acceleration,
                                                        joint_velocity,
                                                        cartesian_mass_matrix,
                                                        jacobian,
                                                        dt,
                                                        parameters,
                                                        parameter_status,
                                                        state,
                                                        scratch,
                                                        variable_impedance_wrench->values,
                                                        variable_impedance_status);
  if (!result.ok()) {
    return result;
  }
  result = step_cartesian_force_controller(external_wrench,
                                           desired_wrench,
                                           feedforward_wrench,
                                           cartesian_pose_error,
                                           dt,
                                           use_shaping_function,
                                           parameters,
                                           parameter_status,
                                           state,
                                           force_wrench->values,
                                           force_status);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  if (cartesian_pose_error.dtype == DType::kFloat32) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] = variable_impedance_status.flags[env] | force_status.flags[env];
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<float>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            dim,
            detail::unified_force_impedance_value(
                internal::read_matrix_value<float>(variable_impedance_wrench->values, env, dim),
                internal::read_matrix_value<float>(force_wrench->values, env, dim)));
      }
    }
  } else {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] = variable_impedance_status.flags[env] | force_status.flags[env];
      if (has_status_flag(status.flags[env], StatusFlag::kSafeMode)) {
        zero_cartesian_output<double>(cartesian_wrench, env);
        continue;
      }
      for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
        internal::write_matrix_value(
            cartesian_wrench,
            env,
            dim,
            detail::unified_force_impedance_value(
                internal::read_matrix_value<double>(variable_impedance_wrench->values, env, dim),
                internal::read_matrix_value<double>(force_wrench->values, env, dim)));
      }
    }
  }
  return internal::ok();
}

Result validate_force_controller_state(StateView& state,
                                       const DType dtype,
                                       const std::size_t num_envs) {
  StateEntryView* differentiator_previous = find_state(state, kDifferentiatorPreviousError);
  StateEntryView* differentiator_initialized = find_state(state, kDifferentiatorInitialized);
  StateEntryView* integrator_previous = find_state(state, kIntegratorPreviousError);
  StateEntryView* integrated_error = find_state(state, kIntegratedError);
  StateEntryView* integrator_initialized = find_state(state, kIntegratorInitialized);
  if (differentiator_previous == nullptr || differentiator_initialized == nullptr ||
      integrator_previous == nullptr || integrated_error == nullptr ||
      integrator_initialized == nullptr) {
    return internal::invalid_view("force controller state entries are incomplete");
  }
  Result result = internal::validate_matching_output(
      differentiator_previous->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result = validate_initialized_state(differentiator_initialized->values, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_matching_output(
      integrator_previous->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  result =
      internal::validate_matching_output(integrated_error->values, dtype, num_envs, kCartesianDim);
  if (!result.ok()) {
    return result;
  }
  return validate_initialized_state(integrator_initialized->values, num_envs);
}

template <typename Scalar>
bool force_parameters_are_valid(const ParameterSetView& parameters,
                                const std::size_t env,
                                const bool use_shaping_function) {
  const char* gain_names[] = {kProportionalGain, kDifferentiatorGain, kIntegralGain};
  for (const char* name : gain_names) {
    const ParameterView* gain = find_parameter(parameters, name);
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      if (!std::isfinite(
              internal::read_vector_parameter<Scalar>(gain, env, dim, static_cast<Scalar>(0)))) {
        return false;
      }
    }
  }
  const ParameterView* u0 = find_parameter(parameters, kU0);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
    const Scalar upper_value = internal::read_vector_parameter<Scalar>(
        upper, env, dim, std::numeric_limits<Scalar>::max());
    const Scalar lower_value = internal::read_vector_parameter<Scalar>(
        lower, env, dim, -std::numeric_limits<Scalar>::max());
    const Scalar u0_value =
        internal::read_vector_parameter<Scalar>(u0, env, dim, static_cast<Scalar>(0));
    if (!std::isfinite(upper_value) || !std::isfinite(lower_value) || !std::isfinite(u0_value) ||
        upper_value < lower_value) {
      return false;
    }
  }
  return !use_shaping_function ||
         shaping_parameters_are_valid<Scalar>(parameters, env, kCartesianDim);
}

template <typename Scalar>
Result apply_force_parameters(const ParameterSetView& parameters,
                              const std::size_t num_envs,
                              const bool use_shaping_function,
                              MutableStatusView parameter_status,
                              const bool clear_status) {
  Result result = internal::validate_status_view(parameter_status, num_envs);
  if (!result.ok()) {
    return result;
  }
  if (clear_status) {
    internal::clear_status(parameter_status, num_envs);
  }
  const char* gain_names[] = {kProportionalGain, kDifferentiatorGain, kIntegralGain};
  for (const char* name : gain_names) {
    result = internal::validate_vector_parameter_metadata(find_parameter(parameters, name),
                                                          "force controller gain is missing",
                                                          dtype_for<Scalar>(),
                                                          num_envs,
                                                          kCartesianDim,
                                                          true);
    if (!result.ok()) {
      return result;
    }
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kU0),
                                                        "u_0 parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        false);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kUpperLimit),
                                                        "upper_limit parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        false);
  if (!result.ok()) {
    return result;
  }
  result = internal::validate_vector_parameter_metadata(find_parameter(parameters, kLowerLimit),
                                                        "lower_limit parameter is missing",
                                                        dtype_for<Scalar>(),
                                                        num_envs,
                                                        kCartesianDim,
                                                        false);
  if (!result.ok()) {
    return result;
  }
  if (use_shaping_function) {
    result =
        internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                     "maximum_distance parameter is missing",
                                                     dtype_for<Scalar>(),
                                                     num_envs,
                                                     kCartesianDim,
                                                     true);
    if (!result.ok()) {
      return result;
    }
    result =
        internal::validate_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                     "shaping_distance parameter is missing",
                                                     dtype_for<Scalar>(),
                                                     num_envs,
                                                     kCartesianDim,
                                                     true);
    if (!result.ok()) {
      return result;
    }
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!force_parameters_are_valid<Scalar>(parameters, env, use_shaping_function)) {
      parameter_status.flags[env] |= status_bit(StatusFlag::kInvalidParameter);
    }
  }
  return internal::ok();
}

PrepareResult prepare_cartesian_force_controller(const TensorView& external_wrench,
                                                 const TensorView& desired_wrench,
                                                 const TensorView& feedforward_wrench,
                                                 const TensorView& pose_error,
                                                 const TensorView& dt,
                                                 const bool use_shaping_function,
                                                 const ParameterSetView& parameters,
                                                 StateView& state,
                                                 const MutableTensorView& commanded_wrench) {
  (void)state;
  PrepareResult prepare_result;
  if (external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || pose_error.device == Device::kCuda ||
      dt.device == Device::kCuda || commanded_wrench.device == Device::kCuda ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    std::size_t num_envs = 0;
    std::size_t signal_dim = 0;
    prepare_result.result = validate_cuda_signal_metadata(
        external_wrench, "external_wrench must be CUDA [num_envs, 6]", &num_envs, &signal_dim);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (signal_dim != kCartesianDim) {
      prepare_result.result =
          internal::invalid_view("Cartesian signal must have shape [num_envs, 6]");
      return prepare_result;
    }
    const TensorView inputs[] = {desired_wrench, feedforward_wrench, pose_error};
    for (const TensorView& input : inputs) {
      std::size_t input_envs = 0;
      std::size_t input_dims = 0;
      prepare_result.result = validate_cuda_signal_metadata(
          input, "force controller input is invalid", &input_envs, &input_dims);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
      if (input_envs != num_envs || input_dims != kCartesianDim ||
          input.dtype != external_wrench.dtype) {
        prepare_result.result =
            internal::invalid_view("force controller inputs must share dtype and num_envs");
        return prepare_result;
      }
    }
    prepare_result.result = validate_cuda_scalar_batch_metadata(
        dt, "dt tensor data is null", external_wrench.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        validate_cuda_cartesian_output_metadata(commanded_wrench, external_wrench.dtype, num_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    const char* gain_names[] = {kProportionalGain, kDifferentiatorGain, kIntegralGain};
    for (const char* name : gain_names) {
      prepare_result.result =
          validate_cuda_vector_parameter_metadata(find_parameter(parameters, name),
                                                  "force controller gain is missing",
                                                  external_wrench.dtype,
                                                  num_envs,
                                                  kCartesianDim);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
    }
    const char* optional_names[] = {kU0, kUpperLimit, kLowerLimit};
    for (const char* name : optional_names) {
      prepare_result.result = validate_cuda_optional_vector_parameter_metadata(
          find_parameter(parameters, name), external_wrench.dtype, num_envs, kCartesianDim);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
    }
    if (use_shaping_function) {
      prepare_result.result =
          validate_cuda_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                  "maximum_distance parameter is missing",
                                                  external_wrench.dtype,
                                                  num_envs,
                                                  kCartesianDim);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
      prepare_result.result =
          validate_cuda_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                  "shaping_distance parameter is missing",
                                                  external_wrench.dtype,
                                                  num_envs,
                                                  kCartesianDim);
      if (!prepare_result.result.ok()) {
        return prepare_result;
      }
    }
    prepare_result.num_envs = num_envs;
    prepare_result.signal_dim = kCartesianDim;
    prepare_result.state.push_back(internal::matrix_descriptor(kDifferentiatorPreviousError,
                                                               external_wrench.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "previous force error"));
    prepare_result.state.push_back(internal::vector_descriptor(kDifferentiatorInitialized,
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "differentiator initialized flag"));
    prepare_result.state.push_back(internal::matrix_descriptor(kIntegratorPreviousError,
                                                               external_wrench.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "previous force error"));
    prepare_result.state.push_back(
        internal::matrix_descriptor(kIntegratedError,
                                    external_wrench.dtype,
                                    num_envs,
                                    kCartesianDim,
                                    "newton_second_and_newton_meter_second",
                                    "task_space",
                                    "integrated force error"));
    prepare_result.state.push_back(internal::vector_descriptor(kIntegratorInitialized,
                                                               DType::kBool,
                                                               num_envs,
                                                               "unitless",
                                                               "none",
                                                               "integrator initialized flag"));
    prepare_result.outputs.push_back(internal::matrix_descriptor("commanded_wrench",
                                                                 external_wrench.dtype,
                                                                 num_envs,
                                                                 kCartesianDim,
                                                                 "newton_and_newton_meter",
                                                                 "task_space",
                                                                 "force controller output"));
    mark_descriptors_cuda(&prepare_result);
    return prepare_result;
#else
    prepare_result.result = internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
    return prepare_result;
#endif
  }
  std::size_t num_envs = 0;
  prepare_result.result =
      validate_cartesian_signal(external_wrench, "external_wrench data is null", &num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  const TensorView inputs[] = {desired_wrench, feedforward_wrench, pose_error};
  for (const TensorView& input : inputs) {
    std::size_t input_envs = 0;
    prepare_result.result =
        validate_cartesian_signal(input, "force controller input is invalid", &input_envs);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    if (input_envs != num_envs || input.dtype != external_wrench.dtype) {
      prepare_result.result =
          internal::invalid_view("force controller inputs must share dtype and num_envs");
      return prepare_result;
    }
  }
  prepare_result.result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  prepare_result.result =
      validate_cartesian_output(commanded_wrench, external_wrench.dtype, num_envs);
  if (!prepare_result.result.ok()) {
    return prepare_result;
  }
  const char* gain_names[] = {kProportionalGain, kDifferentiatorGain, kIntegralGain};
  for (const char* name : gain_names) {
    prepare_result.result =
        internal::validate_vector_parameter_metadata(find_parameter(parameters, name),
                                                     "force controller gain is missing",
                                                     external_wrench.dtype,
                                                     num_envs,
                                                     kCartesianDim,
                                                     true);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
  }
  if (use_shaping_function) {
    prepare_result.result =
        internal::validate_vector_parameter_metadata(find_parameter(parameters, kMaximumDistance),
                                                     "maximum_distance parameter is missing",
                                                     external_wrench.dtype,
                                                     num_envs,
                                                     kCartesianDim,
                                                     true);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
    prepare_result.result =
        internal::validate_vector_parameter_metadata(find_parameter(parameters, kShapingDistance),
                                                     "shaping_distance parameter is missing",
                                                     external_wrench.dtype,
                                                     num_envs,
                                                     kCartesianDim,
                                                     true);
    if (!prepare_result.result.ok()) {
      return prepare_result;
    }
  }
  prepare_result.result = internal::ok();
  prepare_result.num_envs = num_envs;
  prepare_result.signal_dim = kCartesianDim;
  prepare_result.state.push_back(internal::matrix_descriptor(kDifferentiatorPreviousError,
                                                             external_wrench.dtype,
                                                             num_envs,
                                                             kCartesianDim,
                                                             "newton_and_newton_meter",
                                                             "task_space",
                                                             "previous force error"));
  prepare_result.state.push_back(internal::vector_descriptor(kDifferentiatorInitialized,
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "differentiator initialized flag"));
  prepare_result.state.push_back(internal::matrix_descriptor(kIntegratorPreviousError,
                                                             external_wrench.dtype,
                                                             num_envs,
                                                             kCartesianDim,
                                                             "newton_and_newton_meter",
                                                             "task_space",
                                                             "previous force error"));
  prepare_result.state.push_back(
      internal::matrix_descriptor(kIntegratedError,
                                  external_wrench.dtype,
                                  num_envs,
                                  kCartesianDim,
                                  "newton_second_and_newton_meter_second",
                                  "task_space",
                                  "integrated force error"));
  prepare_result.state.push_back(internal::vector_descriptor(kIntegratorInitialized,
                                                             DType::kBool,
                                                             num_envs,
                                                             "unitless",
                                                             "none",
                                                             "integrator initialized flag"));
  prepare_result.outputs.push_back(internal::matrix_descriptor("commanded_wrench",
                                                               external_wrench.dtype,
                                                               num_envs,
                                                               kCartesianDim,
                                                               "newton_and_newton_meter",
                                                               "task_space",
                                                               "force controller output"));
  return prepare_result;
}

Result apply_cartesian_force_controller_parameters(const ParameterSetView& parameters,
                                                   const std::size_t num_envs,
                                                   const bool use_shaping_function,
                                                   MutableStatusView parameter_status) {
  if (utility_lifecycle_uses_cuda(parameters, parameter_status)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return apply_cartesian_force_controller_parameters_cuda(
        parameters, num_envs, use_shaping_function, parameter_status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  const DType dtype = first_parameter_dtype(parameters);
  return dtype == DType::kFloat32
             ? apply_force_parameters<float>(
                   parameters, num_envs, use_shaping_function, parameter_status)
             : apply_force_parameters<double>(
                   parameters, num_envs, use_shaping_function, parameter_status);
}

template <typename Scalar>
Result reset_force_controller_impl(const std::size_t num_envs,
                                   StateView& state,
                                   const TensorView& reset_mask,
                                   MutableStatusView status) {
  Result result = internal::validate_status_view(status, num_envs);
  if (!result.ok()) {
    return result;
  }
  internal::clear_status(status, num_envs);
  result = internal::validate_reset_mask(reset_mask, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_force_controller_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  StateEntryView* differentiator_previous = find_state(state, kDifferentiatorPreviousError);
  StateEntryView* differentiator_initialized = find_state(state, kDifferentiatorInitialized);
  StateEntryView* integrator_previous = find_state(state, kIntegratorPreviousError);
  StateEntryView* integrated_error = find_state(state, kIntegratedError);
  StateEntryView* integrator_initialized = find_state(state, kIntegratorInitialized);
  auto* diff_initialized = static_cast<std::uint8_t*>(differentiator_initialized->values.data);
  auto* int_initialized = static_cast<std::uint8_t*>(integrator_initialized->values.data);
  for (std::size_t env = 0; env < num_envs; ++env) {
    if (!internal::reset_mask_value(reset_mask, env)) {
      continue;
    }
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      internal::write_matrix_value(
          differentiator_previous->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(integrator_previous->values, env, dim, static_cast<Scalar>(0));
      internal::write_matrix_value(integrated_error->values, env, dim, static_cast<Scalar>(0));
    }
    diff_initialized[env *
                     static_cast<std::size_t>(differentiator_initialized->values.strides[0])] = 0U;
    int_initialized[env * static_cast<std::size_t>(integrator_initialized->values.strides[0])] = 0U;
    status.flags[env] |= status_bit(StatusFlag::kResetApplied);
  }
  return internal::ok();
}

Result reset_cartesian_force_controller(const std::size_t num_envs,
                                        StateView& state,
                                        const TensorView& reset_mask,
                                        MutableStatusView status) {
  if (reset_mask.device == Device::kCuda || status.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_cartesian_force_controller_cuda(num_envs, state, reset_mask, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  StateEntryView* integrated_error = find_state(state, kIntegratedError);
  if (integrated_error == nullptr) {
    return internal::invalid_view("integrated_error state is missing");
  }
  if (integrated_error->values.dtype == DType::kFloat32) {
    return reset_force_controller_impl<float>(num_envs, state, reset_mask, status);
  }
  return reset_force_controller_impl<double>(num_envs, state, reset_mask, status);
}

template <typename Scalar>
Result step_force_controller_impl(const TensorView& external_wrench,
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
  std::size_t num_envs = 0;
  Result result =
      validate_cartesian_signal(external_wrench, "external_wrench data is null", &num_envs);
  if (!result.ok()) {
    return result;
  }
  const TensorView inputs[] = {desired_wrench, feedforward_wrench, pose_error};
  for (const TensorView& input : inputs) {
    std::size_t input_envs = 0;
    result = validate_cartesian_signal(input, "force controller input is invalid", &input_envs);
    if (!result.ok()) {
      return result;
    }
    if (input_envs != num_envs || input.dtype != external_wrench.dtype) {
      return internal::invalid_view("force controller inputs must share dtype and num_envs");
    }
  }
  result = internal::validate_scalar_batch(
      dt, "dt tensor data is null", external_wrench.dtype, num_envs);
  if (!result.ok()) {
    return result;
  }
  result = validate_cartesian_output(commanded_wrench, external_wrench.dtype, num_envs);
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
  result = validate_force_controller_state(state, dtype_for<Scalar>(), num_envs);
  if (!result.ok()) {
    return result;
  }
  result = apply_force_parameters<Scalar>(parameters, num_envs, use_shaping_function, status);
  if (!result.ok()) {
    return result;
  }
  if (parameter_status.flags != nullptr) {
    for (std::size_t env = 0; env < num_envs; ++env) {
      status.flags[env] |= parameter_status.flags[env];
    }
  }

  StateEntryView* differentiator_previous = find_state(state, kDifferentiatorPreviousError);
  StateEntryView* differentiator_initialized = find_state(state, kDifferentiatorInitialized);
  StateEntryView* integrator_previous = find_state(state, kIntegratorPreviousError);
  StateEntryView* integrated_error = find_state(state, kIntegratedError);
  StateEntryView* integrator_initialized = find_state(state, kIntegratorInitialized);
  auto* diff_initialized = static_cast<std::uint8_t*>(differentiator_initialized->values.data);
  auto* int_initialized = static_cast<std::uint8_t*>(integrator_initialized->values.data);
  const ParameterView* proportional_gain = find_parameter(parameters, kProportionalGain);
  const ParameterView* differentiator_gain = find_parameter(parameters, kDifferentiatorGain);
  const ParameterView* integral_gain = find_parameter(parameters, kIntegralGain);
  const ParameterView* u0 = find_parameter(parameters, kU0);
  const ParameterView* upper = find_parameter(parameters, kUpperLimit);
  const ParameterView* lower = find_parameter(parameters, kLowerLimit);
  const ParameterView* maximum = find_parameter(parameters, kMaximumDistance);
  const ParameterView* shaping = find_parameter(parameters, kShapingDistance);

  for (std::size_t env = 0; env < num_envs; ++env) {
    const Scalar dt_value = internal::read_scalar_value<Scalar>(dt, env);
    if (!std::isfinite(dt_value) || dt_value <= static_cast<Scalar>(0)) {
      status.flags[env] |= status_bit(StatusFlag::kInvalidDt);
    }
    if (!internal::matrix_env_is_finite<Scalar>(external_wrench, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(desired_wrench, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(feedforward_wrench, env, kCartesianDim) ||
        !internal::matrix_env_is_finite<Scalar>(pose_error, env, kCartesianDim)) {
      status.flags[env] |= status_bit(StatusFlag::kNonFiniteInput);
    }
    if ((status.flags[env] &
         (status_bit(StatusFlag::kInvalidParameter) | status_bit(StatusFlag::kInvalidInput) |
          status_bit(StatusFlag::kNonFiniteInput) | status_bit(StatusFlag::kInvalidDt))) != 0U) {
      status.flags[env] |= status_bit(StatusFlag::kSafeMode);
      zero_cartesian_output<Scalar>(commanded_wrench, env);
      continue;
    }

    const std::size_t diff_initialized_offset =
        env * static_cast<std::size_t>(differentiator_initialized->values.strides[0]);
    const std::size_t int_initialized_offset =
        env * static_cast<std::size_t>(integrator_initialized->values.strides[0]);
    Scalar derivative[6]{};
    Scalar integral_output[6]{};
    Scalar error[6]{};
    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      error[dim] =
          detail::force_error_value(internal::read_matrix_value<Scalar>(desired_wrench, env, dim),
                                    internal::read_matrix_value<Scalar>(external_wrench, env, dim));
      if (diff_initialized[diff_initialized_offset] != 0U) {
        derivative[dim] = detail::differentiator_value(
            error[dim],
            internal::read_matrix_value<Scalar>(differentiator_previous->values, env, dim),
            dt_value);
      }
      if (int_initialized[int_initialized_offset] == 0U) {
        internal::write_matrix_value(
            integrated_error->values,
            env,
            dim,
            internal::read_vector_parameter<Scalar>(u0, env, dim, static_cast<Scalar>(0)));
      } else {
        const Scalar upper_value = internal::read_vector_parameter<Scalar>(
            upper, env, dim, std::numeric_limits<Scalar>::max());
        const Scalar lower_value = internal::read_vector_parameter<Scalar>(
            lower, env, dim, -std::numeric_limits<Scalar>::max());
        const auto update = detail::integrator_trapezoid_value(
            internal::read_matrix_value<Scalar>(integrated_error->values, env, dim),
            error[dim],
            internal::read_matrix_value<Scalar>(integrator_previous->values, env, dim),
            dt_value,
            lower_value,
            upper_value);
        if (update.saturated) {
          status.flags[env] |= status_bit(StatusFlag::kSaturation);
        }
        internal::write_matrix_value(integrated_error->values, env, dim, update.value);
        integral_output[dim] = update.value;
      }
    }
    diff_initialized[diff_initialized_offset] = 1U;
    int_initialized[int_initialized_offset] = 1U;

    for (std::size_t dim = 0; dim < kCartesianDim; ++dim) {
      Scalar scale = static_cast<Scalar>(1);
      if (use_shaping_function) {
        const Scalar maximum_value =
            internal::read_vector_parameter<Scalar>(maximum, env, dim, static_cast<Scalar>(0));
        const Scalar shaping_value =
            internal::read_vector_parameter<Scalar>(shaping, env, dim, static_cast<Scalar>(0));
        scale =
            detail::shaping_factor_value(internal::read_matrix_value<Scalar>(pose_error, env, dim),
                                         maximum_value,
                                         shaping_value);
      }
      const Scalar command = detail::cartesian_force_command_value(
          detail::proportional_value(internal::read_vector_parameter<Scalar>(
                                         proportional_gain, env, dim, static_cast<Scalar>(0)),
                                     error[dim]),
          detail::proportional_value(internal::read_vector_parameter<Scalar>(
                                         differentiator_gain, env, dim, static_cast<Scalar>(0)),
                                     derivative[dim]),
          detail::proportional_value(internal::read_vector_parameter<Scalar>(
                                         integral_gain, env, dim, static_cast<Scalar>(0)),
                                     integral_output[dim]),
          internal::read_matrix_value<Scalar>(desired_wrench, env, dim),
          internal::read_matrix_value<Scalar>(feedforward_wrench, env, dim),
          scale);
      internal::write_matrix_value(commanded_wrench, env, dim, command);
      internal::write_matrix_value(differentiator_previous->values, env, dim, error[dim]);
      internal::write_matrix_value(integrator_previous->values, env, dim, error[dim]);
    }
  }
  return internal::ok();
}

Result step_cartesian_force_controller(const TensorView& external_wrench,
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
  if (external_wrench.device == Device::kCuda || desired_wrench.device == Device::kCuda ||
      feedforward_wrench.device == Device::kCuda || pose_error.device == Device::kCuda ||
      dt.device == Device::kCuda || commanded_wrench.device == Device::kCuda ||
      status.device == Device::kCuda ||
      (parameter_status.flags != nullptr && parameter_status.device == Device::kCuda) ||
      parameter_set_has_cuda(parameters)) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_cartesian_force_controller_cuda(external_wrench,
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
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (external_wrench.dtype == DType::kFloat32) {
    return step_force_controller_impl<float>(external_wrench,
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
  if (external_wrench.dtype == DType::kFloat64) {
    return step_force_controller_impl<double>(external_wrench,
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
  return internal::invalid_view("external_wrench dtype must be float32 or float64");
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
