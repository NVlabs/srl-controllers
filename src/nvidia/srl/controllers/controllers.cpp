/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file controllers.cpp
 * @brief Public controller adapters over the shared CPU/CUDA equation kernels.
 */

#include "nvidia/srl/controllers/controllers.hpp"

#include <array>
#include <cstddef>

#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include "tensor/energy_tank.hpp"
#include "tensor/generic_blocks.hpp"
#include "tensor/high_level_controllers.hpp"

#include "nvidia/srl/controllers/tensor/cartesian_blocks.hpp"
#include "nvidia/srl/controllers/tensor/joint_blocks.hpp"
#include "nvidia/srl/controllers/tensor/operational_space_mass.hpp"
#include "nvidia/srl/controllers/tensor/safety.hpp"
#include "nvidia/srl/controllers/tensor/torque_filter.hpp"
#include "tensor/bimanual_task_kinematics.hpp"
#include "tensor/common.hpp"
#include "tensor/task_wrench_to_joint_torque.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace {

using detail::tensor::ParameterSetView;
using detail::tensor::ParameterView;
using detail::tensor::ScratchEntryView;
using detail::tensor::ScratchView;
using detail::tensor::StateEntryView;
using detail::tensor::StateView;

constexpr std::size_t kCartesianDim = 6;

Result invalid_argument(const char* message) noexcept {
  return Result{ResultCode::kInvalidArgument, message};
}

Result invalid_view(const char* message) noexcept {
  return Result{ResultCode::kInvalidView, message};
}

Result validate_config(const ControllerConfig& config, const bool require_joint_dim) noexcept {
  if (config.batch_size == 0) {
    return invalid_argument("batch_size must be greater than zero");
  }
  if (config.dtype != DType::kFloat32 && config.dtype != DType::kFloat64) {
    return invalid_argument("dtype must be float32 or float64");
  }
  if (config.device != Device::kCpu && config.device != Device::kCuda) {
    return invalid_argument("device must be CPU or CUDA");
  }
  if (require_joint_dim && (config.joint_dim == 0 || config.joint_dim > 64)) {
    return invalid_argument("joint_dim must be in [1, 64]");
  }
  if (config.task_dim == 0 || config.task_dim > 12) {
    return invalid_argument("task_dim must be in [1, 12]");
  }
  const std::size_t dimension =
      require_joint_dim && config.joint_dim > config.task_dim ? config.joint_dim : config.task_dim;
  const std::size_t element_bytes =
      config.dtype == DType::kFloat32 ? sizeof(float) : sizeof(double);
  if (config.batch_size > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) /
                              dimension / dimension ||
      config.batch_size >
          std::numeric_limits<std::size_t>::max() / element_bytes / dimension / dimension) {
    return invalid_argument("configured tensor dimensions overflow contiguous storage");
  }
#ifndef NVIDIASRLCONTROLLERS_ENABLE_CUDA
  if (config.device == Device::kCuda) {
    return Result{ResultCode::kBackendUnavailable,
                  "CUDA preparation requires a library built with ENABLE_CUDA=ON"};
  }
#endif
  if (config.safety.safe_output_policy == SafeOutputPolicy::kHoldLastValid &&
      config.safety.max_hold_steps == 0) {
    return invalid_argument("hold-last safety requires max_hold_steps > 0");
  }
  return Result{};
}

Result validate_joint_config(const ControllerConfig& config) noexcept {
  const Result result = validate_config(config, true);
  if (!result.ok()) {
    return result;
  }
  if (config.joint_dim > 32) {
    return invalid_argument("joint and nullspace controllers require joint_dim in [1, 32]");
  }
  return Result{};
}

Result validate_cartesian_config(const ControllerConfig& config,
                                 const bool require_joint_dim,
                                 const bool variable_task_dim = false) noexcept {
  const Result result = validate_config(config, require_joint_dim);
  if (!result.ok()) {
    return result;
  }
  if (config.cartesian_frame != CartesianFrame::kRobotBase &&
      config.cartesian_frame != CartesianFrame::kEndEffector &&
      config.cartesian_frame != CartesianFrame::kMainFrameOrientation) {
    return invalid_argument(
        "Cartesian controllers require an explicit robot-base or end-effector frame");
  }
  if (!variable_task_dim && config.task_dim != kCartesianDim) {
    return invalid_argument("this Cartesian controller requires task_dim == 6");
  }
  return Result{};
}

const char* cartesian_frame_name(const CartesianFrame frame) noexcept {
  if (frame == CartesianFrame::kRobotBase) {
    return "robot_base";
  }
  return frame == CartesianFrame::kEndEffector ? "end_effector" : "main_frame_orientation";
}

StatusView read_status(const MutableStatusView view) noexcept {
  return StatusView{view.flags, view.size, view.device};
}

Result first_failure(const Result first, const Result second) noexcept {
  return first.ok() ? second : first;
}

bool same_shape(const MutableTensorView& left, const MutableTensorView& right) noexcept {
  if (left.dtype != right.dtype || left.device != right.device || left.layout != right.layout ||
      left.rank != right.rank) {
    return false;
  }
  for (int dim = 0; dim < left.rank; ++dim) {
    if (left.shape[dim] != right.shape[dim] || left.strides[dim] != right.strides[dim]) {
      return false;
    }
  }
  return true;
}

Result validate_contiguous_view(const MutableTensorView& view,
                                const ControllerConfig& config,
                                const DType dtype,
                                const int rank,
                                const std::array<std::int64_t, kMaxTensorRank>& shape,
                                const char* message) noexcept {
  if (view.data == nullptr || view.dtype != dtype || view.device != config.device ||
      view.layout != Layout::kRowMajor || view.rank != rank) {
    return invalid_view(message);
  }
  std::int64_t expected_stride = 1;
  for (int reverse = 0; reverse < rank; ++reverse) {
    const int dimension = rank - reverse - 1;
    if (view.shape[dimension] != shape[dimension] || view.strides[dimension] != expected_stride) {
      return invalid_view(message);
    }
    if (shape[dimension] < 0 ||
        (shape[dimension] != 0 &&
         expected_stride > std::numeric_limits<std::int64_t>::max() / shape[dimension])) {
      return invalid_view(message);
    }
    expected_stride *= shape[dimension];
  }
  detail::tensor::internal::ByteSpan span;
  return detail::tensor::internal::byte_span(view.as_const(), &span);
}

Result validate_step_signals(const ControllerConfig& config,
                             const std::size_t dimension,
                             std::initializer_list<TensorView> signals) noexcept {
  for (const TensorView& view : signals) {
    const Result metadata = detail::tensor::internal::validate_tensor_metadata(view);
    if (!metadata.ok()) return metadata;
    if (view.device != config.device || view.dtype != config.dtype ||
        !detail::tensor::internal::is_row_major_matrix(view) ||
        view.shape[0] != static_cast<std::int64_t>(config.batch_size) ||
        view.shape[1] != static_cast<std::int64_t>(dimension)) {
      return invalid_view("step signals must match configured device, dtype and dimensions");
    }
  }
  return {};
}

Result validate_inertial_storage(const ControllerConfig& config,
                                 const InertialStateStorage& storage) noexcept {
  const std::int64_t task_dim = static_cast<std::int64_t>(config.task_dim);
  Result result = validate_contiguous_view(
      storage.previous_mass_matrix,
      config,
      config.dtype,
      3,
      {static_cast<std::int64_t>(config.batch_size), task_dim, task_dim, 0},
      "previous_mass_matrix must be contiguous [batch_size,task_dim,task_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(
      storage.previous_jacobian,
      config,
      config.dtype,
      3,
      {static_cast<std::int64_t>(config.batch_size),
       task_dim,
       static_cast<std::int64_t>(config.joint_dim),
       0},
      "previous_jacobian must be contiguous [batch_size,task_dim,joint_dim]");
  if (!result.ok()) {
    return result;
  }
  return validate_contiguous_view(storage.initialized,
                                  config,
                                  DType::kBool,
                                  1,
                                  {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                  "inertial initialized must be bool [batch_size]");
}

Result validate_force_storage(const ControllerConfig& config,
                              const ForceControllerStateStorage& storage) noexcept {
  const MutableTensorView signals[] = {storage.differentiator_previous_error,
                                       storage.integrator_previous_error,
                                       storage.integrated_error};
  for (const MutableTensorView& signal : signals) {
    const Result result =
        validate_contiguous_view(signal,
                                 config,
                                 config.dtype,
                                 2,
                                 {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
                                 "force-controller signal state must be contiguous [batch_size,6]");
    if (!result.ok()) {
      return result;
    }
  }
  const MutableTensorView initialized[] = {storage.differentiator_initialized,
                                           storage.integrator_initialized};
  for (const MutableTensorView& mask : initialized) {
    const Result result =
        validate_contiguous_view(mask,
                                 config,
                                 DType::kBool,
                                 1,
                                 {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                 "force-controller initialized state must be bool [batch_size]");
    if (!result.ok()) {
      return result;
    }
  }
  return Result{};
}

Result validate_admittance_storage(const ControllerConfig& config,
                                   const AdmittanceStateStorage& storage) noexcept {
  const MutableTensorView signals[] = {storage.pose_offset,
                                       storage.twist,
                                       storage.pose_previous_signal,
                                       storage.twist_previous_signal};
  for (const MutableTensorView& signal : signals) {
    Result result =
        validate_contiguous_view(signal,
                                 config,
                                 config.dtype,
                                 2,
                                 {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
                                 "admittance signal state must be contiguous [batch_size,6]");
    if (!result.ok()) return result;
  }
  const MutableTensorView flags[] = {storage.pose_integrator_initialized,
                                     storage.twist_integrator_initialized};
  for (const MutableTensorView& flag : flags) {
    Result result =
        validate_contiguous_view(flag,
                                 config,
                                 DType::kBool,
                                 1,
                                 {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                 "admittance initialized state must be bool [batch_size]");
    if (!result.ok()) return result;
  }
  return Result{};
}

Result validate_storage(const ControllerConfig& config,
                        const ControllerOutputStorage& storage,
                        const std::size_t output_dim) noexcept {
  const MutableTensorView& output = storage.command;
  if (output.data == nullptr || output.dtype != config.dtype || output.device != config.device ||
      output.layout != Layout::kRowMajor || output.rank != 2 ||
      output.shape[0] != static_cast<std::int64_t>(config.batch_size) ||
      output.shape[1] != static_cast<std::int64_t>(output_dim) || output.strides[1] != 1 ||
      output.strides[0] != static_cast<std::int64_t>(output_dim)) {
    return invalid_view("command output must be contiguous row-major [batch_size, output_dim]");
  }
  if (storage.parameter_status.flags == nullptr ||
      storage.parameter_status.size < config.batch_size ||
      storage.parameter_status.device != config.device || storage.status.flags == nullptr ||
      storage.status.size < config.batch_size || storage.status.device != config.device) {
    return invalid_view("parameter_status and status must match the configured batch and device");
  }
  const Result overlap_result = detail::tensor::internal::validate_disjoint(
      output.as_const(),
      storage.safety.last_valid_output.as_const(),
      "last_valid_output must not alias the command output");
  if (!overlap_result.ok()) {
    return overlap_result;
  }
  if (storage.safety.last_valid_output.data == nullptr ||
      !same_shape(output, storage.safety.last_valid_output) ||
      storage.safety.last_valid_output.layout != Layout::kRowMajor) {
    return invalid_view("caller-owned safety history does not match the command output");
  }
  Result result = validate_contiguous_view(storage.safety.valid_mask,
                                           config,
                                           DType::kBool,
                                           1,
                                           {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                           "valid_mask must be contiguous bool [batch_size]");
  if (!result.ok()) {
    return result;
  }
  return validate_contiguous_view(storage.safety.hold_count,
                                  config,
                                  DType::kUInt64,
                                  1,
                                  {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                  "hold_count must be contiguous uint64 [batch_size]");
}

using detail::tensor::internal::StorageValidation;

void append_output_storage(StorageValidation* validation,
                           const ControllerOutputStorage& storage,
                           int stage,
                           int last,
                           bool inplace = false) noexcept {
  validation->add(storage.command.as_const(), stage, last, inplace);
  validation->add(detail::tensor::internal::status_tensor(storage.status), stage, last);
  validation->history({detail::tensor::internal::status_tensor(storage.parameter_status),
                       storage.safety.last_valid_output.as_const(),
                       storage.safety.valid_mask.as_const(),
                       storage.safety.hold_count.as_const()});
}

void append_inertial_storage(StorageValidation* validation,
                             const InertialStateStorage& storage) noexcept {
  validation->history({storage.previous_mass_matrix.as_const(),
                       storage.previous_jacobian.as_const(),
                       storage.initialized.as_const()});
}

void append_force_storage(StorageValidation* validation,
                          const ForceControllerStateStorage& storage) noexcept {
  validation->history({storage.differentiator_previous_error.as_const(),
                       storage.differentiator_initialized.as_const(),
                       storage.integrator_previous_error.as_const(),
                       storage.integrated_error.as_const(),
                       storage.integrator_initialized.as_const()});
}

void append_storage(StorageValidation* validation,
                    const ControllerOutputStorage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
}

void append_storage(StorageValidation* validation,
                    const TorqueFilter::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last, true);
  validation->history(
      {storage.state.previous_command.as_const(), storage.state.initialized.as_const()});
}

void append_storage(StorageValidation* validation,
                    const OperationalSpaceMass::Storage& storage,
                    int stage,
                    int last) noexcept {
  validation->add(storage.cartesian_mass_matrix.as_const(), stage, last);
  validation->add(detail::tensor::internal::status_tensor(storage.status), stage, stage);
  validation->add(detail::tensor::internal::status_tensor(storage.parameter_status));
  for (const auto& view : {storage.scratch.joint_factor,
                           storage.scratch.solve_rhs,
                           storage.scratch.cartesian_inverse}) {
    validation->add(view.as_const(), stage, stage);
  }
}

void append_storage(StorageValidation* validation,
                    const CartesianImpedanceController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  append_inertial_storage(validation, storage.state);
}

void append_storage(StorageValidation* validation,
                    const CartesianAdmittancePoseController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  validation->history({storage.state.pose_offset.as_const(),
                       storage.state.twist.as_const(),
                       storage.state.pose_previous_signal.as_const(),
                       storage.state.twist_previous_signal.as_const(),
                       storage.state.pose_integrator_initialized.as_const(),
                       storage.state.twist_integrator_initialized.as_const()});
}

void append_storage(StorageValidation* validation,
                    const CartesianUnifiedForceAdmittanceController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  append_force_storage(validation, storage.state.force);
  const AdmittanceStateStorage& state = storage.state.admittance;
  validation->history({state.pose_offset.as_const(),
                       state.twist.as_const(),
                       state.pose_previous_signal.as_const(),
                       state.twist_previous_signal.as_const(),
                       state.pose_integrator_initialized.as_const(),
                       state.twist_integrator_initialized.as_const()});
}

void append_storage(StorageValidation* validation,
                    const CartesianForceController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  append_force_storage(validation, storage.state);
}

void append_storage(StorageValidation* validation,
                    const CartesianUnifiedForceImpedanceController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  append_inertial_storage(validation, storage.state.impedance);
  append_force_storage(validation, storage.state.force);
  for (const auto& view : {storage.scratch.impedance_wrench,
                           storage.scratch.force_wrench,
                           storage.scratch.impedance_status,
                           storage.scratch.force_status}) {
    validation->add(view.as_const(), stage, stage);
  }
}

void append_storage(StorageValidation* validation,
                    const VariableCartesianImpedanceController::Storage& storage,
                    int stage,
                    int last) noexcept {
  append_output_storage(validation, storage, stage, last);
  append_inertial_storage(validation, storage.state.inertial);
  validation->history({storage.state.current_stiffness_matrix.as_const(),
                       storage.state.previous_stiffness_matrix.as_const(),
                       storage.state.stiffness_initialized.as_const()});
  for (const auto& view : {storage.scratch.stiffness_wrench,
                           storage.scratch.damping_wrench,
                           storage.scratch.inertial_wrench,
                           storage.scratch.stiffness_status,
                           storage.scratch.damping_status,
                           storage.scratch.inertial_status}) {
    validation->add(view.as_const(), stage, stage);
  }
}

void append_storage(StorageValidation* validation,
                    const BimanualTaskKinematics::Storage& storage,
                    int stage,
                    int last) noexcept {
  validation->add(storage.task_pose_error.as_const(), stage, last);
  validation->add(storage.task_jacobian.as_const(), stage, last);
  validation->add(detail::tensor::internal::status_tensor(storage.status), stage, stage);
}

void append_storage(StorageValidation* validation,
                    const TaskWrenchToJointTorqueController::Storage& storage,
                    int stage,
                    int last) noexcept {
  validation->add(storage.unfiltered_joint_torque.as_const(), stage, stage, true);
  append_storage(validation, storage.filter, stage, last);
}

template <typename TaskStorage>
void append_storage(StorageValidation* validation,
                    const CartesianTorqueStorage<TaskStorage>& storage,
                    int,
                    int) noexcept {
  append_storage(validation, storage.mass, 0, 1);
  append_storage(validation, storage.task, 1, 3);
  append_storage(validation, storage.nullspace, 2, 3);
  append_storage(validation, storage.output, 3, 3);
}

void append_storage(StorageValidation* validation,
                    const BimanualCartesianImpedanceTorqueController::Storage& storage,
                    int,
                    int) noexcept {
  append_storage(validation, storage.kinematics, 0, 4);
  append_storage(validation, storage.mass, 1, 2);
  append_storage(validation, storage.task, 2, 4);
  append_storage(validation, storage.nullspace, 3, 4);
  append_storage(validation, storage.output, 4, 4);
  validation->add(storage.task_velocity_error.as_const(), 0, 4);
}

template <typename Storage>
Result validate_independent_storage(const Storage& storage) noexcept {
  StorageValidation validation;
  append_storage(&validation, storage, 0, 0);
  return validation.result;
}

template <std::size_t Capacity>
void append_parameter(std::array<ParameterView, Capacity>* entries,
                      std::size_t* count,
                      const char* name,
                      const ParameterTensor& parameter) noexcept {
  if (parameter.values.data != nullptr && *count < Capacity) {
    (*entries)[*count] =
        ParameterView{name, parameter.values, parameter.batching == ParameterBatching::kShared};
    ++(*count);
  }
}

template <std::size_t Capacity>
ParameterSetView parameter_set(const std::array<ParameterView, Capacity>& entries,
                               const std::size_t count) noexcept {
  return ParameterSetView{entries.data(), count};
}

StatusView parameter_status(const ControllerOutputStorage& storage) noexcept {
  return StatusView{storage.parameter_status.flags,
                    storage.parameter_status.size,
                    storage.parameter_status.device};
}

StatusView parameter_status(const OperationalSpaceMass::Storage& storage) noexcept {
  return StatusView{storage.parameter_status.flags,
                    storage.parameter_status.size,
                    storage.parameter_status.device};
}

StateView inertial_state(const InertialStateStorage& storage,
                         std::array<StateEntryView, 3>* entries) noexcept {
  *entries = {StateEntryView{"previous_mass_matrix", storage.previous_mass_matrix},
              StateEntryView{"previous_jacobian", storage.previous_jacobian},
              StateEntryView{"initialized", storage.initialized}};
  return StateView{entries->data(), entries->size()};
}

StateView admittance_state(const AdmittanceStateStorage& storage,
                           std::array<StateEntryView, 6>* entries) noexcept {
  *entries = {StateEntryView{"admittance_pose", storage.pose_offset},
              StateEntryView{"admittance_twist", storage.twist},
              StateEntryView{"pose_previous_signal", storage.pose_previous_signal},
              StateEntryView{"twist_previous_signal", storage.twist_previous_signal},
              StateEntryView{"pose_integrator_initialized", storage.pose_integrator_initialized},
              StateEntryView{"twist_integrator_initialized", storage.twist_integrator_initialized}};
  return StateView{entries->data(), entries->size()};
}

template <std::size_t Capacity>
void append_force_parameters(std::array<ParameterView, Capacity>* entries,
                             std::size_t* count,
                             const ForceControllerParameters& parameters) noexcept {
  append_parameter(entries, count, "proportional_gain", parameters.proportional_gain);
  append_parameter(entries, count, "differentiator_gain", parameters.differentiator_gain);
  append_parameter(entries, count, "integral_gain", parameters.integral_gain);
  append_parameter(entries, count, "u_0", parameters.initial_output);
  append_parameter(entries, count, "maximum_distance", parameters.maximum_distance);
  append_parameter(entries, count, "shaping_distance", parameters.shaping_distance);
}

StateView force_state(const ForceControllerStateStorage& storage,
                      std::array<StateEntryView, 5>* entries) noexcept {
  *entries = {
      StateEntryView{"differentiator_previous_error", storage.differentiator_previous_error},
      StateEntryView{"differentiator_initialized", storage.differentiator_initialized},
      StateEntryView{"integrator_previous_error", storage.integrator_previous_error},
      StateEntryView{"integrated_error", storage.integrated_error},
      StateEntryView{"integrator_initialized", storage.integrator_initialized}};
  return StateView{entries->data(), entries->size()};
}

StateView unified_force_admittance_state(const UnifiedForceAdmittanceStateStorage& storage,
                                         std::array<StateEntryView, 11>* entries) noexcept {
  *entries = {
      StateEntryView{"differentiator_previous_error", storage.force.differentiator_previous_error},
      StateEntryView{"differentiator_initialized", storage.force.differentiator_initialized},
      StateEntryView{"integrator_previous_error", storage.force.integrator_previous_error},
      StateEntryView{"integrated_error", storage.force.integrated_error},
      StateEntryView{"integrator_initialized", storage.force.integrator_initialized},
      StateEntryView{"admittance_pose", storage.admittance.pose_offset},
      StateEntryView{"admittance_twist", storage.admittance.twist},
      StateEntryView{"pose_previous_signal", storage.admittance.pose_previous_signal},
      StateEntryView{"twist_previous_signal", storage.admittance.twist_previous_signal},
      StateEntryView{"pose_integrator_initialized", storage.admittance.pose_integrator_initialized},
      StateEntryView{"twist_integrator_initialized",
                     storage.admittance.twist_integrator_initialized}};
  return StateView{entries->data(), entries->size()};
}

StateView unified_state(const UnifiedControllerStateStorage& storage,
                        std::array<StateEntryView, 8>* entries) noexcept {
  *entries = {
      StateEntryView{"previous_mass_matrix", storage.impedance.previous_mass_matrix},
      StateEntryView{"previous_jacobian", storage.impedance.previous_jacobian},
      StateEntryView{"initialized", storage.impedance.initialized},
      StateEntryView{"differentiator_previous_error", storage.force.differentiator_previous_error},
      StateEntryView{"differentiator_initialized", storage.force.differentiator_initialized},
      StateEntryView{"integrator_previous_error", storage.force.integrator_previous_error},
      StateEntryView{"integrated_error", storage.force.integrated_error},
      StateEntryView{"integrator_initialized", storage.force.integrator_initialized}};
  return StateView{entries->data(), entries->size()};
}

ScratchView unified_scratch(const UnifiedControllerScratchStorage& storage,
                            std::array<ScratchEntryView, 4>* entries) noexcept {
  *entries = {ScratchEntryView{"impedance_wrench", storage.impedance_wrench},
              ScratchEntryView{"force_wrench", storage.force_wrench},
              ScratchEntryView{"impedance_status", storage.impedance_status},
              ScratchEntryView{"force_status", storage.force_status}};
  return ScratchView{entries->data(), entries->size()};
}

StateView variable_impedance_state(const VariableImpedanceStateStorage& storage,
                                   std::array<StateEntryView, 6>* entries) noexcept {
  *entries = {StateEntryView{"current_stiffness_matrix", storage.current_stiffness_matrix},
              StateEntryView{"previous_stiffness_matrix", storage.previous_stiffness_matrix},
              StateEntryView{"stiffness_initialized", storage.stiffness_initialized},
              StateEntryView{"previous_mass_matrix", storage.inertial.previous_mass_matrix},
              StateEntryView{"previous_jacobian", storage.inertial.previous_jacobian},
              StateEntryView{"inertial_initialized", storage.inertial.initialized}};
  return StateView{entries->data(), entries->size()};
}

ScratchView variable_impedance_scratch(const VariableImpedanceScratchStorage& storage,
                                       std::array<ScratchEntryView, 6>* entries) noexcept {
  *entries = {ScratchEntryView{"stiffness_wrench", storage.stiffness_wrench},
              ScratchEntryView{"damping_wrench", storage.damping_wrench},
              ScratchEntryView{"inertial_wrench", storage.inertial_wrench},
              ScratchEntryView{"stiffness_status", storage.stiffness_status},
              ScratchEntryView{"damping_status", storage.damping_status},
              ScratchEntryView{"inertial_status", storage.inertial_status}};
  return ScratchView{entries->data(), entries->size()};
}

Result finish_step(const Result result,
                   const ControllerConfig& config,
                   const ControllerOutputStorage& storage) noexcept {
  if (!result.ok()) {
    return result;
  }
  return detail::tensor::apply_safe_output_policy(
      storage.command, storage.status, storage.safety, config.safety);
}

Result reset_safety(const ControllerConfig& config,
                    const ControllerOutputStorage& storage,
                    const ResetInputs& inputs) noexcept {
  return detail::tensor::reset_safety_state(
      config.batch_size, storage.safety, inputs.mask, storage.status);
}

void describe_output(const ControllerConfig& config,
                     const char* name,
                     const std::size_t dimension,
                     const char* units,
                     const char* frame,
                     Requirements* requirements) noexcept {
  *requirements = Requirements{};
  requirements->batch_size = config.batch_size;
  requirements->joint_dim = config.joint_dim;
  requirements->outputs[0] = TensorRequirement{
      name,
      config.dtype,
      config.device,
      2,
      {static_cast<std::int64_t>(config.batch_size), static_cast<std::int64_t>(dimension), 0, 0},
      units,
      frame};
  requirements->outputs[1] =
      TensorRequirement{"parameter_status",
                        DType::kUInt64,
                        config.device,
                        1,
                        {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                        "status_flags",
                        "none"};
  requirements->outputs[2] =
      TensorRequirement{"status",
                        DType::kUInt64,
                        config.device,
                        1,
                        {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                        "status_flags",
                        "none"};
  requirements->output_count = 3;
}

Result append_safety_requirements(const ControllerConfig& config,
                                  const std::size_t output_dimension,
                                  const char* units,
                                  const char* frame,
                                  Requirements* requirements) noexcept {
  const std::size_t offset = requirements->state_count;
  if (offset > requirements->state.size() - 3) {
    *requirements = Requirements{};
    return invalid_argument("requirements state capacity exceeded");
  }
  requirements->state[offset] = TensorRequirement{"last_valid_output",
                                                  config.dtype,
                                                  config.device,
                                                  2,
                                                  {static_cast<std::int64_t>(config.batch_size),
                                                   static_cast<std::int64_t>(output_dimension),
                                                   0,
                                                   0},
                                                  units,
                                                  frame};
  requirements->state[offset + 1] =
      TensorRequirement{"valid_mask",
                        DType::kBool,
                        config.device,
                        1,
                        {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                        "unitless",
                        "none"};
  requirements->state[offset + 2] =
      TensorRequirement{"hold_count",
                        DType::kUInt64,
                        config.device,
                        1,
                        {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                        "control_steps",
                        "none"};
  requirements->state_count += 3;
  return Result{};
}

constexpr const char* kMassRequirementNames[] = {"mass.joint_mass_factor",
                                                 "mass.mass_inverse_jacobian_transpose",
                                                 "mass.cartesian_inverse_mass_factor",
                                                 "mass.cartesian_mass_matrix",
                                                 "mass.parameter_status",
                                                 "mass.status"};
constexpr const char* kTaskRequirementNames[] = {"task.previous_mass_matrix",
                                                 "task.previous_jacobian",
                                                 "task.initialized",
                                                 "task.last_valid_output",
                                                 "task.valid_mask",
                                                 "task.hold_count",
                                                 "task.current_stiffness_matrix",
                                                 "task.previous_stiffness_matrix",
                                                 "task.stiffness_initialized",
                                                 "task.inertial_initialized",
                                                 "task.differentiator_previous_error",
                                                 "task.differentiator_initialized",
                                                 "task.integrator_previous_error",
                                                 "task.integrated_error",
                                                 "task.integrator_initialized",
                                                 "task.stiffness_wrench",
                                                 "task.damping_wrench",
                                                 "task.inertial_wrench",
                                                 "task.stiffness_status",
                                                 "task.damping_status",
                                                 "task.inertial_status",
                                                 "task.impedance_wrench",
                                                 "task.force_wrench",
                                                 "task.impedance_status",
                                                 "task.force_status",
                                                 "task.cartesian_wrench",
                                                 "task.parameter_status",
                                                 "task.status"};
constexpr const char* kNullspaceRequirementNames[] = {"nullspace.last_valid_output",
                                                      "nullspace.valid_mask",
                                                      "nullspace.hold_count",
                                                      "nullspace.joint_torque",
                                                      "nullspace.parameter_status",
                                                      "nullspace.status"};
constexpr const char* kKinematicsRequirementNames[] = {
    "kinematics.task_pose_error", "kinematics.task_jacobian", "kinematics.status"};

template <std::size_t Capacity>
bool append_requirement(std::array<TensorRequirement, Capacity>* entries,
                        std::size_t* count,
                        TensorRequirement requirement,
                        const char* name) noexcept {
  if (*count >= Capacity || name == nullptr) {
    return false;
  }
  requirement.name = name;
  (*entries)[(*count)++] = requirement;
  return true;
}

template <std::size_t NameCount>
const char* qualify_requirement(const char* name, const char* const (&names)[NameCount]) noexcept {
  if (name == nullptr) {
    return nullptr;
  }
  for (const char* qualified : names) {
    if (std::strcmp(name, std::strchr(qualified, '.') + 1) == 0) {
      return qualified;
    }
  }
  return nullptr;
}

template <std::size_t NameCount>
Result append_intermediate_requirements(const Requirements& leaf,
                                        const char* const (&names)[NameCount],
                                        Requirements* output) noexcept {
  if (leaf.state_count > leaf.state.size() || leaf.scratch_count > leaf.scratch.size() ||
      leaf.output_count > leaf.outputs.size()) {
    return invalid_argument("leaf requirements exceed capacity");
  }
  for (std::size_t i = 0; i < leaf.state_count; ++i) {
    if (!append_requirement(&output->state,
                            &output->state_count,
                            leaf.state[i],
                            qualify_requirement(leaf.state[i].name, names))) {
      return invalid_argument("intermediate state requirements cannot be appended");
    }
  }
  for (std::size_t i = 0; i < leaf.scratch_count; ++i) {
    if (!append_requirement(&output->scratch,
                            &output->scratch_count,
                            leaf.scratch[i],
                            qualify_requirement(leaf.scratch[i].name, names))) {
      return invalid_argument("intermediate scratch requirements cannot be appended");
    }
  }
  for (std::size_t i = 0; i < leaf.output_count; ++i) {
    const TensorRequirement& requirement = leaf.outputs[i];
    const char* name = qualify_requirement(requirement.name, names);
    const bool persistent =
        requirement.name != nullptr && std::strcmp(requirement.name, "parameter_status") == 0;
    const bool appended =
        persistent
            ? append_requirement(&output->state, &output->state_count, requirement, name)
            : append_requirement(&output->scratch, &output->scratch_count, requirement, name);
    if (!appended) {
      return invalid_argument("intermediate output requirements cannot be appended");
    }
  }
  return Result{};
}

Result append_final_requirements(const Requirements& leaf, Requirements* output) noexcept {
  if (leaf.state_count > leaf.state.size() || leaf.scratch_count > leaf.scratch.size() ||
      leaf.output_count > leaf.outputs.size()) {
    return invalid_argument("final requirements exceed capacity");
  }
  for (std::size_t i = 0; i < leaf.state_count; ++i) {
    if (!append_requirement(
            &output->state, &output->state_count, leaf.state[i], leaf.state[i].name)) {
      return invalid_argument("final state requirements exceed capacity");
    }
  }
  for (std::size_t i = 0; i < leaf.scratch_count; ++i) {
    if (!append_requirement(
            &output->scratch, &output->scratch_count, leaf.scratch[i], leaf.scratch[i].name)) {
      return invalid_argument("final scratch requirements exceed capacity");
    }
  }
  for (std::size_t i = 0; i < leaf.output_count; ++i) {
    if (!append_requirement(
            &output->outputs, &output->output_count, leaf.outputs[i], leaf.outputs[i].name)) {
      return invalid_argument("final output requirements exceed capacity");
    }
  }
  return Result{};
}

TensorRequirement bimanual_velocity_requirement(const ControllerConfig& config) noexcept {
  return TensorRequirement{"task_velocity_error",
                           config.dtype,
                           config.device,
                           2,
                           {static_cast<std::int64_t>(config.batch_size), 12, 0, 0},
                           "meter_per_second_and_radian_per_second",
                           "main_frame_orientation_at_task_origins"};
}

template <typename TaskController, typename Config>
Result describe_torque_requirements(const Config& config, Requirements* output) noexcept {
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  *output = Requirements{};
  Requirements assembled;
  assembled.batch_size = config.batch_size;
  assembled.joint_dim = config.joint_dim;
  Requirements leaf;
  Result result;
  if constexpr (std::is_same<TaskController, BimanualCartesianImpedanceController>::value) {
    result = BimanualTaskKinematics::requirements(config, &leaf);
    if (!result.ok()) {
      return result;
    }
    result = append_intermediate_requirements(leaf, kKinematicsRequirementNames, &assembled);
    if (!result.ok()) {
      return result;
    }
    if (!append_requirement(&assembled.scratch,
                            &assembled.scratch_count,
                            bimanual_velocity_requirement(config),
                            "task_velocity_error")) {
      return invalid_argument("bimanual velocity requirements exceed capacity");
    }
  }
  result = OperationalSpaceMass::requirements(config, &leaf);
  if (!result.ok()) {
    return result;
  }
  result = append_intermediate_requirements(leaf, kMassRequirementNames, &assembled);
  if (!result.ok()) {
    return result;
  }
  result = TaskController::requirements(config, &leaf);
  if (!result.ok()) {
    return result;
  }
  result = append_intermediate_requirements(leaf, kTaskRequirementNames, &assembled);
  if (!result.ok()) {
    return result;
  }
  result = NullspaceController::requirements(config, &leaf);
  if (!result.ok()) {
    return result;
  }
  result = append_intermediate_requirements(leaf, kNullspaceRequirementNames, &assembled);
  if (!result.ok()) {
    return result;
  }
  result = TaskWrenchToJointTorqueController::requirements(config, &leaf);
  if (!result.ok()) {
    return result;
  }
  result = append_final_requirements(leaf, &assembled);
  if (!result.ok()) {
    return result;
  }
  *output = assembled;
  return Result{};
}

}  // namespace

Result TorqueFilter::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_config(config, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(
      config, "filtered_command", config.joint_dim, "command_units", "command_space", output);
  output->state_count = 2;
  output->state[0] = TensorRequirement{"previous_command",
                                       config.dtype,
                                       config.device,
                                       2,
                                       {static_cast<std::int64_t>(config.batch_size),
                                        static_cast<std::int64_t>(config.joint_dim),
                                        0,
                                        0},
                                       "command_units",
                                       "command_space"};
  output->state[1] = TensorRequirement{"initialized",
                                       DType::kBool,
                                       config.device,
                                       1,
                                       {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                       "unitless",
                                       "none"};
  return append_safety_requirements(
      config, config.joint_dim, "command_units", "command_space", output);
}

Result TorqueFilter::prepare(const Config& config,
                             const Parameters& parameters,
                             const Storage& storage,
                             const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_config(config, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, config.joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.state.previous_command,
                                    config,
                                    config.dtype,
                                    2,
                                    {static_cast<std::int64_t>(config.batch_size),
                                     static_cast<std::int64_t>(config.joint_dim),
                                     0,
                                     0},
                                    "previous_command must be contiguous [batch_size,joint_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.state.initialized,
                                    config,
                                    DType::kBool,
                                    1,
                                    {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                    "torque-filter initialized must be bool [batch_size]");
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result TorqueFilter::apply_parameters(const Parameters& parameters,
                                      const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 3> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "lower_limit", parameters_.lower_limit);
  append_parameter(&entries, &count, "upper_limit", parameters_.upper_limit);
  append_parameter(&entries, &count, "maximum_rate", parameters_.maximum_rate);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::apply_torque_filter_parameters(parameter_set(entries, count),
                                                     config_.batch_size,
                                                     config_.joint_dim,
                                                     storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result TorqueFilter::reset(const ResetInputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<StateEntryView, 2> state_entries{
      StateEntryView{"previous_command", storage_.state.previous_command},
      StateEntryView{"initialized", storage_.state.initialized}};
  StateView state{state_entries.data(), state_entries.size()};
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result =
      detail::tensor::reset_torque_filter(config_.batch_size, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result TorqueFilter::step(const Inputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 3> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(&parameter_entries, &parameter_count, "lower_limit", parameters_.lower_limit);
  append_parameter(&parameter_entries, &parameter_count, "upper_limit", parameters_.upper_limit);
  append_parameter(&parameter_entries, &parameter_count, "maximum_rate", parameters_.maximum_rate);
  std::array<StateEntryView, 2> state_entries{
      StateEntryView{"previous_command", storage_.state.previous_command},
      StateEntryView{"initialized", storage_.state.initialized}};
  StateView state{state_entries.data(), state_entries.size()};
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::step_torque_filter(inputs.command,
                                         inputs.dt,
                                         parameter_set(parameter_entries, parameter_count),
                                         parameter_status(storage_),
                                         state,
                                         storage_.command,
                                         storage_.status);
  return finish_step(result, config_, storage_);
}

Result JointImpedanceController::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_joint_config(config);
  if (!result.ok()) {
    return result;
  }
  describe_output(config, "joint_torque", config.joint_dim, "newton_meter", "joint_space", output);
  return append_safety_requirements(
      config, config.joint_dim, "newton_meter", "joint_space", output);
}

Result JointImpedanceController::prepare(const Config& config,
                                         const Parameters& parameters,
                                         const Storage& storage,
                                         const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_joint_config(config);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, config.joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result JointImpedanceController::apply_parameters(const Parameters& parameters,
                                                  const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_factors", parameters_.damping_factors);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::apply_joint_impedance_controller_parameters(parameter_set(entries, count),
                                                                  config_.batch_size,
                                                                  config_.joint_dim,
                                                                  storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result JointImpedanceController::reset(const ResetInputs& inputs,
                                       const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_joint_impedance_controller(
      config_.batch_size, inputs.mask, storage_.status);
  if (!result.ok()) {
    return result;
  }
  return reset_safety(config_, storage_, inputs);
}

Result JointImpedanceController::step(const Inputs& inputs,
                                      const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_factors", parameters_.damping_factors);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::step_joint_impedance_controller(inputs.joint_position_error,
                                                      inputs.joint_velocity_error,
                                                      inputs.desired_acceleration,
                                                      inputs.joint_mass_matrix,
                                                      parameter_set(entries, count),
                                                      parameter_status(storage_),
                                                      storage_.command,
                                                      storage_.status);
  return finish_step(result, config_, storage_);
}

Result NullspaceController::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_joint_config(config);
  if (!result.ok()) {
    return result;
  }
  describe_output(config, "joint_torque", config.joint_dim, "newton_meter", "joint_space", output);
  return append_safety_requirements(
      config, config.joint_dim, "newton_meter", "joint_space", output);
}

Result NullspaceController::prepare(const Config& config,
                                    const Parameters& parameters,
                                    const Storage& storage,
                                    const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_joint_config(config);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, config.joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result NullspaceController::apply_parameters(const Parameters& parameters,
                                             const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_matrix", parameters_.damping_matrix);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::apply_nullspace_controller_parameters(parameter_set(entries, count),
                                                            config_.batch_size,
                                                            config_.joint_dim,
                                                            storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result NullspaceController::reset(const ResetInputs& inputs,
                                  const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result =
      detail::tensor::reset_nullspace_controller(config_.batch_size, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result NullspaceController::step(const Inputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  if (inputs.jacobian.rank != 3 ||
      inputs.jacobian.shape[1] != static_cast<std::int64_t>(config_.task_dim)) {
    return invalid_view("jacobian task dimension does not match ControllerConfig::task_dim");
  }
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_matrix", parameters_.damping_matrix);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_nullspace_controller(inputs.joint_position_error,
                                                                  inputs.joint_velocity,
                                                                  inputs.joint_mass_matrix,
                                                                  inputs.jacobian,
                                                                  parameter_set(entries, count),
                                                                  parameter_status(storage_),
                                                                  storage_.command,
                                                                  storage_.status);
  return finish_step(result, config_, storage_);
}

Result OperationalSpaceMass::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "cartesian_mass_matrix",
                  config.task_dim * config.task_dim,
                  "operational_space_mass_units",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->outputs[0].rank = 3;
  output->outputs[0].shape = {static_cast<std::int64_t>(config.batch_size),
                              static_cast<std::int64_t>(config.task_dim),
                              static_cast<std::int64_t>(config.task_dim),
                              0};
  output->scratch_count = 3;
  output->scratch[0] = TensorRequirement{"joint_mass_factor",
                                         config.dtype,
                                         config.device,
                                         3,
                                         {static_cast<std::int64_t>(config.batch_size),
                                          static_cast<std::int64_t>(config.joint_dim),
                                          static_cast<std::int64_t>(config.joint_dim),
                                          0},
                                         "mass",
                                         "joint_space"};
  output->scratch[1] = TensorRequirement{"mass_inverse_jacobian_transpose",
                                         config.dtype,
                                         config.device,
                                         3,
                                         {static_cast<std::int64_t>(config.batch_size),
                                          static_cast<std::int64_t>(config.joint_dim),
                                          static_cast<std::int64_t>(config.task_dim),
                                          0},
                                         "inverse_mass_times_jacobian",
                                         "joint_to_task"};
  output->scratch[2] = TensorRequirement{"cartesian_inverse_mass_factor",
                                         config.dtype,
                                         config.device,
                                         3,
                                         {static_cast<std::int64_t>(config.batch_size),
                                          static_cast<std::int64_t>(config.task_dim),
                                          static_cast<std::int64_t>(config.task_dim),
                                          0},
                                         "inverse_operational_space_mass",
                                         cartesian_frame_name(config.cartesian_frame)};
  return Result{};
}

Result OperationalSpaceMass::prepare(const Config& config,
                                     const Parameters& parameters,
                                     const Storage& storage,
                                     const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(
      storage.cartesian_mass_matrix,
      config,
      config.dtype,
      3,
      {static_cast<std::int64_t>(config.batch_size),
       static_cast<std::int64_t>(config.task_dim),
       static_cast<std::int64_t>(config.task_dim),
       0},
      "cartesian_mass_matrix must be contiguous [batch_size,task_dim,task_dim]");
  if (!result.ok()) {
    return result;
  }
  if (storage.parameter_status.flags == nullptr ||
      storage.parameter_status.size < config.batch_size ||
      storage.parameter_status.device != config.device || storage.status.flags == nullptr ||
      storage.status.size < config.batch_size || storage.status.device != config.device) {
    return invalid_view("parameter_status and status must match the configured batch and device");
  }
  result =
      validate_contiguous_view(storage.scratch.joint_factor,
                               config,
                               config.dtype,
                               3,
                               {static_cast<std::int64_t>(config.batch_size),
                                static_cast<std::int64_t>(config.joint_dim),
                                static_cast<std::int64_t>(config.joint_dim),
                                0},
                               "joint_factor must be contiguous [batch_size,joint_dim,joint_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.scratch.solve_rhs,
                                    config,
                                    config.dtype,
                                    3,
                                    {static_cast<std::int64_t>(config.batch_size),
                                     static_cast<std::int64_t>(config.joint_dim),
                                     static_cast<std::int64_t>(config.task_dim),
                                     0},
                                    "solve_rhs must be contiguous [batch_size,joint_dim,task_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(
      storage.scratch.cartesian_inverse,
      config,
      config.dtype,
      3,
      {static_cast<std::int64_t>(config.batch_size),
       static_cast<std::int64_t>(config.task_dim),
       static_cast<std::int64_t>(config.task_dim),
       0},
      "cartesian_inverse must be contiguous [batch_size,task_dim,task_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result OperationalSpaceMass::apply_parameters(const Parameters& parameters,
                                              const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "regularization", parameters_.regularization);
  append_parameter(&entries, &count, "minimum_pivot", parameters_.minimum_pivot);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::apply_operational_space_mass_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result OperationalSpaceMass::reset(const ResetInputs& inputs,
                                   const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::reset_operational_space_mass(
      config_.batch_size, inputs.mask, storage_.status);
}

Result OperationalSpaceMass::step(const Inputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 2> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(
      &parameter_entries, &parameter_count, "regularization", parameters_.regularization);
  append_parameter(
      &parameter_entries, &parameter_count, "minimum_pivot", parameters_.minimum_pivot);
  std::array<ScratchEntryView, 3> scratch_entries{
      ScratchEntryView{"joint_mass_factor", storage_.scratch.joint_factor},
      ScratchEntryView{"mass_inverse_jacobian_transpose", storage_.scratch.solve_rhs},
      ScratchEntryView{"cartesian_inverse_mass_factor", storage_.scratch.cartesian_inverse}};
  ScratchView scratch{scratch_entries.data(), scratch_entries.size()};
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::step_operational_space_mass(
      inputs.joint_mass_matrix,
      inputs.jacobian,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      scratch,
      storage_.cartesian_mass_matrix,
      storage_.status);
}

Result CartesianImpedanceController::requirements(const Config& config,
                                                  Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "cartesian_wrench",
                  config.task_dim,
                  "newton_and_newton_meter",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 3;
  output->state[0] = TensorRequirement{"previous_mass_matrix",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size),
                                        static_cast<std::int64_t>(config.task_dim),
                                        static_cast<std::int64_t>(config.task_dim),
                                        0},
                                       "operational_space_mass_units",
                                       cartesian_frame_name(config.cartesian_frame)};
  output->state[1] = TensorRequirement{"previous_jacobian",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size),
                                        static_cast<std::int64_t>(config.task_dim),
                                        static_cast<std::int64_t>(config.joint_dim),
                                        0},
                                       "unitless",
                                       "task_to_joint"};
  output->state[2] = TensorRequirement{"initialized",
                                       DType::kBool,
                                       config.device,
                                       1,
                                       {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                       "unitless",
                                       "none"};
  return append_safety_requirements(config,
                                    config.task_dim,
                                    "newton_and_newton_meter",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result CartesianImpedanceController::prepare(const Config& config,
                                             const Parameters& parameters,
                                             const Storage& storage,
                                             const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, config.task_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_inertial_storage(config, storage.state);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianImpedanceController::apply_parameters(const Parameters& parameters,
                                                      const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_factors", parameters_.damping_factors);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::apply_cartesian_impedance_controller_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result CartesianImpedanceController::reset(const ResetInputs& inputs,
                                           const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<StateEntryView, 3> entries{};
  StateView state = inertial_state(storage_.state, &entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_cartesian_impedance_controller(
      config_.batch_size, config_.joint_dim, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result CartesianImpedanceController::step(const Inputs& inputs,
                                          const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 2> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(
      &parameter_entries, &parameter_count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "damping_factors", parameters_.damping_factors);
  std::array<StateEntryView, 3> state_entries{};
  StateView state = inertial_state(storage_.state, &state_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_cartesian_impedance_controller(
      inputs.cartesian_pose_error,
      inputs.cartesian_velocity_error,
      inputs.desired_acceleration,
      inputs.joint_velocity,
      inputs.cartesian_mass_matrix,
      inputs.jacobian,
      inputs.dt,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      state,
      storage_.command,
      storage_.status);
  return finish_step(result, config_, storage_);
}

Result BimanualCartesianImpedanceController::requirements(const Config& config,
                                                          Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (config.joint_dim != 14 || config.task_dim != 12) {
    return invalid_argument("bimanual impedance requires joint_dim == 14 and task_dim == 12");
  }
  if (config.cartesian_frame != CartesianFrame::kMainFrameOrientation) {
    return invalid_argument("bimanual impedance requires the main-frame-orientation contract");
  }
  return CartesianImpedanceController::requirements(config, output);
}

Result BimanualCartesianImpedanceController::prepare(const Config& config,
                                                     const Parameters& parameters,
                                                     const Storage& storage,
                                                     const ExecutionContext context) noexcept {
  if (config.joint_dim != 14 || config.task_dim != 12) {
    return invalid_argument("bimanual impedance requires joint_dim == 14 and task_dim == 12");
  }
  if (config.cartesian_frame != CartesianFrame::kMainFrameOrientation) {
    return invalid_argument("bimanual impedance requires the main-frame-orientation contract");
  }
  return controller_.prepare(config, parameters, storage, context);
}

Result BimanualCartesianImpedanceController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  return controller_.apply_parameters(parameters, context);
}

Result BimanualCartesianImpedanceController::reset(const ResetInputs& inputs,
                                                   const ExecutionContext context) noexcept {
  return controller_.reset(inputs, context);
}

Result BimanualCartesianImpedanceController::step(const Inputs& inputs,
                                                  const ExecutionContext context) noexcept {
  return controller_.step(inputs, context);
}

Result BimanualTaskKinematics::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_config(config, true);
  if (!result.ok()) {
    return result;
  }
  if (config.joint_dim != 14 || config.task_dim != 12) {
    return invalid_argument("bimanual task kinematics requires joint_dim == 14 and task_dim == 12");
  }
  if (config.cartesian_frame != CartesianFrame::kMainFrameOrientation) {
    return invalid_argument("bimanual task kinematics requires main-frame-orientation output");
  }
  result = detail::tensor::validate_bimanual_frame_config(config.frames);
  if (!result.ok()) {
    return result;
  }
  if (config.coordination_mode != BimanualCoordinationMode::kDecoupledCooperative &&
      config.coordination_mode != BimanualCoordinationMode::kCoupledRigidTransform) {
    return invalid_argument("unsupported bimanual coordination mode");
  }
  *output = Requirements{};
  output->batch_size = config.batch_size;
  output->joint_dim = config.joint_dim;
  output->outputs[0] = TensorRequirement{"task_pose_error",
                                         config.dtype,
                                         config.device,
                                         2,
                                         {static_cast<std::int64_t>(config.batch_size), 12, 0, 0},
                                         "meter_and_radian",
                                         "main_frame_orientation_at_task_origins"};
  output->outputs[1] = TensorRequirement{"task_jacobian",
                                         config.dtype,
                                         config.device,
                                         3,
                                         {static_cast<std::int64_t>(config.batch_size), 12, 14, 0},
                                         "task_velocity_per_joint_velocity",
                                         "main_frame_orientation_at_task_origins"};
  output->outputs[2] = TensorRequirement{"status",
                                         DType::kUInt64,
                                         config.device,
                                         1,
                                         {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                         "status_flags",
                                         "none"};
  output->output_count = 3;
  return Result{};
}

Result BimanualTaskKinematics::prepare(const Config& config, const Storage& storage) noexcept {
  prepared_ = false;
  Requirements ignored;
  Result result = requirements(config, &ignored);
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.task_pose_error,
                                    config,
                                    config.dtype,
                                    2,
                                    {static_cast<std::int64_t>(config.batch_size), 12, 0, 0},
                                    "task_pose_error must be contiguous [batch_size,12]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.task_jacobian,
                                    config,
                                    config.dtype,
                                    3,
                                    {static_cast<std::int64_t>(config.batch_size), 12, 14, 0},
                                    "task_jacobian must be contiguous [batch_size,12,14]");
  if (!result.ok()) {
    return result;
  }
  if (storage.status.flags == nullptr || storage.status.size < config.batch_size ||
      storage.status.device != config.device) {
    return invalid_view("status must match the configured bimanual batch and device");
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  storage_ = storage;
  prepared_ = true;
  return Result{};
}

Result BimanualTaskKinematics::reset(const ResetInputs& inputs,
                                     const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::reset_bimanual_task_kinematics(config_.batch_size,
                                                        inputs.mask,
                                                        storage_.task_pose_error,
                                                        storage_.task_jacobian,
                                                        storage_.status);
}

Result BimanualTaskKinematics::step(const Inputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::step_bimanual_task_kinematics(config_.coordination_mode,
                                                       config_.frames,
                                                       inputs.desired_absolute_pose,
                                                       inputs.arm1_tcp_pose,
                                                       inputs.arm2_tcp_pose,
                                                       inputs.desired_relative_pose,
                                                       inputs.arm1_jacobian,
                                                       inputs.arm2_jacobian,
                                                       storage_.task_pose_error,
                                                       storage_.task_jacobian,
                                                       storage_.status);
}

Result CartesianAdmittancePoseController::requirements(const Config& config,
                                                       Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "commanded_acceleration",
                  6,
                  "meter_per_second_squared_and_radian_per_second_squared",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 6;
  const char* names[] = {"admittance_pose",
                         "admittance_twist",
                         "pose_previous_signal",
                         "twist_previous_signal",
                         "pose_integrator_initialized",
                         "twist_integrator_initialized"};
  for (std::size_t index = 0; index < 4; ++index) {
    output->state[index] =
        TensorRequirement{names[index],
                          config.dtype,
                          config.device,
                          2,
                          {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
                          index == 0 ? "meter_and_radian" : "cartesian_rate",
                          cartesian_frame_name(config.cartesian_frame)};
  }
  for (std::size_t index = 4; index < 6; ++index) {
    output->state[index] =
        TensorRequirement{names[index],
                          DType::kBool,
                          config.device,
                          1,
                          {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                          "unitless",
                          "none"};
  }
  return append_safety_requirements(config,
                                    kCartesianDim,
                                    "meter_per_second_squared_and_radian_per_second_squared",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result CartesianAdmittancePoseController::prepare(const Config& config,
                                                  const Parameters& parameters,
                                                  const Storage& storage,
                                                  const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, 6);
  if (!result.ok()) {
    return result;
  }
  const MutableTensorView signals[] = {storage.state.pose_offset,
                                       storage.state.twist,
                                       storage.state.pose_previous_signal,
                                       storage.state.twist_previous_signal};
  for (const MutableTensorView& signal : signals) {
    result = validate_contiguous_view(signal,
                                      config,
                                      config.dtype,
                                      2,
                                      {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
                                      "admittance signal state must be contiguous [batch_size,6]");
    if (!result.ok()) {
      return result;
    }
  }
  const MutableTensorView initialized[] = {storage.state.pose_integrator_initialized,
                                           storage.state.twist_integrator_initialized};
  for (const MutableTensorView& mask : initialized) {
    result = validate_contiguous_view(mask,
                                      config,
                                      DType::kBool,
                                      1,
                                      {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                      "admittance initialized state must be bool [batch_size]");
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianAdmittancePoseController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 7> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_matrix", parameters_.damping_matrix);
  append_parameter(&entries, &count, "virtual_mass_matrix", parameters_.virtual_mass_matrix);
  append_parameter(&entries, &count, "force_filter_threshold", parameters_.force_filter_threshold);
  append_parameter(&entries, &count, "acceleration_limit", parameters_.acceleration_limit);
  append_parameter(&entries, &count, "twist_limit", parameters_.twist_limit);
  append_parameter(&entries, &count, "pose_limit", parameters_.pose_limit);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::apply_cartesian_admittance_controller_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result CartesianAdmittancePoseController::reset(const ResetInputs& inputs,
                                                const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<StateEntryView, 6> entries{};
  StateView state = admittance_state(storage_.state, &entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_cartesian_admittance_controller(
      config_.batch_size, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result CartesianAdmittancePoseController::step(const Inputs& inputs,
                                               const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 7> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(
      &parameter_entries, &parameter_count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "damping_matrix", parameters_.damping_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "virtual_mass_matrix", parameters_.virtual_mass_matrix);
  append_parameter(&parameter_entries,
                   &parameter_count,
                   "force_filter_threshold",
                   parameters_.force_filter_threshold);
  append_parameter(
      &parameter_entries, &parameter_count, "acceleration_limit", parameters_.acceleration_limit);
  append_parameter(&parameter_entries, &parameter_count, "twist_limit", parameters_.twist_limit);
  append_parameter(&parameter_entries, &parameter_count, "pose_limit", parameters_.pose_limit);
  std::array<StateEntryView, 6> state_entries{};
  StateView state = admittance_state(storage_.state, &state_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_cartesian_admittance_controller(
      inputs.external_wrench,
      inputs.feedforward_wrench,
      inputs.dt,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      state,
      storage_.command,
      storage_.status);
  return finish_step(result, config_, storage_);
}

Result OperationalSpaceController::requirements(const Config& config,
                                                Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(config, "joint_torque", config.joint_dim, "newton_meter", "joint_space", output);
  return append_safety_requirements(
      config, config.joint_dim, "newton_meter", "joint_space", output);
}

Result OperationalSpaceController::prepare(const Config& config,
                                           const Parameters& parameters,
                                           const Storage& storage,
                                           const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, config.joint_dim);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result OperationalSpaceController::apply_parameters(const Parameters& parameters,
                                                    const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "position_gain", parameters_.position_gain);
  append_parameter(&entries, &count, "velocity_gain", parameters_.velocity_gain);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::apply_operational_space_controller_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result OperationalSpaceController::reset(const ResetInputs& inputs,
                                         const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_operational_space_controller(
      config_.batch_size, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result OperationalSpaceController::step(const Inputs& inputs,
                                        const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "position_gain", parameters_.position_gain);
  append_parameter(&entries, &count, "velocity_gain", parameters_.velocity_gain);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::step_operational_space_controller(inputs.cartesian_pose_error,
                                                        inputs.cartesian_velocity_error,
                                                        inputs.desired_acceleration,
                                                        inputs.jacobian_dot_joint_velocity,
                                                        inputs.cartesian_mass_matrix,
                                                        inputs.jacobian,
                                                        inputs.joint_bias_torque,
                                                        parameter_set(entries, count),
                                                        parameter_status(storage_),
                                                        storage_.command,
                                                        storage_.status);
  return finish_step(result, config_, storage_);
}

Result OperationalSpaceInverseDynamicsController::requirements(const Config& config,
                                                               Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  return OperationalSpaceController::requirements(config, output);
}

Result OperationalSpaceInverseDynamicsController::prepare(const Config& config,
                                                          const Parameters& parameters,
                                                          const Storage& storage,
                                                          const ExecutionContext context) noexcept {
  return OperationalSpaceController::prepare(config, parameters, storage, context);
}

Result OperationalSpaceInverseDynamicsController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  return OperationalSpaceController::apply_parameters(parameters, context);
}

Result OperationalSpaceInverseDynamicsController::reset(const ResetInputs& inputs,
                                                        const ExecutionContext context) noexcept {
  return OperationalSpaceController::reset(inputs, context);
}

Result OperationalSpaceInverseDynamicsController::step(const Inputs& inputs,
                                                       const ExecutionContext context) noexcept {
  return OperationalSpaceController::step(inputs, context);
}

Result CartesianForceController::requirements(const Config& config, Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "commanded_wrench",
                  6,
                  "newton_and_newton_meter",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 5;
  const char* names[] = {"differentiator_previous_error",
                         "differentiator_initialized",
                         "integrator_previous_error",
                         "integrated_error",
                         "integrator_initialized"};
  for (std::size_t index = 0; index < 5; ++index) {
    const bool flag = index == 1 || index == 4;
    output->state[index] = TensorRequirement{
        names[index],
        flag ? DType::kBool : config.dtype,
        config.device,
        flag ? 1 : 2,
        flag
            ? std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       0,
                                                       0,
                                                       0}
            : std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       6,
                                                       0,
                                                       0},
        flag ? "unitless" : "wrench_error",
        flag ? "none" : cartesian_frame_name(config.cartesian_frame)};
  }
  return append_safety_requirements(config,
                                    kCartesianDim,
                                    "newton_and_newton_meter",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result CartesianForceController::prepare(const Config& config,
                                         const Parameters& parameters,
                                         const Storage& storage,
                                         const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, 6);
  if (!result.ok()) {
    return result;
  }
  result = validate_force_storage(config, storage.state);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianForceController::apply_parameters(const Parameters& parameters,
                                                  const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 6> entries{};
  std::size_t count = 0;
  append_force_parameters(&entries, &count, parameters_);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::apply_cartesian_force_controller_parameters(parameter_set(entries, count),
                                                                  config_.batch_size,
                                                                  config_.use_shaping_function,
                                                                  storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result CartesianForceController::reset(const ResetInputs& inputs,
                                       const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<StateEntryView, 5> entries{};
  StateView state = force_state(storage_.state, &entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_cartesian_force_controller(
      config_.batch_size, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result CartesianForceController::step(const Inputs& inputs,
                                      const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 6> parameter_entries{};
  std::size_t parameter_count = 0;
  append_force_parameters(&parameter_entries, &parameter_count, parameters_);
  std::array<StateEntryView, 5> state_entries{};
  StateView state = force_state(storage_.state, &state_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_cartesian_force_controller(
      inputs.external_wrench,
      inputs.desired_wrench,
      inputs.feedforward_wrench,
      inputs.pose_error,
      inputs.dt,
      config_.use_shaping_function,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      state,
      storage_.command,
      storage_.status);
  return finish_step(result, config_, storage_);
}

Result StaticJointImpedanceController::requirements(const Config& config,
                                                    Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) return invalid_argument("requirements output is null");
  Result result = validate_config(config, true);
  if (!result.ok()) return result;
  if (config.mode != ComponentMode::kDiagonal && config.mode != ComponentMode::kDense)
    return invalid_argument("static joint impedance mode must be diagonal or dense");
  describe_output(config, "joint_torque", config.joint_dim, "newton_meter", "joint_space", output);
  return append_safety_requirements(
      config, config.joint_dim, "newton_meter", "joint_space", output);
}

Result StaticJointImpedanceController::prepare(const Config& config,
                                               const Parameters& parameters,
                                               const Storage& storage,
                                               const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_config(config, true);
  if (!result.ok()) return result;
  if (config.mode != ComponentMode::kDiagonal && config.mode != ComponentMode::kDense)
    return invalid_argument("static joint impedance mode must be diagonal or dense");
  result = validate_storage(config, storage, config.joint_dim);
  if (!result.ok()) return result;
  result = validate_independent_storage(storage);
  if (!result.ok()) return result;
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result StaticJointImpedanceController::apply_parameters(const Parameters& parameters,
                                                        const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness", parameters.stiffness);
  append_parameter(&entries, &count, "damping", parameters.damping);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::apply_static_impedance_parameters(config_.mode,
                                                                    parameter_set(entries, count),
                                                                    config_.batch_size,
                                                                    config_.joint_dim,
                                                                    storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result StaticJointImpedanceController::reset(const ResetInputs& inputs,
                                             const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result =
      detail::tensor::reset_high_level_stateless(config_.batch_size, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result StaticJointImpedanceController::step(const Inputs& inputs,
                                            const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  const Result signals = validate_step_signals(
      config_, config_.joint_dim, {inputs.joint_position_error, inputs.joint_velocity_error});
  if (!signals.ok()) return signals;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness", parameters_.stiffness);
  append_parameter(&entries, &count, "damping", parameters_.damping);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::step_static_impedance(config_.mode,
                                                        inputs.joint_position_error,
                                                        inputs.joint_velocity_error,
                                                        parameter_set(entries, count),
                                                        parameter_status(storage_),
                                                        storage_.command,
                                                        storage_.status);
  return finish_step(result, config_, storage_);
}

Result StaticCartesianImpedanceController::requirements(const Config& config,
                                                        Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) return invalid_argument("requirements output is null");
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) return result;
  describe_output(config,
                  "cartesian_wrench",
                  6,
                  "newton_and_newton_meter",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  return append_safety_requirements(
      config, 6, "newton_and_newton_meter", cartesian_frame_name(config.cartesian_frame), output);
}

Result StaticCartesianImpedanceController::prepare(const Config& config,
                                                   const Parameters& parameters,
                                                   const Storage& storage,
                                                   const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) return result;
  result = validate_storage(config, storage, 6);
  if (!result.ok()) return result;
  result = validate_independent_storage(storage);
  if (!result.ok()) return result;
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result StaticCartesianImpedanceController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  parameters_ = parameters;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness", parameters.stiffness_matrix);
  append_parameter(&entries, &count, "damping", parameters.damping_matrix);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::apply_static_impedance_parameters(ComponentMode::kDense,
                                                                    parameter_set(entries, count),
                                                                    config_.batch_size,
                                                                    6,
                                                                    storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result StaticCartesianImpedanceController::reset(const ResetInputs& inputs,
                                                 const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result =
      detail::tensor::reset_high_level_stateless(config_.batch_size, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result StaticCartesianImpedanceController::step(const Inputs& inputs,
                                                const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  const Result signals = validate_step_signals(
      config_, 6, {inputs.cartesian_pose_error, inputs.cartesian_velocity_error});
  if (!signals.ok()) return signals;
  std::array<ParameterView, 2> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping", parameters_.damping_matrix);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::step_static_impedance(ComponentMode::kDense,
                                                        inputs.cartesian_pose_error,
                                                        inputs.cartesian_velocity_error,
                                                        parameter_set(entries, count),
                                                        parameter_status(storage_),
                                                        storage_.command,
                                                        storage_.status);
  return finish_step(result, config_, storage_);
}

Result CartesianUnifiedForceAdmittanceController::requirements(const Config& config,
                                                               Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) return invalid_argument("requirements output is null");
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) return result;
  describe_output(config,
                  "commanded_acceleration",
                  6,
                  "meter_per_second_squared_and_radian_per_second_squared",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 11;
  const char* names[] = {"differentiator_previous_error",
                         "differentiator_initialized",
                         "integrator_previous_error",
                         "integrated_error",
                         "integrator_initialized",
                         "admittance_pose",
                         "admittance_twist",
                         "pose_previous_signal",
                         "twist_previous_signal",
                         "pose_integrator_initialized",
                         "twist_integrator_initialized"};
  for (std::size_t index = 0; index < 11; ++index) {
    const bool flag = index == 1 || index == 4 || index == 9 || index == 10;
    output->state[index] = TensorRequirement{
        names[index],
        flag ? DType::kBool : config.dtype,
        config.device,
        flag ? 1 : 2,
        flag
            ? std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       0,
                                                       0,
                                                       0}
            : std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       6,
                                                       0,
                                                       0},
        flag ? "unitless" : "cartesian_state",
        flag ? "none" : cartesian_frame_name(config.cartesian_frame)};
  }
  return append_safety_requirements(config,
                                    6,
                                    "meter_per_second_squared_and_radian_per_second_squared",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result CartesianUnifiedForceAdmittanceController::prepare(const Config& config,
                                                          const Parameters& parameters,
                                                          const Storage& storage,
                                                          const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, false);
  if (!result.ok()) return result;
  result = validate_storage(config, storage, 6);
  if (!result.ok()) return result;
  result = validate_force_storage(config, storage.state.force);
  if (!result.ok()) return result;
  result = validate_admittance_storage(config, storage.state.admittance);
  if (!result.ok()) return result;
  result = validate_independent_storage(storage);
  if (!result.ok()) return result;
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceAdmittanceController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  parameters_ = parameters;
  std::array<ParameterView, 13> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters.stiffness_matrix);
  append_parameter(&entries, &count, "damping_matrix", parameters.damping_matrix);
  append_parameter(&entries, &count, "virtual_mass_matrix", parameters.virtual_mass_matrix);
  append_parameter(&entries, &count, "acceleration_limit", parameters.acceleration_limit);
  append_parameter(&entries, &count, "twist_limit", parameters.twist_limit);
  append_parameter(&entries, &count, "pose_limit", parameters.pose_limit);
  append_parameter(&entries, &count, "proportional_gain", parameters.force.proportional_gain);
  append_parameter(&entries, &count, "differentiator_gain", parameters.force.differentiator_gain);
  append_parameter(&entries, &count, "integral_gain", parameters.force.integral_gain);
  append_parameter(&entries, &count, "initial_integral", parameters.force.initial_integral);
  append_parameter(&entries, &count, "integral_lower_limit", parameters.force.integral_lower_limit);
  append_parameter(&entries, &count, "integral_upper_limit", parameters.force.integral_upper_limit);
  append_parameter(&entries, &count, "error_deadzone", parameters.force.error_deadzone);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::apply_unified_force_admittance_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceAdmittanceController::reset(const ResetInputs& inputs,
                                                        const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  std::array<StateEntryView, 11> entries{};
  StateView state = unified_force_admittance_state(storage_.state, &entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_unified_force_admittance(
      config_.batch_size, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result CartesianUnifiedForceAdmittanceController::step(const Inputs& inputs,
                                                       const ExecutionContext context) noexcept {
  if (!prepared_) return invalid_argument("controller is not prepared");
  const Result signals = validate_step_signals(
      config_, 6, {inputs.desired_wrench, inputs.external_wrench, inputs.feedforward_wrench});
  if (!signals.ok()) return signals;
  const Result dt_metadata = detail::tensor::internal::validate_tensor_metadata(inputs.dt);
  if (!dt_metadata.ok()) return dt_metadata;
  if (inputs.dt.device != config_.device || inputs.dt.dtype != config_.dtype ||
      !(inputs.dt.rank == 0 ||
        (inputs.dt.rank == 1 &&
         inputs.dt.shape[0] >= static_cast<std::int64_t>(config_.batch_size) &&
         inputs.dt.strides[0] >= 1))) {
    return invalid_view("dt must be a configured scalar or batch array");
  }
  std::array<ParameterView, 13> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_matrix", parameters_.damping_matrix);
  append_parameter(&entries, &count, "virtual_mass_matrix", parameters_.virtual_mass_matrix);
  append_parameter(&entries, &count, "acceleration_limit", parameters_.acceleration_limit);
  append_parameter(&entries, &count, "twist_limit", parameters_.twist_limit);
  append_parameter(&entries, &count, "pose_limit", parameters_.pose_limit);
  append_parameter(&entries, &count, "proportional_gain", parameters_.force.proportional_gain);
  append_parameter(&entries, &count, "differentiator_gain", parameters_.force.differentiator_gain);
  append_parameter(&entries, &count, "integral_gain", parameters_.force.integral_gain);
  append_parameter(&entries, &count, "initial_integral", parameters_.force.initial_integral);
  append_parameter(
      &entries, &count, "integral_lower_limit", parameters_.force.integral_lower_limit);
  append_parameter(
      &entries, &count, "integral_upper_limit", parameters_.force.integral_upper_limit);
  append_parameter(&entries, &count, "error_deadzone", parameters_.force.error_deadzone);
  std::array<StateEntryView, 11> state_entries{};
  StateView state = unified_force_admittance_state(storage_.state, &state_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::step_unified_force_admittance(inputs.desired_wrench,
                                                                inputs.external_wrench,
                                                                inputs.feedforward_wrench,
                                                                inputs.dt,
                                                                parameter_set(entries, count),
                                                                parameter_status(storage_),
                                                                state,
                                                                storage_.command,
                                                                storage_.status);
  return finish_step(result, config_, storage_);
}

Result CartesianUnifiedForceImpedanceController::requirements(const Config& config,
                                                              Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "cartesian_wrench",
                  6,
                  "newton_and_newton_meter",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 8;
  output->scratch_count = 4;
  output->state[0] = TensorRequirement{"previous_mass_matrix",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size), 6, 6, 0},
                                       "operational_space_mass_units",
                                       cartesian_frame_name(config.cartesian_frame)};
  output->state[1] = TensorRequirement{"previous_jacobian",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size),
                                        6,
                                        static_cast<std::int64_t>(config.joint_dim),
                                        0},
                                       "unitless",
                                       "task_to_joint"};
  output->state[2] = TensorRequirement{"initialized",
                                       DType::kBool,
                                       config.device,
                                       1,
                                       {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                       "unitless",
                                       "none"};
  const char* force_names[] = {"differentiator_previous_error",
                               "differentiator_initialized",
                               "integrator_previous_error",
                               "integrated_error",
                               "integrator_initialized"};
  for (std::size_t index = 0; index < 5; ++index) {
    const bool flag = index == 1 || index == 4;
    output->state[index + 3] = TensorRequirement{
        force_names[index],
        flag ? DType::kBool : config.dtype,
        config.device,
        flag ? 1 : 2,
        flag
            ? std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       0,
                                                       0,
                                                       0}
            : std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       6,
                                                       0,
                                                       0},
        flag ? "unitless" : "wrench_error",
        flag ? "none" : cartesian_frame_name(config.cartesian_frame)};
  }
  const char* scratch_names[] = {
      "impedance_wrench", "force_wrench", "impedance_status", "force_status"};
  for (std::size_t index = 0; index < 4; ++index) {
    const bool status = index >= 2;
    output->scratch[index] = TensorRequirement{
        scratch_names[index],
        status ? DType::kUInt64 : config.dtype,
        config.device,
        status ? 1 : 2,
        status
            ? std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       0,
                                                       0,
                                                       0}
            : std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       6,
                                                       0,
                                                       0},
        status ? "status_flags" : "newton_and_newton_meter",
        status ? "none" : cartesian_frame_name(config.cartesian_frame)};
  }
  return append_safety_requirements(config,
                                    kCartesianDim,
                                    "newton_and_newton_meter",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result CartesianUnifiedForceImpedanceController::prepare(const Config& config,
                                                         const Parameters& parameters,
                                                         const Storage& storage,
                                                         const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, 6);
  if (!result.ok()) {
    return result;
  }
  result = validate_inertial_storage(config, storage.state.impedance);
  if (!result.ok()) {
    return result;
  }
  result = validate_force_storage(config, storage.state.force);
  if (!result.ok()) {
    return result;
  }
  const MutableTensorView wrench_scratch[] = {storage.scratch.impedance_wrench,
                                              storage.scratch.force_wrench};
  for (const MutableTensorView& scratch : wrench_scratch) {
    result = validate_contiguous_view(
        scratch,
        config,
        config.dtype,
        2,
        {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
        "unified-controller wrench scratch must be contiguous [batch_size,6]");
    if (!result.ok()) {
      return result;
    }
  }
  const MutableTensorView status_scratch[] = {storage.scratch.impedance_status,
                                              storage.scratch.force_status};
  for (const MutableTensorView& scratch : status_scratch) {
    result =
        validate_contiguous_view(scratch,
                                 config,
                                 DType::kUInt64,
                                 1,
                                 {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                 "unified-controller status scratch must be uint64 [batch_size]");
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceImpedanceController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 8> entries{};
  std::size_t count = 0;
  append_parameter(&entries, &count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(&entries, &count, "damping_factors", parameters_.damping_factors);
  append_force_parameters(&entries, &count, parameters_.force);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result =
      detail::tensor::apply_cartesian_unified_force_impedance_controller_parameters(
          parameter_set(entries, count),
          config_.batch_size,
          config_.use_shaping_function,
          storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceImpedanceController::reset(const ResetInputs& inputs,
                                                       const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<StateEntryView, 8> entries{};
  StateView state = unified_state(storage_.state, &entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_cartesian_unified_force_impedance_controller(
      config_.batch_size, config_.joint_dim, state, inputs.mask, storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result CartesianUnifiedForceImpedanceController::step(const Inputs& inputs,
                                                      const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 8> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(
      &parameter_entries, &parameter_count, "stiffness_matrix", parameters_.stiffness_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "damping_factors", parameters_.damping_factors);
  append_force_parameters(&parameter_entries, &parameter_count, parameters_.force);
  std::array<StateEntryView, 8> state_entries{};
  StateView state = unified_state(storage_.state, &state_entries);
  std::array<ScratchEntryView, 4> scratch_entries{};
  ScratchView scratch = unified_scratch(storage_.scratch, &scratch_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_cartesian_unified_force_impedance_controller(
      inputs.cartesian_pose_error,
      inputs.cartesian_velocity_error,
      inputs.desired_acceleration,
      inputs.joint_velocity,
      inputs.cartesian_mass_matrix,
      inputs.jacobian,
      inputs.external_wrench,
      inputs.desired_wrench,
      inputs.feedforward_wrench,
      inputs.dt,
      config_.use_shaping_function,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      state,
      scratch,
      storage_.command,
      storage_.status);
  return finish_step(result, config_, storage_);
}

Result VariableCartesianImpedanceController::requirements(const Config& config,
                                                          Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  describe_output(config,
                  "cartesian_wrench",
                  6,
                  "newton_and_newton_meter",
                  cartesian_frame_name(config.cartesian_frame),
                  output);
  output->state_count = 6;
  output->scratch_count = 6;
  output->state[0] = TensorRequirement{"current_stiffness_matrix",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size), 6, 6, 0},
                                       "newton_per_meter_and_newton_meter_per_radian",
                                       cartesian_frame_name(config.cartesian_frame)};
  output->state[1] = output->state[0];
  output->state[1].name = "previous_stiffness_matrix";
  output->state[2] = TensorRequirement{"stiffness_initialized",
                                       DType::kBool,
                                       config.device,
                                       1,
                                       {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                       "unitless",
                                       "none"};
  output->state[3] = TensorRequirement{"previous_mass_matrix",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size), 6, 6, 0},
                                       "operational_space_mass_units",
                                       cartesian_frame_name(config.cartesian_frame)};
  output->state[4] = TensorRequirement{"previous_jacobian",
                                       config.dtype,
                                       config.device,
                                       3,
                                       {static_cast<std::int64_t>(config.batch_size),
                                        6,
                                        static_cast<std::int64_t>(config.joint_dim),
                                        0},
                                       "unitless",
                                       "task_to_joint"};
  output->state[5] = TensorRequirement{"inertial_initialized",
                                       DType::kBool,
                                       config.device,
                                       1,
                                       {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                       "unitless",
                                       "none"};
  const char* scratch_names[] = {"stiffness_wrench",
                                 "damping_wrench",
                                 "inertial_wrench",
                                 "stiffness_status",
                                 "damping_status",
                                 "inertial_status"};
  for (std::size_t index = 0; index < 6; ++index) {
    const bool status = index >= 3;
    output->scratch[index] = TensorRequirement{
        scratch_names[index],
        status ? DType::kUInt64 : config.dtype,
        config.device,
        status ? 1 : 2,
        status
            ? std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       0,
                                                       0,
                                                       0}
            : std::array<std::int64_t, kMaxTensorRank>{static_cast<std::int64_t>(config.batch_size),
                                                       6,
                                                       0,
                                                       0},
        status ? "status_flags" : "newton_and_newton_meter",
        status ? "none" : cartesian_frame_name(config.cartesian_frame)};
  }
  return append_safety_requirements(config,
                                    kCartesianDim,
                                    "newton_and_newton_meter",
                                    cartesian_frame_name(config.cartesian_frame),
                                    output);
}

Result VariableCartesianImpedanceController::prepare(const Config& config,
                                                     const Parameters& parameters,
                                                     const Storage& storage,
                                                     const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true);
  if (!result.ok()) {
    return result;
  }
  result = validate_storage(config, storage, 6);
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.state.current_stiffness_matrix,
                                    config,
                                    config.dtype,
                                    3,
                                    {static_cast<std::int64_t>(config.batch_size), 6, 6, 0},
                                    "current_stiffness_matrix must be contiguous [batch_size,6,6]");
  if (!result.ok()) {
    return result;
  }
  result =
      validate_contiguous_view(storage.state.previous_stiffness_matrix,
                               config,
                               config.dtype,
                               3,
                               {static_cast<std::int64_t>(config.batch_size), 6, 6, 0},
                               "previous_stiffness_matrix must be contiguous [batch_size,6,6]");
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.state.stiffness_initialized,
                                    config,
                                    DType::kBool,
                                    1,
                                    {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                    "stiffness_initialized must be bool [batch_size]");
  if (!result.ok()) {
    return result;
  }
  result = validate_inertial_storage(config, storage.state.inertial);
  if (!result.ok()) {
    return result;
  }
  const MutableTensorView wrench_scratch[] = {storage.scratch.stiffness_wrench,
                                              storage.scratch.damping_wrench,
                                              storage.scratch.inertial_wrench};
  for (const MutableTensorView& scratch : wrench_scratch) {
    result = validate_contiguous_view(
        scratch,
        config,
        config.dtype,
        2,
        {static_cast<std::int64_t>(config.batch_size), 6, 0, 0},
        "variable-impedance wrench scratch must be contiguous [batch_size,6]");
    if (!result.ok()) {
      return result;
    }
  }
  const MutableTensorView status_scratch[] = {storage.scratch.stiffness_status,
                                              storage.scratch.damping_status,
                                              storage.scratch.inertial_status};
  for (const MutableTensorView& scratch : status_scratch) {
    result =
        validate_contiguous_view(scratch,
                                 config,
                                 DType::kUInt64,
                                 1,
                                 {static_cast<std::int64_t>(config.batch_size), 0, 0, 0},
                                 "variable-impedance status scratch must be uint64 [batch_size]");
    if (!result.ok()) {
      return result;
    }
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = true;
  result = apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result VariableCartesianImpedanceController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  parameters_ = parameters;
  std::array<ParameterView, 5> entries{};
  std::size_t count = 0;
  append_parameter(
      &entries, &count, "initial_stiffness_matrix", parameters_.initial_stiffness_matrix);
  append_parameter(&entries, &count, "maximum_stiffness", parameters_.maximum_stiffness);
  append_parameter(&entries, &count, "minimum_stiffness", parameters_.minimum_stiffness);
  append_parameter(&entries, &count, "maximum_stiffness_rate", parameters_.maximum_stiffness_rate);
  append_parameter(&entries, &count, "damping_factors", parameters_.damping_factors);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::apply_variable_cartesian_impedance_controller_parameters(
      parameter_set(entries, count), config_.batch_size, storage_.parameter_status);
  prepared_ = result.ok();
  return result;
}

Result VariableCartesianImpedanceController::reset(const ResetInputs& inputs,
                                                   const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 5> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(&parameter_entries,
                   &parameter_count,
                   "initial_stiffness_matrix",
                   parameters_.initial_stiffness_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "maximum_stiffness", parameters_.maximum_stiffness);
  append_parameter(
      &parameter_entries, &parameter_count, "minimum_stiffness", parameters_.minimum_stiffness);
  append_parameter(&parameter_entries,
                   &parameter_count,
                   "maximum_stiffness_rate",
                   parameters_.maximum_stiffness_rate);
  append_parameter(
      &parameter_entries, &parameter_count, "damping_factors", parameters_.damping_factors);
  std::array<StateEntryView, 6> state_entries{};
  StateView state = variable_impedance_state(storage_.state, &state_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = detail::tensor::reset_variable_cartesian_impedance_controller(
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      config_.batch_size,
      config_.joint_dim,
      state,
      inputs.mask,
      storage_.status);
  return result.ok() ? reset_safety(config_, storage_, inputs) : result;
}

Result VariableCartesianImpedanceController::step(const Inputs& inputs,
                                                  const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  std::array<ParameterView, 5> parameter_entries{};
  std::size_t parameter_count = 0;
  append_parameter(&parameter_entries,
                   &parameter_count,
                   "initial_stiffness_matrix",
                   parameters_.initial_stiffness_matrix);
  append_parameter(
      &parameter_entries, &parameter_count, "maximum_stiffness", parameters_.maximum_stiffness);
  append_parameter(
      &parameter_entries, &parameter_count, "minimum_stiffness", parameters_.minimum_stiffness);
  append_parameter(&parameter_entries,
                   &parameter_count,
                   "maximum_stiffness_rate",
                   parameters_.maximum_stiffness_rate);
  append_parameter(
      &parameter_entries, &parameter_count, "damping_factors", parameters_.damping_factors);
  std::array<StateEntryView, 6> state_entries{};
  StateView state = variable_impedance_state(storage_.state, &state_entries);
  std::array<ScratchEntryView, 6> scratch_entries{};
  ScratchView scratch = variable_impedance_scratch(storage_.scratch, &scratch_entries);
  detail::tensor::ScopedTensorExecutionContext scope(context);
  const Result result = detail::tensor::step_variable_cartesian_impedance_controller(
      inputs.cartesian_pose_error,
      inputs.requested_stiffness_matrix,
      inputs.cartesian_velocity_error,
      inputs.desired_acceleration,
      inputs.joint_velocity,
      inputs.cartesian_mass_matrix,
      inputs.jacobian,
      inputs.dt,
      parameter_set(parameter_entries, parameter_count),
      parameter_status(storage_),
      state,
      scratch,
      storage_.command,
      storage_.status);
  return finish_step(result, config_, storage_);
}

Result TaskWrenchToJointTorqueController::component_requirements(
    const Config& config, ComponentRequirements* output) noexcept {
  if (output != nullptr) *output = ComponentRequirements{};
  if (!output) return invalid_argument("component requirements output is null");
  Requirements flat;
  const auto result = requirements(config, &flat);
  if (!result.ok()) return result;
  output->unfiltered_joint_torque = flat.scratch[0];
  flat.scratch = {};
  flat.scratch_count = 0;
  output->filter = flat;
  return {};
}

Result TaskWrenchToJointTorqueController::requirements(const Config& config,
                                                       Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  if (output == nullptr) {
    return invalid_argument("requirements output is null");
  }
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  result = TorqueFilter::requirements(config, output);
  if (!result.ok()) {
    return result;
  }
  output->scratch[0] = TensorRequirement{"unfiltered_joint_torque",
                                         config.dtype,
                                         config.device,
                                         2,
                                         {static_cast<std::int64_t>(config.batch_size),
                                          static_cast<std::int64_t>(config.joint_dim),
                                          0,
                                          0},
                                         "newton_meter",
                                         "joint_space"};
  output->scratch_count = 1;
  return Result{};
}

Result TaskWrenchToJointTorqueController::prepare(const Config& config,
                                                  const Parameters& parameters,
                                                  const Storage& storage,
                                                  const ExecutionContext context) noexcept {
  prepared_ = false;
  Result result = validate_cartesian_config(config, true, true);
  if (!result.ok()) {
    return result;
  }
  result =
      validate_contiguous_view(storage.unfiltered_joint_torque,
                               config,
                               config.dtype,
                               2,
                               {static_cast<std::int64_t>(config.batch_size),
                                static_cast<std::int64_t>(config.joint_dim),
                                0,
                                0},
                               "unfiltered_joint_torque must be contiguous [batch_size,joint_dim]");
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  config_ = config;
  storage_ = storage;
  result = filter_.prepare(config, parameters, storage.filter, context);
  prepared_ = result.ok();
  return result;
}

Result TaskWrenchToJointTorqueController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  const Result result = filter_.apply_parameters(parameters, context);
  prepared_ = result.ok();
  return result;
}

Result TaskWrenchToJointTorqueController::reset(const ResetInputs& inputs,
                                                const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  return filter_.reset(inputs, context);
}

Result TaskWrenchToJointTorqueController::step(const Inputs& inputs,
                                               const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result =
      detail::tensor::step_task_wrench_to_joint_torque(inputs.task_wrench,
                                                       inputs.task_jacobian,
                                                       inputs.joint_bias_torque,
                                                       inputs.joint_feedback_torque,
                                                       inputs.task_status,
                                                       inputs.feedback_status,
                                                       storage_.unfiltered_joint_torque);
  if (!result.ok()) {
    return result;
  }
  return filter_.step(TorqueFilter::Inputs{storage_.unfiltered_joint_torque.as_const(), inputs.dt},
                      context);
}

Result CartesianImpedanceTorqueController::component_requirements(
    const Config& config, ComponentRequirements* output) noexcept {
  if (output != nullptr) *output = ComponentRequirements{};
  if (!output) return invalid_argument("component requirements output is null");
  ComponentRequirements complete;
  Result result;
  result = OperationalSpaceMass::requirements(config, &complete.mass);
  if (!result.ok()) return result;
  result = CartesianImpedanceController::requirements(config, &complete.task);
  if (!result.ok()) return result;
  result = NullspaceController::requirements(config, &complete.nullspace);
  if (!result.ok()) return result;
  result = TaskWrenchToJointTorqueController::component_requirements(config, &complete.output);
  if (!result.ok()) return result;
  *output = complete;
  return {};
}

Result CartesianImpedanceTorqueController::requirements(const Config& config,
                                                        Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  return describe_torque_requirements<CartesianImpedanceController>(config, output);
}

Result CartesianImpedanceTorqueController::prepare(const Config& config,
                                                   const Parameters& parameters,
                                                   const Storage& storage,
                                                   const ExecutionContext context) noexcept {
  prepared_ = false;
  Requirements ignored;
  Result result = requirements(config, &ignored);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  result = mass_.prepare(config, parameters.mass, storage.mass, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.prepare(config, parameters.task, storage.task, context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.prepare(config, parameters.nullspace, storage.nullspace, context);
  if (!result.ok()) {
    return result;
  }
  result = output_.prepare(config, parameters.output, storage.output, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianImpedanceTorqueController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.apply_parameters(parameters.mass, context);
  result = first_failure(result, task_.apply_parameters(parameters.task, context));
  result = first_failure(result, nullspace_.apply_parameters(parameters.nullspace, context));
  result = first_failure(result, output_.apply_parameters(parameters.output, context));
  prepared_ = result.ok();
  return result;
}

Result CartesianImpedanceTorqueController::reset(const ResetInputs& inputs,
                                                 const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.reset(inputs, context);
  result = first_failure(result, task_.reset(inputs, context));
  result = first_failure(result, nullspace_.reset(inputs, context));
  return first_failure(result, output_.reset(inputs, context));
}

Result CartesianImpedanceTorqueController::step(const Inputs& inputs,
                                                const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.step({inputs.joint_mass_matrix, inputs.jacobian}, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.step({inputs.cartesian_pose_error,
                       inputs.cartesian_velocity_error,
                       inputs.desired_acceleration,
                       inputs.joint_velocity,
                       mass_.outputs().cartesian_mass_matrix.as_const(),
                       inputs.jacobian,
                       inputs.dt},
                      context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.step({inputs.joint_position_error,
                            inputs.joint_velocity,
                            inputs.joint_mass_matrix,
                            inputs.jacobian},
                           context);
  if (!result.ok()) {
    return result;
  }
  const auto task_outputs = task_.outputs();
  const auto feedback_outputs = nullspace_.outputs();
  return output_.step({task_outputs.cartesian_wrench.as_const(),
                       inputs.jacobian,
                       inputs.joint_bias_torque,
                       feedback_outputs.joint_torque.as_const(),
                       read_status(task_outputs.status),
                       read_status(feedback_outputs.status),
                       inputs.dt},
                      context);
}

Result VariableCartesianImpedanceTorqueController::component_requirements(
    const Config& config, ComponentRequirements* output) noexcept {
  if (output != nullptr) *output = ComponentRequirements{};
  if (!output) return invalid_argument("component requirements output is null");
  ComponentRequirements complete;
  Result result;
  result = OperationalSpaceMass::requirements(config, &complete.mass);
  if (!result.ok()) return result;
  result = VariableCartesianImpedanceController::requirements(config, &complete.task);
  if (!result.ok()) return result;
  result = NullspaceController::requirements(config, &complete.nullspace);
  if (!result.ok()) return result;
  result = TaskWrenchToJointTorqueController::component_requirements(config, &complete.output);
  if (!result.ok()) return result;
  *output = complete;
  return {};
}

Result VariableCartesianImpedanceTorqueController::requirements(const Config& config,
                                                                Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  return describe_torque_requirements<VariableCartesianImpedanceController>(config, output);
}

Result VariableCartesianImpedanceTorqueController::prepare(
    const Config& config,
    const Parameters& parameters,
    const Storage& storage,
    const ExecutionContext context) noexcept {
  prepared_ = false;
  Requirements ignored;
  Result result = requirements(config, &ignored);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  result = mass_.prepare(config, parameters.mass, storage.mass, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.prepare(config, parameters.task, storage.task, context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.prepare(config, parameters.nullspace, storage.nullspace, context);
  if (!result.ok()) {
    return result;
  }
  result = output_.prepare(config, parameters.output, storage.output, context);
  prepared_ = result.ok();
  return result;
}

Result VariableCartesianImpedanceTorqueController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.apply_parameters(parameters.mass, context);
  result = first_failure(result, task_.apply_parameters(parameters.task, context));
  result = first_failure(result, nullspace_.apply_parameters(parameters.nullspace, context));
  result = first_failure(result, output_.apply_parameters(parameters.output, context));
  prepared_ = result.ok();
  return result;
}

Result VariableCartesianImpedanceTorqueController::reset(const ResetInputs& inputs,
                                                         const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.reset(inputs, context);
  result = first_failure(result, task_.reset(inputs, context));
  result = first_failure(result, nullspace_.reset(inputs, context));
  return first_failure(result, output_.reset(inputs, context));
}

Result VariableCartesianImpedanceTorqueController::step(const Inputs& inputs,
                                                        const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.step({inputs.joint_mass_matrix, inputs.jacobian}, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.step({inputs.cartesian_pose_error,
                       inputs.requested_stiffness_matrix,
                       inputs.cartesian_velocity_error,
                       inputs.desired_acceleration,
                       inputs.joint_velocity,
                       mass_.outputs().cartesian_mass_matrix.as_const(),
                       inputs.jacobian,
                       inputs.dt},
                      context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.step({inputs.joint_position_error,
                            inputs.joint_velocity,
                            inputs.joint_mass_matrix,
                            inputs.jacobian},
                           context);
  if (!result.ok()) {
    return result;
  }
  const auto task_outputs = task_.outputs();
  const auto feedback_outputs = nullspace_.outputs();
  return output_.step({task_outputs.cartesian_wrench.as_const(),
                       inputs.jacobian,
                       inputs.joint_bias_torque,
                       feedback_outputs.joint_torque.as_const(),
                       read_status(task_outputs.status),
                       read_status(feedback_outputs.status),
                       inputs.dt},
                      context);
}

Result CartesianUnifiedForceImpedanceTorqueController::component_requirements(
    const Config& config, ComponentRequirements* output) noexcept {
  if (output != nullptr) *output = ComponentRequirements{};
  if (!output) return invalid_argument("component requirements output is null");
  ComponentRequirements complete;
  Result result;
  result = OperationalSpaceMass::requirements(config, &complete.mass);
  if (!result.ok()) return result;
  result = CartesianUnifiedForceImpedanceController::requirements(config, &complete.task);
  if (!result.ok()) return result;
  result = NullspaceController::requirements(config, &complete.nullspace);
  if (!result.ok()) return result;
  result = TaskWrenchToJointTorqueController::component_requirements(config, &complete.output);
  if (!result.ok()) return result;
  *output = complete;
  return {};
}

Result CartesianUnifiedForceImpedanceTorqueController::requirements(const Config& config,
                                                                    Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  return describe_torque_requirements<CartesianUnifiedForceImpedanceController>(config, output);
}

Result CartesianUnifiedForceImpedanceTorqueController::prepare(
    const Config& config,
    const Parameters& parameters,
    const Storage& storage,
    const ExecutionContext context) noexcept {
  prepared_ = false;
  Requirements ignored;
  Result result = requirements(config, &ignored);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  result = mass_.prepare(config, parameters.mass, storage.mass, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.prepare(config, parameters.task, storage.task, context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.prepare(config, parameters.nullspace, storage.nullspace, context);
  if (!result.ok()) {
    return result;
  }
  result = output_.prepare(config, parameters.output, storage.output, context);
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceImpedanceTorqueController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.apply_parameters(parameters.mass, context);
  result = first_failure(result, task_.apply_parameters(parameters.task, context));
  result = first_failure(result, nullspace_.apply_parameters(parameters.nullspace, context));
  result = first_failure(result, output_.apply_parameters(parameters.output, context));
  prepared_ = result.ok();
  return result;
}

Result CartesianUnifiedForceImpedanceTorqueController::reset(
    const ResetInputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.reset(inputs, context);
  result = first_failure(result, task_.reset(inputs, context));
  result = first_failure(result, nullspace_.reset(inputs, context));
  return first_failure(result, output_.reset(inputs, context));
}

Result CartesianUnifiedForceImpedanceTorqueController::step(
    const Inputs& inputs, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.step({inputs.joint_mass_matrix, inputs.jacobian}, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.step({inputs.cartesian_pose_error,
                       inputs.cartesian_velocity_error,
                       inputs.desired_acceleration,
                       inputs.joint_velocity,
                       mass_.outputs().cartesian_mass_matrix.as_const(),
                       inputs.jacobian,
                       inputs.external_wrench,
                       inputs.desired_wrench,
                       inputs.feedforward_wrench,
                       inputs.dt},
                      context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.step({inputs.joint_position_error,
                            inputs.joint_velocity,
                            inputs.joint_mass_matrix,
                            inputs.jacobian},
                           context);
  if (!result.ok()) {
    return result;
  }
  const auto task_outputs = task_.outputs();
  const auto feedback_outputs = nullspace_.outputs();
  return output_.step({task_outputs.cartesian_wrench.as_const(),
                       inputs.jacobian,
                       inputs.joint_bias_torque,
                       feedback_outputs.joint_torque.as_const(),
                       read_status(task_outputs.status),
                       read_status(feedback_outputs.status),
                       inputs.dt},
                      context);
}

Result BimanualCartesianImpedanceTorqueController::component_requirements(
    const Config& config, ComponentRequirements* output) noexcept {
  if (output != nullptr) *output = ComponentRequirements{};
  if (!output) return invalid_argument("component requirements output is null");
  ComponentRequirements complete;
  Result result;
  result = BimanualTaskKinematics::requirements(config, &complete.kinematics);
  if (!result.ok()) return result;
  result = OperationalSpaceMass::requirements(config, &complete.mass);
  if (!result.ok()) return result;
  result = BimanualCartesianImpedanceController::requirements(config, &complete.task);
  if (!result.ok()) return result;
  result = NullspaceController::requirements(config, &complete.nullspace);
  if (!result.ok()) return result;
  result = TaskWrenchToJointTorqueController::component_requirements(config, &complete.output);
  if (!result.ok()) return result;
  complete.velocity.batch_size = config.batch_size;
  complete.velocity.joint_dim = config.joint_dim;
  complete.velocity.scratch[0] = bimanual_velocity_requirement(config);
  complete.velocity.scratch_count = 1;
  *output = complete;
  return {};
}

Result BimanualCartesianImpedanceTorqueController::requirements(const Config& config,
                                                                Requirements* output) noexcept {
  if (output != nullptr) *output = Requirements{};
  return describe_torque_requirements<BimanualCartesianImpedanceController>(config, output);
}

Result BimanualCartesianImpedanceTorqueController::prepare(
    const Config& config,
    const Parameters& parameters,
    const Storage& storage,
    const ExecutionContext context) noexcept {
  prepared_ = false;
  Requirements ignored;
  Result result = requirements(config, &ignored);
  if (!result.ok()) {
    return result;
  }
  result = validate_independent_storage(storage);
  if (!result.ok()) {
    return result;
  }
  result = kinematics_.prepare(config, storage.kinematics);
  if (!result.ok()) {
    return result;
  }
  result = mass_.prepare(config, parameters.mass, storage.mass, context);
  if (!result.ok()) {
    return result;
  }
  result = task_.prepare(config, parameters.task, storage.task, context);
  if (!result.ok()) {
    return result;
  }
  result = nullspace_.prepare(config, parameters.nullspace, storage.nullspace, context);
  if (!result.ok()) {
    return result;
  }
  result = validate_contiguous_view(storage.task_velocity_error,
                                    config,
                                    config.dtype,
                                    2,
                                    {static_cast<std::int64_t>(config.batch_size), 12, 0, 0},
                                    "task_velocity_error must be contiguous [batch_size,12]");
  if (!result.ok()) {
    return result;
  }
  result = output_.prepare(config, parameters.output, storage.output, context);
  task_velocity_error_ = storage.task_velocity_error;
  config_ = config;
  parameters_ = parameters;
  storage_ = storage;
  prepared_ = result.ok();
  return result;
}

Result BimanualCartesianImpedanceTorqueController::apply_parameters(
    const Parameters& parameters, const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = mass_.apply_parameters(parameters.mass, context);
  result = first_failure(result, task_.apply_parameters(parameters.task, context));
  result = first_failure(result, nullspace_.apply_parameters(parameters.nullspace, context));
  result = first_failure(result, output_.apply_parameters(parameters.output, context));
  parameters_ = parameters;
  prepared_ = result.ok();
  return result;
}

Result BimanualCartesianImpedanceTorqueController::reset(const ResetInputs& inputs,
                                                         const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  Result result = kinematics_.reset(inputs, context);
  result = first_failure(result, mass_.reset(inputs, context));
  result = first_failure(result, task_.reset(inputs, context));
  result = first_failure(result, nullspace_.reset(inputs, context));
  return first_failure(result, output_.reset(inputs, context));
}

Result BimanualCartesianImpedanceTorqueController::step(const Inputs& inputs,
                                                        const ExecutionContext context) noexcept {
  if (!prepared_) {
    return invalid_argument("controller is not prepared");
  }
  detail::tensor::ScopedTensorExecutionContext scope(context);
  Result result = kinematics_.step(inputs.kinematics, context);
  if (!result.ok()) {
    return result;
  }
  const auto kinematics_outputs = kinematics_.outputs();
  result =
      mass_.step({inputs.joint_mass_matrix, kinematics_outputs.task_jacobian.as_const()}, context);
  if (!result.ok()) {
    return result;
  }
  result =
      detail::tensor::step_bimanual_task_velocity_error(inputs.desired_task_velocity,
                                                        inputs.joint_velocity,
                                                        kinematics_outputs.task_jacobian.as_const(),
                                                        task_velocity_error_);
  if (!result.ok()) {
    return result;
  }
  result = task_.step({kinematics_outputs.task_pose_error.as_const(),
                       task_velocity_error_.as_const(),
                       inputs.desired_acceleration,
                       inputs.joint_velocity,
                       mass_.outputs().cartesian_mass_matrix.as_const(),
                       kinematics_outputs.task_jacobian.as_const(),
                       inputs.dt},
                      context);
  if (!result.ok()) {
    return result;
  }
  std::array<ParameterView, 2> nullspace_parameters{};
  std::size_t nullspace_parameter_count = 0;
  append_parameter(&nullspace_parameters,
                   &nullspace_parameter_count,
                   "stiffness_matrix",
                   parameters_.nullspace.stiffness_matrix);
  append_parameter(&nullspace_parameters,
                   &nullspace_parameter_count,
                   "damping_matrix",
                   parameters_.nullspace.damping_matrix);
  result = detail::tensor::step_bimanual_independent_nullspace_controller(
      inputs.joint_position_error,
      inputs.joint_velocity,
      inputs.joint_mass_matrix,
      inputs.kinematics.arm1_jacobian,
      inputs.kinematics.arm2_jacobian,
      parameter_set(nullspace_parameters, nullspace_parameter_count),
      parameter_status(storage_.nullspace),
      storage_.nullspace.command,
      storage_.nullspace.status);
  result = finish_step(result, config_, storage_.nullspace);
  if (!result.ok()) {
    return result;
  }
  const auto task_outputs = task_.outputs();
  const auto feedback_outputs = nullspace_.outputs();
  return output_.step({task_outputs.cartesian_wrench.as_const(),
                       kinematics_outputs.task_jacobian.as_const(),
                       inputs.joint_bias_torque,
                       feedback_outputs.joint_torque.as_const(),
                       read_status(task_outputs.status),
                       read_status(feedback_outputs.status),
                       inputs.dt},
                      context);
}

#include "public_blocks_support.inc"

#include "public_blocks.inc"

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
