/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file controllers_python.hpp
 * @brief Private pybind bridge for the public controller lifecycle.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "nvidia/srl/controllers/controllers.hpp"
#include "nvidia/srl/controllers/equation_adapter.hpp"
#include "nvidia/srl/controllers/example_specs.hpp"
#include "tensor/common.hpp"

namespace py = pybind11;

namespace nvidia {
namespace srl {
namespace controllers {
namespace controllers_python {

struct ParsedArray {
  TensorView view{};
  bool read_only{false};
};

inline std::string enum_value(const py::handle object) {
  return py::str(object.attr("value")).cast<std::string>();
}

inline std::string dtype_name(const DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat64:
      return "float64";
    case DType::kBool:
      return "bool";
    case DType::kUInt64:
      return "uint64";
    case DType::kUnknown:
      return "unknown";
  }
  return "unknown";
}

inline std::string device_name(const Device device) {
  switch (device) {
    case Device::kCpu:
      return "cpu";
    case Device::kCuda:
      return "cuda";
    case Device::kUnknown:
      return "unknown";
  }
  return "unknown";
}

inline std::string layout_name(const Layout layout) {
  switch (layout) {
    case Layout::kRowMajor:
      return "row_major";
    case Layout::kStrided:
      return "strided";
    case Layout::kUnknown:
      return "unknown";
  }
  return "unknown";
}

inline DType dtype_from_typestr(const std::string& typestr, std::size_t* item_size) {
  *item_size = 0;
  if (typestr == "|b1" || typestr == "?") {
    *item_size = 1;
    return DType::kBool;
  }
  if (typestr.size() != 3 || (typestr[0] != '<' && typestr[0] != '>' && typestr[0] != '=')) {
    return DType::kUnknown;
  }
  const std::uint16_t endian_probe = 1;
  const char native_order = *reinterpret_cast<const unsigned char*>(&endian_probe) == 1 ? '<' : '>';
  if (typestr[0] != '=' && typestr[0] != native_order) {
    throw std::invalid_argument(
        "array byte order must be native; non-native endian is unsupported");
  }
  const std::string scalar = typestr.substr(1);
  const DType dtype = scalar == "f4"   ? DType::kFloat32
                      : scalar == "f8" ? DType::kFloat64
                      : scalar == "u8" ? DType::kUInt64
                                       : DType::kUnknown;
  *item_size = detail::tensor::internal::dtype_size(dtype);
  return dtype;
}

inline py::tuple array_tuple(const py::handle object, const char* message) {
  if (!py::isinstance<py::tuple>(object)) {
    throw std::invalid_argument(message);
  }
  return py::reinterpret_borrow<py::tuple>(object);
}

template <typename Integer>
Integer array_integer(const py::handle object) {
  if (!py::isinstance<py::int_>(object) || py::isinstance<py::bool_>(object)) {
    throw std::invalid_argument("array dimensions, strides and addresses must be integers");
  }
  try {
    return object.cast<Integer>();
  } catch (const py::cast_error&) {
    throw std::invalid_argument("array integer metadata is outside its supported range");
  }
}

inline std::array<std::int64_t, kMaxTensorRank> parse_shape(py::handle object, int* rank) {
  const py::tuple shape_tuple = array_tuple(object, "array shape must be a tuple");
  if (shape_tuple.size() > kMaxTensorRank) {
    throw std::invalid_argument("tensor rank exceeds kMaxTensorRank");
  }
  std::array<std::int64_t, kMaxTensorRank> shape{};
  *rank = static_cast<int>(shape_tuple.size());
  for (py::size_t index = 0; index < shape_tuple.size(); ++index) {
    shape[index] = array_integer<std::int64_t>(shape_tuple[index]);
    if (shape[index] < 0) {
      throw std::invalid_argument("array extents must be nonnegative");
    }
  }
  return shape;
}

inline std::array<std::int64_t, kMaxTensorRank> contiguous_strides(
    const std::array<std::int64_t, kMaxTensorRank>& shape, const int rank) {
  std::array<std::int64_t, kMaxTensorRank> strides{};
  if (rank == 0) {
    return strides;
  }
  strides[rank - 1] = 1;
  for (int index = rank - 2; index >= 0; --index) {
    if (shape[index + 1] != 0 &&
        strides[index + 1] > std::numeric_limits<std::int64_t>::max() / shape[index + 1]) {
      throw std::invalid_argument("array contiguous strides overflow");
    }
    strides[index] = strides[index + 1] * shape[index + 1];
  }
  return strides;
}

inline ParsedArray parse_array(const py::handle object) {
  const py::object owner = py::reinterpret_borrow<py::object>(object);
  Device device = Device::kCpu;
  py::dict interface;
  if (py::hasattr(owner, "__cuda_array_interface__")) {
    interface = owner.attr("__cuda_array_interface__").cast<py::dict>();
    device = Device::kCuda;
  } else if (py::hasattr(owner, "__array_interface__")) {
    interface = owner.attr("__array_interface__").cast<py::dict>();
  } else {
    throw std::invalid_argument("object does not expose an array interface");
  }
  if (!interface.contains("data") || !interface.contains("typestr") ||
      !interface.contains("shape")) {
    throw std::invalid_argument("array interface is missing data, typestr, or shape");
  }

  const py::tuple data_tuple = array_tuple(interface["data"], "array data must be a tuple");
  if (data_tuple.size() != 2 || !py::isinstance<py::bool_>(data_tuple[1])) {
    throw std::invalid_argument("array data must contain an address and a boolean read-only flag");
  }
  const auto address = array_integer<std::uintptr_t>(data_tuple[0]);
  const bool read_only = data_tuple[1].cast<bool>();
  if (!py::isinstance<py::str>(interface["typestr"])) {
    throw std::invalid_argument("array typestr must be a string");
  }
  std::size_t item_size = 0;
  const DType dtype = dtype_from_typestr(interface["typestr"].cast<std::string>(), &item_size);
  if (dtype == DType::kUnknown || item_size == 0) {
    throw std::invalid_argument("unsupported dtype in array interface");
  }

  int rank = 0;
  const auto shape = parse_shape(interface["shape"], &rank);
  auto strides = contiguous_strides(shape, rank);
  if (interface.contains("strides") && !interface["strides"].is_none()) {
    const py::tuple byte_strides =
        array_tuple(interface["strides"], "array strides must be a tuple or None");
    if (byte_strides.size() != static_cast<py::size_t>(rank)) {
      throw std::invalid_argument("array interface strides rank mismatch");
    }
    for (int index = 0; index < rank; ++index) {
      const auto byte_stride = array_integer<std::int64_t>(byte_strides[index]);
      if (byte_stride < 0) {
        throw std::invalid_argument("negative array strides are unsupported");
      }
      if (byte_stride % static_cast<std::int64_t>(item_size) != 0) {
        throw std::invalid_argument("array byte stride is not divisible by item size");
      }
      strides[index] = byte_stride / static_cast<std::int64_t>(item_size);
    }
  }
  const Layout layout =
      strides == contiguous_strides(shape, rank) ? Layout::kRowMajor : Layout::kStrided;
  const TensorView view{
      reinterpret_cast<const void*>(address), dtype, device, layout, rank, shape, strides};
  detail::tensor::internal::ByteSpan span;
  const Result result = detail::tensor::internal::byte_span(view, &span);
  if (!result.ok()) {
    throw std::invalid_argument(result.message);
  }
  return ParsedArray{view, read_only};
}

inline MutableTensorView parse_mutable_array(const py::handle object) {
  const ParsedArray parsed = parse_array(object);
  if (parsed.read_only) {
    throw std::invalid_argument("mutable tensor view received a read-only array");
  }
  return MutableTensorView{const_cast<void*>(parsed.view.data),
                           parsed.view.dtype,
                           parsed.view.device,
                           parsed.view.layout,
                           parsed.view.rank,
                           parsed.view.shape,
                           parsed.view.strides};
}

inline TensorView parse_optional_array(const py::handle object) {
  return object.is_none() ? TensorView{} : parse_array(object).view;
}

inline StatusView parse_read_status(const py::handle object) {
  const TensorView view = parse_array(object).view;
  if (view.dtype != DType::kUInt64 || view.rank != 1 || view.shape[0] < 0 ||
      view.layout != Layout::kRowMajor || view.strides[0] != 1) {
    throw std::invalid_argument("status arrays must be contiguous uint64 [batch_size]");
  }
  return StatusView{static_cast<const std::uint64_t*>(view.data),
                    static_cast<std::size_t>(view.shape[0]),
                    view.device};
}

inline MutableStatusView parse_status(const py::handle object) {
  const MutableTensorView view = parse_mutable_array(object);
  const StatusView status = parse_read_status(object);
  return MutableStatusView{static_cast<std::uint64_t*>(view.data), status.size, status.device};
}

inline py::dict describe_array(const py::handle object) {
  const ParsedArray parsed = parse_array(object);
  py::dict output;
  output["address"] = reinterpret_cast<std::uintptr_t>(parsed.view.data);
  output["dtype"] = dtype_name(parsed.view.dtype);
  output["device"] = device_name(parsed.view.device);
  output["layout"] = layout_name(parsed.view.layout);
  output["rank"] = parsed.view.rank;
  output["shape"] = parsed.view.shape;
  output["strides"] = parsed.view.strides;
  output["read_only"] = parsed.read_only;
  return output;
}

inline Device parse_device(const py::handle object) {
  const std::string value = enum_value(object);
  if (value == "cpu") {
    return Device::kCpu;
  }
  if (value == "cuda") {
    return Device::kCuda;
  }
  return Device::kUnknown;
}

inline DType parse_dtype(const py::handle object) {
  const std::string value = enum_value(object);
  if (value == "float32") {
    return DType::kFloat32;
  }
  if (value == "float64") {
    return DType::kFloat64;
  }
  return DType::kUnknown;
}

inline CartesianFrame parse_frame(const py::handle object) {
  if (object.is_none()) {
    return CartesianFrame::kUnspecified;
  }
  const std::string value = enum_value(object);
  if (value == "robot_base") {
    return CartesianFrame::kRobotBase;
  }
  if (value == "end_effector") {
    return CartesianFrame::kEndEffector;
  }
  if (value == "main_frame_orientation") {
    return CartesianFrame::kMainFrameOrientation;
  }
  return CartesianFrame::kUnspecified;
}

inline SafeOutputPolicy parse_safe_output_policy(const py::handle object) {
  return enum_value(object) == "hold_last_valid" ? SafeOutputPolicy::kHoldLastValid
                                                 : SafeOutputPolicy::kZero;
}

inline ControllerConfig parse_config(const py::handle object) {
  ControllerConfig config;
  config.device = parse_device(object.attr("device"));
  config.dtype = parse_dtype(object.attr("dtype"));
  config.batch_size = object.attr("batch_size").cast<std::size_t>();
  config.joint_dim = object.attr("joint_dim").cast<std::size_t>();
  config.safety.safe_output_policy = parse_safe_output_policy(object.attr("safe_output_policy"));
  config.safety.max_hold_steps = object.attr("max_hold_steps").cast<std::uint64_t>();
  config.cartesian_frame = parse_frame(object.attr("cartesian_frame"));
  config.task_dim = object.attr("task_dim").cast<std::size_t>();
  return config;
}

inline CompositeControllerConfig parse_composite_config(const py::handle object) {
  CompositeControllerConfig config;
  static_cast<ControllerConfig&>(config) = parse_config(object);
  config.use_shaping_function = object.attr("use_shaping_function").cast<bool>();
  return config;
}

inline BimanualBase parse_bimanual_base(const py::handle object) {
  return enum_value(object) == "arm2" ? BimanualBase::kArm2 : BimanualBase::kArm1;
}

inline BimanualMainFrame parse_bimanual_main_frame(const py::handle object) {
  const std::string value = enum_value(object);
  if (value == "arm2_base") {
    return BimanualMainFrame::kArm2Base;
  }
  if (value == "explicit") {
    return BimanualMainFrame::kExplicit;
  }
  return BimanualMainFrame::kArm1Base;
}

inline BimanualCoordinationMode parse_bimanual_coordination_mode(const py::handle object) {
  return enum_value(object) == "coupled_rigid_transform"
             ? BimanualCoordinationMode::kCoupledRigidTransform
             : BimanualCoordinationMode::kDecoupledCooperative;
}

inline BimanualTaskConfig parse_bimanual_task_config(const py::handle object) {
  BimanualTaskConfig config;
  static_cast<ControllerConfig&>(config) = parse_config(object);
  const py::object frames = object.attr("frames");
  config.frames.arm1_T_arm2 = frames.attr("arm1_t_arm2").cast<std::array<double, 16>>();
  config.frames.main_frame = parse_bimanual_main_frame(frames.attr("main_frame"));
  config.frames.explicit_reference = parse_bimanual_base(frames.attr("explicit_reference"));
  config.frames.reference_T_main = frames.attr("reference_t_main").cast<std::array<double, 16>>();
  config.coordination_mode = parse_bimanual_coordination_mode(object.attr("coordination_mode"));
  return config;
}

inline ExecutionContext parse_context(const py::handle object) {
  if (object.is_none()) {
    return ExecutionContext{};
  }
  const py::object stream = object.attr("cuda_stream");
  if (stream.is_none()) {
    return ExecutionContext{};
  }
  return ExecutionContext{reinterpret_cast<void*>(stream.cast<std::uintptr_t>())};
}

inline ParameterTensor parse_parameter(const py::handle object) {
  if (object.is_none()) {
    return ParameterTensor{};
  }
  const auto batching = enum_value(object.attr("batching")) == "shared"
                            ? ParameterBatching::kShared
                            : ParameterBatching::kPerBatch;
  return ParameterTensor{parse_array(object.attr("values")).view, batching};
}

inline SafetyStateView parse_safety(const py::handle object) {
  return SafetyStateView{parse_mutable_array(object.attr("last_valid_output")),
                         parse_mutable_array(object.attr("valid_mask")),
                         parse_mutable_array(object.attr("hold_count"))};
}

inline ControllerOutputStorage parse_output_storage(const py::handle object) {
  return ControllerOutputStorage{parse_mutable_array(object.attr("command")),
                                 parse_status(object.attr("parameter_status")),
                                 parse_status(object.attr("status")),
                                 parse_safety(object.attr("safety"))};
}

inline InertialStateStorage parse_inertial_state(const py::handle object) {
  return InertialStateStorage{parse_mutable_array(object.attr("previous_mass_matrix")),
                              parse_mutable_array(object.attr("previous_jacobian")),
                              parse_mutable_array(object.attr("initialized"))};
}

inline BimanualCartesianImpedanceTorqueController::Parameters parse_bimanual_torque_parameters(
    const py::handle object) {
  const py::object mass = object.attr("mass");
  const py::object task = object.attr("task");
  const py::object nullspace = object.attr("nullspace");
  const py::object output = object.attr("output");
  return {
      {parse_parameter(mass.attr("regularization")), parse_parameter(mass.attr("minimum_pivot"))},
      {parse_parameter(task.attr("stiffness_matrix")),
       parse_parameter(task.attr("damping_factors"))},
      {parse_parameter(nullspace.attr("stiffness_matrix")),
       parse_parameter(nullspace.attr("damping_matrix"))},
      {parse_parameter(output.attr("lower_limit")),
       parse_parameter(output.attr("upper_limit")),
       parse_parameter(output.attr("maximum_rate"))}};
}

inline BimanualCartesianImpedanceTorqueController::Storage parse_bimanual_torque_storage(
    const py::handle object) {
  const py::object kinematics = object.attr("kinematics");
  const py::object mass = object.attr("mass");
  const py::object mass_scratch = mass.attr("scratch");
  const py::object task = object.attr("task");
  const py::object output = object.attr("output");
  const py::object filter = output.attr("filter");
  const py::object filter_state = filter.attr("state");
  BimanualCartesianImpedanceTorqueController::Storage storage;
  storage.kinematics = {parse_mutable_array(kinematics.attr("task_pose_error")),
                        parse_mutable_array(kinematics.attr("task_jacobian")),
                        parse_status(kinematics.attr("status"))};
  storage.mass = {parse_mutable_array(mass.attr("cartesian_mass_matrix")),
                  parse_status(mass.attr("parameter_status")),
                  parse_status(mass.attr("status")),
                  {parse_mutable_array(mass_scratch.attr("joint_factor")),
                   parse_mutable_array(mass_scratch.attr("solve_rhs")),
                   parse_mutable_array(mass_scratch.attr("cartesian_inverse"))}};
  static_cast<ControllerOutputStorage&>(storage.task) = parse_output_storage(task.attr("output"));
  storage.task.state = parse_inertial_state(task.attr("state"));
  storage.nullspace = parse_output_storage(object.attr("nullspace"));
  storage.output.unfiltered_joint_torque =
      parse_mutable_array(output.attr("unfiltered_joint_torque"));
  static_cast<ControllerOutputStorage&>(storage.output.filter) =
      parse_output_storage(filter.attr("output"));
  storage.output.filter.state = {parse_mutable_array(filter_state.attr("previous_command")),
                                 parse_mutable_array(filter_state.attr("initialized"))};
  storage.task_velocity_error = parse_mutable_array(object.attr("task_velocity_error"));
  return storage;
}

inline ForceControllerParameters parse_force_parameters(const py::handle object) {
  return ForceControllerParameters{parse_parameter(object.attr("proportional_gain")),
                                   parse_parameter(object.attr("differentiator_gain")),
                                   parse_parameter(object.attr("integral_gain")),
                                   parse_parameter(object.attr("initial_output")),
                                   parse_parameter(object.attr("maximum_distance")),
                                   parse_parameter(object.attr("shaping_distance"))};
}

inline ForceControllerStateStorage parse_force_state(const py::handle object) {
  return ForceControllerStateStorage{
      parse_mutable_array(object.attr("differentiator_previous_error")),
      parse_mutable_array(object.attr("differentiator_initialized")),
      parse_mutable_array(object.attr("integrator_previous_error")),
      parse_mutable_array(object.attr("integrated_error")),
      parse_mutable_array(object.attr("integrator_initialized"))};
}

inline AdmittanceStateStorage parse_admittance_state(const py::handle object) {
  return AdmittanceStateStorage{parse_mutable_array(object.attr("pose_offset")),
                                parse_mutable_array(object.attr("twist")),
                                parse_mutable_array(object.attr("pose_previous_signal")),
                                parse_mutable_array(object.attr("twist_previous_signal")),
                                parse_mutable_array(object.attr("pose_integrator_initialized")),
                                parse_mutable_array(object.attr("twist_integrator_initialized"))};
}

inline JointBlockConfig parse_static_joint_config(const py::handle object) {
  JointBlockConfig config;
  static_cast<ControllerConfig&>(config) = parse_config(object);
  const std::string mode = enum_value(object.attr("mode"));
  config.mode = mode == "diagonal" ? ComponentMode::kDiagonal
                : mode == "dense"  ? ComponentMode::kDense
                                   : ComponentMode::kUnsupported;
  return config;
}

inline ForceErrorPidParameters parse_force_error_pid_parameters(const py::handle object) {
  return ForceErrorPidParameters{parse_parameter(object.attr("proportional_gain")),
                                 parse_parameter(object.attr("differentiator_gain")),
                                 parse_parameter(object.attr("integral_gain")),
                                 parse_parameter(object.attr("initial_integral")),
                                 parse_parameter(object.attr("integral_lower_limit")),
                                 parse_parameter(object.attr("integral_upper_limit")),
                                 parse_parameter(object.attr("error_deadzone"))};
}

inline ResetInputs parse_reset(const py::handle object) {
  return ResetInputs{parse_optional_array(object)};
}

inline void ensure_result(const Result result, const char* operation) {
  if (!result.ok()) {
    throw std::invalid_argument(std::string(operation) + ": " + result.message);
  }
}

template <typename Controller>
struct BoundController {
  Controller controller{};
  py::object parameters_owner{};
  py::object storage_owner{};
};

template <typename Callable>
inline void invoke(const char* operation, Callable&& callable) {
  Result result;
  {
    py::gil_scoped_release release;
    result = callable();
  }
  ensure_result(result, operation);
}

inline void bind_torque_filter(py::module_& module) {
  using Controller = TorqueFilter;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "TorqueFilter")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Config parsed_config = parse_config(config);
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("lower_limit")),
                    parse_parameter(parameters.attr("upper_limit")),
                    parse_parameter(parameters.attr("maximum_rate"))};
                const py::object output = storage.attr("output");
                const py::object state = storage.attr("state");
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(output);
                parsed_storage.state =
                    TorqueFilterStateStorage{parse_mutable_array(state.attr("previous_command")),
                                             parse_mutable_array(state.attr("initialized"))};
                const ExecutionContext parsed_context = parse_context(context);
                invoke("prepare torque filter", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{parse_parameter(parameters.attr("lower_limit")),
                                                parse_parameter(parameters.attr("upper_limit")),
                                                parse_parameter(parameters.attr("maximum_rate"))};
            const ExecutionContext parsed_context = parse_context(context);
            invoke("apply torque filter parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const ResetInputs inputs = parse_reset(mask);
            const ExecutionContext parsed_context = parse_context(context);
            invoke("reset torque filter",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding, py::object command, py::object dt, py::object context) {
            const Controller::Inputs inputs{parse_array(command).view, parse_array(dt).view};
            const ExecutionContext parsed_context = parse_context(context);
            invoke("step torque filter",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("command"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_joint_impedance(py::module_& module) {
  using Controller = JointImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "JointImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_factors"))};
                const auto parsed_storage = parse_output_storage(storage);
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare joint impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_factors"))};
            const auto parsed_context = parse_context(context);
            invoke("apply joint impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset joint impedance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object position_error,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object mass_matrix,
             py::object context) {
            const Controller::Inputs inputs{parse_array(position_error).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(mass_matrix).view};
            const auto parsed_context = parse_context(context);
            invoke("step joint impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("joint_position_error"),
          py::arg("joint_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("joint_mass_matrix"),
          py::arg("context") = py::none());
}

inline void bind_nullspace(py::module_& module) {
  using Controller = NullspaceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "NullspaceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_matrix"))};
                const auto parsed_storage = parse_output_storage(storage);
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare nullspace controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_matrix"))};
            const auto parsed_context = parse_context(context);
            invoke("apply nullspace parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset nullspace controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object position_error,
             py::object joint_velocity,
             py::object mass_matrix,
             py::object jacobian,
             py::object context) {
            const Controller::Inputs inputs{parse_array(position_error).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(mass_matrix).view,
                                            parse_array(jacobian).view};
            const auto parsed_context = parse_context(context);
            invoke("step nullspace controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("joint_position_error"),
          py::arg("joint_velocity"),
          py::arg("joint_mass_matrix"),
          py::arg("jacobian"),
          py::arg("context") = py::none());
}

inline void bind_operational_space_mass(py::module_& module) {
  using Controller = OperationalSpaceMass;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "OperationalSpaceMass")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("regularization")),
                    parse_parameter(parameters.attr("minimum_pivot"))};
                const py::object scratch = storage.attr("scratch");
                const Controller::Storage parsed_storage{
                    parse_mutable_array(storage.attr("cartesian_mass_matrix")),
                    parse_status(storage.attr("parameter_status")),
                    parse_status(storage.attr("status")),
                    {parse_mutable_array(scratch.attr("joint_factor")),
                     parse_mutable_array(scratch.attr("solve_rhs")),
                     parse_mutable_array(scratch.attr("cartesian_inverse"))}};
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare operational-space mass", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{parse_parameter(parameters.attr("regularization")),
                                                parse_parameter(parameters.attr("minimum_pivot"))};
            const auto parsed_context = parse_context(context);
            invoke("apply operational-space mass parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset operational-space mass",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding, py::object mass_matrix, py::object jacobian, py::object context) {
            const Controller::Inputs inputs{parse_array(mass_matrix).view,
                                            parse_array(jacobian).view};
            const auto parsed_context = parse_context(context);
            invoke("step operational-space mass",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("joint_mass_matrix"),
          py::arg("jacobian"),
          py::arg("context") = py::none());
}

inline void bind_cartesian_impedance(py::module_& module) {
  using Controller = CartesianImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "CartesianImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_factors"))};
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state = parse_inertial_state(storage.attr("state"));
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare Cartesian impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_factors"))};
            const auto parsed_context = parse_context(context);
            invoke("apply Cartesian impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset Cartesian impedance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object joint_velocity,
             py::object cartesian_mass,
             py::object jacobian,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(cartesian_mass).view,
                                            parse_array(jacobian).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step Cartesian impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("cartesian_pose_error"),
          py::arg("cartesian_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("joint_velocity"),
          py::arg("cartesian_mass_matrix"),
          py::arg("jacobian"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_bimanual_cartesian_impedance(py::module_& module) {
  using Controller = BimanualCartesianImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "BimanualCartesianImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_factors"))};
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state = parse_inertial_state(storage.attr("state"));
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare bimanual Cartesian impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_factors"))};
            const auto parsed_context = parse_context(context);
            invoke("apply bimanual Cartesian impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset bimanual Cartesian impedance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object joint_velocity,
             py::object task_mass,
             py::object task_jacobian,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(task_mass).view,
                                            parse_array(task_jacobian).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step bimanual Cartesian impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("task_pose_error"),
          py::arg("task_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("joint_velocity"),
          py::arg("task_mass_matrix"),
          py::arg("task_jacobian"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_bimanual_task_kinematics(py::module_& module) {
  using Controller = BimanualTaskKinematics;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "BimanualTaskKinematics")
      .def(py::init([](py::object config, py::object storage) {
             auto binding = std::make_unique<Binding>();
             const Controller::Storage parsed_storage{
                 parse_mutable_array(storage.attr("task_pose_error")),
                 parse_mutable_array(storage.attr("task_jacobian")),
                 parse_status(storage.attr("status"))};
             const auto parsed_config = parse_bimanual_task_config(config);
             invoke("prepare bimanual task kinematics",
                    [&]() { return binding->controller.prepare(parsed_config, parsed_storage); });
             binding->storage_owner = storage;
             return binding;
           }),
           py::arg("config"),
           py::arg("storage"))
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset bimanual task kinematics",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object desired_absolute_pose,
             py::object arm1_tcp_pose,
             py::object arm2_tcp_pose,
             py::object desired_relative_pose,
             py::object arm1_jacobian,
             py::object arm2_jacobian,
             py::object context) {
            const Controller::Inputs inputs{parse_array(desired_absolute_pose).view,
                                            parse_array(arm1_tcp_pose).view,
                                            parse_array(arm2_tcp_pose).view,
                                            parse_array(desired_relative_pose).view,
                                            parse_array(arm1_jacobian).view,
                                            parse_array(arm2_jacobian).view};
            const auto parsed_context = parse_context(context);
            invoke("step bimanual task kinematics",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("desired_absolute_pose"),
          py::arg("arm1_tcp_pose"),
          py::arg("arm2_tcp_pose"),
          py::arg("desired_relative_pose"),
          py::arg("arm1_jacobian"),
          py::arg("arm2_jacobian"),
          py::arg("context") = py::none());
}

inline void bind_cartesian_admittance(py::module_& module) {
  using Controller = CartesianAdmittancePoseController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "CartesianAdmittancePoseController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_matrix")),
                    parse_parameter(parameters.attr("virtual_mass_matrix")),
                    parse_parameter(parameters.attr("force_filter_threshold")),
                    parse_parameter(parameters.attr("acceleration_limit")),
                    parse_parameter(parameters.attr("twist_limit")),
                    parse_parameter(parameters.attr("pose_limit"))};
                const py::object state = storage.attr("state");
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state = AdmittanceStateStorage{
                    parse_mutable_array(state.attr("pose_offset")),
                    parse_mutable_array(state.attr("twist")),
                    parse_mutable_array(state.attr("pose_previous_signal")),
                    parse_mutable_array(state.attr("twist_previous_signal")),
                    parse_mutable_array(state.attr("pose_integrator_initialized")),
                    parse_mutable_array(state.attr("twist_integrator_initialized"))};
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare Cartesian admittance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_matrix")),
                parse_parameter(parameters.attr("virtual_mass_matrix")),
                parse_parameter(parameters.attr("force_filter_threshold")),
                parse_parameter(parameters.attr("acceleration_limit")),
                parse_parameter(parameters.attr("twist_limit")),
                parse_parameter(parameters.attr("pose_limit"))};
            const auto parsed_context = parse_context(context);
            invoke("apply Cartesian admittance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset Cartesian admittance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object external_wrench,
             py::object feedforward_wrench,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(external_wrench).view,
                                            parse_array(feedforward_wrench).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step Cartesian admittance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("external_wrench"),
          py::arg("feedforward_wrench"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_operational_space(py::module_& module) {
  using Controller = OperationalSpaceInverseDynamicsController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "OperationalSpaceInverseDynamicsController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("position_gain")),
                    parse_parameter(parameters.attr("velocity_gain"))};
                const auto parsed_storage = parse_output_storage(storage);
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare operational-space controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{parse_parameter(parameters.attr("position_gain")),
                                                parse_parameter(parameters.attr("velocity_gain"))};
            const auto parsed_context = parse_context(context);
            invoke("apply operational-space parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset operational-space controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object jacobian_dot_joint_velocity,
             py::object cartesian_mass,
             py::object jacobian,
             py::object joint_bias_torque,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(jacobian_dot_joint_velocity).view,
                                            parse_array(cartesian_mass).view,
                                            parse_array(jacobian).view,
                                            parse_array(joint_bias_torque).view};
            const auto parsed_context = parse_context(context);
            invoke("step operational-space controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("cartesian_pose_error"),
          py::arg("cartesian_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("jacobian_dot_joint_velocity"),
          py::arg("cartesian_mass_matrix"),
          py::arg("jacobian"),
          py::arg("joint_bias_torque"),
          py::arg("context") = py::none());
  module.attr("OperationalSpaceController") =
      module.attr("OperationalSpaceInverseDynamicsController");
}

inline void bind_cartesian_force(py::module_& module) {
  using Controller = CartesianForceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "CartesianForceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state = parse_force_state(storage.attr("state"));
                const auto parsed_parameters = parse_force_parameters(parameters);
                const auto parsed_config = parse_composite_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare Cartesian force controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const auto parsed = parse_force_parameters(parameters);
            const auto parsed_context = parse_context(context);
            invoke("apply Cartesian force parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset Cartesian force controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object external_wrench,
             py::object desired_wrench,
             py::object feedforward_wrench,
             py::object pose_error,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(external_wrench).view,
                                            parse_array(desired_wrench).view,
                                            parse_array(feedforward_wrench).view,
                                            parse_array(pose_error).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step Cartesian force controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("external_wrench"),
          py::arg("desired_wrench"),
          py::arg("feedforward_wrench"),
          py::arg("pose_error"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_static_joint_impedance(py::module_& module) {
  using Controller = StaticJointImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "StaticJointImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness")),
                    parse_parameter(parameters.attr("damping"))};
                const auto parsed_storage = parse_output_storage(storage);
                const auto parsed_config = parse_static_joint_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare static joint impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{parse_parameter(parameters.attr("stiffness")),
                                                parse_parameter(parameters.attr("damping"))};
            const auto parsed_context = parse_context(context);
            invoke("apply static joint impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto parsed_mask = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset static joint impedance controller",
                   [&]() { return binding.controller.reset(parsed_mask, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object position_error,
             py::object velocity_error,
             py::object context) {
            const Controller::Inputs inputs{parse_array(position_error).view,
                                            parse_array(velocity_error).view};
            const auto parsed_context = parse_context(context);
            invoke("step static joint impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("joint_position_error"),
          py::arg("joint_velocity_error"),
          py::arg("context") = py::none());
}

inline void bind_static_cartesian_impedance(py::module_& module) {
  using Controller = StaticCartesianImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "StaticCartesianImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_matrix"))};
                const auto parsed_config = parse_config(config);
                const auto parsed_storage = parse_output_storage(storage);
                const auto parsed_context = parse_context(context);
                invoke("prepare static Cartesian impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_matrix"))};
            const auto parsed_context = parse_context(context);
            invoke("apply static Cartesian impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto parsed_mask = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset static Cartesian impedance controller",
                   [&]() { return binding.controller.reset(parsed_mask, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object velocity_error,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(velocity_error).view};
            const auto parsed_context = parse_context(context);
            invoke("step static Cartesian impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("cartesian_pose_error"),
          py::arg("cartesian_velocity_error"),
          py::arg("context") = py::none());
}

inline void bind_unified_force_admittance(py::module_& module) {
  using Controller = CartesianUnifiedForceAdmittanceController;
  using Binding = BoundController<Controller>;
  auto parse_parameters = [](const py::handle object) {
    return Controller::Parameters{parse_parameter(object.attr("stiffness_matrix")),
                                  parse_parameter(object.attr("damping_matrix")),
                                  parse_parameter(object.attr("virtual_mass_matrix")),
                                  parse_parameter(object.attr("acceleration_limit")),
                                  parse_parameter(object.attr("twist_limit")),
                                  parse_parameter(object.attr("pose_limit")),
                                  parse_force_error_pid_parameters(object.attr("force"))};
  };
  py::class_<Binding>(module, "CartesianUnifiedForceAdmittanceController")
      .def(py::init([parse_parameters](py::object config,
                                       py::object parameters,
                                       py::object storage,
                                       py::object context) {
             auto binding = std::make_unique<Binding>();
             Controller::Storage parsed_storage;
             static_cast<ControllerOutputStorage&>(parsed_storage) =
                 parse_output_storage(storage.attr("output"));
             const py::object state = storage.attr("state");
             parsed_storage.state.force = parse_force_state(state.attr("force"));
             parsed_storage.state.admittance = parse_admittance_state(state.attr("admittance"));
             const auto parsed_config = parse_config(config);
             const auto parsed_parameters = parse_parameters(parameters);
             const auto parsed_context = parse_context(context);
             invoke("prepare unified force admittance controller", [&]() {
               return binding->controller.prepare(
                   parsed_config, parsed_parameters, parsed_storage, parsed_context);
             });
             binding->parameters_owner = parameters;
             binding->storage_owner = storage;
             return binding;
           }),
           py::arg("config"),
           py::arg("parameters"),
           py::arg("storage"),
           py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [parse_parameters](Binding& binding, py::object parameters, py::object context) {
            const auto parsed_parameters = parse_parameters(parameters);
            const auto parsed_context = parse_context(context);
            invoke("apply unified force admittance parameters", [&]() {
              return binding.controller.apply_parameters(parsed_parameters, parsed_context);
            });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto parsed_mask = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset unified force admittance controller",
                   [&]() { return binding.controller.reset(parsed_mask, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object desired,
             py::object external,
             py::object feedforward,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(desired).view,
                                            parse_array(external).view,
                                            parse_array(feedforward).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step unified force admittance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("desired_wrench"),
          py::arg("external_wrench"),
          py::arg("feedforward_wrench"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_unified_controller(py::module_& module) {
  using Controller = CartesianUnifiedForceImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "CartesianUnifiedForceImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("stiffness_matrix")),
                    parse_parameter(parameters.attr("damping_factors")),
                    parse_force_parameters(parameters.attr("force"))};
                const py::object state = storage.attr("state");
                const py::object scratch = storage.attr("scratch");
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state =
                    UnifiedControllerStateStorage{parse_inertial_state(state.attr("impedance")),
                                                  parse_force_state(state.attr("force"))};
                parsed_storage.scratch = UnifiedControllerScratchStorage{
                    parse_mutable_array(scratch.attr("impedance_wrench")),
                    parse_mutable_array(scratch.attr("force_wrench")),
                    parse_mutable_array(scratch.attr("impedance_status")),
                    parse_mutable_array(scratch.attr("force_status"))};
                const auto parsed_config = parse_composite_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare unified force/impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("stiffness_matrix")),
                parse_parameter(parameters.attr("damping_factors")),
                parse_force_parameters(parameters.attr("force"))};
            const auto parsed_context = parse_context(context);
            invoke("apply unified force/impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset unified force/impedance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object joint_velocity,
             py::object cartesian_mass,
             py::object jacobian,
             py::object external_wrench,
             py::object desired_wrench,
             py::object feedforward_wrench,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(cartesian_mass).view,
                                            parse_array(jacobian).view,
                                            parse_array(external_wrench).view,
                                            parse_array(desired_wrench).view,
                                            parse_array(feedforward_wrench).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step unified force/impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("cartesian_pose_error"),
          py::arg("cartesian_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("joint_velocity"),
          py::arg("cartesian_mass_matrix"),
          py::arg("jacobian"),
          py::arg("external_wrench"),
          py::arg("desired_wrench"),
          py::arg("feedforward_wrench"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_variable_impedance(py::module_& module) {
  using Controller = VariableCartesianImpedanceController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "VariableCartesianImpedanceController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("initial_stiffness_matrix")),
                    parse_parameter(parameters.attr("maximum_stiffness")),
                    parse_parameter(parameters.attr("minimum_stiffness")),
                    parse_parameter(parameters.attr("maximum_stiffness_rate")),
                    parse_parameter(parameters.attr("damping_factors"))};
                const py::object state = storage.attr("state");
                const py::object scratch = storage.attr("scratch");
                Controller::Storage parsed_storage;
                static_cast<ControllerOutputStorage&>(parsed_storage) =
                    parse_output_storage(storage.attr("output"));
                parsed_storage.state = VariableImpedanceStateStorage{
                    parse_mutable_array(state.attr("current_stiffness_matrix")),
                    parse_mutable_array(state.attr("previous_stiffness_matrix")),
                    parse_mutable_array(state.attr("stiffness_initialized")),
                    parse_inertial_state(state.attr("inertial"))};
                parsed_storage.scratch = VariableImpedanceScratchStorage{
                    parse_mutable_array(scratch.attr("stiffness_wrench")),
                    parse_mutable_array(scratch.attr("damping_wrench")),
                    parse_mutable_array(scratch.attr("inertial_wrench")),
                    parse_mutable_array(scratch.attr("stiffness_status")),
                    parse_mutable_array(scratch.attr("damping_status")),
                    parse_mutable_array(scratch.attr("inertial_status"))};
                const auto parsed_config = parse_config(config);
                const auto parsed_context = parse_context(context);
                invoke("prepare variable Cartesian impedance controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{
                parse_parameter(parameters.attr("initial_stiffness_matrix")),
                parse_parameter(parameters.attr("maximum_stiffness")),
                parse_parameter(parameters.attr("minimum_stiffness")),
                parse_parameter(parameters.attr("maximum_stiffness_rate")),
                parse_parameter(parameters.attr("damping_factors"))};
            const auto parsed_context = parse_context(context);
            invoke("apply variable Cartesian impedance parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const auto inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset variable Cartesian impedance controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object pose_error,
             py::object requested_stiffness,
             py::object velocity_error,
             py::object desired_acceleration,
             py::object joint_velocity,
             py::object cartesian_mass,
             py::object jacobian,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(pose_error).view,
                                            parse_array(requested_stiffness).view,
                                            parse_array(velocity_error).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(cartesian_mass).view,
                                            parse_array(jacobian).view,
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step variable Cartesian impedance controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("cartesian_pose_error"),
          py::arg("requested_stiffness_matrix"),
          py::arg("cartesian_velocity_error"),
          py::arg("desired_acceleration"),
          py::arg("joint_velocity"),
          py::arg("cartesian_mass_matrix"),
          py::arg("jacobian"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_task_wrench_to_joint_torque(py::module_& module) {
  using Controller = TaskWrenchToJointTorqueController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "TaskWrenchToJointTorqueController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Config parsed_config = parse_config(config);
                const Controller::Parameters parsed_parameters{
                    parse_parameter(parameters.attr("lower_limit")),
                    parse_parameter(parameters.attr("upper_limit")),
                    parse_parameter(parameters.attr("maximum_rate"))};
                const py::object filter = storage.attr("filter");
                const py::object output = filter.attr("output");
                const py::object state = filter.attr("state");
                Controller::Storage parsed_storage;
                parsed_storage.unfiltered_joint_torque =
                    parse_mutable_array(storage.attr("unfiltered_joint_torque"));
                static_cast<ControllerOutputStorage&>(parsed_storage.filter) =
                    parse_output_storage(output);
                parsed_storage.filter.state =
                    TorqueFilterStateStorage{parse_mutable_array(state.attr("previous_command")),
                                             parse_mutable_array(state.attr("initialized"))};
                const auto parsed_context = parse_context(context);
                invoke("prepare task-wrench to joint-torque controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed{parse_parameter(parameters.attr("lower_limit")),
                                                parse_parameter(parameters.attr("upper_limit")),
                                                parse_parameter(parameters.attr("maximum_rate"))};
            const auto parsed_context = parse_context(context);
            invoke("apply task-wrench torque-filter parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const ResetInputs inputs = parse_reset(mask);
            const auto parsed_context = parse_context(context);
            invoke("reset task-wrench to joint-torque controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object task_wrench,
             py::object task_jacobian,
             py::object joint_bias_torque,
             py::object joint_feedback_torque,
             py::object task_status,
             py::object feedback_status,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{parse_array(task_wrench).view,
                                            parse_array(task_jacobian).view,
                                            parse_array(joint_bias_torque).view,
                                            parse_array(joint_feedback_torque).view,
                                            parse_read_status(task_status),
                                            parse_read_status(feedback_status),
                                            parse_array(dt).view};
            const auto parsed_context = parse_context(context);
            invoke("step task-wrench to joint-torque controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("task_wrench"),
          py::arg("task_jacobian"),
          py::arg("joint_bias_torque"),
          py::arg("joint_feedback_torque"),
          py::arg("task_status"),
          py::arg("feedback_status"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline void bind_bimanual_impedance_torque(py::module_& module) {
  using Controller = BimanualCartesianImpedanceTorqueController;
  using Binding = BoundController<Controller>;
  py::class_<Binding>(module, "BimanualCartesianImpedanceTorqueController")
      .def(
          py::init(
              [](py::object config, py::object parameters, py::object storage, py::object context) {
                auto binding = std::make_unique<Binding>();
                const Controller::Config parsed_config = parse_bimanual_task_config(config);
                const Controller::Parameters parsed_parameters =
                    parse_bimanual_torque_parameters(parameters);
                const Controller::Storage parsed_storage = parse_bimanual_torque_storage(storage);
                const ExecutionContext parsed_context = parse_context(context);
                invoke("prepare bimanual Cartesian impedance torque controller", [&]() {
                  return binding->controller.prepare(
                      parsed_config, parsed_parameters, parsed_storage, parsed_context);
                });
                binding->parameters_owner = parameters;
                binding->storage_owner = storage;
                return binding;
              }),
          py::arg("config"),
          py::arg("parameters"),
          py::arg("storage"),
          py::arg("context") = py::none())
      .def(
          "apply_parameters",
          [](Binding& binding, py::object parameters, py::object context) {
            const Controller::Parameters parsed = parse_bimanual_torque_parameters(parameters);
            const ExecutionContext parsed_context = parse_context(context);
            invoke("apply bimanual Cartesian impedance torque parameters",
                   [&]() { return binding.controller.apply_parameters(parsed, parsed_context); });
            binding.parameters_owner = parameters;
          },
          py::arg("parameters"),
          py::arg("context") = py::none())
      .def(
          "reset",
          [](Binding& binding, py::object mask, py::object context) {
            const ResetInputs inputs = parse_reset(mask);
            const ExecutionContext parsed_context = parse_context(context);
            invoke("reset bimanual Cartesian impedance torque controller",
                   [&]() { return binding.controller.reset(inputs, parsed_context); });
          },
          py::arg("mask") = py::none(),
          py::arg("context") = py::none())
      .def(
          "step",
          [](Binding& binding,
             py::object desired_absolute_pose,
             py::object arm1_tcp_pose,
             py::object arm2_tcp_pose,
             py::object desired_relative_pose,
             py::object arm1_jacobian,
             py::object arm2_jacobian,
             py::object desired_task_velocity,
             py::object desired_acceleration,
             py::object joint_position_error,
             py::object joint_velocity,
             py::object joint_mass_matrix,
             py::object joint_bias_torque,
             py::object dt,
             py::object context) {
            const Controller::Inputs inputs{{parse_array(desired_absolute_pose).view,
                                             parse_array(arm1_tcp_pose).view,
                                             parse_array(arm2_tcp_pose).view,
                                             parse_array(desired_relative_pose).view,
                                             parse_array(arm1_jacobian).view,
                                             parse_array(arm2_jacobian).view},
                                            parse_array(desired_task_velocity).view,
                                            parse_array(desired_acceleration).view,
                                            parse_array(joint_position_error).view,
                                            parse_array(joint_velocity).view,
                                            parse_array(joint_mass_matrix).view,
                                            parse_array(joint_bias_torque).view,
                                            parse_array(dt).view};
            const ExecutionContext parsed_context = parse_context(context);
            invoke("step bimanual Cartesian impedance torque controller",
                   [&]() { return binding.controller.step(inputs, parsed_context); });
          },
          py::arg("desired_absolute_pose"),
          py::arg("arm1_tcp_pose"),
          py::arg("arm2_tcp_pose"),
          py::arg("desired_relative_pose"),
          py::arg("arm1_jacobian"),
          py::arg("arm2_jacobian"),
          py::arg("desired_task_velocity"),
          py::arg("desired_acceleration"),
          py::arg("joint_position_error"),
          py::arg("joint_velocity"),
          py::arg("joint_mass_matrix"),
          py::arg("joint_bias_torque"),
          py::arg("dt"),
          py::arg("context") = py::none());
}

inline const char* robot_family_name(const ExampleRobotFamily family) noexcept {
  switch (family) {
    case ExampleRobotFamily::kFranka:
      return "franka";
    case ExampleRobotFamily::kRizon:
      return "rizon";
    case ExampleRobotFamily::kFlexivDual:
      return "flexiv_dual";
    case ExampleRobotFamily::kUniversalRobots:
      return "universal_robots";
  }
  return "unknown";
}

inline const char* controller_kind_name(const ExampleControllerKind kind) noexcept {
  switch (kind) {
    case ExampleControllerKind::kCartesianVirtualWallsTorque:
      return "cartesian_virtual_walls_torque";
    case ExampleControllerKind::kCartesianPotentialFieldTorque:
      return "cartesian_potential_field_torque";
    case ExampleControllerKind::kCartesianForceImpedanceWithTankTorque:
      return "cartesian_force_impedance_with_tank_torque";
    case ExampleControllerKind::kCartesianAdmittanceTorque:
      return "cartesian_admittance_torque";
    case ExampleControllerKind::kCartesianUnifiedForceImpedanceTorque:
      return "cartesian_unified_force_impedance_torque";
    case ExampleControllerKind::kCartesianImpedanceTorque:
      return "cartesian_impedance_torque";
    case ExampleControllerKind::kJointImpedance:
      return "joint_impedance";
    case ExampleControllerKind::kOperationalSpace:
      return "operational_space";
    case ExampleControllerKind::kVariableCartesianImpedanceTorque:
      return "variable_cartesian_impedance_torque";
    case ExampleControllerKind::kBimanualCartesianImpedanceTorque:
      return "bimanual_cartesian_impedance_torque";
    case ExampleControllerKind::kCartesianAdmittancePose:
      return "cartesian_admittance_pose";
    case ExampleControllerKind::kStaticJointImpedance:
      return "static_joint_impedance";
    case ExampleControllerKind::kStaticCartesianImpedanceTorque:
      return "static_cartesian_impedance_torque";
    case ExampleControllerKind::kDualCartesianImpedanceTorque:
      return "dual_cartesian_impedance_torque";
    case ExampleControllerKind::kCartesianUnifiedForceAdmittancePose:
      return "cartesian_unified_force_admittance_pose";
  }
  return "unknown";
}

inline void bind_example_specs(py::module_& module) {
  module.def("controller_example_specs", []() {
    py::list output;
    for (const ControllerExampleSpec& spec : all_controller_example_specs()) {
      py::dict item;
      item["hardware_target"] = spec.hardware_target;
      item["isaac_launcher"] = spec.isaac_launcher;
      item["robot_family"] = robot_family_name(spec.robot_family);
      item["controller_kind"] = controller_kind_name(spec.controller_kind);
      item["default_robot_model"] = spec.default_robot_model;
      item["supported_robot_models"] = spec.supported_robot_models;
      item["joint_dim"] = spec.joint_dim;
      item["task_dim"] = spec.task_dim;
      item["cartesian_frame"] =
          spec.cartesian_frame == CartesianFrame::kRobotBase
              ? "robot_base"
              : (spec.cartesian_frame == CartesianFrame::kEndEffector
                     ? "end_effector"
                     : (spec.cartesian_frame == CartesianFrame::kMainFrameOrientation
                            ? "main_frame_orientation"
                            : "unspecified"));
      item["nominal_period_seconds"] = spec.nominal_period_seconds;
      item["default_duration_seconds"] = spec.default_duration_seconds;
      item["controlled_force_axis"] = spec.controlled_force_axis;
      output.append(item);
    }
    return output;
  });
}

#include "equations_python.inc"
#include "public_blocks_python.inc"

inline void bind_composition(py::module_& module) {
  module.def("merge_status", [](py::tuple inputs, py::object output, py::object context) {
    std::vector<StatusView> views;
    views.reserve(inputs.size());
    for (const auto input : inputs) {
      const auto status = parse_status(input);
      views.push_back({status.flags, status.size, status.device});
    }
    const auto destination = parse_status(output);
    const auto execution = parse_context(context);
    invoke("merge status",
           [&]() { return merge_status(views.data(), views.size(), destination, execution); });
  });
  module.def("reset_safety_state",
             [](py::object state, py::object mask, py::object status, py::object context) {
               const auto parsed_state = parse_safety(state);
               const auto parsed_mask = parse_optional_array(mask);
               const auto parsed_status = parse_status(status);
               const auto execution = parse_context(context);
               invoke("reset safety", [&]() {
                 return reset_safety_state(
                     parsed_status.size, parsed_state, parsed_mask, parsed_status, execution);
               });
             });
  module.def("apply_safe_output_policy",
             [](py::object output,
                py::object status,
                py::object state,
                py::object policy,
                std::uint64_t max_hold_steps,
                py::object context) {
               const auto parsed_output = parse_mutable_array(output);
               const auto parsed_status = parse_status(status);
               const auto parsed_state = parse_safety(state);
               const StepOptions options{parse_safe_output_policy(policy), max_hold_steps};
               const auto execution = parse_context(context);
               invoke("apply output policy", [&]() {
                 return apply_safe_output_policy(
                     parsed_output, parsed_status, parsed_state, options, execution);
               });
             });
}

inline void bind_controllers(py::module_& parent) {
  py::module_ module = parent.def_submodule("_controller_api");
  module.def("describe_array", &describe_array, py::arg("array_like"));
  module.def("arrays_overlap", [](py::handle left, py::handle right) {
    detail::tensor::internal::ByteSpan left_span;
    detail::tensor::internal::ByteSpan right_span;
    const Result left_result =
        detail::tensor::internal::byte_span(parse_array(left).view, &left_span);
    const Result right_result =
        detail::tensor::internal::byte_span(parse_array(right).view, &right_span);
    ensure_result(left_result, "validate left array");
    ensure_result(right_result, "validate right array");
    return detail::tensor::internal::overlaps(left_span, right_span);
  });
  module.def("validate_storage_lifetimes", [](const py::list& entries) {
    detail::tensor::internal::StorageValidation validation;
    for (const py::handle entry : entries) {
      const auto fields = py::cast<py::tuple>(entry);
      if (fields.size() != 4) {
        throw std::invalid_argument("storage entries require an array, lifetime, and alias policy");
      }
      validation.add(parse_mutable_array(fields[0]).as_const(),
                     fields[1].cast<int>(),
                     fields[2].cast<int>(),
                     fields[3].cast<bool>());
      ensure_result(validation.result, "validate composite storage");
    }
  });
  module.attr("STATUS_UNSUPPORTED_MODE") = status_bit(StatusFlag::kUnsupportedMode);
  module.attr("STATUS_INVALID_PARAMETER") = status_bit(StatusFlag::kInvalidParameter);
  module.attr("STATUS_INVALID_INPUT") = status_bit(StatusFlag::kInvalidInput);
  module.attr("STATUS_NON_FINITE_INPUT") = status_bit(StatusFlag::kNonFiniteInput);
  module.attr("STATUS_INVALID_DT") = status_bit(StatusFlag::kInvalidDt);
  module.attr("STATUS_SATURATION") = status_bit(StatusFlag::kSaturation);
  module.attr("STATUS_RESET_APPLIED") = status_bit(StatusFlag::kResetApplied);
  module.attr("STATUS_SAFE_MODE") = status_bit(StatusFlag::kSafeMode);
  module.attr("STATUS_OUTPUT_HELD") = status_bit(StatusFlag::kOutputHeld);
  module.attr("STATUS_NO_VALID_OUTPUT") = status_bit(StatusFlag::kNoValidOutput);
  module.attr("STATUS_HOLD_EXPIRED") = status_bit(StatusFlag::kHoldExpired);
  bind_torque_filter(module);
  bind_joint_impedance(module);
  bind_nullspace(module);
  bind_operational_space_mass(module);
  bind_cartesian_impedance(module);
  bind_bimanual_cartesian_impedance(module);
  bind_bimanual_task_kinematics(module);
  bind_cartesian_admittance(module);
  bind_operational_space(module);
  bind_cartesian_force(module);
  bind_static_joint_impedance(module);
  bind_static_cartesian_impedance(module);
  bind_unified_force_admittance(module);
  bind_unified_controller(module);
  bind_variable_impedance(module);
  bind_task_wrench_to_joint_torque(module);
  bind_bimanual_impedance_torque(module);
  bind_example_specs(module);
  bind_public_blocks(module);
  bind_equations(module);
  bind_composition(module);
}

}  // namespace controllers_python
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
