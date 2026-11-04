/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "nvidia/srl/controllers/controllers.hpp"

namespace nvidia::srl::controllers::test_support {

template <typename T>
class DescriptorStorage {
 public:
  using Values =
      std::variant<std::vector<T>, std::vector<std::uint64_t>, std::vector<std::uint8_t>>;
  struct Buffer {
    TensorRequirement requirement;
    std::array<std::int64_t, kMaxTensorRank> strides{};
    Values values;
    bool scratch;
    std::size_t bindings{0};
  };

  explicit DescriptorStorage(const Requirements& requirements) {
    allocate(requirements.state, requirements.state_count, false);
    allocate(requirements.scratch, requirements.scratch_count, true);
    allocate(requirements.outputs, requirements.output_count, false);
  }

  MutableTensorView view(const std::string& name) {
    Buffer& buffer = buffers_.at(name);
    ++buffer.bindings;
    void* pointer = std::visit([](auto& values) -> void* { return values.data(); }, buffer.values);
    const TensorRequirement& requirement = buffer.requirement;
    return {pointer,
            requirement.dtype,
            requirement.device,
            Layout::kRowMajor,
            requirement.rank,
            requirement.shape,
            buffer.strides};
  }

  MutableStatusView status(const std::string& name) {
    const MutableTensorView tensor = view(name);
    if (tensor.dtype != DType::kUInt64 || tensor.rank != 1) {
      throw std::invalid_argument("status descriptor is not uint64 [batch]");
    }
    return {static_cast<std::uint64_t*>(tensor.data),
            static_cast<std::size_t>(tensor.shape[0]),
            tensor.device};
  }

  template <typename Element>
  std::vector<Element>& values(const std::string& name) {
    return std::get<std::vector<Element>>(buffers_.at(name).values);
  }

  bool all_bound_once() const {
    return std::all_of(buffers_.begin(), buffers_.end(), [](const auto& entry) {
      return entry.second.bindings == 1;
    });
  }

  void poison_scratch() {
    for (auto& entry : buffers_) {
      Buffer& buffer = entry.second;
      if (buffer.scratch) {
        std::visit(
            [](auto& values) {
              using Element = typename std::decay_t<decltype(values)>::value_type;
              const Element poison = std::is_floating_point<Element>::value
                                         ? std::numeric_limits<Element>::quiet_NaN()
                                         : std::numeric_limits<Element>::max();
              std::fill(values.begin(), values.end(), poison);
            },
            buffer.values);
      }
    }
  }

 private:
  template <std::size_t Capacity>
  void allocate(const std::array<TensorRequirement, Capacity>& requirements,
                std::size_t count,
                bool scratch) {
    if (count > Capacity) throw std::invalid_argument("descriptor count exceeds capacity");
    for (std::size_t i = 0; i < count; ++i) {
      const TensorRequirement& requirement = requirements[i];
      if (requirement.name == nullptr || requirement.device != Device::kCpu ||
          requirement.rank < 1 || requirement.rank > kMaxTensorRank) {
        throw std::invalid_argument("invalid CPU allocation descriptor");
      }
      Buffer buffer{requirement, {}, {}, scratch};
      std::int64_t elements = 1;
      for (int dim = requirement.rank - 1; dim >= 0; --dim) {
        const std::int64_t extent = requirement.shape[dim];
        if (extent <= 0 || elements > std::numeric_limits<std::int64_t>::max() / extent) {
          throw std::invalid_argument("allocation descriptor shape overflows");
        }
        buffer.strides[dim] = elements;
        elements *= extent;
      }
      if (requirement.dtype == dtype_for<T>()) {
        buffer.values = std::vector<T>(elements);
      } else if (requirement.dtype == DType::kUInt64) {
        buffer.values = std::vector<std::uint64_t>(elements);
      } else if (requirement.dtype == DType::kBool) {
        buffer.values = std::vector<std::uint8_t>(elements);
      } else {
        throw std::invalid_argument("allocation descriptor has unexpected dtype");
      }
      if (!buffers_.emplace(requirement.name, std::move(buffer)).second) {
        throw std::invalid_argument("duplicate allocation descriptor name");
      }
    }
  }

  std::map<std::string, Buffer> buffers_;
};

template <typename T>
ControllerOutputStorage bind_output(DescriptorStorage<T>& buffers,
                                    const std::string& prefix,
                                    const std::string& command) {
  return {buffers.view(prefix + command),
          buffers.status(prefix + "parameter_status"),
          buffers.status(prefix + "status"),
          {buffers.view(prefix + "last_valid_output"),
           buffers.view(prefix + "valid_mask"),
           buffers.view(prefix + "hold_count")}};
}

template <typename T>
InertialStateStorage bind_inertial(DescriptorStorage<T>& buffers, const std::string& initialized) {
  return {buffers.view("task.previous_mass_matrix"),
          buffers.view("task.previous_jacobian"),
          buffers.view(initialized)};
}

template <typename T>
void bind_task(DescriptorStorage<T>& buffers, CartesianImpedanceController::Storage& storage) {
  static_cast<ControllerOutputStorage&>(storage) =
      bind_output(buffers, "task.", "cartesian_wrench");
  storage.state = bind_inertial(buffers, "task.initialized");
}

template <typename T>
void bind_task(DescriptorStorage<T>& buffers,
               VariableCartesianImpedanceController::Storage& storage) {
  static_cast<ControllerOutputStorage&>(storage) =
      bind_output(buffers, "task.", "cartesian_wrench");
  storage.state = {buffers.view("task.current_stiffness_matrix"),
                   buffers.view("task.previous_stiffness_matrix"),
                   buffers.view("task.stiffness_initialized"),
                   bind_inertial(buffers, "task.inertial_initialized")};
  storage.scratch = {buffers.view("task.stiffness_wrench"),
                     buffers.view("task.damping_wrench"),
                     buffers.view("task.inertial_wrench"),
                     buffers.view("task.stiffness_status"),
                     buffers.view("task.damping_status"),
                     buffers.view("task.inertial_status")};
}

template <typename T>
void bind_task(DescriptorStorage<T>& buffers,
               CartesianUnifiedForceImpedanceController::Storage& storage) {
  static_cast<ControllerOutputStorage&>(storage) =
      bind_output(buffers, "task.", "cartesian_wrench");
  storage.state = {bind_inertial(buffers, "task.initialized"),
                   {buffers.view("task.differentiator_previous_error"),
                    buffers.view("task.differentiator_initialized"),
                    buffers.view("task.integrator_previous_error"),
                    buffers.view("task.integrated_error"),
                    buffers.view("task.integrator_initialized")}};
  storage.scratch = {buffers.view("task.impedance_wrench"),
                     buffers.view("task.force_wrench"),
                     buffers.view("task.impedance_status"),
                     buffers.view("task.force_status")};
}

template <typename T, typename Controller>
class CompositeFixture {
 public:
  static constexpr bool kBimanual =
      std::is_same<Controller, BimanualCartesianImpedanceTorqueController>::value;
  static constexpr bool kVariable =
      std::is_same<Controller, VariableCartesianImpedanceTorqueController>::value;
  static constexpr bool kUnified =
      std::is_same<Controller, CartesianUnifiedForceImpedanceTorqueController>::value;

  explicit CompositeFixture(std::size_t batch)
      : config(make_config(batch)), requirements(query(config)), buffers(requirements) {
    bind_storage();
    initialize_parameters();
    initialize_inputs();
  }
  CompositeFixture(const CompositeFixture&) = delete;
  CompositeFixture& operator=(const CompositeFixture&) = delete;

  Result prepare() { return controller.prepare(config, parameters, storage); }
  Result reset() { return controller.reset(); }
  Result step() { return controller.step(inputs); }

  typename Controller::Config config;
  Requirements requirements;
  DescriptorStorage<T> buffers;
  typename Controller::Storage storage{};
  typename Controller::Parameters parameters{};
  typename Controller::Inputs inputs{};
  Controller controller;
  std::vector<T> joint_mass;
  std::vector<T> jacobian;
  std::vector<T> pose_error;
  std::vector<T> velocity_error;
  std::vector<T> acceleration;
  std::vector<T> joint_error;
  std::vector<T> joint_velocity;
  std::vector<T> joint_bias;
  std::vector<T> dt;
  std::vector<T> requested_stiffness;
  std::vector<T> external_wrench;
  std::vector<T> desired_wrench;
  std::vector<T> feedforward_wrench;

 private:
  static typename Controller::Config make_config(std::size_t batch) {
    typename Controller::Config config;
    config.batch_size = batch;
    config.dtype = dtype_for<T>();
    config.joint_dim = kBimanual ? 14 : 7;
    config.task_dim = kBimanual ? 12 : 6;
    config.cartesian_frame =
        kBimanual ? CartesianFrame::kMainFrameOrientation : CartesianFrame::kRobotBase;
    config.safety = {SafeOutputPolicy::kHoldLastValid, 2};
    return config;
  }

  static Requirements query(const typename Controller::Config& config) {
    Requirements requirements;
    const Result result = Controller::requirements(config, &requirements);
    if (!result.ok()) throw std::invalid_argument(result.message);
    return requirements;
  }

  static std::vector<T> diagonal(std::size_t batch, std::size_t dimension, T value) {
    std::vector<T> matrix(batch * dimension * dimension, T{0});
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t i = 0; i < dimension; ++i) {
        matrix[(b * dimension + i) * dimension + i] = value;
      }
    }
    return matrix;
  }

  void bind_storage() {
    storage.mass = {buffers.view("mass.cartesian_mass_matrix"),
                    buffers.status("mass.parameter_status"),
                    buffers.status("mass.status"),
                    {buffers.view("mass.joint_mass_factor"),
                     buffers.view("mass.mass_inverse_jacobian_transpose"),
                     buffers.view("mass.cartesian_inverse_mass_factor")}};
    bind_task(buffers, storage.task);
    storage.nullspace = bind_output(buffers, "nullspace.", "joint_torque");
    storage.output.unfiltered_joint_torque = buffers.view("unfiltered_joint_torque");
    static_cast<ControllerOutputStorage&>(storage.output.filter) =
        bind_output(buffers, "", "filtered_command");
    storage.output.filter.state = {buffers.view("previous_command"), buffers.view("initialized")};
    if constexpr (kBimanual) {
      storage.kinematics = {buffers.view("kinematics.task_pose_error"),
                            buffers.view("kinematics.task_jacobian"),
                            buffers.status("kinematics.status")};
      storage.task_velocity_error = buffers.view("task_velocity_error");
    }
  }

  void initialize_parameters() {
    const std::size_t n = config.joint_dim;
    const std::size_t m = config.task_dim;
    const Device cpu = Device::kCpu;
    stiffness_ = diagonal(1, m, T{10});
    joint_stiffness_ = diagonal(1, n, T{1});
    joint_damping_ = diagonal(1, n, T{1});
    damping_.assign(m, T{1});
    lower_.assign(n, T{-100});
    upper_.assign(n, T{100});
    rate_.assign(n, T{1000});
    regularization_ = {T{0.01}};
    parameters.mass = {shared(read_view(regularization_, cpu, 1)), {}};
    parameters.nullspace = {shared(read_view(joint_stiffness_, cpu, n, n)),
                            shared(read_view(joint_damping_, cpu, n, n))};
    parameters.output = {shared(read_view(lower_, cpu, n)),
                         shared(read_view(upper_, cpu, n)),
                         shared(read_view(rate_, cpu, n))};
    if constexpr (kVariable) {
      maximum_stiffness_.assign(m, T{100});
      minimum_stiffness_.assign(m, T{1});
      stiffness_rate_.assign(m, T{10});
      parameters.task = {shared(read_view(stiffness_, cpu, m, m)),
                         shared(read_view(maximum_stiffness_, cpu, m)),
                         shared(read_view(minimum_stiffness_, cpu, m)),
                         shared(read_view(stiffness_rate_, cpu, m)),
                         shared(read_view(damping_, cpu, m))};
    } else {
      parameters.task.stiffness_matrix = shared(read_view(stiffness_, cpu, m, m));
      parameters.task.damping_factors = shared(read_view(damping_, cpu, m));
      if constexpr (kUnified) {
        proportional_.assign(m, T{0.5});
        differentiator_.assign(m, T{0.25});
        integral_.assign(m, T{0.1});
        initial_force_.assign(m, T{0});
        parameters.task.force = {shared(read_view(proportional_, cpu, m)),
                                 shared(read_view(differentiator_, cpu, m)),
                                 shared(read_view(integral_, cpu, m)),
                                 shared(read_view(initial_force_, cpu, m)),
                                 {},
                                 {}};
      }
    }
  }

  void initialize_inputs() {
    const std::size_t b = config.batch_size;
    const std::size_t n = config.joint_dim;
    const std::size_t m = config.task_dim;
    const Device cpu = Device::kCpu;
    joint_mass = diagonal(b, n, T{1});
    jacobian.assign(b * m * n, T{0});
    for (std::size_t row = 0; row < b; ++row) {
      for (std::size_t i = 0; i < m; ++i) jacobian[(row * m + i) * n + i] = T{1};
    }
    pose_error.assign(b * m, T{0.02});
    velocity_error.assign(b * m, T{0.01});
    acceleration.assign(b * m, T{0.03});
    joint_error.assign(b * n, T{0.04});
    joint_velocity.assign(b * n, T{0.01});
    joint_bias.assign(b * n, T{0.1});
    dt.assign(b, T{0.001});
    inputs.desired_acceleration = read_view(acceleration, cpu, b, m);
    inputs.joint_position_error = read_view(joint_error, cpu, b, n);
    inputs.joint_velocity = read_view(joint_velocity, cpu, b, n);
    inputs.joint_mass_matrix = read_view(joint_mass, cpu, b, n, n);
    inputs.joint_bias_torque = read_view(joint_bias, cpu, b, n);
    inputs.dt = read_view(dt, cpu, b);
    if constexpr (kBimanual) {
      inputs.desired_task_velocity = read_view(velocity_error, cpu, b, m);
      initialize_kinematics();
    } else {
      inputs.cartesian_velocity_error = read_view(velocity_error, cpu, b, m);
      inputs.cartesian_pose_error = read_view(pose_error, cpu, b, m);
      inputs.jacobian = read_view(jacobian, cpu, b, m, n);
      if constexpr (kVariable) {
        requested_stiffness = diagonal(b, m, T{11});
        inputs.requested_stiffness_matrix = read_view(requested_stiffness, cpu, b, m, m);
      }
      if constexpr (kUnified) {
        external_wrench.assign(b * m, T{0.2});
        desired_wrench.assign(b * m, T{0.3});
        feedforward_wrench.assign(b * m, T{0.1});
        inputs.external_wrench = read_view(external_wrench, cpu, b, m);
        inputs.desired_wrench = read_view(desired_wrench, cpu, b, m);
        inputs.feedforward_wrench = read_view(feedforward_wrench, cpu, b, m);
      }
    }
  }

  void initialize_kinematics() {
    const std::size_t b = config.batch_size;
    const Device cpu = Device::kCpu;
    pose_ = diagonal(b, 4, T{1});
    desired_pose_ = pose_;
    arm_jacobian_.assign(b * 6 * 7, T{0});
    for (std::size_t row = 0; row < b; ++row) {
      desired_pose_[row * 16 + 3] = T{0.02};
      for (std::size_t i = 0; i < 6; ++i) {
        arm_jacobian_[(row * 6 + i) * 7 + i] = T{1};
      }
    }
    inputs.kinematics = {read_view(desired_pose_, cpu, b, 4, 4),
                         read_view(pose_, cpu, b, 4, 4),
                         read_view(pose_, cpu, b, 4, 4),
                         read_view(desired_pose_, cpu, b, 4, 4),
                         read_view(arm_jacobian_, cpu, b, 6, 7),
                         read_view(arm_jacobian_, cpu, b, 6, 7)};
  }

  std::vector<T> stiffness_, joint_stiffness_, joint_damping_, damping_, lower_, upper_, rate_;
  std::vector<T> regularization_, maximum_stiffness_, minimum_stiffness_, stiffness_rate_;
  std::vector<T> proportional_, differentiator_, integral_, initial_force_;
  std::vector<T> pose_, desired_pose_, arm_jacobian_;
};

}  // namespace nvidia::srl::controllers::test_support
