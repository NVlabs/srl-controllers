/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file controllers.hpp
 * @brief Allocation-free array-view controllers shared by CPU and CUDA callers.
 */

#pragma once

#include <string>

#include "nvidia/srl/controllers/composition.hpp"
#include "nvidia/srl/controllers/types.hpp"
#include "nvidia/srl/controllers/version.hpp"

namespace nvidia {
namespace srl {
namespace controllers {

//! Return the current full version as a string.
NVIDIA_SRL_CONTROLLERS_API std::string get_full_version();

struct ResetInputs {
  TensorView mask{};  ///< Boolean tensor [batch_size].
};

struct InertialStateStorage {
  MutableTensorView previous_mass_matrix{};
  MutableTensorView previous_jacobian{};
  MutableTensorView initialized{};
};

struct AdmittanceStateStorage {
  MutableTensorView pose_offset{};
  MutableTensorView twist{};
  MutableTensorView pose_previous_signal{};
  MutableTensorView twist_previous_signal{};
  MutableTensorView pose_integrator_initialized{};
  MutableTensorView twist_integrator_initialized{};
};

struct TorqueFilterStateStorage {
  MutableTensorView previous_command{};
  MutableTensorView initialized{};
};

struct ForceControllerParameters {
  ParameterTensor proportional_gain{};
  ParameterTensor differentiator_gain{};
  ParameterTensor integral_gain{};
  ParameterTensor initial_output{};
  ParameterTensor maximum_distance{};
  ParameterTensor shaping_distance{};
};

struct ForceControllerStateStorage {
  MutableTensorView differentiator_previous_error{};
  MutableTensorView differentiator_initialized{};
  MutableTensorView integrator_previous_error{};
  MutableTensorView integrated_error{};
  MutableTensorView integrator_initialized{};
};

/** @brief Force-error PID parameters used by the unified force/admittance controller. */
struct ForceErrorPidParameters {
  ParameterTensor proportional_gain{};
  ParameterTensor differentiator_gain{};
  ParameterTensor integral_gain{};
  ParameterTensor initial_integral{};
  ParameterTensor integral_lower_limit{};
  ParameterTensor integral_upper_limit{};
  ParameterTensor error_deadzone{};
};

struct UnifiedForceAdmittanceStateStorage {
  ForceControllerStateStorage force{};
  AdmittanceStateStorage admittance{};
};

struct JointBlockConfig : ControllerConfig {
  ComponentMode mode{ComponentMode::kDense};
};

struct CompositeControllerConfig : ControllerConfig {
  bool use_shaping_function{false};
};

struct UnifiedControllerStateStorage {
  InertialStateStorage impedance{};
  ForceControllerStateStorage force{};
};

struct UnifiedControllerScratchStorage {
  MutableTensorView impedance_wrench{};
  MutableTensorView force_wrench{};
  MutableTensorView impedance_status{};
  MutableTensorView force_status{};
};

struct VariableImpedanceStateStorage {
  MutableTensorView current_stiffness_matrix{};
  MutableTensorView previous_stiffness_matrix{};
  MutableTensorView stiffness_initialized{};
  InertialStateStorage inertial{};
};

struct VariableImpedanceScratchStorage {
  MutableTensorView stiffness_wrench{};
  MutableTensorView damping_wrench{};
  MutableTensorView inertial_wrench{};
  MutableTensorView stiffness_status{};
  MutableTensorView damping_status{};
  MutableTensorView inertial_status{};
};

struct ControllerOutputStorage {
  MutableTensorView command{};
  MutableStatusView parameter_status{};
  MutableStatusView status{};
  SafetyStateView safety{};
};

/**
 * @brief Common allocation-free lifecycle implemented by every controller below.
 *
 * `requirements` describes caller-owned storage without creating a controller. `prepare` validates
 * and retains non-owning views, then applies the complete parameter set. A failed preparation,
 * including a failed re-preparation, leaves the object unprepared so later lifecycle calls fail
 * closed. `reset()` resets every batch row; pass a ResetInputs mask for selective reset.
 *
 * CPU calls execute synchronously. CUDA calls only enqueue work on the stream in ExecutionContext;
 * the controller never creates or synchronizes a stream. Callers must keep all viewed memory alive
 * and correctly ordered until the controller is prepared again or destroyed.
 */

class NVIDIA_SRL_CONTROLLERS_API TorqueFilter {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor lower_limit{};
    ParameterTensor upper_limit{};
    ParameterTensor maximum_rate{};
  };
  struct Inputs {
    TensorView command{};
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    TorqueFilterStateStorage state{};
  };
  struct Outputs {
    MutableTensorView filtered_command{};
    MutableStatusView status{};
  };

  TorqueFilter() = default;
  TorqueFilter(const TorqueFilter&) = delete;
  TorqueFilter& operator=(const TorqueFilter&) = delete;
  TorqueFilter(TorqueFilter&&) = default;
  TorqueFilter& operator=(TorqueFilter&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API JointImpedanceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView joint_position_error{};
    TensorView joint_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_mass_matrix{};
  };
  using Storage = ControllerOutputStorage;
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };

  JointImpedanceController() = default;
  JointImpedanceController(const JointImpedanceController&) = delete;
  JointImpedanceController& operator=(const JointImpedanceController&) = delete;
  JointImpedanceController(JointImpedanceController&&) = default;
  JointImpedanceController& operator=(JointImpedanceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API NullspaceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_matrix{};
  };
  struct Inputs {
    TensorView joint_position_error{};
    TensorView joint_velocity{};
    TensorView joint_mass_matrix{};
    TensorView jacobian{};
  };
  using Storage = ControllerOutputStorage;
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };

  NullspaceController() = default;
  NullspaceController(const NullspaceController&) = delete;
  NullspaceController& operator=(const NullspaceController&) = delete;
  NullspaceController(NullspaceController&&) = default;
  NullspaceController& operator=(NullspaceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API OperationalSpaceMass {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor regularization{};
    ParameterTensor minimum_pivot{};  ///< Optional; an empty view selects the dtype default.
  };
  struct Inputs {
    TensorView joint_mass_matrix{};
    TensorView jacobian{};
  };
  struct Scratch {
    MutableTensorView joint_factor{};
    MutableTensorView solve_rhs{};
    MutableTensorView cartesian_inverse{};
  };
  struct Storage {
    MutableTensorView cartesian_mass_matrix{};
    MutableStatusView parameter_status{};
    MutableStatusView status{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_mass_matrix{};
    MutableStatusView status{};
  };

  OperationalSpaceMass() = default;
  OperationalSpaceMass(const OperationalSpaceMass&) = delete;
  OperationalSpaceMass& operator=(const OperationalSpaceMass&) = delete;
  OperationalSpaceMass(OperationalSpaceMass&&) = default;
  OperationalSpaceMass& operator=(OperationalSpaceMass&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept {
    return Outputs{storage_.cartesian_mass_matrix, storage_.status};
  }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API CartesianImpedanceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    InertialStateStorage state{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };

  CartesianImpedanceController() = default;
  CartesianImpedanceController(const CartesianImpedanceController&) = delete;
  CartesianImpedanceController& operator=(const CartesianImpedanceController&) = delete;
  CartesianImpedanceController(CartesianImpedanceController&&) = default;
  CartesianImpedanceController& operator=(CartesianImpedanceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/**
 * @brief Fixed 12D absolute-plus-relative wrapper over the Cartesian impedance core.
 *
 * The two six-dimensional task blocks use the same full impedance law as
 * CartesianImpedanceController:
 *
 *   wrench = K e + D(Lambda, K, zeta) e_dot
 *            + 0.5 Lambda_dot e_dot - Lambda J_dot q_dot + Lambda x_dd_desired.
 *
 * The first block is the absolute task and the second block is the relative task. Their wrenches
 * are mapped to both arms through the corresponding rows of the bimanual task Jacobian.
 */
class NVIDIA_SRL_CONTROLLERS_API BimanualCartesianImpedanceController {
 public:
  using Config = ControllerConfig;
  using Parameters = CartesianImpedanceController::Parameters;
  using Inputs = CartesianImpedanceController::Inputs;
  using Storage = CartesianImpedanceController::Storage;
  using Outputs = CartesianImpedanceController::Outputs;

  BimanualCartesianImpedanceController() = default;
  BimanualCartesianImpedanceController(const BimanualCartesianImpedanceController&) = delete;
  BimanualCartesianImpedanceController& operator=(const BimanualCartesianImpedanceController&) =
      delete;
  BimanualCartesianImpedanceController(BimanualCartesianImpedanceController&&) = default;
  BimanualCartesianImpedanceController& operator=(BimanualCartesianImpedanceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return controller_.outputs(); }
  bool prepared() const noexcept { return controller_.prepared(); }

 private:
  CartesianImpedanceController controller_{};
};

struct BimanualTaskConfig : ControllerConfig {
  BimanualFrameConfig frames{};
  BimanualCoordinationMode coordination_mode{BimanualCoordinationMode::kDecoupledCooperative};
};

/**
 * @brief Kinematics for one absolute and one relative Cartesian task.
 *
 * Poses are batched row-major homogeneous transforms. Jacobians and task vectors use
 * [linear; angular] ordering. Arm Jacobians are expressed in their own base frames at their TCP
 * origins. The block constructs the absolute midpoint and its Jacobian from the two TCP states.
 * Outputs are rotated into the configured main-frame orientation without moving either task
 * origin. Decoupled mode uses p_abs=(p1+p2)/2 and p_rel=p2-p1. Coupled mode uses the complete
 * moving-frame arm1-TCP-to-arm2-TCP transform and includes its translational lever arm.
 */
class NVIDIA_SRL_CONTROLLERS_API BimanualTaskKinematics {
 public:
  using Config = BimanualTaskConfig;
  struct Inputs {
    TensorView desired_absolute_pose{};  ///< [batch_size,4,4], arm-1 base to target task.
    TensorView arm1_tcp_pose{};          ///< [batch_size,4,4], arm-1 base to arm-1 TCP.
    TensorView arm2_tcp_pose{};          ///< [batch_size,4,4], arm-2 base to arm-2 TCP.
    /**
     * [batch_size,4,4]. Rotation is the arm-1-TCP to arm-2-TCP target. Translation is a
     * shared-base displacement in decoupled mode and the transform translation in coupled mode.
     */
    TensorView desired_relative_pose{};
    TensorView arm1_jacobian{};  ///< [batch_size,6,7], expressed in arm-1 base.
    TensorView arm2_jacobian{};  ///< [batch_size,6,7], expressed in arm-2 base.
  };
  struct Storage {
    MutableTensorView task_pose_error{};  ///< [batch_size,12].
    MutableTensorView task_jacobian{};    ///< [batch_size,12,14].
    MutableStatusView status{};
  };
  struct Outputs {
    MutableTensorView task_pose_error{};
    MutableTensorView task_jacobian{};
    MutableStatusView status{};
  };

  BimanualTaskKinematics() = default;
  BimanualTaskKinematics(const BimanualTaskKinematics&) = delete;
  BimanualTaskKinematics& operator=(const BimanualTaskKinematics&) = delete;
  BimanualTaskKinematics(BimanualTaskKinematics&&) = default;
  BimanualTaskKinematics& operator=(BimanualTaskKinematics&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config, const Storage& storage) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept {
    return Outputs{storage_.task_pose_error, storage_.task_jacobian, storage_.status};
  }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API CartesianAdmittancePoseController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_matrix{};
    ParameterTensor virtual_mass_matrix{};
    ParameterTensor force_filter_threshold{};
    // Empty motion-limit parameters disable their respective clamps.
    ParameterTensor acceleration_limit{};
    ParameterTensor twist_limit{};
    ParameterTensor pose_limit{};
  };
  struct Inputs {
    TensorView external_wrench{};     ///< Environment-on-robot wrench, N and Nm.
    TensorView feedforward_wrench{};  ///< Additional admittance drive, N and Nm.
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    AdmittanceStateStorage state{};
  };
  struct Outputs {
    MutableTensorView commanded_acceleration{};
    MutableTensorView commanded_twist{};
    MutableTensorView commanded_pose_offset{};
    MutableStatusView status{};
  };

  CartesianAdmittancePoseController() = default;
  CartesianAdmittancePoseController(const CartesianAdmittancePoseController&) = delete;
  CartesianAdmittancePoseController& operator=(const CartesianAdmittancePoseController&) = delete;
  CartesianAdmittancePoseController(CartesianAdmittancePoseController&&) = default;
  CartesianAdmittancePoseController& operator=(CartesianAdmittancePoseController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept {
    return Outputs{
        storage_.command, storage_.state.twist, storage_.state.pose_offset, storage_.status};
  }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API OperationalSpaceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor position_gain{};
    ParameterTensor velocity_gain{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView jacobian_dot_joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView joint_bias_torque{};
  };
  using Storage = ControllerOutputStorage;
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };

  OperationalSpaceController() = default;
  OperationalSpaceController(const OperationalSpaceController&) = delete;
  OperationalSpaceController& operator=(const OperationalSpaceController&) = delete;
  OperationalSpaceController(OperationalSpaceController&&) = default;
  OperationalSpaceController& operator=(OperationalSpaceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Task-space inverse dynamics with acceleration-level pose and velocity feedback.
 *
 * OperationalSpaceController remains available for source and binary compatibility.
 */
class NVIDIA_SRL_CONTROLLERS_API OperationalSpaceInverseDynamicsController
    : public OperationalSpaceController {
 public:
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
};

class NVIDIA_SRL_CONTROLLERS_API CartesianForceController {
 public:
  using Config = CompositeControllerConfig;
  using Parameters = ForceControllerParameters;
  struct Inputs {
    TensorView external_wrench{};
    TensorView desired_wrench{};
    TensorView feedforward_wrench{};
    TensorView pose_error{};
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    ForceControllerStateStorage state{};
  };
  struct Outputs {
    MutableTensorView commanded_wrench{};
    MutableStatusView status{};
  };

  CartesianForceController() = default;
  CartesianForceController(const CartesianForceController&) = delete;
  CartesianForceController& operator=(const CartesianForceController&) = delete;
  CartesianForceController(CartesianForceController&&) = default;
  CartesianForceController& operator=(CartesianForceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Static joint impedance with caller-supplied stiffness and damping. */
class NVIDIA_SRL_CONTROLLERS_API StaticJointImpedanceController {
 public:
  using Config = JointBlockConfig;
  struct Parameters {
    ParameterTensor stiffness{};  ///< [joint_dim] or [joint_dim,joint_dim].
    ParameterTensor damping{};    ///< [joint_dim] or [joint_dim,joint_dim].
  };
  struct Inputs {
    TensorView joint_position_error{};  ///< desired - measured, rad.
    TensorView joint_velocity_error{};  ///< desired - measured, rad/s.
  };
  using Storage = ControllerOutputStorage;
  struct Outputs {
    MutableTensorView joint_torque{};  ///< Nm.
    MutableStatusView status{};
  };

  StaticJointImpedanceController() = default;
  StaticJointImpedanceController(const StaticJointImpedanceController&) = delete;
  StaticJointImpedanceController& operator=(const StaticJointImpedanceController&) = delete;
  StaticJointImpedanceController(StaticJointImpedanceController&&) = default;
  StaticJointImpedanceController& operator=(StaticJointImpedanceController&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.command, storage_.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Static 6D Cartesian impedance in the configured Cartesian frame. */
class NVIDIA_SRL_CONTROLLERS_API StaticCartesianImpedanceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};  ///< [6,6] or [batch_size,6,6].
    ParameterTensor damping_matrix{};    ///< [6,6] or [batch_size,6,6].
  };
  struct Inputs {
    TensorView cartesian_pose_error{};      ///< desired - measured, m and rotation-vector rad.
    TensorView cartesian_velocity_error{};  ///< desired - measured, m/s and rad/s.
  };
  using Storage = ControllerOutputStorage;
  struct Outputs {
    MutableTensorView cartesian_wrench{};  ///< N and Nm in Config::cartesian_frame.
    MutableStatusView status{};
  };

  StaticCartesianImpedanceController() = default;
  StaticCartesianImpedanceController(const StaticCartesianImpedanceController&) = delete;
  StaticCartesianImpedanceController& operator=(const StaticCartesianImpedanceController&) = delete;
  StaticCartesianImpedanceController(StaticCartesianImpedanceController&&) = default;
  StaticCartesianImpedanceController& operator=(StaticCartesianImpedanceController&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.command, storage_.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Fused force-error PID and 6D Cartesian admittance controller. */
class NVIDIA_SRL_CONTROLLERS_API CartesianUnifiedForceAdmittanceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_matrix{};
    ParameterTensor virtual_mass_matrix{};
    // Empty motion-limit parameters disable their respective clamps.
    ParameterTensor acceleration_limit{};
    ParameterTensor twist_limit{};
    ParameterTensor pose_limit{};
    ForceErrorPidParameters force{};  ///< Integral bounds may also be empty in this controller.
  };
  struct Inputs {
    TensorView desired_wrench{};      ///< Desired environment-on-robot wrench, N and Nm.
    TensorView external_wrench{};     ///< Measured environment-on-robot wrench, N and Nm.
    TensorView feedforward_wrench{};  ///< N and Nm in Config::cartesian_frame.
    TensorView dt{};                  ///< Positive seconds, scalar or [batch_size].
  };
  struct Storage : ControllerOutputStorage {
    UnifiedForceAdmittanceStateStorage state{};
  };
  struct Outputs {
    MutableTensorView commanded_acceleration{};
    MutableTensorView commanded_twist{};
    MutableTensorView commanded_pose_offset{};
    MutableStatusView status{};
  };

  CartesianUnifiedForceAdmittanceController() = default;
  CartesianUnifiedForceAdmittanceController(const CartesianUnifiedForceAdmittanceController&) =
      delete;
  CartesianUnifiedForceAdmittanceController& operator=(
      const CartesianUnifiedForceAdmittanceController&) = delete;
  CartesianUnifiedForceAdmittanceController(CartesianUnifiedForceAdmittanceController&&) = default;
  CartesianUnifiedForceAdmittanceController& operator=(
      CartesianUnifiedForceAdmittanceController&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept {
    return {storage_.command,
            storage_.state.admittance.twist,
            storage_.state.admittance.pose_offset,
            storage_.status};
  }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API CartesianUnifiedForceImpedanceController {
 public:
  using Config = CompositeControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
    ParameterTensor damping_factors{};
    ForceControllerParameters force{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView external_wrench{};
    TensorView desired_wrench{};
    TensorView feedforward_wrench{};
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    UnifiedControllerStateStorage state{};
    UnifiedControllerScratchStorage scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };

  CartesianUnifiedForceImpedanceController() = default;
  CartesianUnifiedForceImpedanceController(const CartesianUnifiedForceImpedanceController&) =
      delete;
  CartesianUnifiedForceImpedanceController& operator=(
      const CartesianUnifiedForceImpedanceController&) = delete;
  CartesianUnifiedForceImpedanceController(CartesianUnifiedForceImpedanceController&&) = default;
  CartesianUnifiedForceImpedanceController& operator=(CartesianUnifiedForceImpedanceController&&) =
      default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

class NVIDIA_SRL_CONTROLLERS_API VariableCartesianImpedanceController {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor initial_stiffness_matrix{};
    ParameterTensor maximum_stiffness{};
    ParameterTensor minimum_stiffness{};
    ParameterTensor maximum_stiffness_rate{};
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView requested_stiffness_matrix{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView dt{};
  };
  struct Storage : ControllerOutputStorage {
    VariableImpedanceStateStorage state{};
    VariableImpedanceScratchStorage scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };

  VariableCartesianImpedanceController() = default;
  VariableCartesianImpedanceController(const VariableCartesianImpedanceController&) = delete;
  VariableCartesianImpedanceController& operator=(const VariableCartesianImpedanceController&) =
      delete;
  VariableCartesianImpedanceController(VariableCartesianImpedanceController&&) = default;
  VariableCartesianImpedanceController& operator=(VariableCartesianImpedanceController&&) = default;

  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return Outputs{storage_.command, storage_.status}; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** Allocations matching TaskWrenchToJointTorqueController::Storage. */
struct TaskWrenchToJointTorqueRequirements {
  TensorRequirement unfiltered_joint_torque{};
  Requirements filter{};
};

/**
 * @brief Maps a task wrench and optional joint feedback to a filtered joint-torque command.
 *
 * For every batch row this controller computes
 * `tau = joint_bias_torque + joint_feedback_torque + task_jacobian^T * task_wrench`, then applies
 * the configured magnitude and slew-rate filter. Task vectors use `[linear; angular]` ordering;
 * wrenches use `[force_N; torque_Nm]`; all joint-torque inputs and outputs are in Nm. Every input
 * must use the frame declared by `ControllerConfig::cartesian_frame`. An upstream safe-mode status
 * forces this controller's final command through its own safe-output policy.
 */
class NVIDIA_SRL_CONTROLLERS_API TaskWrenchToJointTorqueController {
 public:
  using Config = ControllerConfig;
  using Parameters = TorqueFilter::Parameters;
  struct Inputs {
    TensorView task_wrench{};            ///< [batch_size,task_dim], N then Nm.
    TensorView task_jacobian{};          ///< [batch_size,task_dim,joint_dim].
    TensorView joint_bias_torque{};      ///< [batch_size,joint_dim], Nm.
    TensorView joint_feedback_torque{};  ///< [batch_size,joint_dim], Nm.
    StatusView task_status{};            ///< Upstream task-controller status.
    StatusView feedback_status{};        ///< Upstream joint-feedback status.
    TensorView dt{};                     ///< Scalar or [batch_size], seconds.
  };
  struct Storage {
    MutableTensorView unfiltered_joint_torque{};  ///< [batch_size,joint_dim], Nm.
    TorqueFilter::Storage filter{};
  };
  using Outputs = TorqueFilter::Outputs;

  TaskWrenchToJointTorqueController() = default;
  TaskWrenchToJointTorqueController(const TaskWrenchToJointTorqueController&) = delete;
  TaskWrenchToJointTorqueController& operator=(const TaskWrenchToJointTorqueController&) = delete;
  TaskWrenchToJointTorqueController(TaskWrenchToJointTorqueController&&) = default;
  TaskWrenchToJointTorqueController& operator=(TaskWrenchToJointTorqueController&&) = default;

  using ComponentRequirements = TaskWrenchToJointTorqueRequirements;
  static Result component_requirements(const Config& config,
                                       ComponentRequirements* output) noexcept;
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return filter_.outputs(); }

 private:
  Config config_{};
  Storage storage_{};
  TorqueFilter filter_{};
  bool prepared_{false};
};

/** Complete allocations for each stage of a Cartesian torque pipeline. */
struct CartesianTorqueRequirements {
  Requirements mass{};
  Requirements task{};
  Requirements nullspace{};
  TaskWrenchToJointTorqueRequirements output{};
};
struct BimanualCartesianTorqueRequirements : CartesianTorqueRequirements {
  Requirements kinematics{};
  Requirements velocity{};
};

template <typename TaskParameters>
struct CartesianTorqueParameters {
  OperationalSpaceMass::Parameters mass{};
  TaskParameters task{};
  NullspaceController::Parameters nullspace{};
  TaskWrenchToJointTorqueController::Parameters output{};
};

template <typename TaskStorage>
struct CartesianTorqueStorage {
  OperationalSpaceMass::Storage mass{};
  TaskStorage task{};
  NullspaceController::Storage nullspace{};
  TaskWrenchToJointTorqueController::Storage output{};
};

/** @brief Complete Cartesian impedance to filtered joint-torque controller. */
class NVIDIA_SRL_CONTROLLERS_API CartesianImpedanceTorqueController {
 public:
  using Config = ControllerConfig;
  using Parameters = CartesianTorqueParameters<CartesianImpedanceController::Parameters>;
  struct Inputs {
    TensorView cartesian_pose_error{};      ///< [batch_size,6], m then rotation-vector rad.
    TensorView cartesian_velocity_error{};  ///< [batch_size,6], m/s then rad/s.
    TensorView desired_acceleration{};      ///< [batch_size,6], m/s^2 then rad/s^2.
    TensorView joint_position_error{};      ///< [batch_size,joint_dim], rad.
    TensorView joint_velocity{};            ///< [batch_size,joint_dim], rad/s.
    TensorView joint_mass_matrix{};         ///< [batch_size,joint_dim,joint_dim].
    TensorView jacobian{};                  ///< [batch_size,6,joint_dim], linear then angular.
    TensorView joint_bias_torque{};         ///< [batch_size,joint_dim], Nm.
    TensorView dt{};                        ///< Scalar or [batch_size], seconds.
  };
  using Storage = CartesianTorqueStorage<CartesianImpedanceController::Storage>;
  using Outputs = TaskWrenchToJointTorqueController::Outputs;

  using ComponentRequirements = CartesianTorqueRequirements;
  static Result component_requirements(const Config& config,
                                       ComponentRequirements* output) noexcept;
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return output_.outputs(); }

 private:
  OperationalSpaceMass mass_{};
  CartesianImpedanceController task_{};
  NullspaceController nullspace_{};
  TaskWrenchToJointTorqueController output_{};
  bool prepared_{false};
};

/** @brief Complete variable Cartesian impedance to filtered joint-torque controller. */
class NVIDIA_SRL_CONTROLLERS_API VariableCartesianImpedanceTorqueController {
 public:
  using Config = ControllerConfig;
  using Parameters = CartesianTorqueParameters<VariableCartesianImpedanceController::Parameters>;
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView requested_stiffness_matrix{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_position_error{};
    TensorView joint_velocity{};
    TensorView joint_mass_matrix{};
    TensorView jacobian{};
    TensorView joint_bias_torque{};
    TensorView dt{};
  };
  using Storage = CartesianTorqueStorage<VariableCartesianImpedanceController::Storage>;
  using Outputs = TaskWrenchToJointTorqueController::Outputs;

  using ComponentRequirements = CartesianTorqueRequirements;
  static Result component_requirements(const Config& config,
                                       ComponentRequirements* output) noexcept;
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return output_.outputs(); }

 private:
  OperationalSpaceMass mass_{};
  VariableCartesianImpedanceController task_{};
  NullspaceController nullspace_{};
  TaskWrenchToJointTorqueController output_{};
  bool prepared_{false};
};

/** @brief Complete unified force/impedance to filtered joint-torque controller. */
class NVIDIA_SRL_CONTROLLERS_API CartesianUnifiedForceImpedanceTorqueController {
 public:
  using Config = CompositeControllerConfig;
  using Parameters =
      CartesianTorqueParameters<CartesianUnifiedForceImpedanceController::Parameters>;
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_position_error{};
    TensorView joint_velocity{};
    TensorView joint_mass_matrix{};
    TensorView jacobian{};
    TensorView joint_bias_torque{};
    TensorView external_wrench{};
    TensorView desired_wrench{};
    TensorView feedforward_wrench{};
    TensorView dt{};
  };
  using Storage = CartesianTorqueStorage<CartesianUnifiedForceImpedanceController::Storage>;
  using Outputs = TaskWrenchToJointTorqueController::Outputs;

  using ComponentRequirements = CartesianTorqueRequirements;
  static Result component_requirements(const Config& config,
                                       ComponentRequirements* output) noexcept;
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return output_.outputs(); }

 private:
  OperationalSpaceMass mass_{};
  CartesianUnifiedForceImpedanceController task_{};
  NullspaceController nullspace_{};
  TaskWrenchToJointTorqueController output_{};
  bool prepared_{false};
};

/** @brief Complete bimanual absolute-relative impedance to filtered joint-torque controller. */
class NVIDIA_SRL_CONTROLLERS_API BimanualCartesianImpedanceTorqueController {
 public:
  using Config = BimanualTaskConfig;
  struct Parameters {
    OperationalSpaceMass::Parameters mass{};
    BimanualCartesianImpedanceController::Parameters task{};
    NullspaceController::Parameters nullspace{};
    TaskWrenchToJointTorqueController::Parameters output{};
  };
  struct Inputs {
    BimanualTaskKinematics::Inputs kinematics{};
    TensorView desired_task_velocity{};  ///< [batch_size,12], main-frame task coordinates.
    TensorView desired_acceleration{};   ///< [batch_size,12].
    TensorView joint_position_error{};   ///< [batch_size,14], rad.
    TensorView joint_velocity{};         ///< [batch_size,14], rad/s.
    TensorView joint_mass_matrix{};      ///< [batch_size,14,14].
    TensorView joint_bias_torque{};      ///< [batch_size,14], Nm.
    TensorView dt{};
  };
  struct Storage {
    BimanualTaskKinematics::Storage kinematics{};
    OperationalSpaceMass::Storage mass{};
    BimanualCartesianImpedanceController::Storage task{};
    NullspaceController::Storage nullspace{};
    TaskWrenchToJointTorqueController::Storage output{};
    MutableTensorView task_velocity_error{};  ///< Scratch [batch_size,12].
  };
  using Outputs = TaskWrenchToJointTorqueController::Outputs;

  using ComponentRequirements = BimanualCartesianTorqueRequirements;
  static Result component_requirements(const Config& config,
                                       ComponentRequirements* output) noexcept;
  static Result requirements(const Config& config, Requirements* requirements) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return output_.outputs(); }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  BimanualTaskKinematics kinematics_{};
  OperationalSpaceMass mass_{};
  BimanualCartesianImpedanceController task_{};
  NullspaceController nullspace_{};
  TaskWrenchToJointTorqueController output_{};
  MutableTensorView task_velocity_error_{};
  bool prepared_{false};
};

// BEGIN GENERATED PUBLIC BLOCKS

struct SignalBlockConfig {
  Device device{Device::kCpu};
  DType dtype{DType::kFloat64};
  std::size_t batch_size{0};
  std::size_t signal_dim{0};
  StepOptions safety{};
};

/** @brief Public generic proportional block; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API ProportionalBlock {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {
    ParameterTensor gain{};
  };
  struct Inputs {
    TensorView input_signal{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView output_signal{};
    MutableStatusView status{};
  };
  ProportionalBlock() = default;
  ProportionalBlock(const ProportionalBlock&) = delete;
  ProportionalBlock& operator=(const ProportionalBlock&) = delete;
  ProportionalBlock(ProportionalBlock&&) = default;
  ProportionalBlock& operator=(ProportionalBlock&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public generic integrator block; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API IntegratorBlock {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {
    ParameterTensor initial_output{};
    ParameterTensor lower_limit{};
    ParameterTensor upper_limit{};
  };
  struct Inputs {
    TensorView input_signal{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView previous_signal{};
    MutableTensorView integrated_signal{};
    MutableTensorView initialized{};
  };
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView output_signal{};
    MutableStatusView status{};
  };
  IntegratorBlock() = default;
  IntegratorBlock(const IntegratorBlock&) = delete;
  IntegratorBlock& operator=(const IntegratorBlock&) = delete;
  IntegratorBlock(IntegratorBlock&&) = default;
  IntegratorBlock& operator=(IntegratorBlock&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public generic differentiator block; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API DifferentiatorBlock {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {};
  struct Inputs {
    TensorView input_signal{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView previous_signal{};
    MutableTensorView initialized{};
  };
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView output_signal{};
    MutableStatusView status{};
  };
  DifferentiatorBlock() = default;
  DifferentiatorBlock(const DifferentiatorBlock&) = delete;
  DifferentiatorBlock& operator=(const DifferentiatorBlock&) = delete;
  DifferentiatorBlock(DifferentiatorBlock&&) = default;
  DifferentiatorBlock& operator=(DifferentiatorBlock&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public joint stiffness; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API JointStiffness {
 public:
  using Config = JointBlockConfig;
  struct Parameters {
    ParameterTensor stiffness{};
  };
  struct Inputs {
    TensorView joint_position_error{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };
  JointStiffness() = default;
  JointStiffness(const JointStiffness&) = delete;
  JointStiffness& operator=(const JointStiffness&) = delete;
  JointStiffness(JointStiffness&&) = default;
  JointStiffness& operator=(JointStiffness&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public joint damping; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API JointDamping {
 public:
  using Config = JointBlockConfig;
  struct Parameters {
    ParameterTensor damping{};
  };
  struct Inputs {
    TensorView joint_velocity_error{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };
  JointDamping() = default;
  JointDamping(const JointDamping&) = delete;
  JointDamping& operator=(const JointDamping&) = delete;
  JointDamping(JointDamping&&) = default;
  JointDamping& operator=(JointDamping&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public joint damping double diagonalization; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API JointDampingDoubleDiagonalization {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView joint_velocity_error{};
    TensorView joint_mass_matrix{};
    TensorView stiffness_matrix{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView joint_torque{};
    MutableStatusView status{};
  };
  JointDampingDoubleDiagonalization() = default;
  JointDampingDoubleDiagonalization(const JointDampingDoubleDiagonalization&) = delete;
  JointDampingDoubleDiagonalization& operator=(const JointDampingDoubleDiagonalization&) = delete;
  JointDampingDoubleDiagonalization(JointDampingDoubleDiagonalization&&) = default;
  JointDampingDoubleDiagonalization& operator=(JointDampingDoubleDiagonalization&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public cartesian stiffness; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CartesianStiffness {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor stiffness_matrix{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  CartesianStiffness() = default;
  CartesianStiffness(const CartesianStiffness&) = delete;
  CartesianStiffness& operator=(const CartesianStiffness&) = delete;
  CartesianStiffness(CartesianStiffness&&) = default;
  CartesianStiffness& operator=(CartesianStiffness&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public variable cartesian stiffness; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API VariableCartesianStiffness {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor initial_stiffness_matrix{};
    ParameterTensor maximum_stiffness{};
    ParameterTensor minimum_stiffness{};
    ParameterTensor maximum_stiffness_rate{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView stiffness_matrix{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView current_stiffness_matrix{};
    MutableTensorView previous_stiffness_matrix{};
    MutableTensorView stiffness_initialized{};
  };
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  VariableCartesianStiffness() = default;
  VariableCartesianStiffness(const VariableCartesianStiffness&) = delete;
  VariableCartesianStiffness& operator=(const VariableCartesianStiffness&) = delete;
  VariableCartesianStiffness(VariableCartesianStiffness&&) = default;
  VariableCartesianStiffness& operator=(VariableCartesianStiffness&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public cartesian damping; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CartesianDamping {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor damping_matrix{};
  };
  struct Inputs {
    TensorView cartesian_velocity_error{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  CartesianDamping() = default;
  CartesianDamping(const CartesianDamping&) = delete;
  CartesianDamping& operator=(const CartesianDamping&) = delete;
  CartesianDamping(CartesianDamping&&) = default;
  CartesianDamping& operator=(CartesianDamping&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public cartesian damping double diagonalization; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CartesianDampingDoubleDiagonalization {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView cartesian_velocity_error{};
    TensorView cartesian_mass_matrix{};
    TensorView stiffness_matrix{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  CartesianDampingDoubleDiagonalization() = default;
  CartesianDampingDoubleDiagonalization(const CartesianDampingDoubleDiagonalization&) = delete;
  CartesianDampingDoubleDiagonalization& operator=(const CartesianDampingDoubleDiagonalization&) =
      delete;
  CartesianDampingDoubleDiagonalization(CartesianDampingDoubleDiagonalization&&) = default;
  CartesianDampingDoubleDiagonalization& operator=(CartesianDampingDoubleDiagonalization&&) =
      default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public cartesian damping factorization design; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CartesianDampingFactorizationDesign {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor damping_factors{};
  };
  struct Inputs {
    TensorView cartesian_velocity_error{};
    TensorView cartesian_mass_matrix{};
    TensorView stiffness_matrix{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  CartesianDampingFactorizationDesign() = default;
  CartesianDampingFactorizationDesign(const CartesianDampingFactorizationDesign&) = delete;
  CartesianDampingFactorizationDesign& operator=(const CartesianDampingFactorizationDesign&) =
      delete;
  CartesianDampingFactorizationDesign(CartesianDampingFactorizationDesign&&) = default;
  CartesianDampingFactorizationDesign& operator=(CartesianDampingFactorizationDesign&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public cartesian virtual mass; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CartesianVirtualMass {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor virtual_mass_matrix{};
  };
  struct Inputs {
    TensorView force{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_acceleration{};
    MutableStatusView status{};
  };
  CartesianVirtualMass() = default;
  CartesianVirtualMass(const CartesianVirtualMass&) = delete;
  CartesianVirtualMass& operator=(const CartesianVirtualMass&) = delete;
  CartesianVirtualMass(CartesianVirtualMass&&) = default;
  CartesianVirtualMass& operator=(CartesianVirtualMass&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public shaping function; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API ShapingFunction {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {
    ParameterTensor maximum_distance{};
    ParameterTensor shaping_distance{};
  };
  struct Inputs {
    TensorView error{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView shaping_factor{};
    MutableStatusView status{};
  };
  ShapingFunction() = default;
  ShapingFunction(const ShapingFunction&) = delete;
  ShapingFunction& operator=(const ShapingFunction&) = delete;
  ShapingFunction(ShapingFunction&&) = default;
  ShapingFunction& operator=(ShapingFunction&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public command observer block; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API CommandObserver {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {};
  struct Inputs {
    TensorView command{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    MutableTensorView command_is_valid{};
    MutableStatusView parameter_status{}, status{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView command_is_valid{};
    MutableStatusView status{};
  };
  CommandObserver() = default;
  CommandObserver(const CommandObserver&) = delete;
  CommandObserver& operator=(const CommandObserver&) = delete;
  CommandObserver(CommandObserver&&) = default;
  CommandObserver& operator=(CommandObserver&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.command_is_valid, storage_.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public flow funnel; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API FlowFunnel {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor funnel_radius{};
    ParameterTensor max_distance{};
    ParameterTensor funnel_velocity{};
    ParameterTensor funnel_start_point{};
    ParameterTensor funnel_end_point{};
  };
  struct Inputs {
    TensorView cartesian_pose{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_flow{};
    MutableStatusView status{};
  };
  FlowFunnel() = default;
  FlowFunnel(const FlowFunnel&) = delete;
  FlowFunnel& operator=(const FlowFunnel&) = delete;
  FlowFunnel(FlowFunnel&&) = default;
  FlowFunnel& operator=(FlowFunnel&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public inverse apf; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API InverseAPF {
 public:
  using Config = ControllerConfig;
  struct Parameters {
    ParameterTensor alpha{};
    ParameterTensor gamma{};
    ParameterTensor d_min{};
    ParameterTensor d_max{};
    ParameterTensor deadzone{};
  };
  struct Inputs {
    TensorView current_pose{};
    TensorView goal_pose{};
    TensorView cartesian_twist{};
  };
  struct State {};
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  InverseAPF() = default;
  InverseAPF(const InverseAPF&) = delete;
  InverseAPF& operator=(const InverseAPF&) = delete;
  InverseAPF(InverseAPF&&) = default;
  InverseAPF& operator=(InverseAPF&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public inertial coupling; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API InertialCoupling {
 public:
  using Config = ControllerConfig;
  struct Parameters {};
  struct Inputs {
    TensorView cartesian_velocity_error{};
    TensorView joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView previous_mass_matrix{};
    MutableTensorView previous_jacobian{};
    MutableTensorView initialized{};
  };
  struct Scratch {};
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  InertialCoupling() = default;
  InertialCoupling(const InertialCoupling&) = delete;
  InertialCoupling& operator=(const InertialCoupling&) = delete;
  InertialCoupling(InertialCoupling&&) = default;
  InertialCoupling& operator=(InertialCoupling&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public energy tank; see the block catalog for units. */
class NVIDIA_SRL_CONTROLLERS_API EnergyTank {
 public:
  using Config = SignalBlockConfig;
  struct Parameters {
    ParameterTensor upper_energy_limit{};
    ParameterTensor lower_energy_limit{};
    ParameterTensor initial_energy{};
    ParameterTensor empty_threshold{};
    ParameterTensor power_noise{};
  };
  struct Inputs {
    TensorView power_signal{};
    TensorView additional_power{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView state_variable{};
  };
  struct Scratch {};
  struct Storage {
    MutableTensorView scale{};
    MutableTensorView filtered_power{};
    MutableStatusView parameter_status{}, status{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView scale{};
    MutableTensorView filtered_power{};
    MutableStatusView status{};
  };
  EnergyTank() = default;
  EnergyTank(const EnergyTank&) = delete;
  EnergyTank& operator=(const EnergyTank&) = delete;
  EnergyTank(EnergyTank&&) = default;
  EnergyTank& operator=(EnergyTank&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept {
    return {storage_.scale, storage_.filtered_power, storage_.status};
  }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};

/** @brief Public variable cartesian unified force impedance controller; see the block catalog for
 * units. */
class NVIDIA_SRL_CONTROLLERS_API VariableCartesianUnifiedForceImpedanceController {
 public:
  using Config = CompositeControllerConfig;
  struct Parameters {
    ParameterTensor initial_stiffness_matrix{};
    ParameterTensor maximum_stiffness{};
    ParameterTensor minimum_stiffness{};
    ParameterTensor maximum_stiffness_rate{};
    ParameterTensor damping_factors{};
    ParameterTensor proportional_gain{};
    ParameterTensor differentiator_gain{};
    ParameterTensor integral_gain{};
    ParameterTensor initial_output{};
    ParameterTensor maximum_distance{};
    ParameterTensor shaping_distance{};
  };
  struct Inputs {
    TensorView cartesian_pose_error{};
    TensorView stiffness_matrix{};
    TensorView cartesian_velocity_error{};
    TensorView desired_acceleration{};
    TensorView joint_velocity{};
    TensorView cartesian_mass_matrix{};
    TensorView jacobian{};
    TensorView external_wrench{};
    TensorView desired_wrench{};
    TensorView feedforward_wrench{};
    TensorView dt{};
  };
  struct State {
    MutableTensorView current_stiffness_matrix{};
    MutableTensorView previous_stiffness_matrix{};
    MutableTensorView stiffness_initialized{};
    MutableTensorView previous_mass_matrix{};
    MutableTensorView previous_jacobian{};
    MutableTensorView inertial_initialized{};
    MutableTensorView differentiator_previous_error{};
    MutableTensorView integrator_previous_error{};
    MutableTensorView integrated_error{};
    MutableTensorView differentiator_initialized{};
    MutableTensorView integrator_initialized{};
  };
  struct Scratch {
    MutableTensorView stiffness_wrench{};
    MutableTensorView damping_wrench{};
    MutableTensorView inertial_wrench{};
    MutableTensorView stiffness_status{};
    MutableTensorView damping_status{};
    MutableTensorView inertial_status{};
    MutableTensorView variable_impedance_wrench{};
    MutableTensorView force_wrench{};
    MutableTensorView variable_impedance_status{};
    MutableTensorView force_status{};
  };
  struct Storage {
    ControllerOutputStorage output{};
    State state{};
    Scratch scratch{};
  };
  struct Outputs {
    MutableTensorView cartesian_wrench{};
    MutableStatusView status{};
  };
  VariableCartesianUnifiedForceImpedanceController() = default;
  VariableCartesianUnifiedForceImpedanceController(
      const VariableCartesianUnifiedForceImpedanceController&) = delete;
  VariableCartesianUnifiedForceImpedanceController& operator=(
      const VariableCartesianUnifiedForceImpedanceController&) = delete;
  VariableCartesianUnifiedForceImpedanceController(
      VariableCartesianUnifiedForceImpedanceController&&) = default;
  VariableCartesianUnifiedForceImpedanceController& operator=(
      VariableCartesianUnifiedForceImpedanceController&&) = default;
  static Result requirements(const Config& config, Requirements* output) noexcept;
  Result prepare(const Config& config,
                 const Parameters& parameters,
                 const Storage& storage,
                 ExecutionContext context = {}) noexcept;
  Result apply_parameters(const Parameters& parameters, ExecutionContext context = {}) noexcept;
  Result reset(const ResetInputs& inputs = {}, ExecutionContext context = {}) noexcept;
  Result step(const Inputs& inputs, ExecutionContext context = {}) noexcept;
  Outputs outputs() const noexcept { return {storage_.output.command, storage_.output.status}; }
  bool prepared() const noexcept { return prepared_; }

 private:
  Config config_{};
  Parameters parameters_{};
  Storage storage_{};
  bool prepared_{false};
};
// END GENERATED PUBLIC BLOCKS

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
