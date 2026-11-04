# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Platform-neutral controller API for CPU and CUDA array providers.

The module depends only on Python's array protocols. NumPy, Torch, CuPy, and other
providers remain caller choices; this package does not import any of them. All
state, scratch, output, status, and safety arrays are caller-owned and retained
strongly for the lifetime of a controller.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

# NVIDIA
from nvidia.srl._controllers.nvidia_srl_controllers_cpp import _controller_api as _backend


class Device(str, Enum):
    """Execution device used by every array in one controller instance."""

    CPU = "cpu"
    CUDA = "cuda"


class DType(str, Enum):
    """Floating-point dtype used by controller inputs, parameters, and outputs."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class CartesianFrame(str, Enum):
    """Frame in which Cartesian vectors, matrices, and wrenches are expressed."""

    ROBOT_BASE = "robot_base"
    END_EFFECTOR = "end_effector"
    MAIN_FRAME_ORIENTATION = "main_frame_orientation"


class BimanualBase(str, Enum):
    """Robot base used as the reference for an explicit bimanual main frame."""

    ARM1 = "arm1"
    ARM2 = "arm2"


class BimanualMainFrame(str, Enum):
    """Orientation source for absolute and relative bimanual task coordinates."""

    ARM1_BASE = "arm1_base"
    ARM2_BASE = "arm2_base"
    EXPLICIT = "explicit"


class BimanualCoordinationMode(str, Enum):
    """Coordinate construction used by absolute-relative bimanual tasks."""

    DECOUPLED_COOPERATIVE = "decoupled_cooperative"
    COUPLED_RIGID_TRANSFORM = "coupled_rigid_transform"


class ParameterBatching(str, Enum):
    """Whether a parameter is shared or has one leading row per batch entry."""

    SHARED = "shared"
    PER_BATCH = "per_batch"


class SafeOutputPolicy(str, Enum):
    """Command emitted for a row whose status is unsafe."""

    ZERO = "zero"
    HOLD_LAST_VALID = "hold_last_valid"


@dataclass(frozen=True)
class ExecutionContext:
    """Per-call execution context; ``cuda_stream`` is a CUDA stream pointer."""

    cuda_stream: int | None = None


@dataclass(frozen=True)
class ControllerConfig:
    """Static controller contract shared by the CPU and CUDA implementations."""

    device: Device
    dtype: DType
    batch_size: int
    joint_dim: int = 0
    cartesian_frame: CartesianFrame | None = None
    safe_output_policy: SafeOutputPolicy = SafeOutputPolicy.ZERO
    max_hold_steps: int = 0
    use_shaping_function: bool = False
    task_dim: int = 6

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.joint_dim < 0 or self.joint_dim > 64:
            raise ValueError("joint_dim must be in [0, 64]")
        if self.task_dim <= 0 or self.task_dim > 12:
            raise ValueError("task_dim must be in [1, 12]")
        if self.safe_output_policy is SafeOutputPolicy.HOLD_LAST_VALID and self.max_hold_steps <= 0:
            raise ValueError("hold-last safety requires max_hold_steps > 0")


@dataclass(frozen=True)
class ControllerExampleSpec:
    """Typed hardware/simulation example-pair identity and execution contract."""

    hardware_target: str
    isaac_launcher: str
    robot_family: str
    controller_kind: str
    default_robot_model: str
    supported_robot_models: tuple[str, ...]
    joint_dim: int
    task_dim: int
    cartesian_frame: CartesianFrame | None
    nominal_period_seconds: float
    default_duration_seconds: float
    controlled_force_axis: int | None


def controller_example_specs() -> tuple[ControllerExampleSpec, ...]:
    """Return all canonical real-robot/Isaac Lab example pairs from the C++ registry."""
    frames = {
        "robot_base": CartesianFrame.ROBOT_BASE,
        "end_effector": CartesianFrame.END_EFFECTOR,
        "main_frame_orientation": CartesianFrame.MAIN_FRAME_ORIENTATION,
        "unspecified": None,
    }
    return tuple(
        ControllerExampleSpec(
            hardware_target=item["hardware_target"],
            isaac_launcher=item["isaac_launcher"],
            robot_family=item["robot_family"],
            controller_kind=item["controller_kind"],
            default_robot_model=item["default_robot_model"],
            supported_robot_models=tuple(item["supported_robot_models"].split(",")),
            joint_dim=item["joint_dim"],
            task_dim=item["task_dim"],
            cartesian_frame=frames[item["cartesian_frame"]],
            nominal_period_seconds=item["nominal_period_seconds"],
            default_duration_seconds=item["default_duration_seconds"],
            controlled_force_axis=(
                None if item["controlled_force_axis"] < 0 else item["controlled_force_axis"]
            ),
        )
        for item in _backend.controller_example_specs()
    )


def find_controller_example_spec(hardware_target: str) -> ControllerExampleSpec:
    """Return one canonical example spec, raising ``KeyError`` for unknown targets."""
    for spec in controller_example_specs():
        if spec.hardware_target == hardware_target:
            return spec
    raise KeyError(f"unknown controller example target: {hardware_target}")


_IDENTITY_TRANSFORM = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


@dataclass(frozen=True)
class BimanualFrameConfig:
    """Fixed rigid transforms for bimanual task kinematics.

    Transforms are row-major 4x4 matrices with translation in metres.
    ``arm1_t_arm2`` maps arm-2-base coordinates into the arm-1 base.
    """

    arm1_t_arm2: tuple[float, ...] = _IDENTITY_TRANSFORM
    main_frame: BimanualMainFrame = BimanualMainFrame.ARM1_BASE
    explicit_reference: BimanualBase = BimanualBase.ARM1
    reference_t_main: tuple[float, ...] = _IDENTITY_TRANSFORM

    def __post_init__(self) -> None:
        if len(self.arm1_t_arm2) != 16:
            raise ValueError("arm1_t_arm2 must contain 16 row-major values")
        if len(self.reference_t_main) != 16:
            raise ValueError("reference_t_main must contain 16 row-major values")


@dataclass(frozen=True)
class BimanualTaskConfig(ControllerConfig):
    """Static 12D absolute-plus-relative task-kinematics contract."""

    frames: BimanualFrameConfig = field(default_factory=BimanualFrameConfig)
    coordination_mode: BimanualCoordinationMode = BimanualCoordinationMode.DECOUPLED_COOPERATIVE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.joint_dim != 14:
            raise ValueError("bimanual task kinematics requires joint_dim == 14")
        if self.task_dim != 12:
            raise ValueError("bimanual task kinematics requires task_dim == 12")
        if self.cartesian_frame is not CartesianFrame.MAIN_FRAME_ORIENTATION:
            raise ValueError("bimanual task kinematics requires MAIN_FRAME_ORIENTATION")


@dataclass(frozen=True)
class Parameter:
    """A parameter array with explicit shared or per-batch interpretation."""

    values: Any
    batching: ParameterBatching


def shared(values: Any) -> Parameter:
    """Mark a parameter array as shared by every batch row."""
    return Parameter(values, ParameterBatching.SHARED)


def per_batch(values: Any) -> Parameter:
    """Mark a parameter array as independently batched by row."""
    return Parameter(values, ParameterBatching.PER_BATCH)


@dataclass(frozen=True)
class TorqueFilterParameters:
    """Command bounds and maximum per-second rate."""

    lower_limit: Parameter
    upper_limit: Parameter
    maximum_rate: Parameter


@dataclass(frozen=True)
class JointImpedanceParameters:
    """Joint stiffness matrix and dimensionless damping factors."""

    stiffness_matrix: Parameter
    damping_factors: Parameter


@dataclass(frozen=True)
class NullspaceParameters:
    """Nullspace stiffness and damping matrices."""

    stiffness_matrix: Parameter
    damping_matrix: Parameter


@dataclass(frozen=True)
class OperationalSpaceMassParameters:
    """Operational-space factorization regularization parameters."""

    regularization: Parameter
    minimum_pivot: Parameter | None = None


@dataclass(frozen=True)
class CartesianImpedanceParameters:
    """Cartesian stiffness matrix and dimensionless damping factors."""

    stiffness_matrix: Parameter
    damping_factors: Parameter


@dataclass(frozen=True)
class CartesianAdmittanceParameters:
    """Cartesian admittance gains and filters; use None to disable a motion limit."""

    stiffness_matrix: Parameter
    damping_matrix: Parameter
    virtual_mass_matrix: Parameter
    force_filter_threshold: Parameter
    acceleration_limit: Parameter | None
    twist_limit: Parameter | None
    pose_limit: Parameter | None


@dataclass(frozen=True)
class OperationalSpaceParameters:
    """Operational-space position and velocity gains."""

    position_gain: Parameter
    velocity_gain: Parameter


@dataclass(frozen=True)
class ForceControllerParameters:
    """Cartesian force-feedback gains and optional shaping limits."""

    proportional_gain: Parameter
    differentiator_gain: Parameter
    integral_gain: Parameter
    initial_output: Parameter
    maximum_distance: Parameter
    shaping_distance: Parameter


@dataclass(frozen=True)
class StaticJointImpedanceParameters:
    """Static joint stiffness and manually selected damping."""

    stiffness: Parameter
    damping: Parameter


@dataclass(frozen=True)
class StaticCartesianImpedanceParameters:
    """Static 6D Cartesian stiffness and damping matrices."""

    stiffness_matrix: Parameter
    damping_matrix: Parameter


@dataclass(frozen=True)
class ForceErrorPidParameters:
    """Force-error PID gains and deadzone; integral bounds may be None in force/admittance."""

    proportional_gain: Parameter
    differentiator_gain: Parameter
    integral_gain: Parameter
    initial_integral: Parameter
    integral_lower_limit: Parameter | None
    integral_upper_limit: Parameter | None
    error_deadzone: Parameter


@dataclass(frozen=True)
class CartesianUnifiedForceAdmittanceParameters:
    """Fused force-error PID and admittance; use None to disable a motion limit."""

    stiffness_matrix: Parameter
    damping_matrix: Parameter
    virtual_mass_matrix: Parameter
    acceleration_limit: Parameter | None
    twist_limit: Parameter | None
    pose_limit: Parameter | None
    force: ForceErrorPidParameters


@dataclass(frozen=True)
class CartesianUnifiedForceImpedanceParameters:
    """Cartesian impedance parameters combined with typed force-loop parameters."""

    stiffness_matrix: Parameter
    damping_factors: Parameter
    force: ForceControllerParameters


@dataclass(frozen=True)
class VariableCartesianImpedanceParameters:
    """Variable-stiffness bounds, rates, and damping factors."""

    initial_stiffness_matrix: Parameter
    maximum_stiffness: Parameter
    minimum_stiffness: Parameter
    maximum_stiffness_rate: Parameter
    damping_factors: Parameter


@dataclass(frozen=True)
class SafetyState:
    """Caller-owned bounded hold-last history."""

    last_valid_output: Any
    valid_mask: Any
    hold_count: Any


@dataclass(frozen=True)
class OutputStorage:
    """Caller-owned command, validation status, runtime status, and safety state."""

    command: Any
    parameter_status: Any
    status: Any
    safety: SafetyState


@dataclass(frozen=True)
class InertialState:
    """History used by Cartesian inertial-coupling terms."""

    previous_mass_matrix: Any
    previous_jacobian: Any
    initialized: Any


@dataclass(frozen=True)
class AdmittanceState:
    """Cartesian pose/twist integrator state in the configured frame."""

    pose_offset: Any
    twist: Any
    pose_previous_signal: Any
    twist_previous_signal: Any
    pose_integrator_initialized: Any
    twist_integrator_initialized: Any


@dataclass(frozen=True)
class OperationalSpaceMassScratch:
    """Factorization workspace for ``OperationalSpaceMass``."""

    joint_factor: Any
    solve_rhs: Any
    cartesian_inverse: Any


@dataclass(frozen=True)
class OperationalSpaceMassStorage:
    """Caller-owned output/status/scratch for operational-space mass."""

    cartesian_mass_matrix: Any
    parameter_status: Any
    status: Any
    scratch: OperationalSpaceMassScratch


@dataclass(frozen=True)
class BimanualTaskStorage:
    """Caller-owned outputs for bimanual task kinematics."""

    task_pose_error: Any
    task_jacobian: Any
    status: Any


@dataclass(frozen=True)
class CartesianImpedanceStorage:
    """Caller-owned Cartesian impedance output, state, status, and safety memory."""

    output: OutputStorage
    state: InertialState


@dataclass(frozen=True)
class CartesianAdmittanceStorage:
    """Caller-owned Cartesian admittance output, state, status, and safety memory."""

    output: OutputStorage
    state: AdmittanceState


@dataclass(frozen=True)
class TorqueFilterState:
    """Previous command and initialization mask used for slew limiting."""

    previous_command: Any
    initialized: Any


@dataclass(frozen=True)
class TorqueFilterStorage:
    """Caller-owned torque-filter output, state, status, and safety memory."""

    output: OutputStorage
    state: TorqueFilterState


@dataclass(frozen=True)
class ForceControllerState:
    """Differentiator and integrator history for Cartesian force feedback."""

    differentiator_previous_error: Any
    differentiator_initialized: Any
    integrator_previous_error: Any
    integrated_error: Any
    integrator_initialized: Any


@dataclass(frozen=True)
class CartesianForceStorage:
    """Caller-owned Cartesian force-controller storage."""

    output: OutputStorage
    state: ForceControllerState


@dataclass(frozen=True)
class UnifiedForceAdmittanceState:
    """Force PID and admittance histories committed atomically per batch row."""

    force: ForceControllerState
    admittance: AdmittanceState


@dataclass(frozen=True)
class CartesianUnifiedForceAdmittanceStorage:
    """Caller-owned output and fused force/admittance state."""

    output: OutputStorage
    state: UnifiedForceAdmittanceState


@dataclass(frozen=True)
class UnifiedControllerState:
    """Inertial and force-loop state for unified force/impedance control."""

    impedance: InertialState
    force: ForceControllerState


@dataclass(frozen=True)
class UnifiedControllerScratch:
    """Caller-owned intermediate wrenches and row-local status arrays."""

    impedance_wrench: Any
    force_wrench: Any
    impedance_status: Any
    force_status: Any


@dataclass(frozen=True)
class UnifiedControllerStorage:
    """Caller-owned unified force/impedance storage."""

    output: OutputStorage
    state: UnifiedControllerState
    scratch: UnifiedControllerScratch


@dataclass(frozen=True)
class VariableImpedanceState:
    """Stiffness-rate and inertial history for variable impedance control."""

    current_stiffness_matrix: Any
    previous_stiffness_matrix: Any
    stiffness_initialized: Any
    inertial: InertialState


@dataclass(frozen=True)
class VariableImpedanceScratch:
    """Caller-owned intermediate wrenches and row-local status arrays."""

    stiffness_wrench: Any
    damping_wrench: Any
    inertial_wrench: Any
    stiffness_status: Any
    damping_status: Any
    inertial_status: Any


@dataclass(frozen=True)
class VariableImpedanceStorage:
    """Caller-owned variable Cartesian impedance storage."""

    output: OutputStorage
    state: VariableImpedanceState
    scratch: VariableImpedanceScratch


@dataclass(frozen=True)
class TaskWrenchToJointTorqueStorage:
    """Caller-owned intermediate torque and final torque-filter storage."""

    unfiltered_joint_torque: Any
    filter: TorqueFilterStorage


@dataclass(frozen=True)
class CartesianTorqueParameters:
    """Mass, Cartesian task, nullspace, and final torque-filter parameters."""

    mass: OperationalSpaceMassParameters
    task: Any
    nullspace: NullspaceParameters
    output: TorqueFilterParameters


@dataclass(frozen=True)
class CartesianTorqueStorage:
    """Caller-owned storage for a complete Cartesian torque controller."""

    mass: OperationalSpaceMassStorage
    task: Any
    nullspace: OutputStorage
    output: TaskWrenchToJointTorqueStorage


@dataclass(frozen=True)
class BimanualCartesianImpedanceTorqueStorage:
    """Caller-owned storage for the complete bimanual torque controller."""

    kinematics: BimanualTaskStorage
    mass: OperationalSpaceMassStorage
    task: CartesianImpedanceStorage
    nullspace: OutputStorage
    output: TaskWrenchToJointTorqueStorage
    task_velocity_error: Any


def _storage_arrays(storage: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not is_dataclass(storage):
        return [(prefix, storage)]
    arrays = []
    for entry in fields(storage):
        name = f"{prefix}.{entry.name}" if prefix else entry.name
        arrays.extend(_storage_arrays(getattr(storage, entry.name), name))
    return arrays


def _validate_composite_storage(
    storage: CartesianTorqueStorage | BimanualCartesianImpedanceTorqueStorage,
) -> None:
    offset = int(isinstance(storage, BimanualCartesianImpedanceTorqueStorage))
    stages = {
        "kinematics": 0,
        "mass": offset,
        "task": offset + 1,
        "nullspace": offset + 2,
        "output": offset + 3,
    }
    final = stages["output"]
    entries = []
    for name, array in _storage_arrays(storage):
        if name == "task_velocity_error":
            entries.append((array, 0, 4, False))
            continue
        path = name.split(".")
        stage = stages[path[0]]
        first, last = stage, stage
        if path[-1] == "parameter_status" or "state" in path or "safety" in path:
            first, last = 0, 4
        elif path[0] == "mass" and path[-1] == "cartesian_mass_matrix":
            last = stages["task"]
        elif path[0] == "task" and "scratch" not in path:
            last = final
        elif path[0] == "nullspace" or (path[0] == "kinematics" and path[-1] != "status"):
            last = final
        inplace = name in {"output.unfiltered_joint_torque", "output.filter.output.command"}
        entries.append((array, first, last, inplace))
    _backend.validate_storage_lifetimes(entries)


def _array_info(array: Any) -> Mapping[str, Any]:
    return _backend.describe_array(array)


def _validate_array(
    array: Any,
    config: ControllerConfig,
    shape: tuple[int, ...],
    name: str,
    *,
    dtype: str | None = None,
) -> None:
    info = _array_info(array)
    actual_shape = tuple(info["shape"][: int(info["rank"])])
    expected_dtype = dtype or config.dtype.value
    if info["device"] != config.device.value:
        raise ValueError(f"{name} must be on {config.device.value}, got {info['device']}")
    if info["dtype"] != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {info['dtype']}")
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {actual_shape}")
    if info["layout"] != "row_major":
        raise ValueError(f"{name} must be contiguous row-major")
    if info["read_only"]:
        raise ValueError(f"{name} must be writable")


def _require_parameters(parameters: Any, expected_type: type[Any]) -> None:
    if not isinstance(parameters, expected_type):
        raise TypeError(
            f"parameters must be {expected_type.__name__}, got {type(parameters).__name__}"
        )


def _require_joint_dim(config: ControllerConfig, component: str) -> None:
    if config.joint_dim <= 0:
        raise ValueError(f"{component} requires joint_dim > 0")


def _require_cartesian_frame(config: ControllerConfig, component: str) -> None:
    if config.cartesian_frame is None:
        raise ValueError(f"{component} requires an explicit Cartesian frame")


def _validate_inertial_state(config: ControllerConfig, state: InertialState) -> None:
    batch = config.batch_size
    task_dim = config.task_dim
    _validate_array(
        state.previous_mass_matrix,
        config,
        (batch, task_dim, task_dim),
        "state.previous_mass_matrix",
    )
    _validate_array(
        state.previous_jacobian,
        config,
        (batch, task_dim, config.joint_dim),
        "state.previous_jacobian",
    )
    _validate_array(
        state.initialized,
        config,
        (batch,),
        "state.initialized",
        dtype="bool",
    )


def _validate_force_state(config: ControllerConfig, state: ForceControllerState) -> None:
    batch = config.batch_size
    for name in (
        "differentiator_previous_error",
        "integrator_previous_error",
        "integrated_error",
    ):
        _validate_array(getattr(state, name), config, (batch, 6), f"state.{name}")
    for name in ("differentiator_initialized", "integrator_initialized"):
        _validate_array(getattr(state, name), config, (batch,), f"state.{name}", dtype="bool")


class _ControllerCommand:
    def __init__(self, config: ControllerConfig, storage: OutputStorage, output_dim: int) -> None:
        self.config = config
        self._output = storage
        _validate_array(storage.command, config, (config.batch_size, output_dim), "command")
        _validate_array(
            storage.parameter_status,
            config,
            (config.batch_size,),
            "parameter_status",
            dtype="uint64",
        )
        _validate_array(storage.status, config, (config.batch_size,), "status", dtype="uint64")
        _validate_array(
            storage.safety.valid_mask,
            config,
            (config.batch_size,),
            "safety.valid_mask",
            dtype="bool",
        )
        _validate_array(
            storage.safety.hold_count,
            config,
            (config.batch_size,),
            "safety.hold_count",
            dtype="uint64",
        )
        _validate_array(
            storage.safety.last_valid_output,
            config,
            (config.batch_size, output_dim),
            "safety.last_valid_output",
        )
        if _backend.arrays_overlap(storage.command, storage.safety.last_valid_output):
            raise ValueError("safety.last_valid_output must not alias command")

    @property
    def output(self) -> Any:
        """Caller-owned command array populated by step."""
        return self._output.command

    @property
    def status(self) -> Any:
        """Caller-owned status array from the latest lifecycle call."""
        return self._output.status


class TorqueFilter(_ControllerCommand):
    """Torque limiter with explicit bounds and per-step slew limits."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("TorqueFilter", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: TorqueFilterParameters,
        storage: TorqueFilterStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the torque filter."""
        _require_joint_dim(config, "torque filter")
        _require_parameters(parameters, TorqueFilterParameters)
        super().__init__(config, storage.output, config.joint_dim)
        self.storage = storage
        _validate_array(
            storage.state.previous_command,
            config,
            (config.batch_size, config.joint_dim),
            "state.previous_command",
        )
        _validate_array(
            storage.state.initialized,
            config,
            (config.batch_size,),
            "state.initialized",
            dtype="bool",
        )
        self._controller = _backend.TorqueFilter(config, parameters, storage, context)
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: TorqueFilterParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply torque-filter parameters."""
        _require_parameters(parameters, TorqueFilterParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(self, command: Any, dt: Any, context: ExecutionContext | None = None) -> None:
        """Apply bounds and slew limits into caller-owned output storage."""
        self._controller.step(command, dt, context)


class JointImpedanceController(_ControllerCommand):
    """Joint impedance controller for one or many CPU/CUDA rows."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("JointImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: JointImpedanceParameters,
        storage: OutputStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the joint impedance controller."""
        _require_joint_dim(config, "joint impedance")
        _require_parameters(parameters, JointImpedanceParameters)
        super().__init__(config, storage, config.joint_dim)
        self._controller = _backend.JointImpedanceController(config, parameters, storage, context)
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: JointImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply joint impedance parameters."""
        _require_parameters(parameters, JointImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        joint_position_error: Any,
        joint_velocity_error: Any,
        desired_acceleration: Any,
        joint_mass_matrix: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute joint commands into caller-owned output storage."""
        self._controller.step(
            joint_position_error,
            joint_velocity_error,
            desired_acceleration,
            joint_mass_matrix,
            context,
        )


class NullspaceController(_ControllerCommand):
    """Dynamically consistent nullspace controller."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("NullspaceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: NullspaceParameters,
        storage: OutputStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the nullspace controller."""
        _require_joint_dim(config, "nullspace controller")
        _require_parameters(parameters, NullspaceParameters)
        super().__init__(config, storage, config.joint_dim)
        self._controller = _backend.NullspaceController(config, parameters, storage, context)
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: NullspaceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply nullspace parameters."""
        _require_parameters(parameters, NullspaceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        jacobian: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute nullspace joint commands into caller-owned output storage."""
        self._controller.step(
            joint_position_error, joint_velocity, joint_mass_matrix, jacobian, context
        )


class OperationalSpaceMass:
    """Operational-space mass component."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("OperationalSpaceMass", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: OperationalSpaceMassParameters,
        storage: OperationalSpaceMassStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the operational-space mass component."""
        _require_joint_dim(config, "operational-space mass")
        _require_cartesian_frame(config, "operational-space mass")
        _require_parameters(parameters, OperationalSpaceMassParameters)
        self.config = config
        self.storage = storage
        batch = config.batch_size
        joints = config.joint_dim
        task_dim = config.task_dim
        _validate_array(
            storage.cartesian_mass_matrix,
            config,
            (batch, task_dim, task_dim),
            "cartesian_mass_matrix",
        )
        _validate_array(
            storage.parameter_status, config, (batch,), "parameter_status", dtype="uint64"
        )
        _validate_array(storage.status, config, (batch,), "status", dtype="uint64")
        _validate_array(
            storage.scratch.joint_factor, config, (batch, joints, joints), "joint_factor"
        )
        _validate_array(storage.scratch.solve_rhs, config, (batch, joints, task_dim), "solve_rhs")
        _validate_array(
            storage.scratch.cartesian_inverse,
            config,
            (batch, task_dim, task_dim),
            "cartesian_inverse",
        )
        self._controller = _backend.OperationalSpaceMass(config, parameters, storage, context)
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: OperationalSpaceMassParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply operational-space mass parameters."""
        _require_parameters(parameters, OperationalSpaceMassParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    @property
    def output(self) -> Any:
        """Caller-owned Cartesian mass matrix populated by step."""
        return self.storage.cartesian_mass_matrix

    @property
    def status(self) -> Any:
        """Caller-owned status array from the latest lifecycle call."""
        return self.storage.status

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        joint_mass_matrix: Any,
        jacobian: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute Cartesian mass matrices into caller-owned output storage."""
        self._controller.step(joint_mass_matrix, jacobian, context)


class CartesianImpedanceController(_ControllerCommand):
    """Cartesian impedance controller with caller-owned inertial state."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianImpedanceParameters,
        storage: CartesianImpedanceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the Cartesian impedance controller."""
        _require_joint_dim(config, "Cartesian impedance")
        _require_cartesian_frame(config, "Cartesian impedance")
        _require_parameters(parameters, CartesianImpedanceParameters)
        super().__init__(config, storage.output, config.task_dim)
        self.storage = storage
        _validate_inertial_state(config, storage.state)
        self._controller = _backend.CartesianImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: CartesianImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply Cartesian impedance parameters."""
        _require_parameters(parameters, CartesianImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_velocity: Any,
        cartesian_mass_matrix: Any,
        jacobian: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute Cartesian impedance wrench into caller-owned output storage."""
        self._controller.step(
            cartesian_pose_error,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            cartesian_mass_matrix,
            jacobian,
            dt,
            context,
        )


class BimanualCartesianImpedanceController(_ControllerCommand):
    """Dedicated 12D absolute-plus-relative impedance controller for two seven-joint arms."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("BimanualCartesianImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianImpedanceParameters,
        storage: CartesianImpedanceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate the fixed bimanual contract and prepare its controller."""
        if config.joint_dim != 14:
            raise ValueError("bimanual Cartesian impedance requires joint_dim == 14")
        if config.task_dim != 12:
            raise ValueError("bimanual Cartesian impedance requires task_dim == 12")
        if config.cartesian_frame is not CartesianFrame.MAIN_FRAME_ORIENTATION:
            raise ValueError("bimanual Cartesian impedance requires MAIN_FRAME_ORIENTATION")
        _require_parameters(parameters, CartesianImpedanceParameters)
        super().__init__(config, storage.output, config.task_dim)
        self.storage = storage
        _validate_inertial_state(config, storage.state)
        self._controller = _backend.BimanualCartesianImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: CartesianImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply the coupled 12D impedance parameters."""
        _require_parameters(parameters, CartesianImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        task_pose_error: Any,
        task_velocity_error: Any,
        desired_acceleration: Any,
        joint_velocity: Any,
        task_mass_matrix: Any,
        task_jacobian: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute the coupled absolute-plus-relative 12D wrench."""
        self._controller.step(
            task_pose_error,
            task_velocity_error,
            desired_acceleration,
            joint_velocity,
            task_mass_matrix,
            task_jacobian,
            dt,
            context,
        )


class BimanualTaskKinematics:
    """Absolute-plus-relative pose error and task-Jacobian computation."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("BimanualTaskKinematics", config)

    def __init__(
        self,
        config: BimanualTaskConfig,
        storage: BimanualTaskStorage,
    ) -> None:
        """Validate caller-owned outputs and prepare the bimanual kinematics block."""
        if not isinstance(config, BimanualTaskConfig):
            raise TypeError(f"config must be BimanualTaskConfig, got {type(config).__name__}")
        self.config = config
        self.storage = storage
        _validate_array(
            storage.task_pose_error,
            config,
            (config.batch_size, config.task_dim),
            "task_pose_error",
        )
        _validate_array(
            storage.task_jacobian,
            config,
            (config.batch_size, config.task_dim, config.joint_dim),
            "task_jacobian",
        )
        _validate_array(
            storage.status,
            config,
            (config.batch_size,),
            "status",
            dtype="uint64",
        )
        self._kinematics = _backend.BimanualTaskKinematics(config, storage)

    @property
    def pose_error(self) -> Any:
        """Caller-owned 12D pose-error array populated by ``step``."""
        return self.storage.task_pose_error

    @property
    def jacobian(self) -> Any:
        """Caller-owned 12x14 task-Jacobian array populated by ``step``."""
        return self.storage.task_jacobian

    @property
    def status(self) -> Any:
        """Caller-owned status array from the latest lifecycle call."""
        return self.storage.status

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._kinematics.reset(reset_mask, context)

    def step(
        self,
        desired_absolute_pose: Any,
        arm1_tcp_pose: Any,
        arm2_tcp_pose: Any,
        desired_relative_pose: Any,
        arm1_jacobian: Any,
        arm2_jacobian: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute direct task errors and Jacobians in main-frame orientation."""
        self._kinematics.step(
            desired_absolute_pose,
            arm1_tcp_pose,
            arm2_tcp_pose,
            desired_relative_pose,
            arm1_jacobian,
            arm2_jacobian,
            context,
        )


class CartesianAdmittancePoseController(_ControllerCommand):
    """Cartesian admittance controller in an explicit Cartesian frame."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianAdmittancePoseController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianAdmittanceParameters,
        storage: CartesianAdmittanceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the Cartesian admittance controller."""
        _require_cartesian_frame(config, "Cartesian admittance")
        _require_parameters(parameters, CartesianAdmittanceParameters)
        super().__init__(config, storage.output, 6)
        self.storage = storage
        batch = config.batch_size
        for name in (
            "pose_offset",
            "twist",
            "pose_previous_signal",
            "twist_previous_signal",
        ):
            _validate_array(getattr(storage.state, name), config, (batch, 6), name)
        for name in ("pose_integrator_initialized", "twist_integrator_initialized"):
            _validate_array(getattr(storage.state, name), config, (batch,), name, dtype="bool")
        self._controller = _backend.CartesianAdmittancePoseController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: CartesianAdmittanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply Cartesian admittance parameters."""
        _require_parameters(parameters, CartesianAdmittanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    @property
    def pose_offset(self) -> Any:
        """Caller-owned Cartesian pose offset populated by step."""
        return self.storage.state.pose_offset

    @property
    def twist(self) -> Any:
        """Caller-owned Cartesian twist populated by step."""
        return self.storage.state.twist

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        external_wrench: Any,
        feedforward_wrench: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Integrate admittance state into caller-owned output storage."""
        self._controller.step(external_wrench, feedforward_wrench, dt, context)


class OperationalSpaceInverseDynamicsController(_ControllerCommand):
    """Torque-level task-space inverse dynamics with acceleration feedback."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("OperationalSpaceInverseDynamicsController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: OperationalSpaceParameters,
        storage: OutputStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the inverse-dynamics controller."""
        _require_joint_dim(config, "operational-space inverse-dynamics controller")
        _require_cartesian_frame(config, "operational-space inverse-dynamics controller")
        _require_parameters(parameters, OperationalSpaceParameters)
        super().__init__(config, storage, config.joint_dim)
        self._controller = _backend.OperationalSpaceInverseDynamicsController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: OperationalSpaceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply operational-space parameters."""
        _require_parameters(parameters, OperationalSpaceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        jacobian_dot_joint_velocity: Any,
        cartesian_mass_matrix: Any,
        jacobian: Any,
        joint_bias_torque: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute bias-compensated operational-space joint torque."""
        self._controller.step(
            cartesian_pose_error,
            cartesian_velocity_error,
            desired_acceleration,
            jacobian_dot_joint_velocity,
            cartesian_mass_matrix,
            jacobian,
            joint_bias_torque,
            context,
        )


OperationalSpaceController = OperationalSpaceInverseDynamicsController


class CartesianForceController(_ControllerCommand):
    """Cartesian force-feedback controller."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianForceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: ForceControllerParameters,
        storage: CartesianForceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the Cartesian force controller."""
        _require_cartesian_frame(config, "Cartesian force controller")
        _require_parameters(parameters, ForceControllerParameters)
        super().__init__(config, storage.output, 6)
        self.storage = storage
        _validate_force_state(config, storage.state)
        self._controller = _backend.CartesianForceController(config, parameters, storage, context)
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: ForceControllerParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply Cartesian force parameters."""
        _require_parameters(parameters, ForceControllerParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        external_wrench: Any,
        desired_wrench: Any,
        feedforward_wrench: Any,
        pose_error: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute force-feedback wrench into caller-owned output storage."""
        self._controller.step(
            external_wrench, desired_wrench, feedforward_wrench, pose_error, dt, context
        )


class StaticJointImpedanceController(_ControllerCommand):
    """Static joint impedance using position and desired-velocity errors."""

    @staticmethod
    def requirements(config: Any) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("StaticJointImpedanceController", config)

    def __init__(
        self,
        config: Any,
        parameters: StaticJointImpedanceParameters,
        storage: OutputStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Prepare diagonal or dense static joint impedance."""
        _require_joint_dim(config, "static joint impedance controller")
        _require_parameters(parameters, StaticJointImpedanceParameters)
        super().__init__(config, storage, config.joint_dim)
        self._controller = _backend.StaticJointImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: StaticJointImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Apply the complete static stiffness and damping set."""
        _require_parameters(parameters, StaticJointImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected safety rows."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        joint_position_error: Any,
        joint_velocity_error: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute ``K*(q_des-q) + D*(dq_des-dq)`` in Nm."""
        self._controller.step(joint_position_error, joint_velocity_error, context)


class StaticCartesianImpedanceController(_ControllerCommand):
    """Static 6D Cartesian impedance in the configured Cartesian frame."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("StaticCartesianImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: StaticCartesianImpedanceParameters,
        storage: OutputStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Prepare static Cartesian stiffness and manual damping."""
        _require_cartesian_frame(config, "static Cartesian impedance controller")
        _require_parameters(parameters, StaticCartesianImpedanceParameters)
        super().__init__(config, storage, 6)
        self._controller = _backend.StaticCartesianImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: StaticCartesianImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Apply the complete static stiffness and damping set."""
        _require_parameters(parameters, StaticCartesianImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected safety rows."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute static stiffness and damping wrench in the configured frame."""
        self._controller.step(cartesian_pose_error, cartesian_velocity_error, context)


class CartesianUnifiedForceAdmittanceController(_ControllerCommand):
    """Fused force-error PID and Cartesian admittance controller."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned state and output storage."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianUnifiedForceAdmittanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianUnifiedForceAdmittanceParameters,
        storage: CartesianUnifiedForceAdmittanceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Prepare fused force/admittance control in an explicit Cartesian frame."""
        _require_cartesian_frame(config, "unified force/admittance controller")
        _require_parameters(parameters, CartesianUnifiedForceAdmittanceParameters)
        super().__init__(config, storage.output, 6)
        self.storage = storage
        _validate_force_state(config, storage.state.force)
        batch = config.batch_size
        for name in ("pose_offset", "twist", "pose_previous_signal", "twist_previous_signal"):
            _validate_array(getattr(storage.state.admittance, name), config, (batch, 6), name)
        for name in ("pose_integrator_initialized", "twist_integrator_initialized"):
            _validate_array(
                getattr(storage.state.admittance, name), config, (batch,), name, dtype="bool"
            )
        self._controller = _backend.CartesianUnifiedForceAdmittanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: CartesianUnifiedForceAdmittanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Apply the complete force/admittance parameter set."""
        _require_parameters(parameters, CartesianUnifiedForceAdmittanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    @property
    def twist(self) -> Any:
        """Integrated commanded Cartesian twist."""
        return self.storage.state.admittance.twist

    @property
    def pose_offset(self) -> Any:
        """Integrated commanded Cartesian pose offset."""
        return self.storage.state.admittance.pose_offset

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected force and admittance state rows."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        desired_wrench: Any,
        external_wrench: Any,
        feedforward_wrench: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute acceleration and atomically commit the full state for successful rows."""
        self._controller.step(desired_wrench, external_wrench, feedforward_wrench, dt, context)


class CartesianUnifiedForceImpedanceController(_ControllerCommand):
    """Sum of Cartesian impedance and Cartesian force feedback."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianUnifiedForceImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianUnifiedForceImpedanceParameters,
        storage: UnifiedControllerStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the unified Cartesian controller."""
        _require_joint_dim(config, "unified force/impedance controller")
        _require_cartesian_frame(config, "unified force/impedance controller")
        _require_parameters(parameters, CartesianUnifiedForceImpedanceParameters)
        super().__init__(config, storage.output, 6)
        self.storage = storage
        _validate_inertial_state(config, storage.state.impedance)
        _validate_force_state(config, storage.state.force)
        for name in ("impedance_wrench", "force_wrench"):
            _validate_array(
                getattr(storage.scratch, name),
                config,
                (config.batch_size, 6),
                f"scratch.{name}",
            )
        for name in ("impedance_status", "force_status"):
            _validate_array(
                getattr(storage.scratch, name),
                config,
                (config.batch_size,),
                f"scratch.{name}",
                dtype="uint64",
            )
        self._controller = _backend.CartesianUnifiedForceImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: CartesianUnifiedForceImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply unified force-impedance parameters."""
        _require_parameters(parameters, CartesianUnifiedForceImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_velocity: Any,
        cartesian_mass_matrix: Any,
        jacobian: Any,
        external_wrench: Any,
        desired_wrench: Any,
        feedforward_wrench: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute the unified Cartesian wrench into caller-owned output storage."""
        self._controller.step(
            cartesian_pose_error,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            cartesian_mass_matrix,
            jacobian,
            external_wrench,
            desired_wrench,
            feedforward_wrench,
            dt,
            context,
        )


class VariableCartesianImpedanceController(_ControllerCommand):
    """Cartesian impedance controller with rate-limited stiffness."""

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("VariableCartesianImpedanceController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: VariableCartesianImpedanceParameters,
        storage: VariableImpedanceStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate storage and prepare the variable impedance controller."""
        _require_joint_dim(config, "variable Cartesian impedance")
        _require_cartesian_frame(config, "variable Cartesian impedance")
        _require_parameters(parameters, VariableCartesianImpedanceParameters)
        super().__init__(config, storage.output, 6)
        self.storage = storage
        batch = config.batch_size
        _validate_array(
            storage.state.current_stiffness_matrix,
            config,
            (batch, 6, 6),
            "state.current_stiffness_matrix",
        )
        _validate_array(
            storage.state.previous_stiffness_matrix,
            config,
            (batch, 6, 6),
            "state.previous_stiffness_matrix",
        )
        _validate_array(
            storage.state.stiffness_initialized,
            config,
            (batch,),
            "state.stiffness_initialized",
            dtype="bool",
        )
        _validate_inertial_state(config, storage.state.inertial)
        for name in ("stiffness_wrench", "damping_wrench", "inertial_wrench"):
            _validate_array(getattr(storage.scratch, name), config, (batch, 6), f"scratch.{name}")
        for name in ("stiffness_status", "damping_status", "inertial_status"):
            _validate_array(
                getattr(storage.scratch, name),
                config,
                (batch,),
                f"scratch.{name}",
                dtype="uint64",
            )
        self._controller = _backend.VariableCartesianImpedanceController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: VariableCartesianImpedanceParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply variable impedance parameters."""
        _require_parameters(parameters, VariableCartesianImpedanceParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        requested_stiffness_matrix: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_velocity: Any,
        cartesian_mass_matrix: Any,
        jacobian: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute the rate-limited impedance wrench into caller-owned output storage."""
        self._controller.step(
            cartesian_pose_error,
            requested_stiffness_matrix,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            cartesian_mass_matrix,
            jacobian,
            dt,
            context,
        )


class TaskWrenchToJointTorqueController(_ControllerCommand):
    """Map a task wrench and joint feedback to a filtered joint-torque command."""

    @staticmethod
    def component_requirements(config: ControllerConfig) -> Any:
        """Describe raw command and nested filter storage."""
        # Local Folder
        from ._metadata import describe_torque_output

        return describe_torque_output(config)

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("TaskWrenchToJointTorqueController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: TorqueFilterParameters,
        storage: TaskWrenchToJointTorqueStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate caller-owned storage and configure the final torque filter."""
        _require_joint_dim(config, "task-wrench to joint-torque controller")
        _require_parameters(parameters, TorqueFilterParameters)
        super().__init__(config, storage.filter.output, config.joint_dim)
        _validate_array(
            storage.unfiltered_joint_torque,
            config,
            (config.batch_size, config.joint_dim),
            "unfiltered_joint_torque",
        )
        self.storage = storage
        self._controller = _backend.TaskWrenchToJointTorqueController(
            config, parameters, storage, context
        )
        self.parameters = parameters

    def apply_parameters(
        self,
        parameters: TorqueFilterParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and apply final torque magnitude and slew-rate limits."""
        _require_parameters(parameters, TorqueFilterParameters)
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        task_wrench: Any,
        task_jacobian: Any,
        joint_bias_torque: Any,
        joint_feedback_torque: Any,
        task_status: Any,
        feedback_status: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute filtered joint torque in Nm from task and joint-space inputs."""
        self._controller.step(
            task_wrench,
            task_jacobian,
            joint_bias_torque,
            joint_feedback_torque,
            task_status,
            feedback_status,
            dt,
            context,
        )


class _CartesianTorqueController:
    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianTorqueParameters,
        storage: CartesianTorqueStorage,
        task: Any,
        context: ExecutionContext | None,
    ) -> None:
        self.config = config
        self.parameters = parameters
        self.storage = storage
        self.mass = OperationalSpaceMass(config, parameters.mass, storage.mass, context)
        self.task = task
        self.nullspace = NullspaceController(
            config, parameters.nullspace, storage.nullspace, context
        )
        self.output_controller = TaskWrenchToJointTorqueController(
            config, parameters.output, storage.output, context
        )

    @property
    def output(self) -> Any:
        """Caller-owned final filtered joint torque in Nm."""
        return self.output_controller.output

    @property
    def status(self) -> Any:
        """Caller-owned final controller status."""
        return self.output_controller.status

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        self.mass.reset(reset_mask, context)
        self.task.reset(reset_mask, context)
        self.nullspace.reset(reset_mask, context)
        self.output_controller.reset(reset_mask, context)

    def _finish_step(
        self,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        jacobian: Any,
        joint_bias_torque: Any,
        dt: Any,
        context: ExecutionContext | None,
    ) -> None:
        self.nullspace.step(
            joint_position_error, joint_velocity, joint_mass_matrix, jacobian, context
        )
        self.output_controller.step(
            self.task.output,
            jacobian,
            joint_bias_torque,
            self.nullspace.output,
            self.task.status,
            self.nullspace.status,
            dt,
            context,
        )


class CartesianImpedanceTorqueController(_CartesianTorqueController):
    """Complete Cartesian impedance, nullspace, bias, and torque-filter controller."""

    @staticmethod
    def component_requirements(config: ControllerConfig) -> Any:
        """Describe storage for every pipeline stage before allocation."""
        # Local Folder
        from ._metadata import describe_components

        return describe_components("CartesianImpedanceController", config, bimanual=False)

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianImpedanceTorqueController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianTorqueParameters,
        storage: CartesianTorqueStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and bind the complete Cartesian torque pipeline."""
        _validate_composite_storage(storage)
        task = CartesianImpedanceController(config, parameters.task, storage.task, context)
        super().__init__(config, parameters, storage, task, context)

    def apply_parameters(
        self, parameters: CartesianTorqueParameters, context: ExecutionContext | None = None
    ) -> None:
        """Apply parameters to every controller stage."""
        self.mass.apply_parameters(parameters.mass, context)
        self.task.apply_parameters(parameters.task, context)
        self.nullspace.apply_parameters(parameters.nullspace, context)
        self.output_controller.apply_parameters(parameters.output, context)
        self.parameters = parameters

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        jacobian: Any,
        joint_bias_torque: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute final filtered joint torque in Nm for one control period."""
        self.mass.step(joint_mass_matrix, jacobian, context)
        self.task.step(
            cartesian_pose_error,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            self.mass.output,
            jacobian,
            dt,
            context,
        )
        self._finish_step(
            joint_position_error,
            joint_velocity,
            joint_mass_matrix,
            jacobian,
            joint_bias_torque,
            dt,
            context,
        )


class VariableCartesianImpedanceTorqueController(_CartesianTorqueController):
    """Complete variable Cartesian impedance to filtered joint torque."""

    @staticmethod
    def component_requirements(config: ControllerConfig) -> Any:
        """Describe storage for every pipeline stage before allocation."""
        # Local Folder
        from ._metadata import describe_components

        return describe_components("VariableCartesianImpedanceController", config, bimanual=False)

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("VariableCartesianImpedanceTorqueController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianTorqueParameters,
        storage: CartesianTorqueStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and bind the variable-impedance torque pipeline."""
        _validate_composite_storage(storage)
        task = VariableCartesianImpedanceController(config, parameters.task, storage.task, context)
        super().__init__(config, parameters, storage, task, context)

    def apply_parameters(
        self, parameters: CartesianTorqueParameters, context: ExecutionContext | None = None
    ) -> None:
        """Apply parameters to every controller stage."""
        self.mass.apply_parameters(parameters.mass, context)
        self.task.apply_parameters(parameters.task, context)
        self.nullspace.apply_parameters(parameters.nullspace, context)
        self.output_controller.apply_parameters(parameters.output, context)
        self.parameters = parameters

    def step(
        self,
        cartesian_pose_error: Any,
        requested_stiffness_matrix: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        jacobian: Any,
        joint_bias_torque: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute final filtered joint torque in Nm for one control period."""
        self.mass.step(joint_mass_matrix, jacobian, context)
        self.task.step(
            cartesian_pose_error,
            requested_stiffness_matrix,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            self.mass.output,
            jacobian,
            dt,
            context,
        )
        self._finish_step(
            joint_position_error,
            joint_velocity,
            joint_mass_matrix,
            jacobian,
            joint_bias_torque,
            dt,
            context,
        )


class CartesianUnifiedForceImpedanceTorqueController(_CartesianTorqueController):
    """Complete unified force/impedance to filtered joint torque."""

    @staticmethod
    def component_requirements(config: ControllerConfig) -> Any:
        """Describe storage for every pipeline stage before allocation."""
        # Local Folder
        from ._metadata import describe_components

        return describe_components(
            "CartesianUnifiedForceImpedanceController", config, bimanual=False
        )

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("CartesianUnifiedForceImpedanceTorqueController", config)

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianTorqueParameters,
        storage: CartesianTorqueStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and bind the unified force/impedance torque pipeline."""
        _validate_composite_storage(storage)
        task = CartesianUnifiedForceImpedanceController(
            config, parameters.task, storage.task, context
        )
        super().__init__(config, parameters, storage, task, context)

    def apply_parameters(
        self, parameters: CartesianTorqueParameters, context: ExecutionContext | None = None
    ) -> None:
        """Apply parameters to every controller stage."""
        self.mass.apply_parameters(parameters.mass, context)
        self.task.apply_parameters(parameters.task, context)
        self.nullspace.apply_parameters(parameters.nullspace, context)
        self.output_controller.apply_parameters(parameters.output, context)
        self.parameters = parameters

    def step(
        self,
        cartesian_pose_error: Any,
        cartesian_velocity_error: Any,
        desired_acceleration: Any,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        jacobian: Any,
        joint_bias_torque: Any,
        external_wrench: Any,
        desired_wrench: Any,
        feedforward_wrench: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute final filtered joint torque in Nm for one control period."""
        self.mass.step(joint_mass_matrix, jacobian, context)
        self.task.step(
            cartesian_pose_error,
            cartesian_velocity_error,
            desired_acceleration,
            joint_velocity,
            self.mass.output,
            jacobian,
            external_wrench,
            desired_wrench,
            feedforward_wrench,
            dt,
            context,
        )
        self._finish_step(
            joint_position_error,
            joint_velocity,
            joint_mass_matrix,
            jacobian,
            joint_bias_torque,
            dt,
            context,
        )


class BimanualCartesianImpedanceTorqueController:
    """Complete bimanual kinematics, impedance, nullspace, and filtered torque controller."""

    @staticmethod
    def component_requirements(config: ControllerConfig) -> Any:
        """Describe storage for every pipeline stage before allocation."""
        # Local Folder
        from ._metadata import describe_components

        return describe_components("BimanualCartesianImpedanceController", config, bimanual=True)

    @staticmethod
    def requirements(config: ControllerConfig) -> Any:
        """Describe caller-owned storage before preparing the controller."""
        # Local Folder
        from ._metadata import describe_requirements

        return describe_requirements("BimanualCartesianImpedanceTorqueController", config)

    def __init__(
        self,
        config: BimanualTaskConfig,
        parameters: CartesianTorqueParameters,
        storage: BimanualCartesianImpedanceTorqueStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and bind the complete 14-joint bimanual pipeline."""
        _validate_composite_storage(storage)
        self.config = config
        self.parameters = parameters
        self.storage = storage
        _validate_array(
            storage.task_velocity_error,
            config,
            (config.batch_size, config.task_dim),
            "task_velocity_error",
        )
        self._controller = _backend.BimanualCartesianImpedanceTorqueController(
            config, parameters, storage, context
        )

    @property
    def output(self) -> Any:
        """Caller-owned final filtered 14-joint torque in Nm."""
        return self.storage.output.filter.output.command

    @property
    def status(self) -> Any:
        """Caller-owned final controller status."""
        return self.storage.output.filter.output.status

    def apply_parameters(
        self, parameters: CartesianTorqueParameters, context: ExecutionContext | None = None
    ) -> None:
        """Apply parameters to every controller stage."""
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset selected batch rows, or every row when no mask is supplied."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        desired_absolute_pose: Any,
        arm1_tcp_pose: Any,
        arm2_tcp_pose: Any,
        desired_relative_pose: Any,
        arm1_jacobian: Any,
        arm2_jacobian: Any,
        desired_task_velocity: Any,
        desired_acceleration: Any,
        joint_position_error: Any,
        joint_velocity: Any,
        joint_mass_matrix: Any,
        joint_bias_torque: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute final filtered 14-joint torque in Nm for one control period."""
        self._controller.step(
            desired_absolute_pose,
            arm1_tcp_pose,
            arm2_tcp_pose,
            desired_relative_pose,
            arm1_jacobian,
            arm2_jacobian,
            desired_task_velocity,
            desired_acceleration,
            joint_position_error,
            joint_velocity,
            joint_mass_matrix,
            joint_bias_torque,
            dt,
            context,
        )


STATUS_UNSUPPORTED_MODE = int(_backend.STATUS_UNSUPPORTED_MODE)
STATUS_INVALID_PARAMETER = int(_backend.STATUS_INVALID_PARAMETER)
STATUS_INVALID_INPUT = int(_backend.STATUS_INVALID_INPUT)
STATUS_NON_FINITE_INPUT = int(_backend.STATUS_NON_FINITE_INPUT)
STATUS_INVALID_DT = int(_backend.STATUS_INVALID_DT)
STATUS_SATURATION = int(_backend.STATUS_SATURATION)
STATUS_RESET_APPLIED = int(_backend.STATUS_RESET_APPLIED)
STATUS_SAFE_MODE = int(_backend.STATUS_SAFE_MODE)
STATUS_OUTPUT_HELD = int(_backend.STATUS_OUTPUT_HELD)
STATUS_NO_VALID_OUTPUT = int(_backend.STATUS_NO_VALID_OUTPUT)
STATUS_HOLD_EXPIRED = int(_backend.STATUS_HOLD_EXPIRED)


__all__ = [
    "AdmittanceState",
    "BimanualBase",
    "BimanualCoordinationMode",
    "BimanualFrameConfig",
    "BimanualMainFrame",
    "BimanualTaskConfig",
    "BimanualTaskStorage",
    "BimanualCartesianImpedanceTorqueController",
    "BimanualCartesianImpedanceTorqueStorage",
    "CartesianAdmittanceParameters",
    "CartesianAdmittanceStorage",
    "CartesianUnifiedForceAdmittanceParameters",
    "CartesianUnifiedForceAdmittanceStorage",
    "CartesianForceStorage",
    "CartesianFrame",
    "CartesianImpedanceParameters",
    "CartesianImpedanceStorage",
    "CartesianImpedanceTorqueController",
    "CartesianTorqueParameters",
    "CartesianTorqueStorage",
    "CartesianUnifiedForceImpedanceParameters",
    "TorqueFilterParameters",
    "TorqueFilterState",
    "TorqueFilterStorage",
    "ControllerConfig",
    "ControllerExampleSpec",
    "DType",
    "Device",
    "ExecutionContext",
    "ForceControllerParameters",
    "ForceControllerState",
    "ForceErrorPidParameters",
    "InertialState",
    "JointImpedanceParameters",
    "NullspaceParameters",
    "OperationalSpaceMassParameters",
    "OperationalSpaceMassScratch",
    "OperationalSpaceMassStorage",
    "OperationalSpaceParameters",
    "OutputStorage",
    "Parameter",
    "ParameterBatching",
    "CartesianAdmittancePoseController",
    "BimanualCartesianImpedanceController",
    "BimanualTaskKinematics",
    "CartesianForceController",
    "CartesianUnifiedForceAdmittanceController",
    "CartesianImpedanceController",
    "CartesianUnifiedForceImpedanceController",
    "CartesianUnifiedForceImpedanceTorqueController",
    "TorqueFilter",
    "JointImpedanceController",
    "NullspaceController",
    "OperationalSpaceMass",
    "OperationalSpaceController",
    "OperationalSpaceInverseDynamicsController",
    "StaticCartesianImpedanceController",
    "StaticCartesianImpedanceParameters",
    "StaticJointImpedanceController",
    "StaticJointImpedanceParameters",
    "VariableCartesianImpedanceController",
    "VariableCartesianImpedanceTorqueController",
    "SafeOutputPolicy",
    "SafetyState",
    "STATUS_HOLD_EXPIRED",
    "STATUS_INVALID_DT",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_PARAMETER",
    "STATUS_NO_VALID_OUTPUT",
    "STATUS_NON_FINITE_INPUT",
    "STATUS_OUTPUT_HELD",
    "STATUS_RESET_APPLIED",
    "STATUS_SAFE_MODE",
    "STATUS_SATURATION",
    "STATUS_UNSUPPORTED_MODE",
    "UnifiedControllerScratch",
    "UnifiedControllerState",
    "UnifiedControllerStorage",
    "UnifiedForceAdmittanceState",
    "VariableCartesianImpedanceParameters",
    "VariableImpedanceScratch",
    "VariableImpedanceState",
    "VariableImpedanceStorage",
    "TaskWrenchToJointTorqueController",
    "TaskWrenchToJointTorqueStorage",
    "per_batch",
    "controller_example_specs",
    "find_controller_example_spec",
    "shared",
]
