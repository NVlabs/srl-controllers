# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Generated typed public block API. See tools/generate_public_blocks.py."""
# Standard Library
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Local Folder
from ._api import (
    ControllerConfig,
    Device,
    DType,
    ExecutionContext,
    OutputStorage,
    Parameter,
    SafeOutputPolicy,
    _backend,
)
from ._metadata import Requirements, describe_requirements


class ComponentMode(str, Enum):
    """Explicit joint gain matrix representation."""

    DIAGONAL = "diagonal"
    DENSE = "dense"


@dataclass(frozen=True)
class SignalBlockConfig:
    """Dimension-independent batched signal configuration."""

    device: Device
    dtype: DType
    batch_size: int
    signal_dim: int
    safe_output_policy: SafeOutputPolicy = SafeOutputPolicy.ZERO
    max_hold_steps: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.signal_dim <= 0:
            raise ValueError("batch_size and signal_dim must be positive")
        if self.safe_output_policy == SafeOutputPolicy.HOLD_LAST_VALID and self.max_hold_steps <= 0:
            raise ValueError("hold-last safety requires max_hold_steps > 0")


@dataclass(frozen=True)
class JointBlockConfig(ControllerConfig):
    """Joint configuration with an explicit diagonal or dense gain mode."""

    mode: ComponentMode = ComponentMode.DENSE


@dataclass(frozen=True)
class ProportionalBlockParameters:
    """Typed caller-owned parameters."""

    gain: Parameter


@dataclass(frozen=True)
class ProportionalBlockState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class ProportionalBlockScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class ProportionalBlockStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: ProportionalBlockState = field(default_factory=ProportionalBlockState)
    scratch: ProportionalBlockScratch = field(default_factory=ProportionalBlockScratch)


class ProportionalBlock:
    """Generic proportional block with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: ProportionalBlockParameters,
        storage: ProportionalBlockStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, ProportionalBlockParameters) or not isinstance(
            storage, ProportionalBlockStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.ProportionalBlock(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("ProportionalBlock", config)

    def apply_parameters(
        self, parameters: ProportionalBlockParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, ProportionalBlockParameters):
            raise TypeError("expected ProportionalBlockParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, input_signal: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(input_signal, context)

    @property
    def output_signal(self) -> Any:
        """Caller-owned output_signal; units: caller_defined."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.output_signal

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class IntegratorBlockParameters:
    """Typed caller-owned parameters."""

    initial_output: Parameter
    lower_limit: Parameter
    upper_limit: Parameter


@dataclass(frozen=True)
class IntegratorBlockState:
    """Typed caller-owned state."""

    previous_signal: Any
    integrated_signal: Any
    initialized: Any


@dataclass(frozen=True)
class IntegratorBlockScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class IntegratorBlockStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: IntegratorBlockState
    scratch: IntegratorBlockScratch = field(default_factory=IntegratorBlockScratch)


class IntegratorBlock:
    """Generic integrator block with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: IntegratorBlockParameters,
        storage: IntegratorBlockStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, IntegratorBlockParameters) or not isinstance(
            storage, IntegratorBlockStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.IntegratorBlock(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("IntegratorBlock", config)

    def apply_parameters(
        self, parameters: IntegratorBlockParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, IntegratorBlockParameters):
            raise TypeError("expected IntegratorBlockParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, input_signal: Any, dt: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(input_signal, dt, context)

    @property
    def output_signal(self) -> Any:
        """Caller-owned output_signal; units: caller_defined."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.output_signal

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class DifferentiatorBlockParameters:
    """Typed caller-owned parameters."""


@dataclass(frozen=True)
class DifferentiatorBlockState:
    """Typed caller-owned state."""

    previous_signal: Any
    initialized: Any


@dataclass(frozen=True)
class DifferentiatorBlockScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class DifferentiatorBlockStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: DifferentiatorBlockState
    scratch: DifferentiatorBlockScratch = field(default_factory=DifferentiatorBlockScratch)


class DifferentiatorBlock:
    """Generic differentiator block with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: DifferentiatorBlockParameters,
        storage: DifferentiatorBlockStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, DifferentiatorBlockParameters) or not isinstance(
            storage, DifferentiatorBlockStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.DifferentiatorBlock(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("DifferentiatorBlock", config)

    def apply_parameters(
        self, parameters: DifferentiatorBlockParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, DifferentiatorBlockParameters):
            raise TypeError("expected DifferentiatorBlockParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, input_signal: Any, dt: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(input_signal, dt, context)

    @property
    def output_signal(self) -> Any:
        """Caller-owned output_signal; units: caller_defined."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.output_signal

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class JointStiffnessParameters:
    """Typed caller-owned parameters."""

    stiffness: Parameter


@dataclass(frozen=True)
class JointStiffnessState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class JointStiffnessScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class JointStiffnessStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: JointStiffnessState = field(default_factory=JointStiffnessState)
    scratch: JointStiffnessScratch = field(default_factory=JointStiffnessScratch)


class JointStiffness:
    """Joint stiffness with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: JointBlockConfig,
        parameters: JointStiffnessParameters,
        storage: JointStiffnessStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, JointStiffnessParameters) or not isinstance(
            storage, JointStiffnessStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.JointStiffness(config, parameters, storage, context)

    @staticmethod
    def requirements(config: JointBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("JointStiffness", config)

    def apply_parameters(
        self, parameters: JointStiffnessParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, JointStiffnessParameters):
            raise TypeError("expected JointStiffnessParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, joint_position_error: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(joint_position_error, context)

    @property
    def joint_torque(self) -> Any:
        """Caller-owned joint_torque; units: Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.joint_torque

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class JointDampingParameters:
    """Typed caller-owned parameters."""

    damping: Parameter


@dataclass(frozen=True)
class JointDampingState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class JointDampingScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class JointDampingStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: JointDampingState = field(default_factory=JointDampingState)
    scratch: JointDampingScratch = field(default_factory=JointDampingScratch)


class JointDamping:
    """Joint damping with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: JointBlockConfig,
        parameters: JointDampingParameters,
        storage: JointDampingStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, JointDampingParameters) or not isinstance(
            storage, JointDampingStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.JointDamping(config, parameters, storage, context)

    @staticmethod
    def requirements(config: JointBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("JointDamping", config)

    def apply_parameters(
        self, parameters: JointDampingParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, JointDampingParameters):
            raise TypeError("expected JointDampingParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, joint_velocity_error: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(joint_velocity_error, context)

    @property
    def joint_torque(self) -> Any:
        """Caller-owned joint_torque; units: Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.joint_torque

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class JointDampingDoubleDiagonalizationParameters:
    """Typed caller-owned parameters."""

    damping_factors: Parameter


@dataclass(frozen=True)
class JointDampingDoubleDiagonalizationState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class JointDampingDoubleDiagonalizationScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class JointDampingDoubleDiagonalizationStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: JointDampingDoubleDiagonalizationState = field(
        default_factory=JointDampingDoubleDiagonalizationState
    )
    scratch: JointDampingDoubleDiagonalizationScratch = field(
        default_factory=JointDampingDoubleDiagonalizationScratch
    )


class JointDampingDoubleDiagonalization:
    """Joint damping double diagonalization with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: JointDampingDoubleDiagonalizationParameters,
        storage: JointDampingDoubleDiagonalizationStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(
            parameters, JointDampingDoubleDiagonalizationParameters
        ) or not isinstance(storage, JointDampingDoubleDiagonalizationStorage):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.JointDampingDoubleDiagonalization(
            config, parameters, storage, context
        )

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("JointDampingDoubleDiagonalization", config)

    def apply_parameters(
        self,
        parameters: JointDampingDoubleDiagonalizationParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, JointDampingDoubleDiagonalizationParameters):
            raise TypeError("expected JointDampingDoubleDiagonalizationParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        joint_velocity_error: Any,
        joint_mass_matrix: Any,
        stiffness_matrix: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(joint_velocity_error, joint_mass_matrix, stiffness_matrix, context)

    @property
    def joint_torque(self) -> Any:
        """Caller-owned joint_torque; units: Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.joint_torque

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CartesianStiffnessParameters:
    """Typed caller-owned parameters."""

    stiffness_matrix: Parameter


@dataclass(frozen=True)
class CartesianStiffnessState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CartesianStiffnessScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CartesianStiffnessStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: CartesianStiffnessState = field(default_factory=CartesianStiffnessState)
    scratch: CartesianStiffnessScratch = field(default_factory=CartesianStiffnessScratch)


class CartesianStiffness:
    """Cartesian stiffness with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianStiffnessParameters,
        storage: CartesianStiffnessStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, CartesianStiffnessParameters) or not isinstance(
            storage, CartesianStiffnessStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CartesianStiffness(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CartesianStiffness", config)

    def apply_parameters(
        self, parameters: CartesianStiffnessParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CartesianStiffnessParameters):
            raise TypeError("expected CartesianStiffnessParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, cartesian_pose_error: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(cartesian_pose_error, context)

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class VariableCartesianStiffnessParameters:
    """Typed caller-owned parameters."""

    initial_stiffness_matrix: Parameter
    maximum_stiffness: Parameter
    minimum_stiffness: Parameter
    maximum_stiffness_rate: Parameter


@dataclass(frozen=True)
class VariableCartesianStiffnessState:
    """Typed caller-owned state."""

    current_stiffness_matrix: Any
    previous_stiffness_matrix: Any
    stiffness_initialized: Any


@dataclass(frozen=True)
class VariableCartesianStiffnessScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class VariableCartesianStiffnessStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: VariableCartesianStiffnessState
    scratch: VariableCartesianStiffnessScratch = field(
        default_factory=VariableCartesianStiffnessScratch
    )


class VariableCartesianStiffness:
    """Variable cartesian stiffness with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: VariableCartesianStiffnessParameters,
        storage: VariableCartesianStiffnessStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, VariableCartesianStiffnessParameters) or not isinstance(
            storage, VariableCartesianStiffnessStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.VariableCartesianStiffness(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("VariableCartesianStiffness", config)

    def apply_parameters(
        self,
        parameters: VariableCartesianStiffnessParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, VariableCartesianStiffnessParameters):
            raise TypeError("expected VariableCartesianStiffnessParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        stiffness_matrix: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(cartesian_pose_error, stiffness_matrix, dt, context)

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CartesianDampingParameters:
    """Typed caller-owned parameters."""

    damping_matrix: Parameter


@dataclass(frozen=True)
class CartesianDampingState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CartesianDampingScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CartesianDampingStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: CartesianDampingState = field(default_factory=CartesianDampingState)
    scratch: CartesianDampingScratch = field(default_factory=CartesianDampingScratch)


class CartesianDamping:
    """Cartesian damping with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianDampingParameters,
        storage: CartesianDampingStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, CartesianDampingParameters) or not isinstance(
            storage, CartesianDampingStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CartesianDamping(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CartesianDamping", config)

    def apply_parameters(
        self, parameters: CartesianDampingParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CartesianDampingParameters):
            raise TypeError("expected CartesianDampingParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, cartesian_velocity_error: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(cartesian_velocity_error, context)

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CartesianDampingDoubleDiagonalizationParameters:
    """Typed caller-owned parameters."""

    damping_factors: Parameter


@dataclass(frozen=True)
class CartesianDampingDoubleDiagonalizationState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CartesianDampingDoubleDiagonalizationScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CartesianDampingDoubleDiagonalizationStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: CartesianDampingDoubleDiagonalizationState = field(
        default_factory=CartesianDampingDoubleDiagonalizationState
    )
    scratch: CartesianDampingDoubleDiagonalizationScratch = field(
        default_factory=CartesianDampingDoubleDiagonalizationScratch
    )


class CartesianDampingDoubleDiagonalization:
    """Cartesian damping double diagonalization with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianDampingDoubleDiagonalizationParameters,
        storage: CartesianDampingDoubleDiagonalizationStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(
            parameters, CartesianDampingDoubleDiagonalizationParameters
        ) or not isinstance(storage, CartesianDampingDoubleDiagonalizationStorage):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CartesianDampingDoubleDiagonalization(
            config, parameters, storage, context
        )

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CartesianDampingDoubleDiagonalization", config)

    def apply_parameters(
        self,
        parameters: CartesianDampingDoubleDiagonalizationParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CartesianDampingDoubleDiagonalizationParameters):
            raise TypeError("expected CartesianDampingDoubleDiagonalizationParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_velocity_error: Any,
        cartesian_mass_matrix: Any,
        stiffness_matrix: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(
            cartesian_velocity_error, cartesian_mass_matrix, stiffness_matrix, context
        )

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CartesianDampingFactorizationDesignParameters:
    """Typed caller-owned parameters."""

    damping_factors: Parameter


@dataclass(frozen=True)
class CartesianDampingFactorizationDesignState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CartesianDampingFactorizationDesignScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CartesianDampingFactorizationDesignStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: CartesianDampingFactorizationDesignState = field(
        default_factory=CartesianDampingFactorizationDesignState
    )
    scratch: CartesianDampingFactorizationDesignScratch = field(
        default_factory=CartesianDampingFactorizationDesignScratch
    )


class CartesianDampingFactorizationDesign:
    """Cartesian damping factorization design with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianDampingFactorizationDesignParameters,
        storage: CartesianDampingFactorizationDesignStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(
            parameters, CartesianDampingFactorizationDesignParameters
        ) or not isinstance(storage, CartesianDampingFactorizationDesignStorage):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CartesianDampingFactorizationDesign(
            config, parameters, storage, context
        )

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CartesianDampingFactorizationDesign", config)

    def apply_parameters(
        self,
        parameters: CartesianDampingFactorizationDesignParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CartesianDampingFactorizationDesignParameters):
            raise TypeError("expected CartesianDampingFactorizationDesignParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_velocity_error: Any,
        cartesian_mass_matrix: Any,
        stiffness_matrix: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(
            cartesian_velocity_error, cartesian_mass_matrix, stiffness_matrix, context
        )

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CartesianVirtualMassParameters:
    """Typed caller-owned parameters."""

    virtual_mass_matrix: Parameter


@dataclass(frozen=True)
class CartesianVirtualMassState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CartesianVirtualMassScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CartesianVirtualMassStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: CartesianVirtualMassState = field(default_factory=CartesianVirtualMassState)
    scratch: CartesianVirtualMassScratch = field(default_factory=CartesianVirtualMassScratch)


class CartesianVirtualMass:
    """Cartesian virtual mass with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: CartesianVirtualMassParameters,
        storage: CartesianVirtualMassStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, CartesianVirtualMassParameters) or not isinstance(
            storage, CartesianVirtualMassStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CartesianVirtualMass(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CartesianVirtualMass", config)

    def apply_parameters(
        self, parameters: CartesianVirtualMassParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CartesianVirtualMassParameters):
            raise TypeError("expected CartesianVirtualMassParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, force: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(force, context)

    @property
    def cartesian_acceleration(self) -> Any:
        """Caller-owned cartesian_acceleration; units: m/s^2_then_rad/s^2."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_acceleration

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class ShapingFunctionParameters:
    """Typed caller-owned parameters."""

    maximum_distance: Parameter
    shaping_distance: Parameter


@dataclass(frozen=True)
class ShapingFunctionState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class ShapingFunctionScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class ShapingFunctionStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: ShapingFunctionState = field(default_factory=ShapingFunctionState)
    scratch: ShapingFunctionScratch = field(default_factory=ShapingFunctionScratch)


class ShapingFunction:
    """Shaping function with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: ShapingFunctionParameters,
        storage: ShapingFunctionStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, ShapingFunctionParameters) or not isinstance(
            storage, ShapingFunctionStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.ShapingFunction(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("ShapingFunction", config)

    def apply_parameters(
        self, parameters: ShapingFunctionParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, ShapingFunctionParameters):
            raise TypeError("expected ShapingFunctionParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, error: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(error, context)

    @property
    def shaping_factor(self) -> Any:
        """Caller-owned shaping_factor; units: unitless."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.shaping_factor

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class CommandObserverParameters:
    """Typed caller-owned parameters."""


@dataclass(frozen=True)
class CommandObserverState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class CommandObserverScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class CommandObserverStorage:
    """Caller-owned arrays described by the public requirements query."""

    command_is_valid: Any
    parameter_status: Any
    status: Any
    state: CommandObserverState = field(default_factory=CommandObserverState)
    scratch: CommandObserverScratch = field(default_factory=CommandObserverScratch)


class CommandObserver:
    """Command observer block with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: CommandObserverParameters,
        storage: CommandObserverStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, CommandObserverParameters) or not isinstance(
            storage, CommandObserverStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.CommandObserver(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("CommandObserver", config)

    def apply_parameters(
        self, parameters: CommandObserverParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, CommandObserverParameters):
            raise TypeError("expected CommandObserverParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, command: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(command, context)

    @property
    def command_is_valid(self) -> Any:
        """Caller-owned command_is_valid; units: bool."""
        return self.storage.command_is_valid

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.command_is_valid

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.status


@dataclass(frozen=True)
class FlowFunnelParameters:
    """Typed caller-owned parameters."""

    funnel_radius: Parameter
    max_distance: Parameter
    funnel_velocity: Parameter
    funnel_start_point: Parameter
    funnel_end_point: Parameter


@dataclass(frozen=True)
class FlowFunnelState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class FlowFunnelScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class FlowFunnelStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: FlowFunnelState = field(default_factory=FlowFunnelState)
    scratch: FlowFunnelScratch = field(default_factory=FlowFunnelScratch)


class FlowFunnel:
    """Flow funnel with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: FlowFunnelParameters,
        storage: FlowFunnelStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, FlowFunnelParameters) or not isinstance(
            storage, FlowFunnelStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.FlowFunnel(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("FlowFunnel", config)

    def apply_parameters(
        self, parameters: FlowFunnelParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, FlowFunnelParameters):
            raise TypeError("expected FlowFunnelParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(self, cartesian_pose: Any, context: ExecutionContext | None = None) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(cartesian_pose, context)

    @property
    def cartesian_flow(self) -> Any:
        """Caller-owned cartesian_flow; units: m/s_then_rad/s."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_flow

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class InverseAPFParameters:
    """Typed caller-owned parameters."""

    alpha: Parameter
    gamma: Parameter
    d_min: Parameter
    d_max: Parameter
    deadzone: Parameter


@dataclass(frozen=True)
class InverseAPFState:
    """Typed caller-owned state."""


@dataclass(frozen=True)
class InverseAPFScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class InverseAPFStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: InverseAPFState = field(default_factory=InverseAPFState)
    scratch: InverseAPFScratch = field(default_factory=InverseAPFScratch)


class InverseAPF:
    """Inverse apf with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: InverseAPFParameters,
        storage: InverseAPFStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, InverseAPFParameters) or not isinstance(
            storage, InverseAPFStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.InverseAPF(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("InverseAPF", config)

    def apply_parameters(
        self, parameters: InverseAPFParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, InverseAPFParameters):
            raise TypeError("expected InverseAPFParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        current_pose: Any,
        goal_pose: Any,
        cartesian_twist: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(current_pose, goal_pose, cartesian_twist, context)

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class InertialCouplingParameters:
    """Typed caller-owned parameters."""


@dataclass(frozen=True)
class InertialCouplingState:
    """Typed caller-owned state."""

    previous_mass_matrix: Any
    previous_jacobian: Any
    initialized: Any


@dataclass(frozen=True)
class InertialCouplingScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class InertialCouplingStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: InertialCouplingState
    scratch: InertialCouplingScratch = field(default_factory=InertialCouplingScratch)


class InertialCoupling:
    """Inertial coupling with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: InertialCouplingParameters,
        storage: InertialCouplingStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, InertialCouplingParameters) or not isinstance(
            storage, InertialCouplingStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.InertialCoupling(config, parameters, storage, context)

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("InertialCoupling", config)

    def apply_parameters(
        self, parameters: InertialCouplingParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, InertialCouplingParameters):
            raise TypeError("expected InertialCouplingParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_velocity_error: Any,
        joint_velocity: Any,
        cartesian_mass_matrix: Any,
        jacobian: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(
            cartesian_velocity_error, joint_velocity, cartesian_mass_matrix, jacobian, dt, context
        )

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


@dataclass(frozen=True)
class EnergyTankParameters:
    """Typed caller-owned parameters."""

    upper_energy_limit: Parameter
    lower_energy_limit: Parameter
    initial_energy: Parameter
    empty_threshold: Parameter
    power_noise: Parameter


@dataclass(frozen=True)
class EnergyTankState:
    """Typed caller-owned state."""

    state_variable: Any


@dataclass(frozen=True)
class EnergyTankScratch:
    """Typed caller-owned scratch."""


@dataclass(frozen=True)
class EnergyTankStorage:
    """Caller-owned arrays described by the public requirements query."""

    scale: Any
    filtered_power: Any
    parameter_status: Any
    status: Any
    state: EnergyTankState
    scratch: EnergyTankScratch = field(default_factory=EnergyTankScratch)


class EnergyTank:
    """Energy tank with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: SignalBlockConfig,
        parameters: EnergyTankParameters,
        storage: EnergyTankStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(parameters, EnergyTankParameters) or not isinstance(
            storage, EnergyTankStorage
        ):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.EnergyTank(config, parameters, storage, context)

    @staticmethod
    def requirements(config: SignalBlockConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("EnergyTank", config)

    def apply_parameters(
        self, parameters: EnergyTankParameters, context: ExecutionContext | None = None
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, EnergyTankParameters):
            raise TypeError("expected EnergyTankParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        power_signal: Any,
        additional_power: Any,
        dt: Any,
        context: ExecutionContext | None = None,
    ) -> None:
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(power_signal, additional_power, dt, context)

    @property
    def scale(self) -> Any:
        """Caller-owned scale; units: unitless."""
        return self.storage.scale

    @property
    def filtered_power(self) -> Any:
        """Caller-owned filtered_power; units: W."""
        return self.storage.filtered_power

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.scale

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.status


@dataclass(frozen=True)
class VariableCartesianUnifiedForceImpedanceControllerParameters:
    """Typed caller-owned parameters."""

    initial_stiffness_matrix: Parameter
    maximum_stiffness: Parameter
    minimum_stiffness: Parameter
    maximum_stiffness_rate: Parameter
    damping_factors: Parameter
    proportional_gain: Parameter
    differentiator_gain: Parameter
    integral_gain: Parameter
    initial_output: Parameter
    maximum_distance: Parameter
    shaping_distance: Parameter


@dataclass(frozen=True)
class VariableCartesianUnifiedForceImpedanceControllerState:
    """Typed caller-owned state."""

    current_stiffness_matrix: Any
    previous_stiffness_matrix: Any
    stiffness_initialized: Any
    previous_mass_matrix: Any
    previous_jacobian: Any
    inertial_initialized: Any
    differentiator_previous_error: Any
    integrator_previous_error: Any
    integrated_error: Any
    differentiator_initialized: Any
    integrator_initialized: Any


@dataclass(frozen=True)
class VariableCartesianUnifiedForceImpedanceControllerScratch:
    """Typed caller-owned scratch."""

    stiffness_wrench: Any
    damping_wrench: Any
    inertial_wrench: Any
    stiffness_status: Any
    damping_status: Any
    inertial_status: Any
    variable_impedance_wrench: Any
    force_wrench: Any
    variable_impedance_status: Any
    force_status: Any


@dataclass(frozen=True)
class VariableCartesianUnifiedForceImpedanceControllerStorage:
    """Caller-owned arrays described by the public requirements query."""

    output: OutputStorage
    state: VariableCartesianUnifiedForceImpedanceControllerState
    scratch: VariableCartesianUnifiedForceImpedanceControllerScratch


class VariableCartesianUnifiedForceImpedanceController:
    """Variable cartesian unified force impedance controller with caller-owned CPU/CUDA arrays."""

    def __init__(
        self,
        config: ControllerConfig,
        parameters: VariableCartesianUnifiedForceImpedanceControllerParameters,
        storage: VariableCartesianUnifiedForceImpedanceControllerStorage,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and prepare the block, retaining parameter and storage owners."""
        if not isinstance(
            parameters, VariableCartesianUnifiedForceImpedanceControllerParameters
        ) or not isinstance(storage, VariableCartesianUnifiedForceImpedanceControllerStorage):
            raise TypeError("Block requires its typed parameters and storage")
        self.config, self.parameters, self.storage = config, parameters, storage
        self._controller = _backend.VariableCartesianUnifiedForceImpedanceController(
            config, parameters, storage, context
        )

    @staticmethod
    def requirements(config: ControllerConfig) -> Requirements:
        """Describe every caller-owned storage array before preparation."""
        return describe_requirements("VariableCartesianUnifiedForceImpedanceController", config)

    def apply_parameters(
        self,
        parameters: VariableCartesianUnifiedForceImpedanceControllerParameters,
        context: ExecutionContext | None = None,
    ) -> None:
        """Validate and replace the complete parameter set."""
        if not isinstance(parameters, VariableCartesianUnifiedForceImpedanceControllerParameters):
            raise TypeError("expected VariableCartesianUnifiedForceImpedanceControllerParameters")
        self._controller.apply_parameters(parameters, context)
        self.parameters = parameters

    def reset(self, reset_mask: Any | None = None, context: ExecutionContext | None = None) -> None:
        """Reset all rows or the selected Boolean batch mask."""
        self._controller.reset(reset_mask, context)

    def step(
        self,
        cartesian_pose_error: Any,
        stiffness_matrix: Any,
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
        """Compute one tick; CUDA work is enqueued on the caller stream."""
        self._controller.step(
            cartesian_pose_error,
            stiffness_matrix,
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

    @property
    def cartesian_wrench(self) -> Any:
        """Caller-owned cartesian_wrench; units: N_then_Nm."""
        return self.storage.output.command

    @property
    def output(self) -> Any:
        """Primary caller-owned output array."""
        return self.cartesian_wrench

    @property
    def status(self) -> Any:
        """Per-row runtime status flags."""
        return self.storage.output.status


__all__ = [
    "ComponentMode",
    "SignalBlockConfig",
    "JointBlockConfig",
    "ProportionalBlockParameters",
    "ProportionalBlockState",
    "ProportionalBlockScratch",
    "ProportionalBlockStorage",
    "ProportionalBlock",
    "IntegratorBlockParameters",
    "IntegratorBlockState",
    "IntegratorBlockScratch",
    "IntegratorBlockStorage",
    "IntegratorBlock",
    "DifferentiatorBlockParameters",
    "DifferentiatorBlockState",
    "DifferentiatorBlockScratch",
    "DifferentiatorBlockStorage",
    "DifferentiatorBlock",
    "JointStiffnessParameters",
    "JointStiffnessState",
    "JointStiffnessScratch",
    "JointStiffnessStorage",
    "JointStiffness",
    "JointDampingParameters",
    "JointDampingState",
    "JointDampingScratch",
    "JointDampingStorage",
    "JointDamping",
    "JointDampingDoubleDiagonalizationParameters",
    "JointDampingDoubleDiagonalizationState",
    "JointDampingDoubleDiagonalizationScratch",
    "JointDampingDoubleDiagonalizationStorage",
    "JointDampingDoubleDiagonalization",
    "CartesianStiffnessParameters",
    "CartesianStiffnessState",
    "CartesianStiffnessScratch",
    "CartesianStiffnessStorage",
    "CartesianStiffness",
    "VariableCartesianStiffnessParameters",
    "VariableCartesianStiffnessState",
    "VariableCartesianStiffnessScratch",
    "VariableCartesianStiffnessStorage",
    "VariableCartesianStiffness",
    "CartesianDampingParameters",
    "CartesianDampingState",
    "CartesianDampingScratch",
    "CartesianDampingStorage",
    "CartesianDamping",
    "CartesianDampingDoubleDiagonalizationParameters",
    "CartesianDampingDoubleDiagonalizationState",
    "CartesianDampingDoubleDiagonalizationScratch",
    "CartesianDampingDoubleDiagonalizationStorage",
    "CartesianDampingDoubleDiagonalization",
    "CartesianDampingFactorizationDesignParameters",
    "CartesianDampingFactorizationDesignState",
    "CartesianDampingFactorizationDesignScratch",
    "CartesianDampingFactorizationDesignStorage",
    "CartesianDampingFactorizationDesign",
    "CartesianVirtualMassParameters",
    "CartesianVirtualMassState",
    "CartesianVirtualMassScratch",
    "CartesianVirtualMassStorage",
    "CartesianVirtualMass",
    "ShapingFunctionParameters",
    "ShapingFunctionState",
    "ShapingFunctionScratch",
    "ShapingFunctionStorage",
    "ShapingFunction",
    "CommandObserverParameters",
    "CommandObserverState",
    "CommandObserverScratch",
    "CommandObserverStorage",
    "CommandObserver",
    "FlowFunnelParameters",
    "FlowFunnelState",
    "FlowFunnelScratch",
    "FlowFunnelStorage",
    "FlowFunnel",
    "InverseAPFParameters",
    "InverseAPFState",
    "InverseAPFScratch",
    "InverseAPFStorage",
    "InverseAPF",
    "InertialCouplingParameters",
    "InertialCouplingState",
    "InertialCouplingScratch",
    "InertialCouplingStorage",
    "InertialCoupling",
    "EnergyTankParameters",
    "EnergyTankState",
    "EnergyTankScratch",
    "EnergyTankStorage",
    "EnergyTank",
    "VariableCartesianUnifiedForceImpedanceControllerParameters",
    "VariableCartesianUnifiedForceImpedanceControllerState",
    "VariableCartesianUnifiedForceImpedanceControllerScratch",
    "VariableCartesianUnifiedForceImpedanceControllerStorage",
    "VariableCartesianUnifiedForceImpedanceController",
]
