# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Tests for the platform-neutral Python controller API."""

from __future__ import annotations

# Standard Library
import dataclasses
import importlib
import importlib.util
import pathlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

# Third Party
import numpy as np
import pytest

# NVIDIA
from nvidia.srl.controllers import (
    STATUS_INVALID_PARAMETER,
    STATUS_SAFE_MODE,
    STATUS_SATURATION,
    AdmittanceState,
    BimanualCartesianImpedanceController,
    BimanualCartesianImpedanceTorqueController,
    BimanualCartesianImpedanceTorqueStorage,
    BimanualFrameConfig,
    BimanualTaskConfig,
    BimanualTaskKinematics,
    BimanualTaskStorage,
    CartesianAdmittanceParameters,
    CartesianAdmittancePoseController,
    CartesianAdmittanceStorage,
    CartesianForceController,
    CartesianForceStorage,
    CartesianFrame,
    CartesianImpedanceController,
    CartesianImpedanceParameters,
    CartesianImpedanceStorage,
    CartesianImpedanceTorqueController,
    CartesianTorqueParameters,
    CartesianTorqueStorage,
    CartesianUnifiedForceAdmittanceController,
    CartesianUnifiedForceAdmittanceParameters,
    CartesianUnifiedForceAdmittanceStorage,
    CartesianUnifiedForceImpedanceController,
    CartesianUnifiedForceImpedanceParameters,
    ComponentMode,
    ControllerConfig,
    Device,
    DType,
    ForceControllerParameters,
    ForceControllerState,
    ForceErrorPidParameters,
    InertialState,
    JointBlockConfig,
    JointImpedanceController,
    JointImpedanceParameters,
    NullspaceController,
    NullspaceParameters,
    OperationalSpaceController,
    OperationalSpaceInverseDynamicsController,
    OperationalSpaceMass,
    OperationalSpaceMassParameters,
    OperationalSpaceMassScratch,
    OperationalSpaceMassStorage,
    OperationalSpaceParameters,
    OutputStorage,
    SafetyState,
    StaticCartesianImpedanceController,
    StaticCartesianImpedanceParameters,
    StaticJointImpedanceController,
    StaticJointImpedanceParameters,
    TaskWrenchToJointTorqueStorage,
    TorqueFilter,
    TorqueFilterParameters,
    TorqueFilterState,
    TorqueFilterStorage,
    UnifiedControllerScratch,
    UnifiedControllerState,
    UnifiedControllerStorage,
    UnifiedForceAdmittanceState,
    VariableCartesianImpedanceController,
    VariableCartesianImpedanceParameters,
    VariableImpedanceScratch,
    VariableImpedanceState,
    VariableImpedanceStorage,
    controller_example_specs,
    find_controller_example_spec,
    per_batch,
    shared,
)


def test_controller_catalog_has_no_prepared_public_tier() -> None:
    # NVIDIA
    import nvidia.srl.controllers as controllers

    assert not any(name.startswith("Prepared") for name in controllers.__all__)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nvidia.srl.controllers.prepared")


def test_registry_has_exactly_one_launcher_for_every_hardware_example() -> None:
    specs = controller_example_specs()
    assert len(specs) == 22
    assert len({spec.hardware_target for spec in specs}) == len(specs)
    assert len({spec.isaac_launcher for spec in specs}) == len(specs)
    example_root = pathlib.Path(__file__).resolve().parents[3] / "examples" / "isaaclab"
    for spec in specs:
        assert find_controller_example_spec(spec.hardware_target) == spec
        assert (example_root / spec.isaac_launcher).is_file()
        assert spec.nominal_period_seconds in {0.001, 0.002}


@pytest.mark.parametrize("robot_family", ["franka", "rizon", "flexiv_dual"])
@pytest.mark.parametrize(
    "controller_kind", ["static_joint_impedance", "static_cartesian_impedance_torque"]
)
def test_single_arm_example_rejects_static_impedance_for_non_ur_robots(
    robot_family: str, controller_kind: str
) -> None:
    example_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "examples"
        / "isaaclab"
        / "_single_arm_parallel.py"
    )
    module_spec = importlib.util.spec_from_file_location("single_arm_example", example_path)
    assert module_spec is not None and module_spec.loader is not None
    example = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(example)
    spec = SimpleNamespace(robot_family=robot_family, controller_kind=controller_kind)
    with pytest.raises(ValueError, match="restricted to Universal Robots"):
        example.run(None, spec=spec)


def _output_storage(batch_size: int, output_dim: int) -> OutputStorage:
    return OutputStorage(
        command=np.zeros((batch_size, output_dim), dtype=np.float64),
        parameter_status=np.zeros(batch_size, dtype=np.uint64),
        status=np.zeros(batch_size, dtype=np.uint64),
        safety=SafetyState(
            last_valid_output=np.zeros((batch_size, output_dim), dtype=np.float64),
            valid_mask=np.zeros(batch_size, dtype=np.bool_),
            hold_count=np.zeros(batch_size, dtype=np.uint64),
        ),
    )


def _inertial_state(batch_size: int, joint_dim: int, task_dim: int = 6) -> InertialState:
    return InertialState(
        previous_mass_matrix=np.zeros((batch_size, task_dim, task_dim), dtype=np.float64),
        previous_jacobian=np.zeros((batch_size, task_dim, joint_dim), dtype=np.float64),
        initialized=np.zeros(batch_size, dtype=np.bool_),
    )


def _force_state(batch_size: int) -> ForceControllerState:
    return ForceControllerState(
        differentiator_previous_error=np.zeros((batch_size, 6), dtype=np.float64),
        differentiator_initialized=np.zeros(batch_size, dtype=np.bool_),
        integrator_previous_error=np.zeros((batch_size, 6), dtype=np.float64),
        integrated_error=np.zeros((batch_size, 6), dtype=np.float64),
        integrator_initialized=np.zeros(batch_size, dtype=np.bool_),
    )


def _force_parameters() -> ForceControllerParameters:
    return ForceControllerParameters(
        proportional_gain=shared(np.ones(6, dtype=np.float64)),
        differentiator_gain=shared(np.zeros(6, dtype=np.float64)),
        integral_gain=shared(np.zeros(6, dtype=np.float64)),
        initial_output=shared(np.zeros(6, dtype=np.float64)),
        maximum_distance=shared(np.full(6, 2.0, dtype=np.float64)),
        shaping_distance=shared(np.ones(6, dtype=np.float64)),
    )


def _admittance_controller(
    batch_size: int,
) -> tuple[CartesianAdmittancePoseController, OutputStorage]:
    zeros = lambda shape: np.zeros(shape, dtype=np.float64)
    bool_zeros = lambda shape: np.zeros(shape, dtype=np.bool_)
    status_zeros = lambda shape: np.zeros(shape, dtype=np.uint64)
    diagonal = np.eye(6, dtype=np.float64)
    parameters = CartesianAdmittanceParameters(
        stiffness_matrix=shared(diagonal.copy()),
        damping_matrix=shared(diagonal.copy()),
        virtual_mass_matrix=shared(diagonal.copy()),
        force_filter_threshold=shared(np.zeros(6, dtype=np.float64)),
        acceleration_limit=shared(np.ones(6, dtype=np.float64)),
        twist_limit=shared(np.ones(6, dtype=np.float64)),
        pose_limit=shared(np.ones(6, dtype=np.float64)),
    )
    output = OutputStorage(
        command=zeros((batch_size, 6)),
        parameter_status=status_zeros(batch_size),
        status=status_zeros(batch_size),
        safety=SafetyState(
            last_valid_output=zeros((batch_size, 6)),
            valid_mask=bool_zeros(batch_size),
            hold_count=status_zeros(batch_size),
        ),
    )
    state = AdmittanceState(
        pose_offset=zeros((batch_size, 6)),
        twist=zeros((batch_size, 6)),
        pose_previous_signal=zeros((batch_size, 6)),
        twist_previous_signal=zeros((batch_size, 6)),
        pose_integrator_initialized=bool_zeros(batch_size),
        twist_integrator_initialized=bool_zeros(batch_size),
    )
    controller = CartesianAdmittancePoseController(
        ControllerConfig(
            device=Device.CPU,
            dtype=DType.FLOAT64,
            batch_size=batch_size,
            cartesian_frame=CartesianFrame.ROBOT_BASE,
        ),
        parameters,
        CartesianAdmittanceStorage(output=output, state=state),
    )
    return controller, output


def test_reset_accepts_absent_and_explicit_masks() -> None:
    controller, output = _admittance_controller(batch_size=2)

    controller.reset()
    np.testing.assert_array_equal(output.safety.valid_mask, (False, False))

    output.safety.valid_mask[:] = True
    output.safety.hold_count[:] = 3
    controller.reset(np.array((True, False), dtype=np.bool_))
    np.testing.assert_array_equal(output.safety.valid_mask, (False, True))
    np.testing.assert_array_equal(output.safety.hold_count, (0, 3))


def test_cpu_step_uses_the_same_controller_lifecycle() -> None:
    controller, output = _admittance_controller(batch_size=2)
    controller.reset()

    controller.step(
        external_wrench=np.zeros((2, 6), dtype=np.float64),
        feedforward_wrench=np.zeros((2, 6), dtype=np.float64),
        dt=np.full(2, 0.01, dtype=np.float64),
    )

    np.testing.assert_array_equal(output.command, np.zeros((2, 6)))
    np.testing.assert_array_equal(output.status, np.zeros(2, dtype=np.uint64))


def test_joint_controller_uses_explicit_per_batch_parameters() -> None:
    batch_size = 2
    joint_dim = 2
    output = OutputStorage(
        command=np.zeros((batch_size, joint_dim), dtype=np.float64),
        parameter_status=np.zeros(batch_size, dtype=np.uint64),
        status=np.zeros(batch_size, dtype=np.uint64),
        safety=SafetyState(
            last_valid_output=np.zeros((batch_size, joint_dim), dtype=np.float64),
            valid_mask=np.zeros(batch_size, dtype=np.bool_),
            hold_count=np.zeros(batch_size, dtype=np.uint64),
        ),
    )
    controller = JointImpedanceController(
        ControllerConfig(
            device=Device.CPU,
            dtype=DType.FLOAT64,
            batch_size=batch_size,
            joint_dim=joint_dim,
        ),
        JointImpedanceParameters(
            stiffness_matrix=per_batch(
                np.array(
                    (
                        ((100.0, 0.0), (0.0, 200.0)),
                        ((300.0, 0.0), (0.0, 400.0)),
                    )
                )
            ),
            damping_factors=per_batch(np.ones((batch_size, joint_dim))),
        ),
        output,
    )
    controller.reset()
    controller.step(
        joint_position_error=np.array(((0.1, -0.1), (0.1, -0.1))),
        joint_velocity_error=np.zeros((batch_size, joint_dim)),
        desired_acceleration=np.zeros((batch_size, joint_dim)),
        joint_mass_matrix=np.repeat(
            np.eye(joint_dim, dtype=np.float64)[None, :, :], batch_size, axis=0
        ),
    )

    np.testing.assert_allclose(output.command, ((10.0, -20.0), (30.0, -40.0)))
    np.testing.assert_array_equal(output.status, np.zeros(batch_size, dtype=np.uint64))


def test_joint_controller_rejects_aliased_safety_history() -> None:
    """Preparation rejects safety history that aliases the protected command."""
    command = np.zeros((1, 2), dtype=np.float64)
    storage = OutputStorage(
        command=command,
        parameter_status=np.zeros(1, dtype=np.uint64),
        status=np.zeros(1, dtype=np.uint64),
        safety=SafetyState(
            last_valid_output=command,
            valid_mask=np.zeros(1, dtype=np.bool_),
            hold_count=np.zeros(1, dtype=np.uint64),
        ),
    )

    with pytest.raises(ValueError, match="must not alias command"):
        JointImpedanceController(
            ControllerConfig(Device.CPU, DType.FLOAT64, batch_size=1, joint_dim=2),
            JointImpedanceParameters(
                stiffness_matrix=shared(np.eye(2, dtype=np.float64)),
                damping_factors=shared(np.ones(2, dtype=np.float64)),
            ),
            storage,
        )


@pytest.mark.parametrize("reverse", (False, True))
def test_joint_controller_rejects_partial_safety_overlap(reverse: bool) -> None:
    backing = np.array((3.0, 4.0, 5.0))
    output = _output_storage(1, 2)
    output = OutputStorage(
        command=backing[int(reverse) : int(reverse) + 2].reshape(1, 2),
        parameter_status=output.parameter_status,
        status=output.status,
        safety=SafetyState(
            last_valid_output=backing[int(not reverse) : int(not reverse) + 2].reshape(1, 2),
            valid_mask=output.safety.valid_mask,
            hold_count=output.safety.hold_count,
        ),
    )
    output.parameter_status[:] = 123
    with pytest.raises(ValueError, match="alias|overlap"):
        JointImpedanceController(
            ControllerConfig(Device.CPU, DType.FLOAT64, 1, 2),
            JointImpedanceParameters(shared(np.eye(2)), shared(np.ones(2))),
            output,
        )
    np.testing.assert_array_equal(backing, (3.0, 4.0, 5.0))
    np.testing.assert_array_equal(output.parameter_status, (123,))


@pytest.mark.parametrize("scalar_dtype", (np.float32, np.float64))
@pytest.mark.parametrize("role", ("input", "parameter", "output", "status"))
def test_non_native_byte_order_is_rejected(scalar_dtype: Any, role: str) -> None:
    native_dtype = np.dtype(scalar_dtype)
    swapped_dtype = native_dtype.newbyteorder("S")
    output = OutputStorage(
        command=np.zeros((1, 2), dtype=swapped_dtype if role == "output" else native_dtype),
        parameter_status=np.zeros(
            1, dtype=np.dtype(np.uint64).newbyteorder("S") if role == "status" else np.uint64
        ),
        status=np.zeros(1, dtype=np.uint64),
        safety=SafetyState(
            last_valid_output=np.zeros((1, 2), dtype=native_dtype),
            valid_mask=np.zeros(1, dtype=np.bool_),
            hold_count=np.zeros(1, dtype=np.uint64),
        ),
    )
    parameters = JointImpedanceParameters(
        shared(np.eye(2, dtype=swapped_dtype if role == "parameter" else native_dtype)),
        shared(np.ones(2, dtype=native_dtype)),
    )
    config = ControllerConfig(
        Device.CPU, DType.FLOAT32 if scalar_dtype is np.float32 else DType.FLOAT64, 1, 2
    )
    with pytest.raises(ValueError, match="byte order|endian"):
        controller = JointImpedanceController(config, parameters, output)
        controller.reset()
        controller.step(
            np.array(((0.1, -0.1),), dtype=swapped_dtype),
            np.zeros((1, 2), dtype=native_dtype),
            np.zeros((1, 2), dtype=native_dtype),
            np.eye(2, dtype=native_dtype)[None],
        )


class _MetadataArray:
    def __init__(self, interface: dict[str, Any], *, cuda: bool = False) -> None:
        setattr(self, "__cuda_array_interface__" if cuda else "__array_interface__", interface)


@pytest.fixture
def array_backend() -> Any:
    # NVIDIA
    from nvidia.srl._controllers import nvidia_srl_controllers_cpp as extension

    return extension._controller_api


@pytest.mark.parametrize("cuda", (False, True), ids=("cpu", "cuda-parser-only"))
@pytest.mark.parametrize(
    "typestr", ("", "f8", "|f8", "<f16", "<i8", "<u4", "<f8junk", "float64", None, 8)
)
def test_array_parser_rejects_malformed_dtype(array_backend: Any, cuda: bool, typestr: Any) -> None:
    owner = np.zeros((1, 2), dtype=np.float64)
    interface = dict(owner.__array_interface__, typestr=typestr)
    with pytest.raises((ValueError, TypeError)):
        array_backend.describe_array(_MetadataArray(interface, cuda=cuda))


@pytest.mark.parametrize("cuda", (False, True), ids=("cpu", "cuda-parser-only"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("shape", (-1, 2)),
        ("shape", (1, 1, 1, 1, 1)),
        ("shape", (2**62, 8)),
        ("shape", (1.5, 2)),
        ("shape", (True, 2)),
        ("shape", [1, 2]),
        ("shape", None),
        ("data", (0, False)),
        ("data", (-1, False)),
        ("data", (True, False)),
        ("data", (None, False)),
        ("data", (8, "false")),
        ("data", (8,)),
        ("data", (8, False, 0)),
        ("strides", (16,)),
        ("strides", (16, 7)),
        ("strides", (16, -8)),
        ("strides", (16, 8.5)),
        ("strides", (16, True)),
        ("strides", (2**64, 8)),
        ("strides", [16, 8]),
    ),
)
def test_array_parser_rejects_malformed_metadata(
    array_backend: Any, cuda: bool, field: str, value: Any
) -> None:
    owner = np.zeros((1, 2), dtype=np.float64)
    interface = dict(owner.__array_interface__)
    interface[field] = value
    with pytest.raises((ValueError, TypeError, OverflowError)):
        array_backend.describe_array(_MetadataArray(interface, cuda=cuda))


@pytest.mark.parametrize("cuda", (False, True), ids=("cpu", "cuda-parser-only"))
@pytest.mark.parametrize("field", ("data", "typestr", "shape"))
def test_array_parser_rejects_missing_metadata(array_backend: Any, cuda: bool, field: str) -> None:
    owner = np.zeros((1, 2), dtype=np.float64)
    interface = dict(owner.__array_interface__)
    del interface[field]
    with pytest.raises(ValueError, match="missing|interface"):
        array_backend.describe_array(_MetadataArray(interface, cuda=cuda))


@pytest.mark.parametrize("scalar_dtype", (np.float32, np.float64, np.uint64, np.bool_))
def test_array_parser_preserves_native_metadata(array_backend: Any, scalar_dtype: Any) -> None:
    owner = np.zeros((2, 3), dtype=scalar_dtype)
    owner.flags.writeable = False
    metadata = array_backend.describe_array(owner)
    assert metadata["address"] == owner.__array_interface__["data"][0]
    expected_dtype = "bool" if scalar_dtype is np.bool_ else np.dtype(scalar_dtype).name
    assert metadata["dtype"] == expected_dtype
    assert metadata["device"] == "cpu"
    assert metadata["layout"] == "row_major"
    assert metadata["rank"] == 2
    assert tuple(metadata["shape"][:2]) == (2, 3)
    assert tuple(metadata["strides"][:2]) == (3, 1)
    assert metadata["read_only"] is True


@pytest.mark.parametrize("scalar_dtype", (np.float32, np.float64))
@pytest.mark.parametrize("kind", ("padded", "broadcast"))
def test_array_parser_preserves_native_strided_metadata(
    array_backend: Any, scalar_dtype: Any, kind: str
) -> None:
    owner = np.ones((2, 4), dtype=scalar_dtype)
    view = owner[:, :2] if kind == "padded" else np.broadcast_to(owner[:1, :2], (2, 2))
    metadata = array_backend.describe_array(view)
    assert metadata["address"] == view.__array_interface__["data"][0]
    assert metadata["layout"] == "strided"
    assert tuple(metadata["shape"][:2]) == (2, 2)
    assert tuple(metadata["strides"][:2]) == tuple(
        stride // view.itemsize for stride in view.strides
    )


@pytest.mark.parametrize("scalar_dtype", (np.float32, np.float64))
@pytest.mark.parametrize("cuda", (False, True), ids=("cpu", "cuda-parser-only"))
def test_array_parser_rejects_unaligned_storage(
    array_backend: Any, scalar_dtype: Any, cuda: bool
) -> None:
    itemsize = np.dtype(scalar_dtype).itemsize
    backing = bytearray(2 * itemsize + 1)
    unaligned = np.ndarray((1, 2), dtype=scalar_dtype, buffer=backing, offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="align"):
        array_backend.describe_array(_MetadataArray(dict(unaligned.__array_interface__), cuda=cuda))


@pytest.mark.parametrize("scalar_dtype", (np.float32, np.float64))
def test_joint_controller_accepts_read_only_native_inputs_and_parameters(scalar_dtype: Any) -> None:
    def readonly(values: Any) -> Any:
        array = np.asarray(values, dtype=scalar_dtype)
        array.flags.writeable = False
        return array

    command = np.zeros((1, 2), dtype=scalar_dtype)
    output = OutputStorage(
        command,
        np.zeros(1, dtype=np.uint64),
        np.zeros(1, dtype=np.uint64),
        SafetyState(command.copy(), np.zeros(1, dtype=np.bool_), np.zeros(1, dtype=np.uint64)),
    )
    controller = JointImpedanceController(
        ControllerConfig(
            Device.CPU, DType.FLOAT32 if scalar_dtype is np.float32 else DType.FLOAT64, 1, 2
        ),
        JointImpedanceParameters(shared(readonly(((4, 0), (0, 9)))), shared(readonly((1, 1)))),
        output,
    )
    controller.reset()
    controller.step(
        readonly(((1, -1),)),
        readonly(((0.5, -0.5),)),
        readonly(((1, 2),)),
        readonly((((1, 0), (0, 1)),)),
    )
    np.testing.assert_allclose(
        command, ((7, -10),), rtol=1e-6 if scalar_dtype is np.float32 else 1e-12
    )
    np.testing.assert_array_equal(output.status, (0,))
    np.testing.assert_array_equal(output.safety.valid_mask, (True,))
    np.testing.assert_array_equal(output.safety.hold_count, (0,))


@pytest.mark.parametrize(
    "role",
    ("command", "parameter_status", "status", "last_valid_output", "valid_mask", "hold_count"),
)
def test_joint_controller_rejects_read_only_writable_roles(role: str) -> None:
    output = _output_storage(1, 2)
    owner = output if hasattr(output, role) else output.safety
    array = getattr(owner, role)
    array.flags.writeable = False
    with pytest.raises(ValueError, match="read.only|writ"):
        JointImpedanceController(
            ControllerConfig(Device.CPU, DType.FLOAT64, 1, 2),
            JointImpedanceParameters(shared(np.eye(2)), shared(np.ones(2))),
            output,
        )
    np.testing.assert_array_equal(output.command, ((0, 0),))
    np.testing.assert_array_equal(output.parameter_status, (0,))


@pytest.mark.parametrize("kind", ("negative", "broadcast", "padded", "transposed", "unaligned"))
def test_joint_controller_rejects_noncontiguous_or_unaligned_input(kind: str) -> None:
    output = _output_storage(2, 2)
    controller = JointImpedanceController(
        ControllerConfig(Device.CPU, DType.FLOAT64, 2, 2),
        JointImpedanceParameters(shared(np.eye(2)), shared(np.ones(2))),
        output,
    )
    controller.reset()
    if kind == "negative":
        position = np.ones((2, 2))[:, ::-1]
    elif kind == "broadcast":
        position = np.broadcast_to(np.ones((1, 2)), (2, 2))
    elif kind == "padded":
        position = np.ones((2, 4))[:, :2]
    elif kind == "transposed":
        position = np.ones((2, 2)).T
    else:
        position = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    before_status = output.status.copy()
    with pytest.raises(ValueError, match="stride|contiguous|align|row.major"):
        controller.step(
            position, np.zeros((2, 2)), np.zeros((2, 2)), np.repeat(np.eye(2)[None], 2, axis=0)
        )
    np.testing.assert_array_equal(output.command, np.zeros((2, 2)))
    np.testing.assert_array_equal(output.status, before_status)
    np.testing.assert_array_equal(output.safety.valid_mask, (False, False))


def _cartesian_torque_fixture() -> (
    tuple[ControllerConfig, CartesianTorqueParameters, CartesianTorqueStorage]
):
    config = ControllerConfig(Device.CPU, DType.FLOAT64, 1, 7, CartesianFrame.ROBOT_BASE)
    parameters = CartesianTorqueParameters(
        OperationalSpaceMassParameters(shared(np.array(1e-6))),
        CartesianImpedanceParameters(shared(np.eye(6) * 4), shared(np.ones(6))),
        NullspaceParameters(shared(np.eye(7)), shared(np.eye(7))),
        TorqueFilterParameters(
            shared(np.full(7, -100.0)), shared(np.full(7, 100.0)), shared(np.full(7, 1000.0))
        ),
    )
    storage = CartesianTorqueStorage(
        OperationalSpaceMassStorage(
            np.zeros((1, 6, 6)),
            np.zeros(1, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            OperationalSpaceMassScratch(
                np.zeros((1, 7, 7)), np.zeros((1, 7, 6)), np.zeros((1, 6, 6))
            ),
        ),
        CartesianImpedanceStorage(_output_storage(1, 6), _inertial_state(1, 7)),
        _output_storage(1, 7),
        TaskWrenchToJointTorqueStorage(
            np.zeros((1, 7)),
            TorqueFilterStorage(
                _output_storage(1, 7),
                TorqueFilterState(np.zeros((1, 7)), np.zeros(1, dtype=np.bool_)),
            ),
        ),
    )
    return config, parameters, storage


def test_composite_rejects_cross_stage_overlap_before_parameter_writes() -> None:
    config, parameters, storage = _cartesian_torque_fixture()
    storage.mass.parameter_status[:] = 123
    task = replace(
        storage.task,
        output=replace(storage.task.output, parameter_status=storage.mass.parameter_status),
    )
    with pytest.raises(ValueError, match="overlap"):
        CartesianImpedanceTorqueController(config, parameters, replace(storage, task=task))
    np.testing.assert_array_equal(storage.mass.parameter_status, (123,))


@pytest.mark.parametrize("inplace", (False, True))
def test_composite_preserves_safe_storage_reuse(inplace: bool) -> None:
    config, parameters, storage = _cartesian_torque_fixture()
    intermediate = (
        storage.output.filter.output.command
        if inplace
        else storage.mass.scratch.joint_factor.reshape(-1)[:7].reshape(1, 7)
    )
    storage = replace(storage, output=replace(storage.output, unfiltered_joint_torque=intermediate))
    controller = CartesianImpedanceTorqueController(config, parameters, storage)
    controller.reset()
    for _ in range(3):
        controller.step(
            np.full((1, 6), 0.1),
            np.zeros((1, 6)),
            np.zeros((1, 6)),
            np.zeros((1, 7)),
            np.zeros((1, 7)),
            np.eye(7)[None],
            np.eye(6, 7)[None],
            np.full((1, 7), 0.2),
            np.array(0.001),
        )
        np.testing.assert_allclose(controller.output, ((0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2),))
        np.testing.assert_array_equal(controller.status, (0,))


def test_controller_rejects_parameters_for_a_different_controller() -> None:
    """Typed parameter schemas reject accidental cross-controller configuration."""
    output = OutputStorage(
        command=np.zeros((1, 2), dtype=np.float64),
        parameter_status=np.zeros(1, dtype=np.uint64),
        status=np.zeros(1, dtype=np.uint64),
        safety=SafetyState(
            last_valid_output=np.zeros((1, 2), dtype=np.float64),
            valid_mask=np.zeros(1, dtype=np.bool_),
            hold_count=np.zeros(1, dtype=np.uint64),
        ),
    )
    wrong_parameters = CartesianAdmittanceParameters(
        stiffness_matrix=shared(np.eye(6, dtype=np.float64)),
        damping_matrix=shared(np.eye(6, dtype=np.float64)),
        virtual_mass_matrix=shared(np.eye(6, dtype=np.float64)),
        force_filter_threshold=shared(np.zeros(6, dtype=np.float64)),
        acceleration_limit=shared(np.ones(6, dtype=np.float64)),
        twist_limit=shared(np.ones(6, dtype=np.float64)),
        pose_limit=shared(np.ones(6, dtype=np.float64)),
    )

    with pytest.raises(TypeError, match="parameters must be JointImpedanceParameters"):
        JointImpedanceController(
            ControllerConfig(Device.CPU, DType.FLOAT64, batch_size=1, joint_dim=2),
            wrong_parameters,  # type: ignore[arg-type]
            output,
        )


def test_extension_exports_only_the_private_controller_backend() -> None:
    # NVIDIA
    from nvidia.srl._controllers import nvidia_srl_controllers_cpp as extension

    assert [name for name in dir(extension) if not name.startswith("__")] == ["_controller_api"]
    assert not hasattr(extension, "tensor")


def test_controller_config_validates_task_dimension() -> None:
    config = ControllerConfig(Device.CPU, DType.FLOAT64, batch_size=1)
    assert config.task_dim == 6
    for task_dim in (0, 13):
        with pytest.raises(ValueError, match=r"task_dim must be in \[1, 12\]"):
            ControllerConfig(Device.CPU, DType.FLOAT64, batch_size=1, task_dim=task_dim)


def test_nullspace_controller_accepts_twelve_task_rows() -> None:
    batch_size = 1
    joint_dim = 14
    config = ControllerConfig(
        Device.CPU,
        DType.FLOAT64,
        batch_size=batch_size,
        joint_dim=joint_dim,
        task_dim=12,
    )
    output = _output_storage(batch_size, joint_dim)
    controller = NullspaceController(
        config,
        NullspaceParameters(
            stiffness_matrix=shared(np.eye(joint_dim, dtype=np.float64)),
            damping_matrix=shared(np.eye(joint_dim, dtype=np.float64)),
        ),
        output,
    )
    controller.reset()
    jacobian = np.zeros((batch_size, 12, joint_dim), dtype=np.float64)
    jacobian[0, :, :12] = np.eye(12, dtype=np.float64)
    controller.step(
        np.zeros((batch_size, joint_dim), dtype=np.float64),
        np.zeros((batch_size, joint_dim), dtype=np.float64),
        np.eye(joint_dim, dtype=np.float64)[None, :, :],
        jacobian,
    )
    np.testing.assert_array_equal(output.command, np.zeros((batch_size, joint_dim)))
    np.testing.assert_array_equal(output.status, np.zeros(batch_size, dtype=np.uint64))


def test_bimanual_impedance_executes_coupled_twelve_dimensional_task() -> None:
    batch_size = 1
    joint_dim = 14
    task_dim = 12
    config = ControllerConfig(
        Device.CPU,
        DType.FLOAT64,
        batch_size=batch_size,
        joint_dim=joint_dim,
        cartesian_frame=CartesianFrame.MAIN_FRAME_ORIENTATION,
        task_dim=task_dim,
    )
    jacobian = np.zeros((batch_size, task_dim, joint_dim), dtype=np.float64)
    jacobian[0, :, :task_dim] = np.eye(task_dim, dtype=np.float64)
    mass_storage = OperationalSpaceMassStorage(
        cartesian_mass_matrix=np.zeros((batch_size, task_dim, task_dim), dtype=np.float64),
        parameter_status=np.zeros(batch_size, dtype=np.uint64),
        status=np.zeros(batch_size, dtype=np.uint64),
        scratch=OperationalSpaceMassScratch(
            joint_factor=np.zeros((batch_size, joint_dim, joint_dim), dtype=np.float64),
            solve_rhs=np.zeros((batch_size, joint_dim, task_dim), dtype=np.float64),
            cartesian_inverse=np.zeros((batch_size, task_dim, task_dim), dtype=np.float64),
        ),
    )
    mass = OperationalSpaceMass(
        config,
        OperationalSpaceMassParameters(regularization=shared(np.array(1.0e-6))),
        mass_storage,
    )
    mass.reset()
    mass.step(np.eye(joint_dim, dtype=np.float64)[None, :, :], jacobian)

    output = _output_storage(batch_size, task_dim)
    controller = BimanualCartesianImpedanceController(
        config,
        CartesianImpedanceParameters(
            stiffness_matrix=shared(np.eye(task_dim, dtype=np.float64)),
            damping_factors=shared(np.ones(task_dim, dtype=np.float64)),
        ),
        CartesianImpedanceStorage(
            output=output,
            state=_inertial_state(batch_size, joint_dim, task_dim),
        ),
    )
    controller.reset()
    controller.step(
        np.zeros((batch_size, task_dim), dtype=np.float64),
        np.zeros((batch_size, task_dim), dtype=np.float64),
        np.zeros((batch_size, task_dim), dtype=np.float64),
        np.zeros((batch_size, joint_dim), dtype=np.float64),
        mass_storage.cartesian_mass_matrix,
        jacobian,
        np.full(batch_size, 0.01, dtype=np.float64),
    )

    np.testing.assert_array_equal(output.command, np.zeros((batch_size, task_dim)))
    np.testing.assert_array_equal(output.status, np.zeros(batch_size, dtype=np.uint64))


def test_bimanual_impedance_rejects_non_bimanual_config() -> None:
    with pytest.raises(ValueError, match="joint_dim == 14"):
        BimanualCartesianImpedanceController(
            ControllerConfig(
                Device.CPU,
                DType.FLOAT64,
                batch_size=1,
                joint_dim=13,
                cartesian_frame=CartesianFrame.MAIN_FRAME_ORIENTATION,
                task_dim=12,
            ),
            CartesianImpedanceParameters(
                stiffness_matrix=shared(np.eye(12, dtype=np.float64)),
                damping_factors=shared(np.ones(12, dtype=np.float64)),
            ),
            CartesianImpedanceStorage(
                output=_output_storage(1, 12),
                state=_inertial_state(1, 13, 12),
            ),
        )


def test_bimanual_task_kinematics_constructs_decoupled_midpoint_coordinates() -> None:
    config = BimanualTaskConfig(
        Device.CPU,
        DType.FLOAT64,
        batch_size=1,
        joint_dim=14,
        cartesian_frame=CartesianFrame.MAIN_FRAME_ORIENTATION,
        task_dim=12,
        frames=BimanualFrameConfig(),
    )
    task_pose_error = np.zeros((1, 12), dtype=np.float64)
    task_jacobian = np.zeros((1, 12, 14), dtype=np.float64)
    status = np.zeros(1, dtype=np.uint64)
    kinematics = BimanualTaskKinematics(
        config,
        BimanualTaskStorage(task_pose_error, task_jacobian, status),
    )
    current_absolute = np.eye(4, dtype=np.float64)[None, :, :].copy()
    desired_absolute = current_absolute.copy()
    desired_absolute[0, 1, 3] = 0.0
    desired_absolute[0, :3, :3] = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    arm1_pose = current_absolute.copy()
    arm2_pose = current_absolute.copy()
    arm2_pose[0, 1, 3] = -0.4
    desired_relative = arm2_pose.copy()
    arm1_jacobian = np.zeros((1, 6, 7), dtype=np.float64)
    arm2_jacobian = np.zeros((1, 6, 7), dtype=np.float64)

    kinematics.reset()
    kinematics.step(
        desired_absolute,
        arm1_pose,
        arm2_pose,
        desired_relative,
        arm1_jacobian,
        arm2_jacobian,
    )

    np.testing.assert_allclose(task_pose_error[0, :3], (0.0, 0.2, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(task_pose_error[0, 3:6], (0.0, 0.0, np.pi / 2.0))
    np.testing.assert_allclose(task_pose_error[0, 6:], np.zeros(6), atol=1.0e-12)
    np.testing.assert_array_equal(task_jacobian, np.zeros_like(task_jacobian))
    np.testing.assert_array_equal(status, np.zeros(1, dtype=np.uint64))


def test_complete_bimanual_controller_uses_independent_arm_nullspaces() -> None:
    batch_size = 1
    joint_dim = 14
    task_dim = 12
    config = BimanualTaskConfig(
        Device.CPU,
        DType.FLOAT64,
        batch_size=batch_size,
        joint_dim=joint_dim,
        cartesian_frame=CartesianFrame.MAIN_FRAME_ORIENTATION,
        task_dim=task_dim,
        frames=BimanualFrameConfig(),
    )
    kinematics_storage = BimanualTaskStorage(
        np.zeros((batch_size, task_dim), dtype=np.float64),
        np.zeros((batch_size, task_dim, joint_dim), dtype=np.float64),
        np.zeros(batch_size, dtype=np.uint64),
    )
    mass_storage = OperationalSpaceMassStorage(
        cartesian_mass_matrix=np.zeros((batch_size, task_dim, task_dim), dtype=np.float64),
        parameter_status=np.zeros(batch_size, dtype=np.uint64),
        status=np.zeros(batch_size, dtype=np.uint64),
        scratch=OperationalSpaceMassScratch(
            joint_factor=np.zeros((batch_size, joint_dim, joint_dim), dtype=np.float64),
            solve_rhs=np.zeros((batch_size, joint_dim, task_dim), dtype=np.float64),
            cartesian_inverse=np.zeros((batch_size, task_dim, task_dim), dtype=np.float64),
        ),
    )
    task_storage = CartesianImpedanceStorage(
        output=_output_storage(batch_size, task_dim),
        state=_inertial_state(batch_size, joint_dim, task_dim),
    )
    nullspace_storage = _output_storage(batch_size, joint_dim)
    filter_output = _output_storage(batch_size, joint_dim)
    output_storage = TaskWrenchToJointTorqueStorage(
        unfiltered_joint_torque=np.zeros((batch_size, joint_dim), dtype=np.float64),
        filter=TorqueFilterStorage(
            output=filter_output,
            state=TorqueFilterState(
                previous_command=np.zeros((batch_size, joint_dim), dtype=np.float64),
                initialized=np.zeros(batch_size, dtype=np.bool_),
            ),
        ),
    )
    stiffness = np.eye(joint_dim, dtype=np.float64) * 5.0
    damping = np.eye(joint_dim, dtype=np.float64)
    parameters = CartesianTorqueParameters(
        mass=OperationalSpaceMassParameters(
            regularization=shared(np.array(1.0e-6, dtype=np.float64)),
            minimum_pivot=shared(np.array(1.0e-12, dtype=np.float64)),
        ),
        task=CartesianImpedanceParameters(
            stiffness_matrix=shared(np.eye(task_dim, dtype=np.float64)),
            damping_factors=shared(np.ones(task_dim, dtype=np.float64)),
        ),
        nullspace=NullspaceParameters(
            stiffness_matrix=shared(stiffness),
            damping_matrix=shared(damping),
        ),
        output=TorqueFilterParameters(
            lower_limit=shared(np.full(joint_dim, -1000.0, dtype=np.float64)),
            upper_limit=shared(np.full(joint_dim, 1000.0, dtype=np.float64)),
            maximum_rate=shared(np.full(joint_dim, 1.0e9, dtype=np.float64)),
        ),
    )
    storage = BimanualCartesianImpedanceTorqueStorage(
        kinematics=kinematics_storage,
        mass=mass_storage,
        task=task_storage,
        nullspace=nullspace_storage,
        output=output_storage,
        task_velocity_error=np.zeros((batch_size, task_dim), dtype=np.float64),
    )
    controller = BimanualCartesianImpedanceTorqueController(config, parameters, storage)
    controller.reset()
    pose = np.eye(4, dtype=np.float64)[None, :, :]
    arm1_jacobian = np.zeros((batch_size, 6, 7), dtype=np.float64)
    arm2_jacobian = np.zeros_like(arm1_jacobian)
    arm1_jacobian[0, :, :6] = np.eye(6, dtype=np.float64)
    arm2_jacobian[0, :, :6] = np.eye(6, dtype=np.float64)
    position_error = np.zeros((batch_size, joint_dim), dtype=np.float64)
    position_error[0, 6] = 0.2
    position_error[0, 13] = -0.4
    controller.step(
        pose,
        pose,
        pose,
        pose,
        arm1_jacobian,
        arm2_jacobian,
        np.zeros((batch_size, task_dim), dtype=np.float64),
        np.zeros((batch_size, task_dim), dtype=np.float64),
        position_error,
        np.zeros((batch_size, joint_dim), dtype=np.float64),
        np.eye(joint_dim, dtype=np.float64)[None, :, :],
        np.zeros((batch_size, joint_dim), dtype=np.float64),
        np.full(batch_size, 0.001, dtype=np.float64),
    )

    np.testing.assert_allclose(controller.output[0, (6, 13)], (1.0, -2.0), atol=1.0e-9)
    np.testing.assert_array_equal(controller.status, np.zeros(batch_size, dtype=np.uint64))


def test_every_controller_executes_through_the_public_lifecycle() -> None:
    batch_size = 1
    joint_dim = 6
    config = ControllerConfig(
        Device.CPU,
        DType.FLOAT64,
        batch_size=batch_size,
        joint_dim=joint_dim,
        cartesian_frame=CartesianFrame.ROBOT_BASE,
    )
    zeros6 = np.zeros((batch_size, 6), dtype=np.float64)
    identity6 = np.eye(6, dtype=np.float64)
    batch_identity6 = identity6[None, :, :].copy()
    dt = np.full(batch_size, 0.01, dtype=np.float64)

    filter_output = _output_storage(batch_size, joint_dim)
    torque_filter = TorqueFilter(
        config,
        TorqueFilterParameters(
            lower_limit=shared(np.full(joint_dim, -1.0, dtype=np.float64)),
            upper_limit=shared(np.ones(joint_dim, dtype=np.float64)),
            maximum_rate=shared(np.full(joint_dim, 10.0, dtype=np.float64)),
        ),
        TorqueFilterStorage(
            output=filter_output,
            state=TorqueFilterState(
                previous_command=np.zeros((batch_size, joint_dim), dtype=np.float64),
                initialized=np.zeros(batch_size, dtype=np.bool_),
            ),
        ),
    )
    torque_filter.reset()
    torque_filter.step(np.full((batch_size, joint_dim), 2.0, dtype=np.float64), dt)
    assert np.all(np.isfinite(filter_output.command))
    assert np.all(np.abs(filter_output.command) <= 1.0)

    nullspace_output = _output_storage(batch_size, joint_dim)
    nullspace = NullspaceController(
        config,
        NullspaceParameters(
            stiffness_matrix=shared(identity6.copy()),
            damping_matrix=shared(identity6.copy()),
        ),
        nullspace_output,
    )
    nullspace.reset()
    nullspace.step(zeros6, zeros6, batch_identity6, batch_identity6)
    np.testing.assert_allclose(nullspace_output.command, zeros6)
    np.testing.assert_array_equal(nullspace_output.status, np.zeros(batch_size, dtype=np.uint64))

    mass_storage = OperationalSpaceMassStorage(
        cartesian_mass_matrix=np.zeros((batch_size, 6, 6), dtype=np.float64),
        parameter_status=np.zeros(batch_size, dtype=np.uint64),
        status=np.zeros(batch_size, dtype=np.uint64),
        scratch=OperationalSpaceMassScratch(
            joint_factor=np.zeros((batch_size, joint_dim, joint_dim), dtype=np.float64),
            solve_rhs=np.zeros((batch_size, joint_dim, 6), dtype=np.float64),
            cartesian_inverse=np.zeros((batch_size, 6, 6), dtype=np.float64),
        ),
    )
    mass = OperationalSpaceMass(
        config,
        OperationalSpaceMassParameters(
            regularization=shared(np.array(1.0e-6)),
        ),
        mass_storage,
    )
    mass.reset()
    mass.step(batch_identity6, batch_identity6)
    np.testing.assert_allclose(
        mass_storage.cartesian_mass_matrix,
        batch_identity6 / (1.0 + 1.0e-6),
        rtol=1.0e-12,
    )

    impedance_output = _output_storage(batch_size, 6)
    impedance = CartesianImpedanceController(
        config,
        CartesianImpedanceParameters(
            stiffness_matrix=shared(identity6.copy()),
            damping_factors=shared(np.ones(6, dtype=np.float64)),
        ),
        CartesianImpedanceStorage(
            output=impedance_output,
            state=_inertial_state(batch_size, joint_dim),
        ),
    )
    impedance.reset()
    impedance.step(zeros6, zeros6, zeros6, zeros6, batch_identity6, batch_identity6, dt)
    np.testing.assert_allclose(impedance_output.command, zeros6)

    operational_output = _output_storage(batch_size, joint_dim)
    assert OperationalSpaceController is OperationalSpaceInverseDynamicsController
    operational = OperationalSpaceInverseDynamicsController(
        config,
        OperationalSpaceParameters(
            position_gain=shared(np.ones(6, dtype=np.float64)),
            velocity_gain=shared(np.ones(6, dtype=np.float64)),
        ),
        operational_output,
    )
    operational.reset()
    operational.step(
        zeros6,
        zeros6,
        zeros6,
        zeros6,
        batch_identity6,
        batch_identity6,
        zeros6,
    )
    np.testing.assert_allclose(operational_output.command, zeros6)

    force_output = _output_storage(batch_size, 6)
    force = CartesianForceController(
        config,
        _force_parameters(),
        CartesianForceStorage(output=force_output, state=_force_state(batch_size)),
    )
    force.reset()
    force.step(zeros6, zeros6, zeros6, zeros6, dt)
    np.testing.assert_allclose(force_output.command, zeros6)

    unified_output = _output_storage(batch_size, 6)
    unified = CartesianUnifiedForceImpedanceController(
        config,
        CartesianUnifiedForceImpedanceParameters(
            stiffness_matrix=shared(identity6.copy()),
            damping_factors=shared(np.ones(6, dtype=np.float64)),
            force=_force_parameters(),
        ),
        UnifiedControllerStorage(
            output=unified_output,
            state=UnifiedControllerState(
                impedance=_inertial_state(batch_size, joint_dim),
                force=_force_state(batch_size),
            ),
            scratch=UnifiedControllerScratch(
                impedance_wrench=np.zeros((batch_size, 6), dtype=np.float64),
                force_wrench=np.zeros((batch_size, 6), dtype=np.float64),
                impedance_status=np.zeros(batch_size, dtype=np.uint64),
                force_status=np.zeros(batch_size, dtype=np.uint64),
            ),
        ),
    )
    unified.reset()
    unified.step(
        zeros6,
        zeros6,
        zeros6,
        zeros6,
        batch_identity6,
        batch_identity6,
        zeros6,
        zeros6,
        zeros6,
        dt,
    )
    np.testing.assert_allclose(unified_output.command, zeros6)

    variable_output = _output_storage(batch_size, 6)
    variable = VariableCartesianImpedanceController(
        config,
        VariableCartesianImpedanceParameters(
            initial_stiffness_matrix=shared(identity6.copy()),
            maximum_stiffness=shared(np.full(6, 2.0, dtype=np.float64)),
            minimum_stiffness=shared(np.zeros(6, dtype=np.float64)),
            maximum_stiffness_rate=shared(np.full(6, 10.0, dtype=np.float64)),
            damping_factors=shared(np.ones(6, dtype=np.float64)),
        ),
        VariableImpedanceStorage(
            output=variable_output,
            state=VariableImpedanceState(
                current_stiffness_matrix=np.zeros((batch_size, 6, 6), dtype=np.float64),
                previous_stiffness_matrix=np.zeros((batch_size, 6, 6), dtype=np.float64),
                stiffness_initialized=np.zeros(batch_size, dtype=np.bool_),
                inertial=_inertial_state(batch_size, joint_dim),
            ),
            scratch=VariableImpedanceScratch(
                stiffness_wrench=np.zeros((batch_size, 6), dtype=np.float64),
                damping_wrench=np.zeros((batch_size, 6), dtype=np.float64),
                inertial_wrench=np.zeros((batch_size, 6), dtype=np.float64),
                stiffness_status=np.zeros(batch_size, dtype=np.uint64),
                damping_status=np.zeros(batch_size, dtype=np.uint64),
                inertial_status=np.zeros(batch_size, dtype=np.uint64),
            ),
        ),
    )
    variable.reset()
    variable.step(
        zeros6,
        batch_identity6,
        zeros6,
        zeros6,
        zeros6,
        batch_identity6,
        batch_identity6,
        dt,
    )
    np.testing.assert_allclose(variable_output.command, zeros6)


def test_static_impedance_and_unified_force_admittance_python_surface() -> None:
    joint_output = _output_storage(1, 2)
    joint = StaticJointImpedanceController(
        JointBlockConfig(
            device=Device.CPU,
            dtype=DType.FLOAT64,
            batch_size=1,
            joint_dim=2,
            mode=ComponentMode.DIAGONAL,
        ),
        StaticJointImpedanceParameters(
            stiffness=shared(np.array([10.0, 20.0])),
            damping=shared(np.array([2.0, 4.0])),
        ),
        joint_output,
    )
    joint.step(np.array([[1.0, -1.0]]), np.array([[0.5, 0.25]]))
    np.testing.assert_allclose(joint_output.command, [[11.0, -19.0]])

    config = ControllerConfig(
        device=Device.CPU,
        dtype=DType.FLOAT64,
        batch_size=1,
        cartesian_frame=CartesianFrame.ROBOT_BASE,
    )
    cartesian_output = _output_storage(1, 6)
    cartesian = StaticCartesianImpedanceController(
        config,
        StaticCartesianImpedanceParameters(
            stiffness_matrix=shared(np.eye(6)), damping_matrix=shared(np.eye(6))
        ),
        cartesian_output,
    )
    cartesian.step(np.ones((1, 6)), np.full((1, 6), 2.0))
    np.testing.assert_allclose(cartesian_output.command, np.full((1, 6), 3.0))

    output = _output_storage(1, 6)
    admittance = AdmittanceState(
        pose_offset=np.zeros((1, 6)),
        twist=np.zeros((1, 6)),
        pose_previous_signal=np.zeros((1, 6)),
        twist_previous_signal=np.zeros((1, 6)),
        pose_integrator_initialized=np.zeros(1, dtype=np.bool_),
        twist_integrator_initialized=np.zeros(1, dtype=np.bool_),
    )
    force_state = _force_state(1)
    unified = CartesianUnifiedForceAdmittanceController(
        config,
        CartesianUnifiedForceAdmittanceParameters(
            stiffness_matrix=shared(np.zeros((6, 6))),
            damping_matrix=shared(np.zeros((6, 6))),
            virtual_mass_matrix=shared(np.eye(6)),
            acceleration_limit=shared(np.full(6, 100.0)),
            twist_limit=shared(np.full(6, 100.0)),
            pose_limit=shared(np.full(6, 100.0)),
            force=ForceErrorPidParameters(
                proportional_gain=shared(np.ones(6)),
                differentiator_gain=shared(np.zeros(6)),
                integral_gain=shared(np.zeros(6)),
                initial_integral=shared(np.zeros(6)),
                integral_lower_limit=shared(np.full(6, -10.0)),
                integral_upper_limit=shared(np.full(6, 10.0)),
                error_deadzone=shared(np.full(6, 0.5)),
            ),
        ),
        CartesianUnifiedForceAdmittanceStorage(
            output=output,
            state=UnifiedForceAdmittanceState(force=force_state, admittance=admittance),
        ),
    )
    unified.step(np.ones((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)), np.ones(1))
    np.testing.assert_allclose(output.command, np.full((1, 6), -0.5))
    pose_before_failure = admittance.pose_offset.copy()
    force_before_failure = force_state.differentiator_previous_error.copy()
    unified.step(np.ones((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)), np.zeros(1))
    np.testing.assert_array_equal(admittance.pose_offset, pose_before_failure)
    np.testing.assert_array_equal(force_state.differentiator_previous_error, force_before_failure)


def _admittance_with_optional_limits(runtime, dtype, kind, limits):
    zeros = lambda shape: runtime.zeros(shape, dtype)
    flags = lambda: runtime.zeros(1, np.bool_)
    parameter = lambda values: shared(runtime.array(np.asarray(values, dtype=dtype)))
    config = ControllerConfig(
        runtime.device,
        DType(np.dtype(dtype).name),
        1,
        cartesian_frame=CartesianFrame.ROBOT_BASE,
    )
    state = AdmittanceState(
        pose_offset=zeros((1, 6)),
        twist=zeros((1, 6)),
        pose_previous_signal=zeros((1, 6)),
        twist_previous_signal=zeros((1, 6)),
        pose_integrator_initialized=flags(),
        twist_integrator_initialized=flags(),
    )
    output = OutputStorage(
        command=zeros((1, 6)),
        parameter_status=runtime.zeros(1, np.uint64),
        status=runtime.zeros(1, np.uint64),
        safety=SafetyState(zeros((1, 6)), flags(), runtime.zeros(1, np.uint64)),
    )
    parameters = dict(
        stiffness_matrix=parameter(np.zeros((6, 6))),
        damping_matrix=parameter(np.zeros((6, 6))),
        virtual_mass_matrix=parameter(np.eye(6)),
        **{
            name: None if value is None else parameter(np.full(6, value))
            for name, value in limits.items()
        },
    )
    if kind == "normal":
        controller = CartesianAdmittancePoseController(
            config,
            CartesianAdmittanceParameters(
                **parameters, force_filter_threshold=parameter(np.zeros(6))
            ),
            CartesianAdmittanceStorage(output=output, state=state),
            runtime.context,
        )
    else:
        force_state = ForceControllerState(
            differentiator_previous_error=zeros((1, 6)),
            differentiator_initialized=flags(),
            integrator_previous_error=zeros((1, 6)),
            integrated_error=zeros((1, 6)),
            integrator_initialized=flags(),
        )
        controller = CartesianUnifiedForceAdmittanceController(
            config,
            CartesianUnifiedForceAdmittanceParameters(
                **parameters,
                force=ForceErrorPidParameters(
                    proportional_gain=parameter(np.zeros(6)),
                    differentiator_gain=parameter(np.zeros(6)),
                    integral_gain=parameter(np.zeros(6)),
                    initial_integral=parameter(np.zeros(6)),
                    integral_lower_limit=None,
                    integral_upper_limit=None,
                    error_deadzone=parameter(np.zeros(6)),
                ),
            ),
            CartesianUnifiedForceAdmittanceStorage(
                output=output, state=UnifiedForceAdmittanceState(force_state, state)
            ),
            runtime.context,
        )
    controller.reset(context=runtime.context)
    return controller, output


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["normal", "force"])
@pytest.mark.parametrize(
    "limit,value,expected",
    [
        (None, None, (10.0, 1.0, 0.05)),
        ("acceleration_limit", 2.0, (2.0, 0.2, 0.01)),
        ("twist_limit", 0.4, (10.0, 0.4, 0.02)),
        ("pose_limit", 0.02, (10.0, 1.0, 0.02)),
    ],
)
def test_admittance_motion_limits_are_independently_optional(
    runtime, dtype, kind, limit, value, expected
):
    limits = dict(acceleration_limit=None, twist_limit=None, pose_limit=None)
    if limit is not None:
        limits[limit] = value
    controller, output = _admittance_with_optional_limits(runtime, dtype, kind, limits)
    signs = np.array([[1, -1, 1, -1, 1, -1]], dtype=dtype)
    zero = runtime.zeros((1, 6), dtype)
    feedforward = runtime.array(signs * 10.0)
    dt = runtime.array(np.array([0.1], dtype=dtype))
    if kind == "normal":
        # The normal controller initializes its trapezoidal integrators on the first step.
        controller.step(zero, feedforward, dt, runtime.context)
        controller.step(zero, feedforward, dt, runtime.context)
    else:
        controller.step(zero, zero, feedforward, dt, runtime.context)
    runtime.synchronize()
    for actual, magnitude in zip(
        (output.command, controller.twist, controller.pose_offset),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(runtime.numpy(actual), signs * magnitude, atol=1.0e-6)
    status = int(runtime.numpy(output.status)[0])
    assert not status & STATUS_SAFE_MODE
    assert bool(status & STATUS_SATURATION) == (limit is not None)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lower,upper", [(None, None), (-0.3, None), (None, 0.4), (-0.3, 0.4)])
def test_force_admittance_integral_bounds_are_optional(runtime, dtype, lower, upper):
    controller, output = _admittance_with_optional_limits(
        runtime,
        dtype,
        "force",
        dict(acceleration_limit=None, twist_limit=None, pose_limit=None),
    )
    bound = lambda value: (
        None if value is None else shared(runtime.array(np.full(6, value, dtype=dtype)))
    )
    force = dataclasses.replace(
        controller.parameters.force,
        integral_gain=shared(runtime.array(np.ones(6, dtype=dtype))),
        integral_lower_limit=bound(lower),
        integral_upper_limit=bound(upper),
    )
    controller.apply_parameters(
        dataclasses.replace(controller.parameters, force=force), runtime.context
    )
    signs = np.array([[1, -1, 1, -1, 1, -1]], dtype=dtype)
    zero = runtime.zeros((1, 6), dtype)
    controller.step(
        runtime.array(signs * 1000.0),
        zero,
        zero,
        runtime.array(np.array([0.1], dtype=dtype)),
        runtime.context,
    )
    runtime.synchronize()
    expected = np.where(
        signs < 0, 100.0 if upper is None else upper, -100.0 if lower is None else lower
    )
    np.testing.assert_allclose(runtime.numpy(output.command), expected, atol=1.0e-5)
    np.testing.assert_allclose(
        runtime.numpy(controller.storage.state.force.integrated_error),
        expected,
        atol=1.0e-5,
    )
    assert int(runtime.numpy(output.status)[0]) == 0


@pytest.mark.parametrize("limit", ["acceleration_limit", "twist_limit", "pose_limit"])
@pytest.mark.parametrize("invalid_value", [-1.0, float("nan"), float("inf")])
def test_force_admittance_still_rejects_invalid_supplied_limits(runtime, limit, invalid_value):
    limits = dict(acceleration_limit=None, twist_limit=None, pose_limit=None)
    limits[limit] = invalid_value
    controller, output = _admittance_with_optional_limits(runtime, np.float64, "force", limits)
    runtime.synchronize()
    assert int(runtime.numpy(output.parameter_status)[0]) & STATUS_INVALID_PARAMETER
    zero = runtime.zeros((1, 6), np.float64)
    controller.step(zero, zero, zero, runtime.array(np.array([0.1])), runtime.context)
    runtime.synchronize()
    assert int(runtime.numpy(output.status)[0]) & STATUS_SAFE_MODE
    np.testing.assert_array_equal(runtime.numpy(output.command), np.zeros((1, 6)))


@pytest.mark.parametrize("kind", ("joint_diagonal", "joint_dense", "cartesian"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_static_controllers_use_caller_stream_and_graph(runtime, kind, dtype):
    dimension = 6 if kind == "cartesian" else 2
    scalar_kind = DType.FLOAT32 if dtype == np.float32 else DType.FLOAT64
    output = runtime.tree(_output_storage(2, dimension))
    output = replace(output, command=runtime.zeros((2, dimension), dtype))
    # All floating storage must match the requested scalar type.
    output = replace(
        output,
        safety=replace(output.safety, last_valid_output=runtime.zeros((2, dimension), dtype)),
    )
    diagonal = kind == "joint_diagonal"
    stiffness = np.full(dimension, 2, dtype) if diagonal else np.eye(dimension, dtype=dtype) * 2
    damping = np.ones(dimension, dtype) if diagonal else np.eye(dimension, dtype=dtype)
    stiffness_parameter = shared(runtime.array(stiffness))
    damping_parameter = shared(runtime.array(damping))
    if kind == "cartesian":
        config = ControllerConfig(
            runtime.device, scalar_kind, 2, cartesian_frame=CartesianFrame.ROBOT_BASE
        )
        parameters = StaticCartesianImpedanceParameters(stiffness_parameter, damping_parameter)
        controller = StaticCartesianImpedanceController(config, parameters, output, runtime.context)
    else:
        config = JointBlockConfig(
            runtime.device,
            scalar_kind,
            2,
            dimension,
            mode=ComponentMode.DIAGONAL if diagonal else ComponentMode.DENSE,
        )
        parameters = StaticJointImpedanceParameters(stiffness_parameter, damping_parameter)
        controller = StaticJointImpedanceController(config, parameters, output, runtime.context)
    controller.reset(context=runtime.context)
    first = runtime.array(np.ones((2, dimension), dtype))
    second = runtime.array(np.ones((2, dimension), dtype))
    runtime.captured_step(lambda: controller.step(first, second, context=runtime.context))
    np.testing.assert_allclose(runtime.numpy(output.command), 3)
    np.testing.assert_array_equal(runtime.numpy(output.status), 0)
