# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Shared Isaac Lab plumbing for the parallel controller examples."""

from __future__ import annotations

# Standard Library
import importlib
import pathlib
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class EnergyTankMetrics:
    """Sampled force-tank diagnostics, separate from actuator saturation."""

    minimum_energy_j: float
    maximum_energy_j: float
    minimum_scale: float
    limited_row_samples: int
    maximum_balance_residual_j: float


@dataclass(frozen=True)
class CartesianFieldMetrics:
    """Observed point-field behavior, in meters and seconds in robot-base axes."""

    minimum_clearance_m: float | None
    maximum_goal_error_m: float | None
    minimum_settled_seconds: float | None
    repulsion_row_samples: int


@dataclass(frozen=True)
class MotionMetrics:
    """Observed simulator results returned by an example run.

    Distances are measured in the robot root frame in meters. Controller
    unsafe count reports distinct stopped rows; saturation counts accumulate
    row-steps across replicated environments. Native commands are recorded
    before converting to simulator actuator storage.
    """

    robot: str
    controller: str
    num_envs: int
    sim_steps: int
    minimum_env_displacement_m: float
    maximum_inter_env_spread_m: float
    unsafe_status_rows: int
    saturation_rows: int
    persistent_saturation_steps: int
    fault_isolation_verified: bool
    buffers_stable: bool
    maximum_tracking_error_m: float | None = None
    maximum_wrist_distance_error_m: float | None = None
    configuration: dict[str, Any] | None = None
    final_native_command_first_env: list[float] | None = None
    simulation_metadata: dict[str, Any] | None = None
    energy_tank: EnergyTankMetrics | None = None
    cartesian_field: CartesianFieldMetrics | None = None


def require_controller_module() -> Any:
    """Import this checkout's platform-neutral controller API."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    python_src = repo_root / "python" / "src"
    nvidia_src = python_src / "nvidia"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))

    nvidia_package = importlib.import_module("nvidia")
    package_path = list(getattr(nvidia_package, "__path__", []))
    if str(nvidia_src) not in package_path:
        nvidia_package.__path__ = [str(nvidia_src), *package_path]
    importlib.invalidate_caches()

    try:
        controller_module = importlib.import_module("nvidia.srl.controllers")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the controllers API. Build this repository "
            "in _build with ENABLE_CUDA=ON and ENABLE_PYBIND=ON using the Isaac "
            "Lab Python interpreter."
        ) from exc
    return controller_module


def current_execution_context(controller_module: Any, torch: Any) -> Any:
    """Return a per-call context for Torch's current CUDA stream."""
    stream = torch.cuda.current_stream()
    return controller_module.ExecutionContext(cuda_stream=int(stream.cuda_stream))


def status_mask(controller_module: Any, names: Iterable[str]) -> int:
    """Combine named controller status flags."""
    mask = 0
    for name in names:
        mask |= int(getattr(controller_module, name))
    return mask


def unsafe_status_mask(controller_module: Any) -> int:
    """Return status bits that require an environment-local safe action."""
    return status_mask(
        controller_module,
        (
            "STATUS_INVALID_INPUT",
            "STATUS_NON_FINITE_INPUT",
            "STATUS_INVALID_PARAMETER",
            "STATUS_INVALID_DT",
            "STATUS_UNSUPPORTED_MODE",
            "STATUS_SAFE_MODE",
        ),
    )


def flagged_rows(torch: Any, status: Any, mask: int) -> Any:
    """Return a CUDA bool vector selecting rows with any requested status bit."""
    return status.to(torch.int64).bitwise_and(mask) != 0


def capture_buffer_pointers(*roots: Any) -> dict[str, int]:
    """Capture all caller-owned tensor addresses in nested dictionaries."""
    pointers: dict[str, int] = {}

    def visit(prefix: str, value: Any) -> None:
        if hasattr(value, "data_ptr"):
            pointers[prefix] = int(value.data_ptr())
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(f"{prefix}.{key}" if prefix else key, value[key])
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(f"{prefix}.{index}", item)
        elif is_dataclass(value):
            for field in fields(value):
                visit(
                    f"{prefix}.{field.name}" if prefix else field.name,
                    getattr(value, field.name),
                )

    for index, root in enumerate(roots):
        visit(str(index), root)
    return pointers


def resolve_exact_joint_ids(robot: Any, joint_names: tuple[str, ...]) -> list[int]:
    """Resolve an ordered exact joint contract, reporting all available names."""
    by_name = {name: index for index, name in enumerate(robot.joint_names)}
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise RuntimeError(
            f"Missing required joints {missing}; available joints: {list(robot.joint_names)}"
        )
    return [by_name[name] for name in joint_names]


def resolve_end_effector(robot: Any, candidates: tuple[str, ...]) -> tuple[int, int, str]:
    """Resolve an end-effector body and its fixed-base PhysX Jacobian index."""
    for name in candidates:
        if name not in robot.body_names:
            continue
        body_index = list(robot.body_names).index(name)
        jacobian_index = body_index - 1 if robot.is_fixed_base else body_index
        jacobian_count = int(robot.root_physx_view.get_jacobians().shape[1])
        if 0 <= jacobian_index < jacobian_count:
            return body_index, jacobian_index, name
    raise RuntimeError(
        f"Could not resolve a Jacobian-capable end-effector from {candidates}; "
        f"available bodies: {list(robot.body_names)}"
    )


def make_parallel_scene(
    robot_cfg: Any,
    num_envs: int,
    env_spacing: float = 1.5,
    contact_body_name: str | None = None,
) -> Any:
    """Create a replicated robot scene with a ground plane and distant light."""
    # Third Party
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    @configclass
    class ParallelRobotSceneCfg(InteractiveSceneCfg):
        """Replicated scene used by the controller rollouts."""

        ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        robot = robot_cfg
        contact = (
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{contact_body_name}",
                update_period=0.0,
                history_length=1,
                track_air_time=False,
            )
            if contact_body_name is not None
            else None
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
        )

    return InteractiveScene(
        ParallelRobotSceneCfg(
            num_envs=num_envs,
            env_spacing=env_spacing,
            lazy_sensor_update=True,
        )
    )


def controller_observation_buffers(torch: Any, physics: dict[str, Any]) -> dict[str, Any]:
    """Allocate float64 controller observations separately from PhysX storage."""
    converted = {
        name: (
            tensor.to(dtype=torch.float64)
            if not name.startswith("_") and tensor.is_floating_point()
            else tensor
        )
        for name, tensor in physics.items()
    }
    converted["_physics"] = physics
    return converted


def update_controller_observations(observations: dict[str, Any]) -> None:
    """Convert observations on the caller stream without replacing buffers."""
    for name, tensor in observations["_physics"].items():
        if not name.startswith("_") and tensor.is_floating_point():
            observations[name].copy_(tensor)


def root_frame_kinematics(
    torch: Any,
    math_utils: Any,
    robot: Any,
    arm_joint_ids: list[int],
    ee_body_index: int,
    jacobian_index: int,
    frame_transforms: Any = None,
) -> dict[str, Any]:
    """Allocate controller-base/TCP observations and fill them from PhysX.

    Positions are meters, joint angles are radians, velocities are per second,
    efforts are Nm, and Isaac Lab quaternions use ``wxyz`` ordering.
    """
    num_envs = robot.data.joint_pos.shape[0]
    joint_count = len(arm_joint_ids)
    total_joint_count = robot.data.joint_pos.shape[1]
    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    # Third Party
    from _ur_adapter import CartesianFrameTransforms

    transforms = frame_transforms or CartesianFrameTransforms()

    def fixed(values: tuple[float, ...]) -> Any:
        return torch.tensor(values, dtype=dtype, device=device).repeat(num_envs, 1)

    buffers = {
        "joint_position": torch.empty((num_envs, joint_count), dtype=dtype, device=device),
        "joint_velocity": torch.empty((num_envs, joint_count), dtype=dtype, device=device),
        "jacobian": torch.empty((num_envs, 6, joint_count), dtype=dtype, device=device),
        "joint_mass_matrix": torch.empty(
            (num_envs, joint_count, joint_count), dtype=dtype, device=device
        ),
        "gravity": torch.empty((num_envs, joint_count), dtype=dtype, device=device),
        "coriolis": torch.empty((num_envs, joint_count), dtype=dtype, device=device),
        "ee_position_b": torch.empty((num_envs, 3), dtype=dtype, device=device),
        "ee_quat_b": torch.empty((num_envs, 4), dtype=dtype, device=device),
        "ee_velocity_b": torch.empty((num_envs, 6), dtype=dtype, device=device),
        "base_quat_w": torch.empty((num_envs, 4), dtype=dtype, device=device),
        "current_pose": (
            torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        ),
        "_joint_ids": torch.tensor(arm_joint_ids, dtype=torch.long, device=device),
        "_base_position_in_root": fixed(transforms.base_position_in_root_m),
        "_base_quaternion_in_root": fixed(transforms.base_quaternion_in_root_wxyz),
        "_tcp_position_in_body": fixed(transforms.tcp_position_in_body_m),
        "_tcp_quaternion_in_body": fixed(transforms.tcp_quaternion_in_body_wxyz),
        "_jacobian_w": torch.empty((num_envs, 6, joint_count), dtype=dtype, device=device),
        "_joint_mass_columns": torch.empty(
            (num_envs, total_joint_count, joint_count), dtype=dtype, device=device
        ),
    }
    update_root_frame_kinematics(
        torch,
        math_utils,
        robot,
        arm_joint_ids,
        ee_body_index,
        jacobian_index,
        buffers,
    )
    return controller_observation_buffers(torch, buffers)


def update_root_frame_kinematics(
    torch: Any,
    math_utils: Any,
    robot: Any,
    arm_joint_ids: list[int],
    ee_body_index: int,
    jacobian_index: int,
    buffers: dict[str, Any],
) -> None:
    """Update caller-owned root-frame observation buffers from PhysX."""
    if "_physics" in buffers:
        update_root_frame_kinematics(
            torch,
            math_utils,
            robot,
            arm_joint_ids,
            ee_body_index,
            jacobian_index,
            buffers["_physics"],
        )
        update_controller_observations(buffers)
        return
    joint_ids = buffers["_joint_ids"]
    torch.index_select(robot.data.joint_pos, 1, joint_ids, out=buffers["joint_position"])
    torch.index_select(robot.data.joint_vel, 1, joint_ids, out=buffers["joint_velocity"])

    jacobian_w_all = robot.root_physx_view.get_jacobians()[:, jacobian_index]
    torch.index_select(jacobian_w_all, 2, joint_ids, out=buffers["_jacobian_w"])
    base_position_w, base_quat_w = math_utils.combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        buffers["_base_position_in_root"],
        buffers["_base_quaternion_in_root"],
    )
    buffers["base_quat_w"].copy_(base_quat_w)
    base_rotation = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat_w))
    torch.bmm(
        base_rotation,
        buffers["_jacobian_w"][:, :3],
        out=buffers["jacobian"][:, :3],
    )
    torch.bmm(
        base_rotation,
        buffers["_jacobian_w"][:, 3:],
        out=buffers["jacobian"][:, 3:],
    )

    joint_mass_all = robot.root_physx_view.get_generalized_mass_matrices()
    torch.index_select(joint_mass_all, 2, joint_ids, out=buffers["_joint_mass_columns"])
    torch.index_select(
        buffers["_joint_mass_columns"],
        1,
        joint_ids,
        out=buffers["joint_mass_matrix"],
    )
    torch.index_select(
        robot.root_physx_view.get_gravity_compensation_forces(),
        1,
        joint_ids,
        out=buffers["gravity"],
    )
    torch.index_select(
        robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces(),
        1,
        joint_ids,
        out=buffers["coriolis"],
    )

    tcp_position_w, tcp_quat_w = math_utils.combine_frame_transforms(
        robot.data.body_pos_w[:, ee_body_index],
        robot.data.body_quat_w[:, ee_body_index],
        buffers["_tcp_position_in_body"],
        buffers["_tcp_quaternion_in_body"],
    )
    # PhysX link Jacobians describe centre-of-mass linear velocities. Shift
    # their evaluation point to the controller TCP, including the COM offset.
    tcp_offset_b = math_utils.quat_rotate_inverse(
        base_quat_w, tcp_position_w - robot.data.body_com_pos_w[:, ee_body_index]
    )
    buffers["jacobian"][:, :3].add_(
        torch.cross(
            buffers["jacobian"][:, 3:].transpose(1, 2), tcp_offset_b[:, None, :], dim=-1
        ).transpose(1, 2)
    )
    ee_position_b, ee_quat_b = math_utils.subtract_frame_transforms(
        base_position_w,
        base_quat_w,
        tcp_position_w,
        tcp_quat_w,
    )
    buffers["ee_position_b"].copy_(ee_position_b)
    buffers["ee_quat_b"].copy_(ee_quat_b)
    # Use the same root-frame Jacobian and selected joint velocities as the
    # controller to avoid the imported asset's first-update velocity transient.
    torch.bmm(
        buffers["jacobian"],
        buffers["joint_velocity"].unsqueeze(-1),
        out=buffers["ee_velocity_b"].unsqueeze(-1),
    )
    buffers["current_pose"][:, :3, :3].copy_(math_utils.matrix_from_quat(ee_quat_b))
    buffers["current_pose"][:, :3, 3].copy_(ee_position_b)


def update_motion_metrics(
    torch: Any,
    ee_position_b: Any,
    initial_position_b: Any,
    maximum_displacement: Any,
    maximum_spread: Any,
    active_mask: Any = None,
) -> None:
    """Update device-resident per-environment displacement and rollout spread."""
    displacement = torch.linalg.vector_norm(ee_position_b - initial_position_b, dim=1)
    torch.maximum(maximum_displacement, displacement, out=maximum_displacement)
    if active_mask is None:
        active_mask = torch.ones(
            ee_position_b.shape[0], dtype=torch.bool, device=ee_position_b.device
        )
    high = ee_position_b.masked_fill(~active_mask[:, None], -float("inf")).amax(dim=0)
    low = ee_position_b.masked_fill(~active_mask[:, None], float("inf")).amin(dim=0)
    spread = torch.where(active_mask.any(), torch.linalg.vector_norm(high - low), 0.0)
    torch.maximum(maximum_spread, spread, out=maximum_spread)


def validate_motion_metrics(
    metrics: MotionMetrics,
    *,
    minimum_displacement_m: float,
    maximum_spread_m: float,
    maximum_tracking_error_m: float | None = None,
) -> None:
    """Raise when a validation rollout did not demonstrate safe synchronized motion."""
    failures = []
    if maximum_tracking_error_m is not None and (
        metrics.maximum_tracking_error_m is None
        or metrics.maximum_tracking_error_m > maximum_tracking_error_m
    ):
        failures.append(
            f"tracking error {metrics.maximum_tracking_error_m} m "
            f"exceeds {maximum_tracking_error_m} m"
        )
    if metrics.minimum_env_displacement_m < minimum_displacement_m:
        failures.append(
            f"minimum per-environment displacement {metrics.minimum_env_displacement_m:.6f} m "
            f"is below {minimum_displacement_m:.6f} m"
        )
    if metrics.num_envs > 1 and metrics.maximum_inter_env_spread_m > maximum_spread_m:
        failures.append(
            f"inter-environment spread {metrics.maximum_inter_env_spread_m:.6f} m "
            f"exceeds {maximum_spread_m:.6f} m"
        )
    if metrics.unsafe_status_rows != 1:
        failures.append(
            f"expected exactly one injected unsafe row, observed {metrics.unsafe_status_rows}"
        )
    if not metrics.fault_isolation_verified:
        failures.append("one-row non-finite fault isolation and latched stop were not observed")
    if not metrics.buffers_stable:
        failures.append("caller-owned controller buffer addresses changed during the rollout")
    if metrics.persistent_saturation_steps > 10:
        failures.append(
            f"actuator saturation persisted for {metrics.persistent_saturation_steps} steps"
        )
    if failures:
        raise RuntimeError("; ".join(failures))


def update_tracking_error(torch: Any, error: Any, stopped: Any, maximum: Any) -> None:
    """Accumulate healthy-row positional tracking error without a host transfer."""
    norm = torch.linalg.vector_norm(error[:, :3], dim=1)
    norm = torch.where(~stopped & torch.isfinite(norm), norm, 0.0)
    maximum.copy_(torch.maximum(maximum, norm.amax()))
