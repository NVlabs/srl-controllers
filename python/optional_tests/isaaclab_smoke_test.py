# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Optional local IsaacLab integration tests for the CUDA controller backend.

This module intentionally lives outside ``python/tests`` and exits before
importing Torch or IsaacLab unless explicitly enabled. It uses IsaacLab as the
robot state and dynamics source, then exercises the private compiled backend
used by the public controller classes.

Frame and safety assumptions:
- IsaacLab quaternions are ``wxyz``.
- Cartesian pose/velocity/model tensors are expressed in the Franka root frame.
- The test performs no planning or collision checking; it only applies bounded
  smoke-test efforts for a few headless simulation steps.
"""

# Standard Library
import importlib
import importlib.util
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

# Third Party
import pytest

ENABLE_ENV_VAR = "NVIDIA_SRL_CONTROLLERS_ENABLE_ISAACLAB_TEST"
ISAACLAB_ROOT_ENV_VAR = "NVIDIA_SRL_CONTROLLERS_ISAACLAB_ROOT"
ENV_COUNTS_ENV_VAR = "NVIDIA_SRL_CONTROLLERS_ISAACLAB_ENV_COUNTS"
STEPS_ENV_VAR = "NVIDIA_SRL_CONTROLLERS_ISAACLAB_STEPS"
CONTROLLERS_ENV_VAR = "NVIDIA_SRL_CONTROLLERS_ISAACLAB_CONTROLLERS"

DEFAULT_ISAACLAB_ROOT = pathlib.Path("/home/ljohannsmeie/software/IsaacLab")
DEFAULT_ENV_COUNTS = (2, 16, 64)
DEFAULT_CONTROLLER_NAMES = (
    "joint_impedance",
    "nullspace",
    "operational_space",
    "cartesian_impedance",
    "cartesian_force",
    "cartesian_unified",
    "variable_cartesian_impedance",
    "variable_cartesian_unified",
    "cartesian_admittance",
)
CARTESIAN_DIM = 6
SIM_DT = 0.01
EFFORT_LIMIT = 40.0
ISAACLAB_TEST_ENABLED = os.environ.get(ENABLE_ENV_VAR, "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


pytestmark = [
    pytest.mark.isaaclab,
    pytest.mark.skipif(
        not ISAACLAB_TEST_ENABLED,
        reason=f"optional IsaacLab integration test disabled; set {ENABLE_ENV_VAR}=1 to run it",
    ),
]


@dataclass
class ControllerCase:
    """Caller-owned tensor buffers and metadata for one controller adapter."""

    name: str
    actuation: str
    parameters: dict[str, Any]
    state: dict[str, Any]
    scratch: dict[str, Any]
    output: Any
    parameter_status: Any
    reset_status: Any
    step_status: Any
    reset_mask: Any
    output_name: str
    scratch_status_names: tuple[str, ...] = ()
    initialized_state_names: tuple[str, ...] = ()
    pointer_snapshot: dict[str, int] | None = None


def _isaaclab_root() -> pathlib.Path:
    """Return the requested local IsaacLab checkout path."""
    return pathlib.Path(os.environ.get(ISAACLAB_ROOT_ENV_VAR, DEFAULT_ISAACLAB_ROOT))


def _parse_env_counts() -> tuple[int, ...]:
    """Parse the optional IsaacLab env-count tier list."""
    raw = os.environ.get(ENV_COUNTS_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_ENV_COUNTS
    env_counts = []
    for item in raw.split(","):
        value = item.strip()
        if value == "":
            continue
        try:
            count = int(value)
        except ValueError:
            pytest.fail(f"{ENV_COUNTS_ENV_VAR} contains a non-integer value: {value!r}")
        if count <= 0:
            pytest.fail(f"{ENV_COUNTS_ENV_VAR} values must be positive, got {count}")
        env_counts.append(count)
    if not env_counts:
        pytest.fail(f"{ENV_COUNTS_ENV_VAR} must contain at least one positive integer")
    return tuple(env_counts)


def _parse_step_count() -> int:
    """Parse the optional IsaacLab per-controller step count."""
    raw = os.environ.get(STEPS_ENV_VAR)
    if raw is None or raw.strip() == "":
        return 5
    try:
        steps = int(raw)
    except ValueError:
        pytest.fail(f"{STEPS_ENV_VAR} must be a positive integer, got {raw!r}")
    if steps <= 0:
        pytest.fail(f"{STEPS_ENV_VAR} must be positive, got {steps}")
    return steps


def _parse_controller_names() -> tuple[str, ...]:
    """Parse the optional controller adapter allowlist."""
    raw = os.environ.get(CONTROLLERS_ENV_VAR)
    if raw is None or raw.strip() == "":
        return DEFAULT_CONTROLLER_NAMES
    selected = tuple(name.strip() for name in raw.split(",") if name.strip())
    if not selected:
        pytest.fail(f"{CONTROLLERS_ENV_VAR} must contain at least one controller name")
    unknown = sorted(set(selected).difference(DEFAULT_CONTROLLER_NAMES))
    if unknown:
        pytest.fail(
            f"{CONTROLLERS_ENV_VAR} contains unsupported controller names {unknown}; "
            f"supported names are {list(DEFAULT_CONTROLLER_NAMES)}"
        )
    return selected


def _prepend_isaaclab_source_paths(root: pathlib.Path) -> None:
    """Make the local IsaacLab checkout importable for this optional test."""
    if not root.is_dir():
        pytest.fail(
            f"IsaacLab checkout not found at {root}. Set {ISAACLAB_ROOT_ENV_VAR} to the "
            "local IsaacLab checkout before enabling this test."
        )
    for relative in ("source/isaaclab_assets", "source/isaaclab"):
        path = root / relative
        if not path.is_dir():
            pytest.fail(f"expected IsaacLab source path does not exist: {path}")
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _make_local_controllers_namespace_visible() -> None:
    """Expose this checkout's ``nvidia.srl`` namespace to the IsaacLab interpreter."""
    repo_python_src = pathlib.Path(__file__).resolve().parents[1] / "src"
    repo_nvidia = repo_python_src / "nvidia"
    if not repo_nvidia.is_dir():
        pytest.fail(
            f"expected local controllers Python namespace path does not exist: {repo_nvidia}"
        )

    repo_python_src_text = str(repo_python_src)
    if repo_python_src_text not in sys.path:
        sys.path.insert(0, repo_python_src_text)

    try:
        nvidia_package = importlib.import_module("nvidia")
    except ImportError as exc:
        pytest.fail(f"could not import top-level nvidia package namespace: {exc}")

    # Some IsaacLab environments install a regular site-packages/nvidia package.
    # In that case PYTHONPATH alone cannot merge this checkout's PEP 420 namespace
    # portion, so explicitly add python/src/nvidia to the loaded package path.
    package_path = list(getattr(nvidia_package, "__path__", []))
    repo_nvidia_text = str(repo_nvidia)
    if repo_nvidia_text not in package_path:
        nvidia_package.__path__ = [repo_nvidia_text, *package_path]
    importlib.invalidate_caches()


def _assert_package_resolves_from_root(package_name: str, root: pathlib.Path) -> None:
    """Verify that an IsaacLab package resolves from the requested checkout."""
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        pytest.fail(
            f"could not import {package_name!r}. Add the IsaacLab source paths to PYTHONPATH "
            f"or run through {root / 'isaaclab.sh'}."
        )
    origin = pathlib.Path(spec.origin).resolve()
    if not origin.is_relative_to(root.resolve()):
        pytest.fail(
            f"{package_name!r} resolved from {origin}, not from the requested IsaacLab "
            f"checkout {root}."
        )


def _import_tensor_extension() -> Any:
    """Import the compiled controller backend or fail actionably."""
    try:
        cpp_module = importlib.import_module("nvidia.srl._controllers.nvidia_srl_controllers_cpp")
    except ImportError as exc:
        pytest.fail(
            "could not import nvidia.srl._controllers.nvidia_srl_controllers_cpp. "
            "Build this repository with ENABLE_CUDA=ON and ENABLE_PYBIND=ON using "
            "the IsaacLab Python interpreter, and include "
            "python/src on PYTHONPATH or run this optional test from the repository "
            "checkout so the build symlink under python/src/nvidia/srl/_controllers "
            "is visible. "
            f"Original import error: {exc}"
        )
    tensor_module = getattr(cpp_module, "tensor", None)
    if tensor_module is None:
        pytest.fail(
            "controllers pybind extension imported, but its private controller "
            "backend is missing. Rebuild with -DENABLE_CUDA=ON -DENABLE_PYBIND=ON."
        )
    return tensor_module


def _require_cuda_torch() -> Any:
    """Import Torch and require a local CUDA device."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        pytest.fail(f"Torch is required in the IsaacLab environment: {exc}")
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for the optional IsaacLab tensor integration test.")
    torch.cuda.set_device(0)
    return torch


def _make_effort_franka_cfg(franka_cfg: Any) -> Any:
    """Return a Franka config that accepts direct effort targets."""
    robot_cfg = franka_cfg.copy()
    for actuator_cfg in robot_cfg.actuators.values():
        actuator_cfg.stiffness = 0.0
        actuator_cfg.damping = 0.0
    return robot_cfg


def _assert_result_ok(result: dict[str, Any], context: str) -> None:
    """Assert that a backend lifecycle call succeeded."""
    assert result["ok"] is True, f"{context} failed: {result}"


def _status_mask(tensor_module: Any, names: tuple[str, ...]) -> int:
    """Combine named controller status flags into one integer mask."""
    mask = 0
    for name in names:
        mask |= int(getattr(tensor_module, name))
    return mask


def _assert_status_clean(torch: Any, status: Any, unsafe_mask: int, context: str) -> None:
    """Assert that all per-environment status flags avoid unsafe bits."""
    unsafe = status.to(torch.int64).bitwise_and(unsafe_mask)
    assert torch.all(unsafe == 0).item(), f"{context} unsafe status flags: {status.cpu()}"


def _assert_status_has(torch: Any, status: Any, required_mask: int, context: str) -> None:
    """Assert that all environments report a required status bit."""
    flags = status.to(torch.int64).bitwise_and(required_mask)
    assert torch.all(flags != 0).item(), f"{context} missing status flag: {status.cpu()}"


def _assert_finite(torch: Any, value: Any, name: str) -> None:
    """Assert that a CUDA tensor contains only finite values."""
    assert value.is_cuda, f"{name} is not a CUDA tensor"
    assert torch.all(torch.isfinite(value)).item(), f"{name} has non-finite values"


def _assert_cuda_row_major(tensor_module: Any, value: Any, name: str) -> None:
    """Assert that pybind sees a tensor as borrowed CUDA row-major storage."""
    descriptor = tensor_module.describe_array(value)
    assert descriptor["device"] == "cuda", f"{name} descriptor is not CUDA: {descriptor}"
    assert descriptor["layout"] == "row_major", f"{name} descriptor is not row-major: {descriptor}"


def _assert_all_buffers_cuda(tensors: dict[str, Any]) -> None:
    """Assert that all caller-owned tensors are device tensors."""
    for name, value in tensors.items():
        assert value.is_cuda, f"{name} is not a CUDA tensor"


def _iter_named_tensors(prefix: str, value: Any) -> list[tuple[str, Any]]:
    """Collect tensor leaves from a nested caller-owned buffer object."""
    if hasattr(value, "data_ptr") and hasattr(value, "is_cuda"):
        return [(prefix, value)]
    if isinstance(value, dict):
        tensors: list[tuple[str, Any]] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else key
            tensors.extend(_iter_named_tensors(child_prefix, value[key]))
        return tensors
    return []


def _capture_case_buffer_pointers(case: ControllerCase) -> dict[str, int]:
    """Capture caller-owned storage pointers for hot-path allocation checks."""
    roots = {
        "parameters": case.parameters,
        "state": case.state,
        "scratch": case.scratch,
        "output": case.output,
        "parameter_status": case.parameter_status,
        "reset_status": case.reset_status,
        "step_status": case.step_status,
        "reset_mask": case.reset_mask,
    }
    return {
        name: int(tensor.data_ptr())
        for root, value in roots.items()
        for name, tensor in _iter_named_tensors(root, value)
    }


def _assert_case_buffer_pointers_stable(case: ControllerCase) -> None:
    """Assert that controller-owned buffer storage did not change across steps."""
    assert case.pointer_snapshot is not None, f"{case.name} has no pointer snapshot"
    current = _capture_case_buffer_pointers(case)
    assert current == case.pointer_snapshot, (
        f"{case.name} caller-owned buffer pointers changed: "
        f"before={case.pointer_snapshot}, after={current}"
    )


def _resolve_single_body(robot: Any, patterns: tuple[str, ...]) -> tuple[int, str]:
    """Resolve one end-effector body by trying a small ordered set of names."""
    for pattern in patterns:
        try:
            body_ids, body_names = robot.find_bodies(pattern)
        except ValueError:
            continue
        if len(body_ids) == 1:
            return body_ids[0], body_names[0]
    pytest.fail(
        f"could not resolve a single Franka end-effector body from patterns {patterns}; "
        f"available bodies: {robot.body_names}"
    )


def _jacobian_body_index(robot: Any, body_index: int) -> int:
    """Return the PhysX Jacobian body index for fixed-base and floating-base assets."""
    if robot.is_fixed_base:
        return body_index - 1
    return body_index


def _pattern(torch: Any, num_envs: int, dim: int, dtype: Any, device: Any, scale: float) -> Any:
    """Return a deterministic nonzero batched pattern tensor."""
    values = torch.linspace(-1.0, 1.0, dim, dtype=dtype, device=device)
    env_scale = 1.0 + 0.01 * torch.arange(num_envs, dtype=dtype, device=device).unsqueeze(1)
    return (scale * env_scale * values.unsqueeze(0)).contiguous()


def _eye(torch: Any, dim: int, dtype: Any, device: Any, scale: float) -> Any:
    """Return a shared scaled identity matrix."""
    return torch.eye(dim, dtype=dtype, device=device) * scale


def _batched_eye(torch: Any, num_envs: int, dim: int, dtype: Any, device: Any, scale: float) -> Any:
    """Return a batched scaled identity matrix."""
    return _eye(torch, dim, dtype, device, scale).unsqueeze(0).repeat(num_envs, 1, 1).contiguous()


def _force_parameters(torch: Any, dtype: Any, device: Any) -> dict[str, Any]:
    """Create non-degenerate bounded force-controller parameters."""
    return {
        "proportional_gain": torch.full((CARTESIAN_DIM,), 0.35, dtype=dtype, device=device),
        "differentiator_gain": torch.full((CARTESIAN_DIM,), 0.02, dtype=dtype, device=device),
        "integral_gain": torch.full((CARTESIAN_DIM,), 0.01, dtype=dtype, device=device),
        "upper_limit": torch.full((CARTESIAN_DIM,), 3.0, dtype=dtype, device=device),
        "lower_limit": torch.full((CARTESIAN_DIM,), -3.0, dtype=dtype, device=device),
    }


def _cartesian_impedance_parameters(torch: Any, dtype: Any, device: Any) -> dict[str, Any]:
    """Create stable Cartesian impedance parameters."""
    return {
        "stiffness_matrix": _eye(torch, CARTESIAN_DIM, dtype, device, 18.0),
        "damping_factors": torch.full((CARTESIAN_DIM,), 0.65, dtype=dtype, device=device),
    }


def _cartesian_unified_parameters(torch: Any, dtype: Any, device: Any) -> dict[str, Any]:
    """Create stable unified force/impedance parameters."""
    parameters = _cartesian_impedance_parameters(torch, dtype, device)
    parameters.update(_force_parameters(torch, dtype, device))
    return parameters


def _variable_cartesian_parameters(torch: Any, dtype: Any, device: Any) -> dict[str, Any]:
    """Create stable variable Cartesian impedance parameters."""
    return {
        "initial_stiffness_matrix": _eye(torch, CARTESIAN_DIM, dtype, device, 12.0),
        "maximum_stiffness": torch.full((CARTESIAN_DIM,), 30.0, dtype=dtype, device=device),
        "minimum_stiffness": torch.full((CARTESIAN_DIM,), 4.0, dtype=dtype, device=device),
        "maximum_stiffness_rate": torch.full((CARTESIAN_DIM,), 800.0, dtype=dtype, device=device),
        "damping_factors": torch.full((CARTESIAN_DIM,), 0.65, dtype=dtype, device=device),
    }


def _variable_unified_parameters(torch: Any, dtype: Any, device: Any) -> dict[str, Any]:
    """Create stable variable unified force/impedance parameters."""
    parameters = _variable_cartesian_parameters(torch, dtype, device)
    parameters.update(_force_parameters(torch, dtype, device))
    return parameters


def _impedance_state(
    torch: Any, num_envs: int, joint_dim: int, dtype: Any, device: Any
) -> dict[str, Any]:
    """Create caller-owned Cartesian impedance state."""
    return {
        "previous_mass_matrix": torch.zeros(
            (num_envs, CARTESIAN_DIM, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "previous_jacobian": torch.zeros(
            (num_envs, CARTESIAN_DIM, joint_dim), dtype=dtype, device=device
        ),
        "initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }


def _force_state(torch: Any, num_envs: int, dtype: Any, device: Any) -> dict[str, Any]:
    """Create caller-owned Cartesian force-controller state."""
    return {
        "differentiator_previous_error": torch.zeros(
            (num_envs, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "differentiator_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "integrator_previous_error": torch.zeros(
            (num_envs, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "integrated_error": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "integrator_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }


def _unified_state(
    torch: Any, num_envs: int, joint_dim: int, dtype: Any, device: Any
) -> dict[str, Any]:
    """Create caller-owned unified force/impedance state."""
    state = _impedance_state(torch, num_envs, joint_dim, dtype, device)
    state.update(_force_state(torch, num_envs, dtype, device))
    return state


def _unified_scratch(torch: Any, num_envs: int, dtype: Any, device: Any) -> dict[str, Any]:
    """Create caller-owned unified force/impedance scratch."""
    return {
        "impedance_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "force_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "impedance_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
        "force_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
    }


def _variable_impedance_state(
    torch: Any, num_envs: int, joint_dim: int, dtype: Any, device: Any
) -> dict[str, Any]:
    """Create caller-owned variable Cartesian impedance state."""
    return {
        "current_stiffness_matrix": torch.zeros(
            (num_envs, CARTESIAN_DIM, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "previous_stiffness_matrix": torch.zeros(
            (num_envs, CARTESIAN_DIM, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "stiffness_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "previous_mass_matrix": torch.zeros(
            (num_envs, CARTESIAN_DIM, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "previous_jacobian": torch.zeros(
            (num_envs, CARTESIAN_DIM, joint_dim), dtype=dtype, device=device
        ),
        "inertial_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }


def _variable_impedance_scratch(
    torch: Any, num_envs: int, dtype: Any, device: Any
) -> dict[str, Any]:
    """Create caller-owned variable Cartesian impedance scratch."""
    return {
        "stiffness_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "damping_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "inertial_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "stiffness_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
        "damping_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
        "inertial_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
    }


def _variable_unified_state(
    torch: Any, num_envs: int, joint_dim: int, dtype: Any, device: Any
) -> dict[str, Any]:
    """Create caller-owned variable unified force/impedance state."""
    state = _variable_impedance_state(torch, num_envs, joint_dim, dtype, device)
    state.update(_force_state(torch, num_envs, dtype, device))
    return state


def _variable_unified_scratch(torch: Any, num_envs: int, dtype: Any, device: Any) -> dict[str, Any]:
    """Create caller-owned variable unified force/impedance scratch."""
    scratch = _variable_impedance_scratch(torch, num_envs, dtype, device)
    scratch.update(
        {
            "variable_impedance_wrench": torch.zeros(
                (num_envs, CARTESIAN_DIM), dtype=dtype, device=device
            ),
            "force_wrench": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
            "variable_impedance_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
            "force_status": torch.zeros(num_envs, dtype=torch.uint64, device=device),
        }
    )
    return scratch


def _admittance_state(torch: Any, num_envs: int, dtype: Any, device: Any) -> dict[str, Any]:
    """Create caller-owned Cartesian admittance state."""
    return {
        "admittance_pose": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "admittance_twist": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "pose_previous_signal": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "twist_previous_signal": torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device),
        "pose_integrator_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "twist_integrator_initialized": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }


def _make_controller_case(
    name: str, torch: Any, num_envs: int, joint_dim: int, dtype: Any, device: Any
) -> ControllerCase:
    """Create one high-level controller adapter with caller-owned CUDA buffers."""
    parameter_status = torch.zeros(num_envs, dtype=torch.uint64, device=device)
    reset_status = torch.zeros(num_envs, dtype=torch.uint64, device=device)
    step_status = torch.zeros(num_envs, dtype=torch.uint64, device=device)
    reset_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

    if name == "joint_impedance":
        parameters = {
            "stiffness_matrix": _eye(torch, joint_dim, dtype, device, 5.0),
            "damping_factors": torch.full((joint_dim,), 0.6, dtype=dtype, device=device),
        }
        output = torch.zeros((num_envs, joint_dim), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "joint_torque",
            parameters,
            {},
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "joint_torque",
        )

    if name == "nullspace":
        parameters = {
            "stiffness_matrix": _eye(torch, joint_dim, dtype, device, 3.0),
            "damping_matrix": _eye(torch, joint_dim, dtype, device, 0.15),
        }
        output = torch.zeros((num_envs, joint_dim), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "joint_torque",
            parameters,
            {},
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "joint_torque",
        )

    if name == "operational_space":
        output = torch.zeros((num_envs, joint_dim), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "joint_torque",
            _cartesian_impedance_parameters(torch, dtype, device),
            {},
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "joint_torque",
        )

    if name == "cartesian_impedance":
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "cartesian_wrench",
            _cartesian_impedance_parameters(torch, dtype, device),
            _impedance_state(torch, num_envs, joint_dim, dtype, device),
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "cartesian_wrench",
            initialized_state_names=("initialized",),
        )

    if name == "cartesian_force":
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "cartesian_wrench",
            _force_parameters(torch, dtype, device),
            _force_state(torch, num_envs, dtype, device),
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "cartesian_wrench",
            initialized_state_names=("differentiator_initialized", "integrator_initialized"),
        )

    if name == "cartesian_unified":
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "cartesian_wrench",
            _cartesian_unified_parameters(torch, dtype, device),
            _unified_state(torch, num_envs, joint_dim, dtype, device),
            _unified_scratch(torch, num_envs, dtype, device),
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "cartesian_wrench",
            scratch_status_names=("impedance_status", "force_status"),
            initialized_state_names=(
                "initialized",
                "differentiator_initialized",
                "integrator_initialized",
            ),
        )

    if name == "variable_cartesian_impedance":
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "cartesian_wrench",
            _variable_cartesian_parameters(torch, dtype, device),
            _variable_impedance_state(torch, num_envs, joint_dim, dtype, device),
            _variable_impedance_scratch(torch, num_envs, dtype, device),
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "cartesian_wrench",
            scratch_status_names=("stiffness_status", "damping_status", "inertial_status"),
            initialized_state_names=("stiffness_initialized", "inertial_initialized"),
        )

    if name == "variable_cartesian_unified":
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "cartesian_wrench",
            _variable_unified_parameters(torch, dtype, device),
            _variable_unified_state(torch, num_envs, joint_dim, dtype, device),
            _variable_unified_scratch(torch, num_envs, dtype, device),
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "cartesian_wrench",
            scratch_status_names=(
                "stiffness_status",
                "damping_status",
                "inertial_status",
                "variable_impedance_status",
                "force_status",
            ),
            initialized_state_names=(
                "stiffness_initialized",
                "inertial_initialized",
                "differentiator_initialized",
                "integrator_initialized",
            ),
        )

    if name == "cartesian_admittance":
        parameters = {
            "stiffness_matrix": _eye(torch, CARTESIAN_DIM, dtype, device, 4.0),
            "damping_matrix": _eye(torch, CARTESIAN_DIM, dtype, device, 1.0),
            "virtual_mass_matrix": _eye(torch, CARTESIAN_DIM, dtype, device, 2.5),
            "force_filter_threshold": torch.full(
                (CARTESIAN_DIM,), 0.01, dtype=dtype, device=device
            ),
            "acceleration_limit": torch.full((CARTESIAN_DIM,), 2.0, dtype=dtype, device=device),
            "twist_limit": torch.full((CARTESIAN_DIM,), 0.25, dtype=dtype, device=device),
            "pose_limit": torch.full((CARTESIAN_DIM,), 0.05, dtype=dtype, device=device),
        }
        output = torch.zeros((num_envs, CARTESIAN_DIM), dtype=dtype, device=device)
        return ControllerCase(
            name,
            "contract_only",
            parameters,
            _admittance_state(torch, num_envs, dtype, device),
            {},
            output,
            parameter_status,
            reset_status,
            step_status,
            reset_mask,
            "commanded_acceleration",
            initialized_state_names=("pose_integrator_initialized", "twist_integrator_initialized"),
        )

    pytest.fail(f"unhandled controller case: {name}")


def _compute_task_tensors(
    torch: Any, math_utils: Any, robot: Any, ee_body_index: int, jacobian_index: int
) -> dict[str, Any]:
    """Read IsaacLab-owned state/model tensors and build controller input tensors."""
    joint_position = robot.data.joint_pos
    joint_velocity = robot.data.joint_vel
    body_position_w = robot.data.body_pos_w
    body_quat_w = robot.data.body_quat_w
    body_velocity_w = robot.data.body_vel_w
    jacobian_w = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, :]
    joint_mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()
    gravity = robot.root_physx_view.get_gravity_compensation_forces()

    base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_quat_w))
    jacobian = torch.cat(
        (
            torch.bmm(base_rot, jacobian_w[:, :3, :]),
            torch.bmm(base_rot, jacobian_w[:, 3:, :]),
        ),
        dim=1,
    ).contiguous()

    ee_position_b, ee_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        body_position_w[:, ee_body_index],
        body_quat_w[:, ee_body_index],
    )
    relative_velocity_w = body_velocity_w[:, ee_body_index, :] - robot.data.root_vel_w
    ee_velocity_b = torch.empty(
        (joint_position.shape[0], CARTESIAN_DIM),
        dtype=joint_position.dtype,
        device=joint_position.device,
    )
    ee_velocity_b[:, :3] = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, :3]
    )
    ee_velocity_b[:, 3:] = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, 3:]
    )

    eye_joint = torch.eye(
        joint_mass_matrix.shape[-1], dtype=joint_position.dtype, device=joint_position.device
    )
    regularized_joint_mass = joint_mass_matrix + eye_joint.unsqueeze(0) * 1.0e-5
    solved = torch.linalg.solve(regularized_joint_mass, jacobian.transpose(1, 2))
    lambda_inv = torch.bmm(jacobian, solved)
    eye_task = torch.eye(CARTESIAN_DIM, dtype=joint_position.dtype, device=joint_position.device)
    cartesian_mass_matrix = torch.linalg.inv(lambda_inv + eye_task.unsqueeze(0) * 1.0e-4)
    # The CUDA damping design validates positive task inertia. The
    # IsaacLab/PhysX model tensors are authoritative; for this short integration
    # test use their derived diagonal operational-space inertia to avoid
    # rejecting a numerically ill-conditioned initial Franka pose.
    cartesian_mass_matrix = 0.5 * (cartesian_mass_matrix + cartesian_mass_matrix.transpose(1, 2))
    cartesian_mass_diagonal = torch.diagonal(cartesian_mass_matrix, dim1=1, dim2=2)
    cartesian_mass_diagonal = cartesian_mass_diagonal.clamp_min(1.0e-2)
    cartesian_mass_matrix = torch.diag_embed(cartesian_mass_diagonal).contiguous()

    num_envs = joint_position.shape[0]
    joint_dim = joint_position.shape[1]
    dtype = joint_position.dtype
    device = joint_position.device
    cartesian_pattern = _pattern(torch, num_envs, CARTESIAN_DIM, dtype, device, 1.0)
    joint_pattern = _pattern(torch, num_envs, joint_dim, dtype, device, 1.0)
    current_pose = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
    current_pose[:, :3, :3] = math_utils.matrix_from_quat(ee_quat_b)
    current_pose[:, :3, 3] = ee_position_b

    return {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "joint_position_error": (joint_pattern * 0.015 - joint_velocity * 0.0).contiguous(),
        "joint_velocity_error": (-joint_velocity + joint_pattern * 0.005).contiguous(),
        "joint_desired_acceleration": (joint_pattern * 0.01).contiguous(),
        "ee_position_b": ee_position_b.contiguous(),
        "ee_quat_b": ee_quat_b.contiguous(),
        "ee_velocity_b": ee_velocity_b.contiguous(),
        "current_pose": current_pose.contiguous(),
        "jacobian": jacobian,
        "joint_mass_matrix": joint_mass_matrix,
        "cartesian_mass_matrix": cartesian_mass_matrix,
        "gravity": gravity,
        "cartesian_pose_error": (cartesian_pattern * 0.01).contiguous(),
        "cartesian_velocity_error": (-ee_velocity_b + cartesian_pattern * 0.005).contiguous(),
        "desired_acceleration": (cartesian_pattern * 0.02).contiguous(),
        "jacobian_dot_joint_velocity": torch.zeros(
            (num_envs, CARTESIAN_DIM), dtype=dtype, device=device
        ),
        "variable_stiffness_matrix": _batched_eye(
            torch, num_envs, CARTESIAN_DIM, dtype, device, 24.0
        ),
        "external_wrench": (cartesian_pattern * 0.05).contiguous(),
        "desired_wrench": (cartesian_pattern * 0.08).contiguous(),
        "feedforward_wrench": (cartesian_pattern * 0.03).contiguous(),
        "dt": torch.full((num_envs,), SIM_DT, dtype=dtype, device=device),
    }


def _assert_model_shapes(observations: dict[str, Any], num_envs: int, joint_dim: int) -> None:
    """Assert that IsaacLab and derived model tensors match the controller contract."""
    assert observations["joint_position"].shape == (num_envs, joint_dim)
    assert observations["joint_velocity"].shape == (num_envs, joint_dim)
    assert observations["joint_position_error"].shape == (num_envs, joint_dim)
    assert observations["joint_velocity_error"].shape == (num_envs, joint_dim)
    assert observations["joint_desired_acceleration"].shape == (num_envs, joint_dim)
    assert observations["jacobian"].shape == (num_envs, CARTESIAN_DIM, joint_dim)
    assert observations["joint_mass_matrix"].shape == (num_envs, joint_dim, joint_dim)
    assert observations["cartesian_mass_matrix"].shape == (
        num_envs,
        CARTESIAN_DIM,
        CARTESIAN_DIM,
    )
    assert observations["current_pose"].shape == (num_envs, 4, 4)
    assert observations["variable_stiffness_matrix"].shape == (
        num_envs,
        CARTESIAN_DIM,
        CARTESIAN_DIM,
    )
    assert observations["gravity"].shape == (num_envs, joint_dim)


def _prepare_controller_case(
    tensor_module: Any, case: ControllerCase, observations: dict[str, Any]
) -> None:
    """Run the prepare lifecycle call for one adapter."""
    if case.name == "joint_impedance":
        result = tensor_module.prepare_joint_impedance_controller(
            observations["joint_position_error"],
            observations["joint_velocity_error"],
            observations["joint_desired_acceleration"],
            observations["joint_mass_matrix"],
            case.parameters,
            case.output,
        )
    elif case.name == "nullspace":
        result = tensor_module.prepare_nullspace_controller(
            observations["joint_position_error"],
            observations["joint_velocity"],
            observations["joint_mass_matrix"],
            observations["jacobian"],
            case.parameters,
            case.output,
        )
    elif case.name == "operational_space":
        result = tensor_module.prepare_operational_space_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["jacobian_dot_joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["gravity"],
            case.parameters,
            case.output,
        )
    elif case.name == "cartesian_impedance":
        result = tensor_module.prepare_cartesian_impedance_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["dt"],
            case.parameters,
            case.state,
            case.output,
        )
    elif case.name == "cartesian_force":
        result = tensor_module.prepare_cartesian_force_controller(
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["cartesian_pose_error"],
            observations["dt"],
            False,
            case.parameters,
            case.state,
            case.output,
        )
    elif case.name == "cartesian_unified":
        result = tensor_module.prepare_cartesian_unified_force_impedance_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            False,
            case.parameters,
            case.state,
            case.scratch,
            case.output,
        )
    elif case.name == "variable_cartesian_impedance":
        result = tensor_module.prepare_variable_cartesian_impedance_controller(
            observations["cartesian_pose_error"],
            observations["variable_stiffness_matrix"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["dt"],
            case.parameters,
            case.state,
            case.scratch,
            case.output,
        )
    elif case.name == "variable_cartesian_unified":
        result = tensor_module.prepare_variable_cartesian_unified_force_impedance_controller(
            observations["cartesian_pose_error"],
            observations["variable_stiffness_matrix"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            False,
            case.parameters,
            case.state,
            case.scratch,
            case.output,
        )
    elif case.name == "cartesian_admittance":
        result = tensor_module.prepare_cartesian_admittance_controller(
            observations["external_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            case.parameters,
            case.state,
            case.output,
        )
    else:
        pytest.fail(f"unhandled prepare case {case.name}")
    _assert_result_ok(result, f"prepare_{case.name}")


def _apply_controller_parameters(
    tensor_module: Any, case: ControllerCase, num_envs: int, joint_dim: int
) -> None:
    """Run the apply-parameters lifecycle call for one adapter."""
    if case.name == "joint_impedance":
        result = tensor_module.apply_joint_impedance_controller_parameters(
            case.parameters, num_envs, joint_dim, case.parameter_status
        )
    elif case.name == "nullspace":
        result = tensor_module.apply_nullspace_controller_parameters(
            case.parameters, num_envs, joint_dim, case.parameter_status
        )
    elif case.name == "operational_space":
        result = tensor_module.apply_operational_space_controller_parameters(
            case.parameters, num_envs, case.parameter_status
        )
    elif case.name == "cartesian_impedance":
        result = tensor_module.apply_cartesian_impedance_controller_parameters(
            case.parameters, num_envs, case.parameter_status
        )
    elif case.name == "cartesian_force":
        result = tensor_module.apply_cartesian_force_controller_parameters(
            case.parameters, num_envs, False, case.parameter_status
        )
    elif case.name == "cartesian_unified":
        result = tensor_module.apply_cartesian_unified_force_impedance_controller_parameters(
            case.parameters, num_envs, False, case.parameter_status
        )
    elif case.name == "variable_cartesian_impedance":
        result = tensor_module.apply_variable_cartesian_impedance_controller_parameters(
            case.parameters, num_envs, case.parameter_status
        )
    elif case.name == "variable_cartesian_unified":
        result = (
            tensor_module.apply_variable_cartesian_unified_force_impedance_controller_parameters(
                case.parameters, num_envs, False, case.parameter_status
            )
        )
    elif case.name == "cartesian_admittance":
        result = tensor_module.apply_cartesian_admittance_controller_parameters(
            case.parameters, num_envs, case.parameter_status
        )
    else:
        pytest.fail(f"unhandled apply case {case.name}")
    _assert_result_ok(result, f"apply_{case.name}_parameters")


def _reset_controller_case(
    tensor_module: Any, case: ControllerCase, num_envs: int, joint_dim: int
) -> None:
    """Run the reset lifecycle call for one adapter."""
    case.reset_status.zero_()
    if case.name == "joint_impedance":
        result = tensor_module.reset_joint_impedance_controller(
            num_envs, case.reset_mask, case.reset_status
        )
    elif case.name == "nullspace":
        result = tensor_module.reset_nullspace_controller(
            num_envs, case.reset_mask, case.reset_status
        )
    elif case.name == "operational_space":
        result = tensor_module.reset_operational_space_controller(
            num_envs, case.reset_mask, case.reset_status
        )
    elif case.name == "cartesian_impedance":
        result = tensor_module.reset_cartesian_impedance_controller(
            num_envs, joint_dim, case.state, case.reset_mask, case.reset_status
        )
    elif case.name == "cartesian_force":
        result = tensor_module.reset_cartesian_force_controller(
            num_envs, case.state, case.reset_mask, case.reset_status
        )
    elif case.name == "cartesian_unified":
        result = tensor_module.reset_cartesian_unified_force_impedance_controller(
            num_envs, joint_dim, case.state, case.reset_mask, case.reset_status
        )
    elif case.name == "variable_cartesian_impedance":
        result = tensor_module.reset_variable_cartesian_impedance_controller(
            case.parameters,
            case.parameter_status,
            num_envs,
            joint_dim,
            case.state,
            case.reset_mask,
            case.reset_status,
        )
    elif case.name == "variable_cartesian_unified":
        result = tensor_module.reset_variable_cartesian_unified_force_impedance_controller(
            case.parameters,
            case.parameter_status,
            num_envs,
            joint_dim,
            case.state,
            case.reset_mask,
            case.reset_status,
        )
    elif case.name == "cartesian_admittance":
        result = tensor_module.reset_cartesian_admittance_controller(
            num_envs, case.state, case.reset_mask, case.reset_status
        )
    else:
        pytest.fail(f"unhandled reset case {case.name}")
    _assert_result_ok(result, f"reset_{case.name}")


def _step_controller_case(
    tensor_module: Any, case: ControllerCase, observations: dict[str, Any]
) -> None:
    """Run one controller step into caller-owned output/status buffers."""
    case.step_status.zero_()
    for scratch_status_name in case.scratch_status_names:
        case.scratch[scratch_status_name].zero_()

    if case.name == "joint_impedance":
        result = tensor_module.step_joint_impedance_controller(
            observations["joint_position_error"],
            observations["joint_velocity_error"],
            observations["joint_desired_acceleration"],
            observations["joint_mass_matrix"],
            case.parameters,
            case.parameter_status,
            case.output,
            case.step_status,
        )
    elif case.name == "nullspace":
        result = tensor_module.step_nullspace_controller(
            observations["joint_position_error"],
            observations["joint_velocity"],
            observations["joint_mass_matrix"],
            observations["jacobian"],
            case.parameters,
            case.parameter_status,
            case.output,
            case.step_status,
        )
    elif case.name == "operational_space":
        result = tensor_module.step_operational_space_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["jacobian_dot_joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["gravity"],
            case.parameters,
            case.parameter_status,
            case.output,
            case.step_status,
        )
    elif case.name == "cartesian_impedance":
        result = tensor_module.step_cartesian_impedance_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["dt"],
            case.parameters,
            case.parameter_status,
            case.state,
            case.output,
            case.step_status,
        )
    elif case.name == "cartesian_force":
        result = tensor_module.step_cartesian_force_controller(
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["cartesian_pose_error"],
            observations["dt"],
            False,
            case.parameters,
            case.parameter_status,
            case.state,
            case.output,
            case.step_status,
        )
    elif case.name == "cartesian_unified":
        result = tensor_module.step_cartesian_unified_force_impedance_controller(
            observations["cartesian_pose_error"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            False,
            case.parameters,
            case.parameter_status,
            case.state,
            case.scratch,
            case.output,
            case.step_status,
        )
    elif case.name == "variable_cartesian_impedance":
        result = tensor_module.step_variable_cartesian_impedance_controller(
            observations["cartesian_pose_error"],
            observations["variable_stiffness_matrix"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["dt"],
            case.parameters,
            case.parameter_status,
            case.state,
            case.scratch,
            case.output,
            case.step_status,
        )
    elif case.name == "variable_cartesian_unified":
        result = tensor_module.step_variable_cartesian_unified_force_impedance_controller(
            observations["cartesian_pose_error"],
            observations["variable_stiffness_matrix"],
            observations["cartesian_velocity_error"],
            observations["desired_acceleration"],
            observations["joint_velocity"],
            observations["cartesian_mass_matrix"],
            observations["jacobian"],
            observations["external_wrench"],
            observations["desired_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            False,
            case.parameters,
            case.parameter_status,
            case.state,
            case.scratch,
            case.output,
            case.step_status,
        )
    elif case.name == "cartesian_admittance":
        result = tensor_module.step_cartesian_admittance_controller(
            observations["external_wrench"],
            observations["feedforward_wrench"],
            observations["dt"],
            case.parameters,
            case.parameter_status,
            case.state,
            case.output,
            case.step_status,
        )
    else:
        pytest.fail(f"unhandled step case {case.name}")
    _assert_result_ok(result, f"step_{case.name}")


def _assert_case_ready(
    torch: Any, tensor_module: Any, case: ControllerCase, unsafe_mask: int
) -> None:
    """Validate controller status, output, scratch status, state, and storage."""
    _assert_status_clean(torch, case.parameter_status, unsafe_mask, f"{case.name} parameter")
    _assert_status_clean(torch, case.step_status, unsafe_mask, f"{case.name} step")
    for scratch_status_name in case.scratch_status_names:
        _assert_status_clean(
            torch,
            case.scratch[scratch_status_name],
            unsafe_mask,
            f"{case.name} scratch {scratch_status_name}",
        )
    _assert_finite(torch, case.output, f"{case.name} {case.output_name}")
    _assert_case_buffer_pointers_stable(case)
    _assert_cuda_row_major(tensor_module, case.output, f"{case.name} {case.output_name}")


def _assert_state_initialized(torch: Any, case: ControllerCase) -> None:
    """Assert stateful controllers advanced their caller-owned initialized flags."""
    for name in case.initialized_state_names:
        initialized = case.state[name]
        assert torch.all(initialized).item(), f"{case.name} did not initialize state {name}"


def _joint_effort_from_case(torch: Any, case: ControllerCase, observations: dict[str, Any]) -> Any:
    """Convert a controller output into bounded direct joint effort targets."""
    if case.actuation == "joint_torque":
        joint_effort = case.output
    elif case.actuation == "cartesian_wrench":
        joint_effort = torch.bmm(
            observations["jacobian"].transpose(1, 2), case.output.unsqueeze(-1)
        ).squeeze(-1)
        joint_effort = joint_effort + observations["gravity"]
    elif case.actuation == "contract_only":
        joint_effort = torch.zeros_like(observations["joint_velocity"])
    else:
        pytest.fail(f"unhandled actuation mode {case.actuation!r} for {case.name}")
    joint_effort = torch.clamp(joint_effort, min=-EFFORT_LIMIT, max=EFFORT_LIMIT)
    _assert_finite(torch, joint_effort, f"{case.name} joint_effort")
    return joint_effort


def _capture_robot_state(robot: Any) -> dict[str, Any]:
    """Capture initial IsaacLab state for repeatable controller cases."""
    return {
        "root_pose": robot.data.root_pose_w.clone(),
        "root_velocity": robot.data.root_vel_w.clone(),
        "joint_position": robot.data.joint_pos.clone(),
        "joint_velocity": robot.data.joint_vel.clone(),
    }


def _restore_robot_state(torch: Any, robot: Any, scene: Any, state: dict[str, Any]) -> None:
    """Restore robot pose, velocity, joint state, and effort targets."""
    robot.write_root_pose_to_sim(state["root_pose"])
    robot.write_root_velocity_to_sim(state["root_velocity"])
    robot.write_joint_state_to_sim(state["joint_position"], state["joint_velocity"])
    robot.set_joint_effort_target(torch.zeros_like(state["joint_position"]))
    robot.write_data_to_sim()
    scene.update(SIM_DT)


def _assert_observations_clean(torch: Any, observations: dict[str, Any]) -> None:
    """Validate finite model tensors used by controller adapters."""
    for name in (
        "joint_position",
        "joint_velocity",
        "joint_position_error",
        "joint_velocity_error",
        "joint_desired_acceleration",
        "ee_position_b",
        "ee_quat_b",
        "ee_velocity_b",
        "current_pose",
        "jacobian",
        "joint_mass_matrix",
        "cartesian_mass_matrix",
        "gravity",
        "cartesian_pose_error",
        "cartesian_velocity_error",
        "desired_acceleration",
        "variable_stiffness_matrix",
        "external_wrench",
        "desired_wrench",
        "feedforward_wrench",
        "dt",
    ):
        _assert_finite(torch, observations[name], name)


def _run_controller_case(
    tensor_module: Any,
    torch: Any,
    math_utils: Any,
    robot: Any,
    scene: Any,
    sim: Any,
    initial_state: dict[str, Any],
    controller_name: str,
    num_envs: int,
    joint_dim: int,
    ee_body_index: int,
    jacobian_index: int,
    step_count: int,
    unsafe_mask: int,
) -> None:
    """Prepare, reset, and step one controller adapter in the IsaacLab scene."""
    _restore_robot_state(torch, robot, scene, initial_state)
    observations = _compute_task_tensors(torch, math_utils, robot, ee_body_index, jacobian_index)
    _assert_model_shapes(observations, num_envs, joint_dim)
    _assert_observations_clean(torch, observations)
    case = _make_controller_case(
        controller_name,
        torch,
        num_envs,
        joint_dim,
        robot.data.joint_pos.dtype,
        robot.data.joint_pos.device,
    )
    _assert_all_buffers_cuda(
        {
            name: tensor
            for root, value in {
                "parameters": case.parameters,
                "state": case.state,
                "scratch": case.scratch,
                "output": case.output,
                "parameter_status": case.parameter_status,
                "reset_status": case.reset_status,
                "step_status": case.step_status,
                "reset_mask": case.reset_mask,
            }.items()
            for name, tensor in _iter_named_tensors(root, value)
        }
    )
    _prepare_controller_case(tensor_module, case, observations)
    _apply_controller_parameters(tensor_module, case, num_envs, joint_dim)
    _assert_status_clean(torch, case.parameter_status, unsafe_mask, f"{case.name} parameter apply")
    _reset_controller_case(tensor_module, case, num_envs, joint_dim)
    _assert_status_has(
        torch, case.reset_status, int(tensor_module.STATUS_RESET_APPLIED), f"{case.name} reset"
    )
    _assert_status_clean(torch, case.reset_status, unsafe_mask, f"{case.name} reset")
    case.pointer_snapshot = _capture_case_buffer_pointers(case)

    for _ in range(step_count):
        observations = _compute_task_tensors(
            torch, math_utils, robot, ee_body_index, jacobian_index
        )
        _assert_model_shapes(observations, num_envs, joint_dim)
        _step_controller_case(tensor_module, case, observations)
        _assert_case_ready(torch, tensor_module, case, unsafe_mask)
        joint_effort = _joint_effort_from_case(torch, case, observations)
        robot.set_joint_effort_target(joint_effort)
        robot.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)

    torch.cuda.synchronize()
    _assert_state_initialized(torch, case)
    _assert_case_buffer_pointers_stable(case)
    _reset_controller_case(tensor_module, case, num_envs, joint_dim)
    _assert_status_has(
        torch,
        case.reset_status,
        int(tensor_module.STATUS_RESET_APPLIED),
        f"{case.name} final reset",
    )
    _assert_status_clean(torch, case.reset_status, unsafe_mask, f"{case.name} final reset")


def _run_status_isolation_case(
    tensor_module: Any,
    torch: Any,
    math_utils: Any,
    robot: Any,
    ee_body_index: int,
    jacobian_index: int,
    unsafe_mask: int,
) -> None:
    """Verify that one bad environment reports safe mode without contaminating others."""
    num_envs = int(robot.data.joint_pos.shape[0])
    if num_envs != 2:
        return
    joint_dim = int(robot.data.joint_pos.shape[1])
    observations = _compute_task_tensors(torch, math_utils, robot, ee_body_index, jacobian_index)
    case = _make_controller_case(
        "operational_space",
        torch,
        num_envs,
        joint_dim,
        robot.data.joint_pos.dtype,
        robot.data.joint_pos.device,
    )
    _prepare_controller_case(tensor_module, case, observations)
    _apply_controller_parameters(tensor_module, case, num_envs, joint_dim)
    _reset_controller_case(tensor_module, case, num_envs, joint_dim)
    isolated_observations = dict(observations)
    bad_pose_error = observations["cartesian_pose_error"].clone()
    bad_pose_error[1, 0] = float("nan")
    isolated_observations["cartesian_pose_error"] = bad_pose_error
    _step_controller_case(tensor_module, case, isolated_observations)
    nonfinite_safe_mask = _status_mask(
        tensor_module, ("STATUS_NON_FINITE_INPUT", "STATUS_SAFE_MODE")
    )
    status_flags = case.step_status.to(torch.int64)
    bad_flags = status_flags.bitwise_and(nonfinite_safe_mask)
    unsafe_flags = status_flags.bitwise_and(unsafe_mask)
    assert bad_flags[0].item() == 0, f"env 0 unexpectedly reported bad flags: {case.step_status}"
    assert (
        bad_flags[1].item() == nonfinite_safe_mask
    ), f"env 1 did not report non-finite safe mode: {case.step_status}"
    assert (
        unsafe_flags[0].item() == 0
    ), f"env 0 unexpectedly reported unsafe flags: {case.step_status}"
    _assert_finite(torch, case.output, "status isolation operational_space output")


def _run_isaaclab_env_count(
    tensor_module: Any,
    torch: Any,
    env_count: int,
    controller_names: tuple[str, ...],
    step_count: int,
) -> None:
    """Create one Franka scene and run the selected controllers at one env count."""
    # NVIDIA / IsaacLab
    # Third Party
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.utils import math as math_utils
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    franka_cfg = _make_effort_franka_cfg(FRANKA_PANDA_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")

    @configclass
    class ControllerSceneCfg(InteractiveSceneCfg):
        """Replicated Franka scene for optional controller integration tests."""

        ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        robot = franka_cfg
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
        )

    sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=SIM_DT))
    try:
        scene = InteractiveScene(
            ControllerSceneCfg(num_envs=env_count, env_spacing=2.0, lazy_sensor_update=True)
        )
        sim.reset()
        scene.update(SIM_DT)

        robot = scene.articulations["robot"]
        assert robot.data.joint_pos.is_cuda
        assert robot.data.joint_vel.is_cuda
        assert robot.data.joint_pos.shape[0] == env_count
        joint_dim = int(robot.data.joint_pos.shape[1])
        ee_body_index, ee_body_name = _resolve_single_body(
            robot, ("panda_hand", "panda_link7", ".*hand.*")
        )
        assert ee_body_name in robot.body_names
        jacobian_index = _jacobian_body_index(robot, ee_body_index)
        if jacobian_index < 0:
            pytest.fail(f"invalid Jacobian body index {jacobian_index} for body {ee_body_name}")

        unsafe_mask = _status_mask(
            tensor_module,
            (
                "STATUS_INVALID_INPUT",
                "STATUS_NON_FINITE_INPUT",
                "STATUS_INVALID_PARAMETER",
                "STATUS_INVALID_DT",
                "STATUS_UNSUPPORTED_MODE",
                "STATUS_SAFE_MODE",
            ),
        )
        stream = torch.cuda.current_stream()
        tensor_module.set_cuda_stream(stream.cuda_stream)
        initial_state = _capture_robot_state(robot)
        for controller_name in controller_names:
            _run_controller_case(
                tensor_module,
                torch,
                math_utils,
                robot,
                scene,
                sim,
                initial_state,
                controller_name,
                env_count,
                joint_dim,
                ee_body_index,
                jacobian_index,
                step_count,
                unsafe_mask,
            )
        _restore_robot_state(torch, robot, scene, initial_state)
        _run_status_isolation_case(
            tensor_module, torch, math_utils, robot, ee_body_index, jacobian_index, unsafe_mask
        )
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


def _run_isaaclab_suite(
    tensor_module: Any,
    torch: Any,
    env_counts: tuple[int, ...],
    controller_names: tuple[str, ...],
    step_count: int,
) -> None:
    """Launch replicated Franka scenes for each requested env-count tier."""
    for env_count in env_counts:
        _run_isaaclab_env_count(tensor_module, torch, env_count, controller_names, step_count)


def test_isaaclab_franka_parallel_high_level_controller_suite() -> None:
    """Run optional headless IsaacLab tests against compiled CUDA tensor controllers."""
    root = _isaaclab_root()
    _prepend_isaaclab_source_paths(root)
    _assert_package_resolves_from_root("isaaclab", root)
    _assert_package_resolves_from_root("isaaclab_assets", root)
    _make_local_controllers_namespace_visible()
    tensor_module = _import_tensor_extension()
    torch = _require_cuda_torch()
    env_counts = _parse_env_counts()
    controller_names = _parse_controller_names()
    step_count = _parse_step_count()

    simulation_app = None
    try:
        # NVIDIA / IsaacLab
        # Third Party
        from isaaclab.app import AppLauncher

        # AppLauncher forwards process argv to Kit; pytest's flags such as
        # ``-m isaaclab`` are not valid Kit arguments.
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            simulation_app = AppLauncher(headless=True).app
        finally:
            sys.argv = original_argv
        _run_isaaclab_suite(tensor_module, torch, env_counts, controller_names, step_count)
    finally:
        tensor_module.set_cuda_stream(None)
        if simulation_app is not None:
            simulation_app.close()
