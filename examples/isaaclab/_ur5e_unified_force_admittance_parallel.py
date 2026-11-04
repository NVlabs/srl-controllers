# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Hardware-configured UR force/admittance with live-frame shaping and Isaac pose actuation."""

from __future__ import annotations

# Standard Library
from typing import Any


def run(
    simulation_app: Any,
    *,
    spec: Any = None,
    num_envs: int = 16,
    duration_s: float = 10.0,
    device: str = "cuda:0",
    validate: bool = False,
    tcp_offset: tuple[float, ...] = (0.0,) * 6,
) -> Any:
    """Run the hardware-configured pose controller through the shared Isaac adapter."""
    # Third Party
    from _parallel_common import require_controller_module
    from _pose_parallel import run as run_pose

    if spec is None:
        spec = require_controller_module().find_controller_example_spec(
            "cartesian_unified_force_admittance_controller"
        )
    return run_pose(
        simulation_app,
        spec=spec,
        num_envs=num_envs,
        duration_s=duration_s,
        device=device,
        validate=validate,
        robot_model="ur5e",
        tcp_offset=tcp_offset,
    )


def main() -> None:
    """Keep the existing alias routed through the canonical CLI."""
    # Third Party
    from _canonical_launcher import main as launch

    launch("cartesian_unified_force_admittance_controller")


if __name__ == "__main__":
    main()
