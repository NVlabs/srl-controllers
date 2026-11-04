# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Compatibility launcher for the FR3 Cartesian impedance companion."""

# Standard Library
import pathlib
import sys
from typing import Any

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

# Third Party
from _canonical_launcher import main  # noqa: E402
from _parallel_common import require_controller_module  # noqa: E402
from _single_arm_parallel import run as _run  # noqa: E402


def run(
    simulation_app: Any,
    *,
    num_envs: int = 16,
    duration_s: float = 10.0,
    device: str = "cuda:0",
    validate: bool = False,
) -> Any:
    """Forward the historical entry point to its registered canonical companion."""
    spec = require_controller_module().find_controller_example_spec(
        "franka_cartesian_impedance_controller"
    )
    return _run(
        simulation_app,
        spec=spec,
        num_envs=num_envs,
        duration_s=duration_s,
        device=device,
        validate=validate,
        robot_model="fr3",
    )


if __name__ == "__main__":
    main("franka_cartesian_impedance_controller")
