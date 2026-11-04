# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Compatibility launcher for the canonical Universal Robots admittance companion."""

# Standard Library
import pathlib
import sys

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

# Third Party
from _canonical_launcher import main as _main  # noqa: E402
from _ur5e_admittance_parallel import run  # noqa: E402

__all__ = ["run"]


if __name__ == "__main__":
    _main("cartesian_admittance_controller")
