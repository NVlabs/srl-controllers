# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Isaac Lab companion for the Franka force/impedance energy-tank example."""

# Third Party
from _canonical_launcher import main

if __name__ == "__main__":
    main("franka_cartesian_force_impedance_controller_with_tank")
