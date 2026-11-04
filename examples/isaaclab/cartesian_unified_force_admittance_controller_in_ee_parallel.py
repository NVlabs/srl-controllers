# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Launch the UR5e end-effector-frame unified force/admittance companion."""

# Third Party
from _canonical_launcher import main

if __name__ == "__main__":
    main("cartesian_unified_force_admittance_controller_in_ee")
