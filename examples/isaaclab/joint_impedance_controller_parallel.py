# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Non-functional UR joint-impedance Isaac companion, retained for diagnosis.

Both UR5e and UR10e oscillate under small disturbances. Equilibrium-only passes
do not qualify stability. See docs/public-api-validation.md for the evidence;
the physical example and shared native controller remain the configuration authority.
"""

# Third Party
from _canonical_launcher import main

if __name__ == "__main__":
    main("joint_impedance_controller")
