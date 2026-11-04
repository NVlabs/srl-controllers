# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Non-functional UR Cartesian-impedance Isaac companion, retained for diagnosis.

Both UR5e and UR10e oscillate under small disturbances. Matching hardware torque
encoding suppresses the numerical equilibrium trigger but does not restore stability.
See docs/public-api-validation.md; hardware and official model parameters are preserved.
"""

# Third Party
from _canonical_launcher import main

if __name__ == "__main__":
    main("cartesian_impedance_controller")
