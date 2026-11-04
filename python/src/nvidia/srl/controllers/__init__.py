# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Platform-neutral controllers, canonical equations and custom composition utilities."""

# NVIDIA
from nvidia.srl._controllers import _get_version
from nvidia.srl.controllers._api import *  # noqa: F401,F403
from nvidia.srl.controllers._api import __all__ as _controller_api

# Local Folder
from . import equations
from ._blocks import *  # noqa: F401,F403
from ._blocks import __all__ as _block_api
from ._composition import apply_safe_output_policy, merge_status, reset_safety_state
from ._metadata import (
    BimanualCartesianTorqueRequirements,
    CartesianTorqueRequirements,
    Requirements,
    TaskWrenchToJointTorqueRequirements,
    TensorRequirement,
)

__all__ = [
    *_controller_api,
    *_block_api,
    "Requirements",
    "TensorRequirement",
    "equations",
    "apply_safe_output_policy",
    "merge_status",
    "reset_safety_state",
    "CartesianTorqueRequirements",
    "BimanualCartesianTorqueRequirements",
    "TaskWrenchToJointTorqueRequirements",
    "__version__",
]
__version__ = _get_version()
del _get_version, _controller_api, _block_api
