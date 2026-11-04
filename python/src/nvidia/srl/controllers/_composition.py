# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Explicit status propagation and final command protection for custom controllers."""

# Standard Library
from typing import Any

# Local Folder
from ._api import ExecutionContext, SafeOutputPolicy, SafetyState, _backend


def merge_status(*inputs: Any, output: Any, context: ExecutionContext | None = None) -> None:
    """OR uint64 [B] flags and mark every fault SafeMode on the caller's stream.

    Output may alias the first input exactly. A structural exception aborts the tick:
    do not execute dependent stages or reuse their previous commands.
    """
    _backend.merge_status(inputs, output, context)


def reset_safety_state(
    state: SafetyState,
    *,
    status: Any,
    reset_mask: Any = None,
    context: ExecutionContext | None = None,
) -> None:
    """Reset caller-owned policy history for all rows or a Boolean [B] mask."""
    _backend.reset_safety_state(state, reset_mask, status, context)


def apply_safe_output_policy(
    output: Any,
    status: Any,
    state: SafetyState,
    *,
    policy: SafeOutputPolicy = SafeOutputPolicy.ZERO,
    max_hold_steps: int = 0,
    context: ExecutionContext | None = None,
) -> None:
    """Protect the final command once per tick using independent safety history.

    Lifecycle blocks already update their own histories. Use a distinct SafetyState for
    this final policy. CUDA is asynchronous: retain all arrays until the stream completes.
    """
    _backend.apply_safe_output_policy(output, status, state, policy, max_hold_steps, context)


__all__ = ["merge_status", "reset_safety_state", "apply_safe_output_policy"]
