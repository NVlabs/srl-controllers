# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Synthetic PID, custom nonlinear feedback and an energy tank, using only public APIs."""

# Standard Library
from dataclasses import fields, is_dataclass
from typing import Any, get_type_hints

# Third Party
import numpy as np

# NVIDIA
import nvidia.srl.controllers as ctl


def allocate_storage(name: str, config: Any) -> Any:
    """Allocate once, before stepping, from the new block's public storage descriptors."""
    requirements = getattr(ctl, name).requirements(config)
    tree: dict[str, Any] = {}
    for item in (*requirements.outputs, *requirements.state, *requirements.scratch):
        node = tree
        path = item.name.split(".")
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = np.zeros(item.shape, dtype=item.dtype)

    def construct(cls: type, values: dict[str, Any]) -> Any:
        types = get_type_hints(cls)
        return cls(
            **{
                field.name: (
                    construct(types[field.name], values.get(field.name, {}))
                    if is_dataclass(types[field.name])
                    else values[field.name]
                )
                for field in fields(cls)
            }
        )

    return construct(getattr(ctl, name + "Storage"), tree)


def run() -> np.ndarray:
    """Compute synthetic commands; there is no robot connection or motion execution."""
    config = ctl.SignalBlockConfig(ctl.Device.CPU, ctl.DType.FLOAT64, 2, 3)
    p = ctl.ProportionalBlock(
        config,
        ctl.ProportionalBlockParameters(ctl.shared(np.full(3, 2.0))),
        allocate_storage("ProportionalBlock", config),
    )
    i = ctl.IntegratorBlock(
        config,
        ctl.IntegratorBlockParameters(
            ctl.shared(np.zeros(3)), ctl.shared(np.full(3, -10.0)), ctl.shared(np.full(3, 10.0))
        ),
        allocate_storage("IntegratorBlock", config),
    )
    d = ctl.DifferentiatorBlock(
        config, ctl.DifferentiatorBlockParameters(), allocate_storage("DifferentiatorBlock", config)
    )
    tank_config = ctl.SignalBlockConfig(ctl.Device.CPU, ctl.DType.FLOAT64, 2, 1)
    tank = ctl.EnergyTank(
        tank_config,
        ctl.EnergyTankParameters(
            *(ctl.shared(np.array(value)) for value in (10.0, 1.0, 5.0, 1.0, 0.0))
        ),
        allocate_storage("EnergyTank", tank_config),
    )
    final = ctl.SafetyState(np.zeros((2, 3)), np.zeros(2, bool), np.zeros(2, np.uint64))
    status = np.zeros(2, np.uint64)
    command, scratch = np.zeros((2, 3)), np.zeros((2, 3))
    power, additional_power = np.zeros(2), np.zeros(2)
    error, velocity, dt = np.full((2, 3), 0.2), np.full((2, 3), 0.1), np.array(0.001)
    for block in (p, i, d, tank):
        block.reset()
    ctl.reset_safety_state(final, status=status)
    for _ in range(5):
        # Structural failures raise: the exception aborts this tick before output is reused.
        p.step(error)
        i.step(error, dt)
        d.step(error, dt)
        np.add(p.output, i.output, out=command)
        np.add(command, d.output, out=command)
        np.tanh(command, out=command)  # The application supplies its own control law.
        # Synthetic translational force [N] and velocity [m/s], in one common frame.
        np.multiply(command, velocity, out=scratch)
        np.sum(scratch, axis=1, out=power)  # W; positive power charges this tank convention.
        tank.step(power, additional_power, dt)
        np.multiply(command, tank.scale[:, None], out=command)
        ctl.merge_status(p.status, i.status, d.status, tank.status, output=status)
        ctl.apply_safe_output_policy(command, status, final)
    assert np.isfinite(command).all() and not np.any(status & ctl.STATUS_SAFE_MODE)
    return command


if __name__ == "__main__":
    print(run())
