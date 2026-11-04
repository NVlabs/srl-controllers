# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Caller-owned force/tank composition shared by the simulator and focused tests."""

from __future__ import annotations

# Standard Library
from typing import Any

# Third Party
from _parallel_common import EnergyTankMetrics, flagged_rows


class ForceTank:
    """Base-frame PD force contribution with per-row energy accounting in J/W/s."""

    def __init__(self, api: Any, torch: Any, config: Any, reference: Any, context: Any) -> None:
        self.torch = torch
        count = reference.shape[0]

        def zeros() -> Any:
            return torch.zeros_like(reference)

        def scalars() -> Any:
            return torch.zeros(count, dtype=reference.dtype, device=reference.device)

        def flags() -> Any:
            return torch.zeros(count, dtype=torch.uint64, device=reference.device)

        def booleans() -> Any:
            return torch.zeros(count, dtype=torch.bool, device=reference.device)

        gain, derivative = zeros()[0], zeros()[0]
        gain[2], derivative[2] = 1.0, 0.01
        force_parameters = api.ForceControllerParameters(
            *(
                api.shared(array)
                for array in (
                    gain,
                    derivative,
                    zeros()[0],
                    zeros()[0],
                    torch.full_like(gain, 0.1),
                    torch.full_like(gain, 0.06),
                )
            )
        )
        force_storage = api.CartesianForceStorage(
            api.OutputStorage(
                zeros(), flags(), flags(), api.SafetyState(zeros(), booleans(), flags())
            ),
            api.ForceControllerState(zeros(), booleans(), zeros(), zeros(), booleans()),
        )
        self.force = api.CartesianForceController(config, force_parameters, force_storage, context)
        tank_config = api.SignalBlockConfig(config.device, config.dtype, count, 1)
        tank_parameters = api.EnergyTankParameters(
            *(
                api.shared(torch.tensor(value, dtype=reference.dtype, device=reference.device))
                for value in (1.0, 0.01, 1.0, 0.1, 0.0)
            )
        )
        tank_storage = api.EnergyTankStorage(
            scalars(), scalars(), flags(), flags(), api.EnergyTankState(scalars())
        )
        self.tank = api.EnergyTank(tank_config, tank_parameters, tank_storage, context)
        self.power, self.additional_power = scalars(), scalars()
        self.power_terms, self.scaled_wrench = zeros(), zeros()
        self.energy, self.previous_energy, self.work, self.credit = (scalars() for _ in range(4))
        self.residual = scalars()
        self.minimum_energy = torch.ones((), dtype=reference.dtype, device=reference.device)
        self.maximum_energy = torch.ones_like(self.minimum_energy)
        self.minimum_scale = torch.ones_like(self.minimum_energy)
        self.maximum_residual = torch.zeros_like(self.minimum_energy)
        self.limited_samples = torch.zeros((), dtype=torch.int64, device=reference.device)

    def reset(self, mask: Any, context: Any) -> None:
        self.force.reset(mask, context)
        self.tank.reset(mask, context)

    def unsafe_rows(self, mask: int) -> Any:
        return flagged_rows(self.torch, self.force.status, mask) | flagged_rows(
            self.torch, self.tank.status, mask
        )

    def step(
        self,
        external: Any,
        desired: Any,
        feedforward: Any,
        pose_error: Any,
        twist: Any,
        dt: Any,
        upstream_unsafe: Any,
        unsafe_mask: int,
        context: Any,
    ) -> Any:
        """Return scaled force wrench; invalid upstream rows retain their tank state."""
        torch = self.torch
        self.force.step(external, desired, feedforward, pose_error, dt, context)
        torch.mul(self.force.output, twist, out=self.power_terms)
        torch.sum(self.power_terms, dim=1, out=self.power)
        self.power.neg_()
        invalid = upstream_unsafe | flagged_rows(torch, self.force.status, unsafe_mask)
        self.power[invalid] = float("nan")
        self.previous_energy.copy_(self.tank.storage.state.state_variable).square_().mul_(0.5)
        self.tank.step(self.power, self.additional_power, dt, context)
        invalid |= flagged_rows(torch, self.tank.status, unsafe_mask)
        torch.mul(self.force.output, self.tank.scale.unsqueeze(-1), out=self.scaled_wrench)
        self.scaled_wrench[invalid] = float("nan")
        self.energy.copy_(self.tank.storage.state.state_variable).square_().mul_(0.5)
        self.work.copy_(self.power).clamp_(max=0).neg_().mul_(dt).mul_(self.tank.scale)
        self.work[invalid] = 0
        self.credit.copy_(self.tank.filtered_power).clamp_(min=0).mul_(self.tank.scale).mul_(dt)
        self.residual.copy_(self.previous_energy).sub_(self.work).neg_().add_(1.0)
        torch.minimum(self.credit, self.residual, out=self.credit)
        self.residual.copy_(self.energy).sub_(self.previous_energy).add_(self.work).sub_(
            self.credit
        )
        self.residual[invalid] = 0
        self.maximum_residual.copy_(torch.maximum(self.maximum_residual, self.residual.abs().max()))
        self.minimum_energy.copy_(torch.minimum(self.minimum_energy, self.energy.min()))
        self.maximum_energy.copy_(torch.maximum(self.maximum_energy, self.energy.max()))
        # Fault outputs are zero; exclude them from normal limiting diagnostics.
        self.residual.copy_(self.tank.scale)
        self.residual[invalid] = 1
        self.minimum_scale.copy_(torch.minimum(self.minimum_scale, self.residual.min()))
        self.limited_samples.add_((self.residual < 1).sum())
        return self.scaled_wrench

    def metrics(self) -> EnergyTankMetrics:
        """Read diagnostics only after the simulation loop has completed."""
        return EnergyTankMetrics(
            minimum_energy_j=float(self.minimum_energy.item()),
            maximum_energy_j=float(self.maximum_energy.item()),
            minimum_scale=float(self.minimum_scale.item()),
            limited_row_samples=int(self.limited_samples.item()),
            maximum_balance_residual_j=float(self.maximum_residual.item()),
        )
