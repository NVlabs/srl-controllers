# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Point fields in robot-base axes, using the canonical checked equations."""

from __future__ import annotations

# Standard Library
from typing import Any

# Third Party
from _parallel_common import CartesianFieldMetrics, flagged_rows


class CartesianFields:
    """Fixed geometry relative to the measured initial task point, in meters.

    Persistent buffers are owned here. Torch temporaries are simulator-only;
    the C++ hardware equivalent uses fixed arrays in its control callback.
    """

    def __init__(self, api: Any, torch: Any, config: Any, reference: Any, kind: str) -> None:
        """Capture the reference point and allocate independent row storage."""
        # NVIDIA
        from nvidia.srl.controllers import equations

        self.api, self.torch, self.equations, self.kind = api, torch, equations, kind
        self.array_view = (
            (lambda tensor: tensor.numpy())
            if config.device == api.Device.CPU
            else (lambda tensor: tensor)
        )
        self.reference = reference.clone()
        self.config = equations.EquationConfig(config.device, config.dtype, reference.shape[0])
        self.goal_offset = reference.new_tensor((0.0, 0.12, 0.0))
        self.obstacle_offset = reference.new_tensor((-0.025, 0.06, 0.0))
        self.force = torch.zeros_like(reference)
        self.displacement, self.obstacle_delta = (torch.zeros_like(reference) for _ in range(2))
        self.distance = reference.new_zeros(reference.shape[0])
        self.clearance, self.repulsion, self.goal_error, self.settled_seconds = (
            torch.zeros_like(self.distance) for _ in range(4)
        )
        self.wall_distance = [torch.zeros_like(self.distance) for _ in range(6)]
        self.wall_velocity = [torch.zeros_like(self.distance) for _ in range(6)]
        self.wall_force = [torch.zeros_like(self.distance) for _ in range(6)]
        self.status = torch.zeros(reference.shape[0], dtype=torch.uint64, device=reference.device)
        self.unsafe = torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)
        self.minimum_clearance = reference.new_tensor(float("inf"))
        self.repulsion_samples = torch.zeros((), dtype=torch.int64, device=reference.device)
        self.parameters = {
            name: api.shared(self.array_view(reference.new_tensor(value)))
            for name, value in {
                "stiffness": 1000.0,
                "damping": 40.0,
                "damping_activation_distance": 0.005,
                "influence_distance": 0.035,
                "gain": 0.00005,
            }.items()
        }

    def step(
        self, position: Any, velocity: Any, elapsed_s: float, dt_s: float, context: Any
    ) -> Any:
        """Return uncapped field force; reject invalid equation inputs and results."""
        torch, api = self.torch, self.api
        self.displacement.copy_(position).sub_(self.reference)
        self.goal_error.copy_(torch.linalg.vector_norm(self.goal_offset - self.displacement, dim=1))
        self.unsafe.copy_(~(torch.isfinite(position).all(1) & torch.isfinite(velocity).all(1)))
        self.force.zero_()
        if self.kind == "cartesian_virtual_walls_torque":
            for axis in range(3):
                for side, sign in enumerate((1.0, -1.0)):
                    index = 2 * axis + side
                    self.wall_distance[index].copy_(self.displacement[:, axis]).mul_(sign).add_(0.1)
                    self.wall_velocity[index].copy_(velocity[:, axis]).mul_(sign)
                    self.equations.virtual_wall_force_value(
                        self.config,
                        signed_distance=api.per_batch(self.array_view(self.wall_distance[index])),
                        normal_velocity=api.per_batch(self.array_view(self.wall_velocity[index])),
                        **{
                            key: self.parameters[key]
                            for key in ("stiffness", "damping", "damping_activation_distance")
                        },
                        output=self.array_view(self.wall_force[index]),
                        status=self.array_view(self.status),
                        context=context,
                    )
                    self.unsafe |= flagged_rows(
                        torch, self.status, api.STATUS_INVALID_INPUT | api.STATUS_NON_FINITE_INPUT
                    )
                self.force[:, axis].copy_(self.wall_force[2 * axis]).sub_(
                    self.wall_force[2 * axis + 1]
                )
        else:
            self.obstacle_delta.copy_(self.displacement).sub_(self.obstacle_offset)
            self.distance.copy_(torch.linalg.vector_norm(self.obstacle_delta, dim=1))
            self.clearance.copy_(self.distance).sub_(0.02)
            self.minimum_clearance.copy_(
                torch.minimum(self.minimum_clearance, self.clearance.min())
            )
            self.unsafe |= ~torch.isfinite(self.clearance)
            # The checked equation rejects nonpositive surface clearance.
            # Non-finite input rows are discarded after evaluating an innocuous distance.
            self.equations.spherical_apf_repulsion_value(
                self.config,
                surface_clearance=api.per_batch(
                    self.array_view(torch.where(self.unsafe, 1.0, self.clearance))
                ),
                influence_distance=self.parameters["influence_distance"],
                gain=self.parameters["gain"],
                output=self.array_view(self.repulsion),
                status=self.array_view(self.status),
                context=context,
            )
            self.unsafe |= flagged_rows(
                torch, self.status, api.STATUS_INVALID_INPUT | api.STATUS_NON_FINITE_INPUT
            )
            self.repulsion_samples.add_(((self.repulsion > 0) & ~self.unsafe).sum())
            fraction = min(1.0, max(0.0, elapsed_s))
            ramp = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
            direction = self.obstacle_delta / torch.where(
                self.unsafe, 1.0, self.distance
            ).unsqueeze(1)
            self.force.copy_(50.0 * (self.goal_offset - self.displacement))
            self.force.add_(self.repulsion.unsqueeze(1) * direction).mul_(ramp).sub_(
                20.0 * velocity
            )
            settled = (
                (self.goal_error <= 0.005)
                & (torch.linalg.vector_norm(velocity, dim=1) <= 0.01)
                & ~self.unsafe
            )
            self.settled_seconds.copy_(torch.where(settled, self.settled_seconds + dt_s, 0.0))
        magnitude = torch.linalg.vector_norm(self.force, dim=1)
        self.unsafe |= ~torch.isfinite(magnitude)
        self.force[self.unsafe] = 0.0
        return self.force

    def buffer_roots(self) -> dict[str, Any]:
        """Expose persistent tensor storage to the example's address checks."""
        buffers = {name: value for name, value in vars(self).items() if hasattr(value, "data_ptr")}
        for name in ("wall_distance", "wall_velocity", "wall_force"):
            buffers[name] = dict(enumerate(getattr(self, name)))
        buffers["parameters"] = self.parameters
        return buffers

    def metrics(self, active_rows: Any = None) -> CartesianFieldMetrics:
        """Read diagnostics; goal attainment covers only rows allowed to execute."""
        is_apf = self.kind == "cartesian_potential_field_torque"
        goal_error = self.goal_error if active_rows is None else self.goal_error[active_rows]
        settled = self.settled_seconds if active_rows is None else self.settled_seconds[active_rows]
        return CartesianFieldMetrics(
            minimum_clearance_m=float(self.minimum_clearance.item()) if is_apf else None,
            maximum_goal_error_m=(
                (float(goal_error.max().item()) if goal_error.numel() else float("nan"))
                if is_apf
                else None
            ),
            minimum_settled_seconds=(
                (float(settled.min().item()) if settled.numel() else 0.0) if is_apf else None
            ),
            repulsion_row_samples=int(self.repulsion_samples.item()),
        )

    def visualize(self, robot: Any, math_utils: Any) -> Any:
        """Spawn virtual geometry with no collision shapes or physical forces."""
        # Third Party
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        is_wall = self.kind == "cartesian_virtual_walls_torque"
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9), opacity=0.18)
        shape = (
            sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2), visual_material=material, collision_props=None
            )
            if is_wall
            else sim_utils.SphereCfg(radius=0.02, visual_material=material, collision_props=None)
        )
        markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/Visuals/CartesianField", markers={"field": shape})
        )
        points = self.reference if is_wall else self.reference + self.obstacle_offset
        root = robot.data.root_state_w
        # Native controller geometry is float64; visualization uses the PhysX dtype.
        positions = root[:, :3] + math_utils.quat_apply(root[:, 3:7], points.to(root.dtype))
        markers.visualize(translations=positions, orientations=root[:, 3:7])
        return markers
