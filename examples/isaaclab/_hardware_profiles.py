# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Immutable hardware configurations for the simulator's native controller blocks.

These values mirror the physical examples, which remain the authority. The
source contracts in ``hardware_profile_contract.json`` detect changes requiring
another audit. Model observations and vendor servo implementations remain
transport inputs, rather than alternate controller parameters.
"""

from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass, replace

JOINT_STIFFNESS = (1500.0, 1200.0, 800.0, 500.0, 300.0, 200.0, 50.0)
NULLSPACE_STIFFNESS = (120.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0)
NULLSPACE_DAMPING = (12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0)
FRANKA_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
RIZON_LIMITS = {
    "rizon4": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon4s": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon4m": (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
    "rizon10": (261.0, 261.0, 123.0, 123.0, 57.0, 57.0, 57.0),
    "rizon10s": (261.0, 261.0, 123.0, 123.0, 57.0, 57.0, 57.0),
}
UR_LIMITS = {
    "ur5e": (150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
    "ur10e": (330.0, 330.0, 150.0, 54.0, 54.0, 54.0),
}


@dataclass(frozen=True)
class HardwareProfile:
    """Resolved configuration; SI units and [translation; rotation] task order."""

    target: str
    model: str
    family: str
    kind: str
    frame: str
    joint_dim: int
    task_dim: int
    period_s: float
    initial_joint_position_rad: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    torque_limits_nm: tuple[float, ...]
    torque_rate_nm_per_s: tuple[float, ...]
    sources: tuple[str, ...]
    damping_factor: float = 0.7
    regularization: float = 1.0e-6
    minimum_pivot: float | None = None
    nullspace_stiffness: tuple[float, ...] = ()
    nullspace_damping: tuple[float, ...] = ()
    admittance_stiffness: tuple[float, ...] = ()
    admittance_mass: tuple[float, ...] = ()
    admittance_damping: tuple[float, ...] = ()
    wrench_deadzone: tuple[float, ...] = ()
    wrench_cutoff_hz: float = 0.0
    force_proportional: tuple[float, ...] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    force_differentiator: tuple[float, ...] = (0.0, 0.0, 0.01, 0.0, 0.0, 0.0)
    force_integral: tuple[float, ...] = (0.0,) * 6
    force_maximum_distance: tuple[float, ...] = (0.1,) * 6
    force_shaping_distance: tuple[float, ...] = (0.06,) * 6
    target_force_n: float = 0.0
    force_ramp_s: float = 2.0
    variable_maximum: tuple[float, ...] = ()
    variable_frequency_hz: float = 0.0
    variable_offset: float = 0.0
    variable_amplitude: float = 0.0
    command_mode: str = "torque"
    torque_filter_period_s: float | None = None
    torque_filter_seed_zero: bool = False
    motor_kd_nm_s_per_rad: float = 0.0
    vendor_gravity_compensation: bool = True
    force_reference_frame: str | None = None
    wrench_seed_first_sample: bool = True
    vendor_cartesian_stiffness: tuple[float, ...] = ()
    vendor_cartesian_damping_factor: tuple[float, ...] = ()

    def variable_z_stiffness(self, elapsed_s: float) -> float:
        """Sample the physical example's requested base-z stiffness."""
        if self.family == "franka":
            return 1000.0 * (
                math.sin(2.0 * math.pi * self.variable_frequency_hz * elapsed_s) + 1.001
            )
        return (
            self.variable_amplitude
            * math.sin(2.0 * math.pi * self.variable_frequency_hz * elapsed_s)
            + self.variable_offset
        )


def hardware_profile(
    target: str, model: str, torque_limits_nm: tuple[float, ...] | None = None
) -> HardwareProfile:
    """Resolve one physical target/model without any simulator tuning."""
    if torque_limits_nm is not None:
        profile = hardware_profile(target, model)
        if len(torque_limits_nm) != profile.joint_dim or any(
            not math.isfinite(limit) or limit <= 0.0 for limit in torque_limits_nm
        ):
            raise ValueError(
                "captured torque limits must be finite, positive and match joint order"
            )
        return replace(profile, torque_limits_nm=tuple(torque_limits_nm))
    ee = target.endswith("_in_ee")
    frame = "end_effector" if ee else "robot_base"
    if target.startswith(("franka_", "rizon_")):
        franka = target.startswith("franka_")
        family = "franka" if franka else "rizon"
        supported = ("fr3", "panda") if franka else tuple(RIZON_LIMITS)
        if model not in supported:
            raise ValueError(f"unsupported {family} model: {model}")
        suffix = target.removeprefix(family + "_").removesuffix("_in_ee")
        kinds = {
            "joint_impedance_controller": "joint_impedance",
            "cartesian_impedance_controller": "cartesian_impedance_torque",
            "variable_cartesian_impedance_controller": "variable_cartesian_impedance_torque",
            "cartesian_force_impedance_controller": "cartesian_unified_force_impedance_torque",
            "cartesian_admittance_controller": "cartesian_admittance_torque",
        }
        if franka:
            kinds["cartesian_force_impedance_controller_with_tank"] = (
                "cartesian_force_impedance_with_tank_torque"
            )
            kinds.update(
                virtual_walls_controller="cartesian_virtual_walls_torque",
                artificial_potential_field_controller="cartesian_potential_field_torque",
            )
        if suffix not in kinds or (ee and suffix != "cartesian_force_impedance_controller"):
            raise ValueError(f"unknown hardware target: {target}")
        kind = kinds[suffix]
        rotational = 100.0 if franka else 200.0
        stiffness = (2000.0,) * 3 + (rotational,) * 3
        limits = FRANKA_LIMITS if franka else RIZON_LIMITS[model]
        initial = (0.0, 0.0, 0.0, -2.3 * math.pi / 4, 0.0, 2.3 * math.pi / 4, math.pi / 4)
        if franka and kind in {
            "joint_impedance",
            "cartesian_impedance_torque",
        }:
            initial = (0.0, 0.0, 0.0, -3.0 * math.pi / 4, 0.0, 3.0 * math.pi / 4, math.pi / 4)
        if not franka:
            initial = (0.0, -15.0 * math.pi / 180, 0.0, 75.0 * math.pi / 180, 0.0, 0.0, 0.0)
        if ee:
            initial = (*initial[:4], math.pi / 4, *initial[5:])
        sources = (
            (f"examples/franka/{target}.cpp", "examples/franka/franka_utils.hpp")
            if franka
            else (
                "examples/rizon/rizon_workspaces.hpp",
                "examples/rizon/rizon_utils.hpp",
                "examples/rizon/rizon_controller_example.cpp",
            )
        )
        profile = HardwareProfile(
            target,
            model,
            family,
            kind,
            frame,
            7,
            6,
            0.001,
            initial,
            stiffness,
            (),
            limits,
            ((999.999,) if franka else (1000.0,)) * 7,
            sources,
            torque_filter_period_s=0.001 if franka else None,
            torque_filter_seed_zero=franka,
        )
        if kind == "joint_impedance":
            return replace(profile, stiffness=JOINT_STIFFNESS, task_dim=0, frame="unspecified")
        if kind == "cartesian_impedance_torque":
            return replace(
                profile,
                nullspace_stiffness=NULLSPACE_STIFFNESS,
                nullspace_damping=NULLSPACE_DAMPING,
            )
        if kind == "variable_cartesian_impedance_torque":
            maximum = (5000.0,) * 3 + (250.0,) * 3 if franka else stiffness
            return replace(
                profile,
                variable_maximum=maximum,
                variable_frequency_hz=0.1,
                variable_offset=1001.0 if franka else 1000.5,
                variable_amplitude=1000.0 if franka else 999.5,
            )
        if kind in {
            "cartesian_unified_force_impedance_torque",
            "cartesian_force_impedance_with_tank_torque",
        }:
            stiffness = (
                (
                    (5000.0, 5000.0, 20.0, 250.0, 250.0, 250.0)
                    if ee
                    else (2000.0, 2000.0, 50.0, 100.0, 100.0, 100.0)
                )
                if franka
                else (2000.0, 2000.0, 50.0, 200.0, 200.0, 200.0)
            )
            return replace(
                profile,
                stiffness=stiffness,
                target_force_n=10.0 if ee else (-10.0 if franka else -15.0),
            )
        if kind in {"cartesian_virtual_walls_torque", "cartesian_potential_field_torque"}:
            is_wall = kind == "cartesian_virtual_walls_torque"
            return replace(
                profile,
                stiffness=(0.0,) * 6 if is_wall else (0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                damping=(0.0,) * 6 if is_wall else (0.0, 0.0, 0.0, 3.0, 3.0, 3.0),
                nullspace_stiffness=() if is_wall else NULLSPACE_STIFFNESS,
                nullspace_damping=() if is_wall else NULLSPACE_DAMPING,
                sources=(
                    *sources,
                    "examples/franka/franka_cartesian_field.hpp",
                    "examples/cartesian_fields.hpp",
                ),
            )
        if franka:
            k = (500.0,) * 3 + (50.0,) * 3
            mass = (1.95,) * 6 if model == "fr3" else (29.25,) * 3 + (3.9,) * 3
            return replace(
                profile,
                admittance_stiffness=k,
                admittance_mass=mass,
                admittance_damping=tuple(4.0 * math.sqrt(m * s) for m, s in zip(mass, k)),
                wrench_deadzone=(0.0,) * 6,
                wrench_cutoff_hz=7.0,
                wrench_seed_first_sample=False,
                command_mode="torque" if model == "fr3" else "pose",
                sources=(*sources, "examples/franka/franka_admittance.hpp"),
            )
        return replace(
            profile,
            admittance_stiffness=(1000.0,) * 3 + (100.0,) * 3,
            admittance_mass=(1.0,) * 6,
            admittance_damping=(100.0,) * 3 + (20.0,) * 3,
            wrench_deadzone=(0.5,) * 6,
            wrench_cutoff_hz=7.0,
            command_mode="pose",
            vendor_cartesian_stiffness=(2000.0,) * 3 + (200.0,) * 3,
            vendor_cartesian_damping_factor=(0.7,) * 6,
        )

    if target in {
        "flexiv_dual_bimanual_cartesian_impedance",
        "flexiv_dual_bimanual_compliant_hold",
    }:
        models = model.split(",")
        if len(models) != 2 or any(m not in RIZON_LIMITS for m in models):
            raise ValueError("dual Flexiv requires two supported Rizon models")
        arm = hardware_profile("rizon_cartesian_impedance_controller", models[0])
        hold = target == "flexiv_dual_bimanual_compliant_hold"
        absolute_stiffness = (300.0,) * 3 + (30.0,) * 3 if hold else (1500.0,) * 3 + (150.0,) * 3
        relative_stiffness = (750.0,) * 3 + (75.0,) * 3 if hold else (500.0,) * 3 + (50.0,) * 3
        return replace(
            arm,
            target=target,
            family="flexiv_dual",
            model=model,
            kind="bimanual_cartesian_impedance_torque",
            frame="main_frame_orientation",
            joint_dim=14,
            task_dim=12,
            stiffness=absolute_stiffness + relative_stiffness,
            nullspace_stiffness=(40.0, 30.0, 20.0, 16.0, 10.0, 6.0, 2.0) * 2,
            nullspace_damping=(5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5) * 2,
            torque_limits_nm=RIZON_LIMITS[models[0]] + RIZON_LIMITS[models[1]],
            torque_rate_nm_per_s=(1000.0,) * 14,
            minimum_pivot=1e-12,
            initial_joint_position_rad=arm.initial_joint_position_rad * 2,
            torque_filter_seed_zero=False,
            sources=("examples/flexiv_dual/consumer/flexiv_dual_controller_example.cpp",),
        )

    kinds = {
        "joint_impedance_controller": "static_joint_impedance",
        "cartesian_impedance_controller": "static_cartesian_impedance_torque",
        "cartesian_admittance_controller": "cartesian_admittance_pose",
        "cartesian_unified_force_admittance_controller": "cartesian_unified_force_admittance_pose",
        "cartesian_unified_force_admittance_controller_in_ee": (
            "cartesian_unified_force_admittance_pose"
        ),
    }
    if (
        target not in kinds
        or model not in UR_LIMITS
        or ("admittance" in target and model != "ur5e")
    ):
        raise ValueError(f"unsupported hardware target/model: {target}/{model}")
    kind = kinds[target]
    initial = (
        0.0,
        -math.pi / 2,
        -math.pi / 2,
        -math.pi / 2,
        3 * math.pi / 4 if ee else math.pi / 2,
        0.0,
    )
    joint = kind == "static_joint_impedance"
    stiffness = (
        (1000.0, 800.0, 600.0, 200.0, 100.0, 50.0) if joint else (1000.0,) * 3 + (100.0,) * 3
    )
    damping = (
        (56.5685424948, 101.1928851256, 37.9473319220, 17.8885438200, 8.9442719100, 4.8989794856)
        if joint
        else (178.8854382000,) * 3 + (18.9736659612,) * 3
    )
    p = HardwareProfile(
        target,
        model,
        "universal_robots",
        kind,
        "unspecified" if joint else frame,
        6,
        0 if joint else 6,
        0.002,
        initial,
        stiffness,
        damping,
        UR_LIMITS[model],
        (500.0,) * 6,
        (
            f"examples/ur/{target.removesuffix('_in_ee')}.cpp",
            "examples/ur/impedance_common.hpp",
            "examples/ur/initial_motion.hpp",
        ),
        torque_filter_seed_zero=False,
    )
    if "admittance" not in kind:
        return p
    unified = "unified" in kind
    k = (1000.0, 1000.0, 10.0, 30.0, 30.0, 30.0) if unified else (1000.0,) * 3 + (30.0,) * 3
    mass = tuple(
        1.3 * v for v in ((15.0, 15.0, 1.5, 0.5, 0.5, 0.5) if unified else (15.0,) * 3 + (0.5,) * 3)
    )
    damping = tuple(
        (
            (3.0 if ee else 1.5) * 2.0 * math.sqrt((s + (10.0 / (0.1 - 0.06) if ee else 0.0)) * m)
            if unified and i == 2
            else 4.0 * math.sqrt(s * m)
        )
        for i, (s, m) in enumerate(zip(k, mass))
    )
    return replace(
        p,
        admittance_stiffness=k,
        admittance_mass=mass,
        admittance_damping=damping,
        wrench_deadzone=(0.5, 0.5, 0.5, 0.05, 0.05, 0.05) if unified else (0.5,) * 6,
        wrench_cutoff_hz=5.0,
        command_mode="pose",
        target_force_n=-10.0 if ee else 10.0,
        force_ramp_s=2.0,
        frame="robot_base",
        force_reference_frame=frame,
        force_proportional=(1.0,) * 6,
        force_differentiator=(0.0,) * 6,
    )
