# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Shared command-line entry point for canonical Isaac Lab companion examples."""

from __future__ import annotations

# Standard Library
import argparse
import dataclasses
import importlib
import json
import math
import sys
import traceback
from typing import Any

# Third Party
from _parallel_common import require_controller_module


def _parse_models(spec: Any) -> tuple[str, ...]:
    return tuple(model.strip() for model in spec.supported_robot_models if model.strip())


def main(hardware_target: str) -> None:
    """Launch the exact simulation companion registered for ``hardware_target``."""
    controller_api = require_controller_module()
    spec = controller_api.find_controller_example_spec(hardware_target)
    non_functional_notice = (
        "NON-FUNCTIONAL Isaac torque companion for UR5e/UR10e: small disturbances cause "
        "oscillation. An equilibrium-only pass does not establish stable control. "
        "Retained for diagnosis; see docs/public-api-validation.md."
        if hardware_target in {"joint_impedance_controller", "cartesian_impedance_controller"}
        else None
    )
    parser = argparse.ArgumentParser(
        description=(
            f"Isaac Lab companion for {spec.hardware_target}; controller={spec.controller_kind}"
            + (f". {non_functional_notice}" if non_functional_notice else "")
        )
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument(
        "--duration",
        type=float,
        default=spec.default_duration_seconds,
        help="rollout duration [s]",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="inject one bad row and enforce motion, isolation, and stable-buffer checks",
    )
    model_choices = _parse_models(spec)
    if spec.robot_family == "flexiv_dual":
        default_left, default_right = spec.default_robot_model.split(",", maxsplit=1)
        parser.add_argument("--left-robot-model", choices=model_choices, default=default_left)
        parser.add_argument("--right-robot-model", choices=model_choices, default=default_right)
        parser.add_argument(
            "--arm1-t-arm2",
            type=lambda value: tuple(float(v) for v in value.split(",")),
            default=None,
            help="row-major rigid transform; translation in meters",
        )
        parser.add_argument(
            "--main-frame",
            choices=("arm1", "arm2", "explicit-arm1", "explicit-arm2"),
            default="arm1",
            help="task-vector axes, matching the paired hardware example",
        )
        parser.add_argument(
            "--reference-t-main",
            type=lambda value: tuple(float(v) for v in value.split(",")),
            default=None,
            help="required for explicit main frames; row-major rigid transform in metres",
        )
    elif len(model_choices) > 1:
        parser.add_argument(
            "--robot-model", choices=model_choices, default=spec.default_robot_model
        )

    if spec.robot_family in {"rizon", "flexiv_dual"}:
        parser.add_argument(
            "--torque-limits-nm",
            required=True,
            type=lambda value: tuple(float(v) for v in value.split(",")),
            help="RobotInfo.tau_max in arm joint order; concatenate left then right",
        )
    if spec.robot_family == "universal_robots":
        parser.add_argument(
            "--tcp-offset",
            type=lambda value: tuple(float(v) for v in value.split(",")),
            default=(0.0,) * 6,
            help="configured flange-to-TCP translation [m] and rotation vector [rad]",
        )

    # Third Party
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be positive")

    if hasattr(args, "torque_limits_nm"):
        if len(args.torque_limits_nm) != spec.joint_dim or any(
            not math.isfinite(v) or v <= 0 for v in args.torque_limits_nm
        ):
            parser.error("--torque-limits-nm must contain finite positive limits in joint order")
    if hasattr(args, "tcp_offset") and (
        len(args.tcp_offset) != 6 or not all(math.isfinite(v) for v in args.tcp_offset)
    ):
        parser.error("--tcp-offset must contain six finite translation/rotation-vector values")

    if spec.robot_family == "flexiv_dual":
        from _flexiv_dual_parallel import _resolve_frames

        try:
            _resolve_frames(
                controller_api, args.arm1_t_arm2, args.main_frame, args.reference_t_main
            )
        except ValueError as exc:
            parser.error(str(exc))

    if non_functional_notice:
        print(f"[WARNING] {non_functional_notice}", file=sys.stderr, flush=True)
    app_launcher = AppLauncher(args)
    try:
        if spec.robot_family in {"franka", "rizon"}:
            runner = importlib.import_module("_single_arm_parallel")
            metrics = runner.run(
                app_launcher.app,
                spec=spec,
                num_envs=args.num_envs,
                duration_s=args.duration,
                device=args.device,
                validate=args.validate,
                robot_model=getattr(args, "robot_model", spec.default_robot_model),
                torque_limits_nm=getattr(args, "torque_limits_nm", None),
            )
        elif spec.robot_family == "universal_robots":
            if spec.controller_kind in {
                "static_joint_impedance",
                "static_cartesian_impedance_torque",
            }:
                runner = importlib.import_module("_single_arm_parallel")
                metrics = runner.run(
                    app_launcher.app,
                    spec=spec,
                    num_envs=args.num_envs,
                    duration_s=args.duration,
                    device=args.device,
                    validate=args.validate,
                    robot_model=getattr(args, "robot_model", spec.default_robot_model),
                    tcp_offset=args.tcp_offset,
                )
            elif spec.controller_kind == "cartesian_unified_force_admittance_pose":
                runner = importlib.import_module("_ur5e_unified_force_admittance_parallel")
                metrics = runner.run(
                    app_launcher.app,
                    spec=spec,
                    num_envs=args.num_envs,
                    duration_s=args.duration,
                    device=args.device,
                    validate=args.validate,
                    tcp_offset=args.tcp_offset,
                )
            else:
                runner = importlib.import_module("_ur5e_admittance_parallel")
                metrics = runner.run(
                    app_launcher.app,
                    num_envs=args.num_envs,
                    duration_s=args.duration,
                    device=args.device,
                    validate=args.validate,
                    tcp_offset=args.tcp_offset,
                )
        else:
            runner = importlib.import_module("_flexiv_dual_parallel")
            metrics = runner.run(
                app_launcher.app,
                spec=spec,
                num_envs=args.num_envs,
                duration_s=args.duration,
                device=args.device,
                validate=args.validate,
                left_robot_model=args.left_robot_model,
                right_robot_model=args.right_robot_model,
                arm1_t_arm2=args.arm1_t_arm2,
                main_frame=args.main_frame,
                reference_t_main=args.reference_t_main,
                torque_limits_nm=args.torque_limits_nm,
            )
        print(json.dumps(dataclasses.asdict(metrics), indent=2, sort_keys=True), flush=True)
    except Exception:
        # Kit shutdown can consume the interpreter's normal unhandled-exception
        # report, so emit it before closing the application.
        traceback.print_exc()
        raise
    finally:
        app_launcher.app.close()
