#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

"""Report CUDA qualification regressions without changing the CI result."""

# Standard Library
import argparse
import json
import re
from pathlib import Path

PROFILE_PATTERN = re.compile(
    r"^NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE\("
    r"(?P<sm>\d+),\s*(?P<dtype>kFloat32|kFloat64),\s*"
    r"(?P<dimension>\d+),\s*(?P<capacity>\d+),\s*(?P<warps>\d+)\)$"
)


def records(path: Path) -> dict[tuple[object, ...], dict[str, object]]:
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record") not in {"raw_controller", "captured_graph"}:
            continue
        key = tuple(record.get(name) for name in
                    ("record", "sm", "scalar", "joint_dim", "batch", "dispatch"))
        result[key] = record
    return result


def profile_table(path: Path) -> dict[tuple[int, str, int], tuple[int, int]]:
    profiles: dict[tuple[int, str, int], tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PROFILE_PATTERN.match(line.strip())
        if match is None:
            continue
        scalar = "float32" if match["dtype"] == "kFloat32" else "float64"
        key = (int(match["sm"]), scalar, int(match["dimension"]))
        value = (int(match["capacity"]), int(match["warps"]))
        if key in profiles:
            raise ValueError(f"duplicate CUDA dispatch profile: {key}")
        profiles[key] = value
    if not profiles:
        raise ValueError(f"no CUDA dispatch profiles found in {path}")
    return profiles


def dispatch_errors(measurement: Path,
                    expected_profiles: dict[tuple[int, str, int], tuple[int, int]]) -> list[str]:
    observed: dict[tuple[int, str, int], tuple[int, int]] = {}
    for line in measurement.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record") != "raw_controller" or record.get("dispatch") != "auto":
            continue
        key = (int(record["sm"]), str(record["scalar"]), int(record["joint_dim"]))
        value = (int(record["warps_per_block"]),)
        previous = observed.get(key)
        if previous is not None and previous[1] != value[0]:
            return [f"auto-dispatch reported inconsistent warp counts for {key}"]
        observed[key] = (expected_profiles.get(key, (0, 0))[0], value[0])

    errors: list[str] = []
    for key, expected in expected_profiles.items():
        actual = observed.get(key)
        if actual is None:
            errors.append(f"auto-dispatch did not report a measurement for {key}")
        elif actual != expected:
            errors.append(f"auto-dispatch mismatch for {key}: expected {expected}, got {actual}")
    unexpected = sorted(set(observed) - set(expected_profiles))
    for key in unexpected:
        errors.append(f"auto-dispatch reported an unqualified profile: {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--warning-only", action="store_true")
    parser.add_argument("--profile-table", type=Path)
    args = parser.parse_args()

    baseline = records(args.baseline)
    measurement = records(args.measurement)
    warning_count = 0
    for key, current in measurement.items():
        reference = baseline.get(key)
        if reference is None:
            continue
        thresholds = reference.get("regression_thresholds_pct", {})
        for metric in ("p50_us", "p99_us"):
            threshold = thresholds.get(metric)
            if threshold is None or not reference.get(metric):
                continue
            regression = ((float(current[metric]) / float(reference[metric])) - 1.0) * 100.0
            if regression > float(threshold):
                warning_count += 1
                print(f"WARNING {key} {metric} regressed {regression:.2f}% "
                      f"(threshold {float(threshold):.2f}%).")
    if warning_count == 0:
        print("No configured CUDA performance regressions detected.")
    if args.profile_table is not None:
        try:
            errors = dispatch_errors(args.measurement, profile_table(args.profile_table))
        except (KeyError, TypeError, ValueError) as error:
            print(f"ERROR unable to validate CUDA auto-dispatch: {error}")
            return 1
        for error in errors:
            print(f"ERROR {error}")
        if errors:
            return 1
        print("Qualified CUDA dispatch profiles were applied by the public auto-dispatch path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
