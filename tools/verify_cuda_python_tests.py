# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Reject missing, skipped or failed mandatory Python GPU correctness coverage."""

from __future__ import annotations

# Standard Library
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_FAMILIES = (
    "test_every_new_block_uses_public_metadata_and_lifecycle",
    "test_checked_equations_match_analytical_examples",
    "test_faults_without_safe_mode_cannot_be_treated_as_healthy",
    "test_final_history_advances_once_and_expires",
    "test_static_controllers_use_caller_stream_and_graph",
    "test_admittance_motion_limits_are_independently_optional",
    "test_force_admittance_integral_bounds_are_optional",
)


def verify_report(root: ET.Element) -> int:
    """Require actual successful CUDA fixture cases for each public API family."""
    cases = root.findall(".//testcase")
    cuda_cases = [
        case
        for case in cases
        if case.get("name", "").partition("[")[2].partition("]")[0].split("-")[0] == "cuda"
        and "cuda-parser-only" not in case.get("name", "")
    ]
    if not cuda_cases:
        raise ValueError("Python correctness report contains no CUDA fixture cases")
    for case in cuda_cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise ValueError(f"GPU case did not execute successfully: {case.get('name')}")
    names = [case.get("name", "") for case in cuda_cases]
    missing = [
        family
        for family in REQUIRED_FAMILIES
        if not any(name.startswith(family + "[") for name in names)
    ]
    if missing:
        raise ValueError(f"Missing mandatory Python GPU families: {missing}")
    for kind in ("joint_diagonal", "joint_dense", "cartesian"):
        if not any(
            name.startswith("test_static_controllers_use_caller_stream_and_graph[") and kind in name
            for name in names
        ):
            raise ValueError(f"Missing static GPU controller mode: {kind}")
    return len(cuda_cases)


def self_test() -> None:
    """Exercise the gate against success and incomplete or skipped GPU reports."""
    valid = ET.Element("testsuites")
    suite = ET.SubElement(valid, "testsuite")
    for family in REQUIRED_FAMILIES:
        suffix = (
            "[cuda]"
            if family == "test_final_history_advances_once_and_expires"
            else "[cuda-cartesian]"
        )
        ET.SubElement(suite, "testcase", name=family + suffix)
    for kind in ("joint_diagonal", "joint_dense"):
        ET.SubElement(
            suite,
            "testcase",
            name="test_static_controllers_use_caller_stream_and_graph[cuda-" + kind + "]",
        )
    verify_report(valid)
    for tag in ("skipped", "failure", "error"):
        invalid = ET.fromstring(ET.tostring(valid))
        ET.SubElement(invalid.findall(".//testcase")[0], tag)
        try:
            verify_report(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Accepted a GPU {tag}")
    for invalid in (
        ET.Element("testsuites"),
        ET.fromstring(ET.tostring(valid).replace(b"cuda", b"cpu")),
        ET.fromstring(ET.tostring(valid).replace(b"cuda", b"cuda-parser-only")),
    ):
        try:
            verify_report(invalid)
        except ValueError:
            continue
        raise AssertionError("Accepted a report without CUDA execution")
    invalid = ET.fromstring(ET.tostring(valid))
    invalid.find("testsuite").remove(invalid.findall(".//testcase")[0])
    try:
        verify_report(invalid)
    except ValueError:
        return
    raise AssertionError("Accepted a missing GPU family")


def main() -> None:
    """Validate a pytest JUnit report or exercise the validator itself."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.report:
        count = verify_report(ET.parse(args.report).getroot())
        print(f"Verified actual execution of {count} mandatory Python CUDA cases")
    elif not args.self_test:
        parser.error("report or --self-test is required")


if __name__ == "__main__":
    main()
