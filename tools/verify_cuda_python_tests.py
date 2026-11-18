# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Reject missing, skipped or failed mandatory Python GPU correctness coverage."""

from __future__ import annotations

# Standard Library
import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_REPORT_NODES = 250_000
MAX_REPORT_DEPTH = 256
REPORT_TIMEOUT_SECONDS = 5


class _ReportTreeBuilder(ET.TreeBuilder):
    """Reject DTDs before a report can define or expand XML entities."""

    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise ValueError("JUnit reports must not contain a document type declaration")


def read_report(path: Path) -> ET.Element:
    """Parse reports in a bounded child; rebuild the result without parsing XML."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--_parse-report", str(path.resolve())],
            capture_output=True,
            timeout=REPORT_TIMEOUT_SECONDS,
            check=False,
            env={"PATH": os.defpath, "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("JUnit parsing exceeded the five-second limit") from error
    if result.returncode:
        raise ValueError("JUnit report rejected: " + result.stderr.decode("utf-8", errors="replace")[:300])
    rows = json.loads(result.stdout)
    elements = []
    for parent, tag, attributes, text, tail in rows:
        element = ET.Element(tag, attributes)
        element.text, element.tail = text, tail
        if parent is not None:
            elements[parent].append(element)
        elements.append(element)
    return elements[0]


def _parse_report(path: Path) -> None:
    """Private worker: no credentials, bounded memory/CPU and UTF-8-only XML."""
    if sys.platform.startswith("linux"):
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (REPORT_TIMEOUT_SECONDS,) * 2)
    with path.open("rb") as report:
        content = report.read(MAX_REPORT_BYTES + 1)
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("JUnit report exceeds the 32 MiB input limit")
    # CTest and pytest emit UTF-8. Do not pass attacker-controlled UTF-16 code
    # units to an older system Expat with incomplete surrogate validation.
    text = content.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("JUnit reports must be UTF-8, not UTF-16 or UTF-32")
    root = ET.fromstring(text, parser=ET.XMLParser(target=_ReportTreeBuilder()))
    rows = []
    pending = [(None, root, 0)]
    while pending:
        parent, element, depth = pending.pop()
        if depth > MAX_REPORT_DEPTH or len(rows) >= MAX_REPORT_NODES:
            raise ValueError("JUnit report exceeds the node or depth limit")
        index = len(rows)
        rows.append((parent, element.tag, element.attrib, element.text, element.tail))
        pending.extend((index, child, depth + 1) for child in reversed(element))
    # The parent decodes only generated JSON and builds Elements directly. It
    # never reparses XML, so Expat complexity stays inside the bounded worker.
    json.dump(rows, sys.stdout, ensure_ascii=True)


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
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "report.xml"
        report.write_bytes(b"<testsuites><testsuite/></testsuites>")
        assert read_report(report).tag == "testsuites"
        malicious = (
            '<!DOCTYPE testsuites [<!ENTITY repeated "text">]>'
            '<testsuites>&repeated;</testsuites>'
        )
        for encoding in ("utf-8", "utf-16"):
            report.write_bytes(malicious.encode(encoding))
            try:
                read_report(report)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted a JUnit report with an entity declaration")
        with report.open("wb") as oversized:
            oversized.truncate(MAX_REPORT_BYTES + 1)
        try:
            read_report(report)
        except ValueError as error:
            assert "input limit" in str(error)
        else:
            raise AssertionError("Accepted an oversized JUnit report")
        report.write_text("<a>" * (MAX_REPORT_DEPTH + 2) + "</a>" * (MAX_REPORT_DEPTH + 2))
        try:
            read_report(report)
        except ValueError as error:
            assert "depth limit" in str(error)
        else:
            raise AssertionError("Accepted an excessive XML nesting depth")
        report.write_text('<testsuites name="é"><testcase>escaped &lt;!DOCTYPE text</testcase></testsuites>')
        parsed = read_report(report)
        assert parsed.attrib["name"] == "é" and parsed[0].text == "escaped <!DOCTYPE text"
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
    parser.add_argument("--_parse-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._parse_report:
        try:
            _parse_report(args._parse_report)
        except (ValueError, OSError, ET.ParseError) as error:
            print(str(error)[:250], file=sys.stderr)
            raise SystemExit(1) from None
        return
    if args.self_test:
        self_test()
    if args.report:
        count = verify_report(read_report(args.report))
        print(f"Verified actual execution of {count} mandatory Python CUDA cases")
    elif not args.self_test:
        parser.error("report or --self-test is required")


if __name__ == "__main__":
    main()
