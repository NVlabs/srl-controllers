#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

"""Validate the selected NVIDIA License text, source headers and retained licenses."""

from __future__ import annotations

# Standard Library
import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PRIMARY_LICENSE = "LicenseRef-NVIDIA-2022-03-22"
PRIMARY_LICENSE_SHA256 = "0799fbb323e03399571dcc8279d7cd7433f2374e33502b2c723961635ea9fea2"
SPDX_IDENTIFIER = re.compile(
    r"^\s*(?:#|//|\*)\s*SPDX-License-Identifier:\s*(.*?)\s*$", re.MULTILINE
)
NVIDIA_COPYRIGHT = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) \d{4}(?:-\d{4})? "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
PROPRIETARY_PHRASES = (
    "property and proprietary " "rights in and to this material",
    "without an express license agreement " "from NVIDIA CORPORATION",
)

CONTROLLED_NAMES = {
    ".clang-format",
    ".gitattributes",
    ".gitignore",
    ".gitlab-ci.yml",
    "CMakeLists.txt",
    "CODEOWNERS",
    "CPPLINT.cfg",
    "Makefile",
    "pyproject.toml",
}
CONTROLLED_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".dockerfile",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".py",
    ".sh",
    ".tpp",
    ".urscript",
}
CONTROLLED_COMPOUND_SUFFIXES = (".hpp.in", ".code-workspace")
LICENSE_EXCEPTIONS = {
    Path("artifacts/cuda/source.tar.gz.sha256"):
        "generated raw sha256sum output; comments would invalidate the machine-readable artifact",
}
RETAINED_LICENSES = {
    Path("examples/franka/franka_utils.hpp"):
        ("Apache-2.0", "Franka-derived source; preserve its upstream license"),
    Path("artifacts/cuda/joint_impedance_cuda_tuning_profiles.inc"):
        ("Apache-2.0", "historical qualification artifact; preserve its original license"),
    Path("artifacts/cuda/cuda_component_tuning_profiles.inc"):
        ("Apache-2.0", "historical qualification artifact; preserve its original license"),
}
RETAINED_DATA = {
    Path("examples/ur/models/ur5e.urdf"),
    Path("examples/ur/models/ur10e.urdf"),
    Path("examples/ur/ur_description_LICENSE"),
}
FRANKA_COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2017 Franka Emika GmbH"


@dataclass(frozen=True)
class LicenseIssue:
    """One validation failure associated with a tracked path."""

    path: Path
    message: str


def is_license_controlled(path: Path) -> bool:
    """Return whether a tracked path must carry a source license header."""
    if path in LICENSE_EXCEPTIONS:
        return False
    if path.name in CONTROLLED_NAMES or path.suffix in CONTROLLED_SUFFIXES:
        return True
    return path.name.endswith(CONTROLLED_COMPOUND_SUFFIXES)


def validate_controlled_text(path: Path, text: str) -> list[LicenseIssue]:
    """Validate the header of one license-controlled text file."""
    header = "\n".join(text.splitlines()[:20])
    issues: list[LicenseIssue] = []
    expected = RETAINED_LICENSES.get(path, (PRIMARY_LICENSE, ""))[0]
    if SPDX_IDENTIFIER.findall(header) != [expected]:
        issues.append(
            LicenseIssue(path, f"expected exactly one {expected} SPDX identifier in first 20 lines")
        )
    if NVIDIA_COPYRIGHT.search(header) is None:
        issues.append(LicenseIssue(path, "missing canonical NVIDIA SPDX copyright in first 20 lines"))
    if path == Path("examples/franka/franka_utils.hpp") and FRANKA_COPYRIGHT not in header:
        issues.append(LicenseIssue(path, "missing retained Franka copyright in first 20 lines"))
    return issues


def validate_retained_data_text(path: Path, text: str) -> list[LicenseIssue]:
    """Reject a conflicting license declaration on retained BSD model material."""
    identifiers = re.findall(r"SPDX-License-Identifier:([^\r\n]*)", text)
    if any(value.strip().removesuffix("-->").strip() != "BSD-3-Clause" for value in identifiers):
        return [LicenseIssue(path, "UR model data must retain BSD-3-Clause terms")]
    return []


def tracked_paths(root: Path) -> list[Path]:
    """Return repository-relative paths tracked by Git."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [Path(value.decode()) for value in result.stdout.split(b"\0") if value]


def read_text(path: Path) -> str | None:
    """Read UTF-8 text, returning None for binary or differently encoded data."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def validate_primary_license(text: str | None) -> list[LicenseIssue]:
    """Require the complete selected catalog text, including its substantive clauses."""
    if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != PRIMARY_LICENSE_SHA256:
        return [
            LicenseIssue(Path("LICENSE"), "expected complete NVIDIA License dated 22 March 2022")
        ]
    return []


def validate_repository(root: Path) -> tuple[list[LicenseIssue], int]:
    """Validate controlled headers, prohibited text, and documented exceptions."""
    issues = validate_primary_license(read_text(root / "LICENSE"))
    controlled_count = 0
    paths = tracked_paths(root)
    tracked = set(paths)

    for exception, reason in LICENSE_EXCEPTIONS.items():
        if exception not in tracked:
            issues.append(LicenseIssue(exception, f"documented exception is not tracked: {reason}"))
    for retained in set(RETAINED_LICENSES) | RETAINED_DATA:
        if retained not in tracked:
            issues.append(LicenseIssue(retained, "documented retained-license path is not tracked"))

    for relative_path in paths:
        text = read_text(root / relative_path)
        if text is None:
            continue
        if relative_path in RETAINED_DATA:
            issues.extend(validate_retained_data_text(relative_path, text))
        for phrase in PROPRIETARY_PHRASES:
            if phrase in text:
                issues.append(LicenseIssue(relative_path, f"contains proprietary notice text: {phrase}"))
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped_line = line.lstrip(" #/*")
            is_copyright_notice = stripped_line.startswith(
                ("SPDX-FileCopyrightText:", "Copyright (c)")
            )
            if is_copyright_notice and "NVIDIA CORPORATION" in line:
                if "NVIDIA CORPORATION & AFFILIATES." not in line:
                    issues.append(
                        LicenseIssue(relative_path, f"line {line_number} uses a noncanonical NVIDIA entity")
                    )
        if is_license_controlled(relative_path):
            controlled_count += 1
            issues.extend(validate_controlled_text(relative_path, text))

    checksum = root / "artifacts/cuda/source.tar.gz.sha256"
    checksum_text = read_text(checksum)
    if checksum_text is None or re.fullmatch(r"[0-9a-f]{64}  -\n?", checksum_text) is None:
        issues.append(LicenseIssue(checksum.relative_to(root), "checksum exception is not raw sha256sum output"))
    return issues, controlled_count


def run_self_test() -> None:
    """Exercise path classification and positive/negative header fixtures."""
    selected = (Path(__file__).resolve().parents[1] / "LICENSE").read_text()
    assert not validate_primary_license(selected)
    assert validate_primary_license(None)
    assert validate_primary_license(selected.replace("research or evaluation", "any commercial"))
    assert validate_primary_license(selected.split("5. Limitation of Liability.")[0])
    assert validate_primary_license(
        selected.replace("NVIDIA License", "NVIDIA Source Code License")
    )
    valid = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        '# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22\n\nprint("ok")\n'
    )
    assert is_license_controlled(Path("sample.py"))
    assert is_license_controlled(Path("CMakeLists.txt"))
    assert is_license_controlled(Path("sample.code-workspace"))
    assert is_license_controlled(Path("generated.hpp.in"))
    assert not is_license_controlled(Path("README.md"))
    assert not is_license_controlled(Path("artifacts/cuda/source.tar.gz.sha256"))
    assert not validate_controlled_text(Path("sample.py"), valid)
    obsolete = valid.replace(PRIMARY_LICENSE, "LicenseRef-NSCL")
    assert validate_controlled_text(Path("sample.py"), obsolete)
    assert len(validate_controlled_text(Path("sample.py"), 'print("missing")\n')) == 2
    wrong_entity = valid.replace("NVIDIA CORPORATION & AFFILIATES.", "NVIDIA CORPORATION")
    assert len(validate_controlled_text(Path("sample.py"), wrong_entity)) == 1
    apache = valid.replace("LicenseRef-NVIDIA-2022-03-22", "Apache-2.0")
    assert len(validate_controlled_text(Path("sample.py"), apache)) == 1
    dual = valid.replace(
        "LicenseRef-NVIDIA-2022-03-22", "LicenseRef-NVIDIA-2022-03-22 OR Apache-2.0"
    )
    assert len(validate_controlled_text(Path("sample.py"), dual)) == 1
    duplicate = valid + "# SPDX-License-Identifier: Apache-2.0\n"
    assert len(validate_controlled_text(Path("sample.py"), duplicate)) == 1
    franka = "# " + FRANKA_COPYRIGHT + "\n" + apache
    retained = Path("examples/franka/franka_utils.hpp")
    assert not validate_controlled_text(retained, franka)
    wrong_franka = franka.replace("Apache-2.0", "LicenseRef-NVIDIA-2022-03-22")
    assert len(validate_controlled_text(retained, wrong_franka)) == 1
    assert len(validate_controlled_text(retained, apache)) == 1
    for artifact in set(RETAINED_LICENSES) - {retained}:
        assert not validate_controlled_text(artifact, apache)
        assert len(validate_controlled_text(artifact, valid)) == 1
    model = Path("examples/ur/models/ur5e.urdf")
    assert not validate_retained_data_text(model, "<robot/>\n")
    bsd = "<!-- SPDX-License-Identifier: BSD-3-Clause -->\n<robot/>\n"
    assert not validate_retained_data_text(model, bsd)
    assert validate_retained_data_text(
        model, bsd.replace("BSD-3-Clause", "LicenseRef-NVIDIA-2022-03-22")
    )
    assert validate_retained_data_text(
        model, bsd.replace("BSD-3-Clause", "BSD-3-Clause AND LicenseRef-NVIDIA-2022-03-22")
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run embedded checker fixtures")
    return parser.parse_args()


def main() -> int:
    """Run the requested validation and return a process status."""
    arguments = parse_arguments()
    if arguments.self_test:
        run_self_test()
        print("License header checker self-test passed.")
        return 0

    root = Path(__file__).resolve().parents[1]
    issues, controlled_count = validate_repository(root)
    for issue in issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)
    if issues:
        print(f"License validation failed with {len(issues)} issue(s).", file=sys.stderr)
        return 1
    print(
        f"Validated {controlled_count} tracked license-controlled files; "
        f"{len(LICENSE_EXCEPTIONS)} documented exception(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
