# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Check legal files in real source, wheel and CPack archives or a native installation."""

from __future__ import annotations

# Standard Library
import argparse
import hashlib
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

PRIMARY_SHA256 = "0799fbb323e03399571dcc8279d7cd7433f2374e33502b2c723961635ea9fea2"
LICENSE_EXPRESSION = "LicenseRef-NVIDIA-2022-03-22 AND Apache-2.0 AND BSD-3-Clause"
LEGAL_FILES = (
    "LICENSE",
    "LICENSES/Apache-2.0.txt",
    "THIRD_PARTY_NOTICES.md",
    "examples/ur/ur_description_LICENSE",
)


def require(condition: bool, message: str) -> None:
    """Keep package validation active even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def read_members(path: Path) -> dict[str, bytes]:
    """Read an installation directory or supported archive without extracting it."""
    if path.is_dir():
        return {
            p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    with tarfile.open(path) as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def validate_members(
    members: dict[str, bytes], expected: dict[str, bytes], wheel: bool = False
) -> None:
    """Check complete legal-file groups and the declared wheel distribution expression."""
    candidates = [
        n[: -len("LICENSE")]
        for n in members
        if n == "LICENSE"
        or (
            n.endswith("/LICENSE")
            and (
                n.count("/") == 1
                or n.endswith("share/doc/nvidia-srl-controllers/LICENSE")
                or ".dist-info/licenses/" in n
            )
        )
    ]
    require(bool(candidates), "No package primary-license location was found")
    for prefix in candidates:
        for name, data in expected.items():
            target = prefix + name
            require(target in members, f"Missing packaged legal file: {target}")
            require(members[target] == data, f"Incorrect packaged legal file: {target}")
    metadata = [n for n in members if n.endswith(".dist-info/METADATA")]
    if wheel:
        require(len(metadata) == 1, "Expected exactly one wheel METADATA file")
    for name in metadata:
        record = BytesParser().parsebytes(members[name])
        require(record["License-Expression"] == LICENSE_EXPRESSION, name)
        require(set(record.get_all("License-File", [])) == set(LEGAL_FILES), name)
    for name in members:
        if name.endswith("/PKG-INFO"):
            record = BytesParser().parsebytes(members[name])
            require(record["License-Expression"] == LICENSE_EXPRESSION, name)
            require(set(record.get_all("License-File", [])) == set(LEGAL_FILES), name)


def self_test(expected: dict[str, bytes]) -> None:
    """Reject missing notices, altered terms, and obsolete wheel metadata."""
    fixture = {"package/" + name: data for name, data in expected.items()}
    validate_members(fixture, expected)
    installed = {
        "share/doc/nvidia-srl-controllers/" + name: data for name, data in expected.items()
    }
    validate_members(installed, expected)
    broken_cases = []
    for name in LEGAL_FILES:
        missing = dict(fixture)
        del missing["package/" + name]
        broken_cases.append(missing)
    altered = dict(fixture)
    altered["package/LICENSE"] = expected["LICENSE"].replace(
        b"research or evaluation", b"commercial deployment"
    )
    broken_cases.append(altered)
    for broken in broken_cases:
        try:
            validate_members(broken, expected)
        except ValueError:
            continue
        raise AssertionError("Invalid package legal files were accepted")
    wheel = {"package.dist-info/licenses/" + name: data for name, data in expected.items()}
    metadata = (
        "Metadata-Version: 2.4\nName: package\nVersion: 0\n"
        f"License-Expression: {LICENSE_EXPRESSION}\n"
        + "".join(f"License-File: {name}\n" for name in LEGAL_FILES)
    ).encode()
    wheel["package.dist-info/METADATA"] = metadata
    validate_members(wheel, expected, wheel=True)
    wheel["package.dist-info/METADATA"] = metadata.replace(
        b"LicenseRef-NVIDIA-2022-03-22", b"LicenseRef-NSCL"
    )
    try:
        validate_members(wheel, expected, wheel=True)
    except ValueError:
        return
    raise AssertionError("Obsolete wheel license metadata was accepted")


def main() -> None:
    """Check supplied package outputs against the tracked legal materials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", type=Path, nargs="*")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = {name: (root / name).read_bytes() for name in LEGAL_FILES}
    require(
        hashlib.sha256(expected["LICENSE"]).hexdigest() == PRIMARY_SHA256,
        "The source primary license differs from the selected catalog text",
    )
    if arguments.self_test:
        self_test(expected)
        print("Packaged-license self-test passed.")
    if not arguments.self_test and not arguments.packages:
        parser.error("Supply a package/archive/install directory or --self-test")
    for path in arguments.packages:
        validate_members(read_members(path), expected, wheel=path.suffix == ".whl")
        print(f"Verified packaged license files: {path}")


if __name__ == "__main__":
    main()
