# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Canonical formatting shared by public API generators and their freshness checks."""

# Standard Library
import subprocess
import sys
from pathlib import Path


def format_source(path: Path, source: str) -> str:
    """Apply the repository formatters before comparing or writing generated sources."""
    commands = (
        [
            [sys.executable, "-m", "isort", "--filename", str(path), "-"],
            [sys.executable, "-m", "black", "--quiet", "--stdin-filename", str(path), "-"],
        ]
        if path.suffix == ".py"
        else [["clang-format", "--assume-filename=" + str(path)]]
    )
    for command in commands:
        source = subprocess.run(
            command, input=source, text=True, capture_output=True, check=True
        ).stdout
    return source
