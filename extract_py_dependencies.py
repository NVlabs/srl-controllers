#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Generate dependency text files from `pyproject.toml` file.

The base dependencies are written to `pip_pkg_list.txt` file and the optional dependencies are
written to `pip_pkg_list_<extra>.txt` files.
"""

# Standard Library
import tomllib
from pathlib import Path


def generate_requirements(pyproject_path: Path, dep_file_name_base: str = "pip_pkg_list") -> None:
    """
    Generate requirements from pyproject.toml.

    The base dependencies are written to `pip_pkg_list.txt` file and the optional dependencies are
    written to `pip_pkg_list_<extra>.txt` files.

    Args:
        pyproject_path: Path to pyproject.toml.
        dep_file_name_base: Base name of the dependency files.
    """
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    base_deps = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    # Write base requirements
    filename = pyproject_path.parent / f"{dep_file_name_base}.txt"
    filename.write_text("\n".join(base_deps) + "\n")
    print(f"Wrote base dependencies to {filename}")

    # Write optional requirements
    for extra, deps in optional_deps.items():
        filename = pyproject_path.parent / f"{dep_file_name_base}_{extra}.txt"
        filename.write_text("\n".join(deps) + "\n")
        print(f"Wrote optional dependencies for '{extra}' to {filename}")


if __name__ == "__main__":
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        print("Error: pyproject.toml not found in current directory.")
    else:
        generate_requirements(pyproject_file)
