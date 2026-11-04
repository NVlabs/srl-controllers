# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Install a native wheel into a fresh environment and test it without checkout imports."""

# Standard Library
import argparse
import os
import shutil
import subprocess
import venv
from pathlib import Path


def main() -> None:
    """Create a fresh isolated consumer and run public lifecycle/equation contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--cuda", action="store_true", help="Require real GPU tests for a CUDA wheel"
    )
    arguments = parser.parse_args()
    wheel = arguments.wheel.resolve(strict=True)
    work = arguments.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    environment = work / "environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", str(wheel), "pytest"], check=True)
    tests = work / "tests"
    tests.mkdir()
    source = Path(__file__).resolve().parents[1] / "python/tests/controllers"
    for name in (
        "__init__.py",
        "conftest.py",
        "controller_api_test.py",
        "public_blocks_test.py",
        "equations_test.py",
        "composition_test.py",
    ):
        shutil.copy2(source / name, tests / name)
    runtime_environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    }
    runtime_environment["CONTROLLERS_TEST_CUDA"] = "1" if arguments.cuda else "0"
    # -I excludes the current directory, user packages and Python environment overrides.
    verify_import = (
        "import pathlib,sys,nvidia.srl.controllers as c; "
        "assert pathlib.Path(c.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); "
        "print('Installed public API:',c.__file__)"
    )
    subprocess.run(
        [str(python), "-I", "-c", verify_import], check=True, cwd=work, env=runtime_environment
    )
    # These source-only tests inspect simulator launchers; they are covered by
    # checkout tests, while the installed consumer exercises their public registry separately.
    selected = (
        "not test_registry_has_exactly_one_launcher_for_every_hardware_example and "
        "not test_single_arm_example_rejects_static_impedance_for_non_ur_robots"
    )
    subprocess.run(
        [str(python), "-I", "-m", "pytest", "-q", "-k", selected, str(tests)],
        check=True,
        cwd=work,
        env=runtime_environment,
    )


if __name__ == "__main__":
    main()
