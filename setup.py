# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Build a self-contained native wheel using the same CMake targets as C++ consumers."""

# Standard Library
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Third Party
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Stage the extension and its controller library together with relative runtime lookup."""

    def build_extension(self, extension):
        """Configure an isolated build and copy native artifacts into the wheel tree."""
        source = Path(__file__).resolve().parent
        build = Path(self.build_temp).resolve() / "controllers"
        destination = Path(self.get_ext_fullpath(extension.name)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DENABLE_PYBIND=ON",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_TOOLS=OFF",
            "-DENABLE_DOCS=OFF",
            "-DENABLE_EXAMPLES=OFF",
            "-DENABLE_COVERAGE=OFF",
            "-DDEVEL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DNVIDIASRLCONTROLLERS_PYTHON_SOURCE_SYMLINK=OFF",
            "-DNVIDIASRLCONTROLLERS_PYTHON_WHEEL=ON",
            (
                "-DCMAKE_INSTALL_RPATH=@loader_path"
                if sys.platform == "darwin"
                else "-DCMAKE_INSTALL_RPATH=$ORIGIN"
            ),
        ]
        arguments += shlex.split(os.environ.get("CONTROLLERS_CMAKE_ARGS", ""))
        subprocess.run(arguments, check=True)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "nvidia_srl_controllers_cpp",
                "--parallel",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
            ],
            check=True,
        )
        native = build / "src/nvidia/srl/controllers"
        extension_files = list(native.glob("nvidia_srl_controllers_cpp.*"))
        extension_files = [
            path for path in extension_files if path.suffix in (".so", ".pyd", ".dylib")
        ]
        if len(extension_files) != 1:
            raise RuntimeError(f"Expected one native extension, found {extension_files}")
        shutil.copy2(extension_files[0], destination)
        libraries = [
            path
            for path in native.iterdir()
            if path.name.startswith(("libnvidia_srl_controllers.", "nvidia_srl_controllers.dll"))
        ]
        if not libraries:
            raise RuntimeError("CMake did not produce the controller shared library")
        for library in libraries:
            shutil.copy2(library, destination.parent / library.name)


setup(
    ext_modules=[Extension("nvidia.srl._controllers.nvidia_srl_controllers_cpp", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
