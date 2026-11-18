# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Read actual CI runtime conditions used by the source finding assessments."""

import hashlib
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import sysconfig


def check_runtime(environment):
    """Fail on known unsafe conditions; return evidence, never a PIC decision."""
    if environment.get("LOCALDOMAIN") != "." or environment.get("LANGUAGE") != "C":
        raise ValueError("CI requires a fixed resolver search list and translation language")
    if platform.machine() != "x86_64" or not sys.platform.startswith("linux"):
        raise ValueError("This finding assessment is scoped to Linux amd64")
    nscd_paths = [p for p in ["/usr/sbin/nscd", "/usr/bin/nscd", "/run/nscd/socket", "/var/run/nscd/socket"]
                  if Path(p).exists()]
    if nscd_paths:
        raise ValueError("nscd is present; its source finding needs a separate assessment")
    executable = Path(sys.executable).resolve()
    if executable.stat().st_mode & (stat.S_ISUID | stat.S_ISGID) or sysconfig.is_python_build():
        raise ValueError("Expected an installed, non-privileged Python interpreter")
    vpath = sysconfig.get_config_var("VPATH")
    landmarks = []
    if isinstance(vpath, str):
        candidate = (executable.parent / vpath / "Modules" / "Setup.local").resolve()
        # Check every existing directory that could supply the build-layout
        # landmark, rather than assuming all non-Windows layouts are safe.
        for directory in [candidate.parent, *candidate.parent.parents]:
            if not directory.exists():
                continue
            info = directory.stat()
            writable = os.access(directory, os.W_OK)
            if info.st_uid != 0 or writable or info.st_mode & 0o022:
                raise ValueError("An untrusted directory can supply Python's build-layout landmark")
            landmarks.append({"path": str(directory), "uid": info.st_uid,
                              "mode": oct(stat.S_IMODE(info.st_mode)), "writable_by_ci_user": writable})

    privileged = []
    unreadable = []

    def inaccessible(error):
        unreadable.append(str(error.filename))

    # Inaccessible directories cannot supply executable paths to this CI user.
    # Virtual kernel filesystems cannot supply the installed loader programs.
    for directory, subdirs, files in os.walk("/", followlinks=False, onerror=inaccessible):
        if directory == "/":
            subdirs[:] = [p for p in subdirs if p not in {"proc", "sys", "dev"}]
        for name in files:
            path = Path(directory) / name
            try:
                info = path.stat()
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0
                        or not info.st_mode & (stat.S_ISUID | stat.S_ISGID)
                        or not os.access(path, os.X_OK)):
                    continue
                with path.open("rb") as stream:
                    if stream.read(4) != b"\x7fELF":
                        continue
                result = subprocess.run(["readelf", "--dynamic", str(path)], env=environment,
                                        capture_output=True, text=True, timeout=15, check=True)
                rpaths = [line.strip() for line in result.stdout.splitlines()
                          if "(RPATH)" in line or "(RUNPATH)" in line]
                for line in rpaths:
                    values = re.search(r"\[(.*)\]", line)
                    if values and any(p.startswith(("$ORIGIN", "${ORIGIN}"))
                                      for p in values[1].split(":")):
                        raise ValueError("Privileged executable has an affected ORIGIN loader path")
                privileged.append({"path": str(path), "rpaths": rpaths,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            except (PermissionError, FileNotFoundError):
                unreadable.append(str(path))
    tinyxml = []
    for path in sorted({p.resolve() for p in Path("/usr/lib/x86_64-linux-gnu").glob("libtinyxml2.so*")}):
        symbols = subprocess.run(["readelf", "--dyn-syms", "--wide", str(path)], env=environment,
                                 capture_output=True, text=True, timeout=15, check=True).stdout
        tinyxml.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "assert_fail_symbol_present": "__assert_fail" in symbols})
    return {
        "contract": "controllers-runtime-security/1", "platform": platform.platform(),
        "architecture": platform.machine(), "python": platform.python_version(),
        "python_executable": str(executable), "python_prefix": sys.base_prefix,
        "python_build_layout": sysconfig.is_python_build(),
        "python_vpath": vpath, "python_landmark_directories": landmarks,
        "python_srcdir": sysconfig.get_config_var("srcdir"),
        "resolver_search_override": environment["LOCALDOMAIN"], "language": environment["LANGUAGE"],
        "nscd_paths_present": nscd_paths, "privileged_elf_programs": privileged,
        "inaccessible_paths": sorted(set(unreadable)), "tinyxml2_libraries": tinyxml,
        "official_approval": "not_established", "network_requests": 0,
    }
