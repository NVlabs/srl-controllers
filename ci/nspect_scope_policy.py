# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Validate non-shipped location configuration before the existing native import."""

import hashlib
import io
from pathlib import Path
import re


def load_exclusions(source_root, parse_allowlist, nspect_id="NSPECT-BE6H-WKN4"):
    """Keep the official Pulse parser and exclude-directory payload unchanged.

    No policy is active until the repository contains a reviewed allowlist file.
    This hook supports location scope only, never vulnerability dispositions.
    """
    path = Path(source_root) / ".nspect-allowlist.toml"
    if path.is_symlink():
        raise ValueError("The release scope policy must be a regular repository file")
    if not path.exists():
        return [], None
    if not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError("Invalid or oversized release scope policy")
    content = path.read_bytes()
    config = parse_allowlist(io.StringIO(content.decode("utf-8"))).get_allow_config()
    if config.get("allow_vulns") or config.get("nvidia_dependencies"):
        raise ValueError("This scope correction does not authorize vulnerability or ownership overrides")
    selected = []
    for entry in config.get("excluded_directories", []):
        if entry.get("nspect_ids") and nspect_id not in entry["nspect_ids"]:
            continue
        if not entry.get("paths") or not entry.get("comment", "").strip():
            raise ValueError("Excluded directories need explicit paths and rationale")
        for path in entry["paths"]:
            if len(path) > 2000 or any(char in path for char in "\r\n\0"):
                raise ValueError("Invalid package-location pattern")
            literal = path
            if path.endswith(r"$(?!\n)"):
                encoded = path[:-len(r"$(?!\n)")]
                literal = re.sub(r"\\(.)", r"\1", encoded)
                if not literal or re.escape(literal) != encoded:
                    raise ValueError("File patterns must match one escaped literal location exactly")
            elif not path.endswith("/"):
                raise ValueError("Use a directory path or an exact anchored file-location pattern")
            if literal.lower() in {"/", "//", "./", ".*/", "/usr/", "/usr/include/", "/usr/lib/", "/lib/"}:
                raise ValueError("A broad system/source root is not a validated package scope")
            # Pulse lowercases patterns before matching. Do not use \\Z: it becomes
            # \\z on the pinned Python runtime. The end/newline guard stays exact
            # under Pulse's re.MULTILINE flag and case normalization.
            re.compile(path.lower(), re.M | re.I)
        selected.append(entry)
    return selected, hashlib.sha256(content).hexdigest()


def preview_locations(exclusions, package_locations, matches):
    """Show all retained locations; missing/blank locations cannot establish exclusion."""
    paths = [path for entry in exclusions for path in entry["paths"]]
    result = []
    for package in package_locations:
        locations = package.get("locations", [])
        excluded = [location for location in locations if matches(location, paths)]
        complete = package.get("complete") is True and bool(locations) and all(locations)
        result.append({"name": package["name"], "version": package["version"],
                       "origin": package.get("origin"), "complete": bool(complete),
                       "locations": locations, "matched": excluded,
                       "fully_excluded": bool(complete and len(excluded) == len(locations))})
    return result
