# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Check local guide links and hardware/companion catalog consistency."""

# Standard Library
import pathlib
import re
import urllib.parse

# NVIDIA
import nvidia.srl.controllers as ctl

ROOT = pathlib.Path(__file__).resolve().parents[3]
GUIDES = (
    tuple(ROOT.glob("*.md"))
    + tuple((ROOT / "docs").glob("*.md"))
    + tuple((ROOT / "examples").glob("*/README.md"))
    + (ROOT / "examples/README.md",)
)


def test_local_guide_links_resolve():
    """Catch broken repository-relative links without requesting network access."""
    failures = []
    for guide in GUIDES:
        content = re.sub(r"```.*?```", "", guide.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", content):
            target = target.strip().strip("<>")
            if urllib.parse.urlsplit(target).scheme or target.startswith("#"):
                continue
            path = urllib.parse.unquote(target.split("#", 1)[0])
            if path and not (guide.parent / path).exists():
                failures.append(f"{guide.relative_to(ROOT)} -> {target}")
    assert not failures, "\n".join(failures)


def test_example_catalog_and_launchers_cover_full_registry():
    """The guide must name every physical target and each launcher must select it."""
    catalog = (ROOT / "examples/README.md").read_text()
    specs = ctl.controller_example_specs()
    assert len(specs) == 22
    for spec in specs:
        assert f"`{spec.hardware_target}`" in catalog
        launcher = ROOT / "examples/isaaclab" / spec.isaac_launcher
        assert f'main("{spec.hardware_target}")' in launcher.read_text()
    assert "two Enter" in (ROOT / "CHANGELOG.md").read_text()
    assert "Rizon4M" in (ROOT / "docs/example-parity.md").read_text()
