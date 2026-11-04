# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Keep the reviewed inventory, installed interfaces and numerical coverage in agreement."""

# Standard Library
import inspect
import json
import re
from pathlib import Path

# NVIDIA
import nvidia.srl.controllers as ctl

# Local Folder
from .equations_test import CASES

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = json.loads((ROOT / "docs/public_api_inventory.json").read_text())


def test_every_controller_declaration_is_public_and_inventoried():
    expected = {entry["name"] for entry in INVENTORY["blocks"]}
    header = (ROOT / "include/nvidia/srl/controllers/controllers.hpp").read_text()
    assert set(re.findall(r"class NVIDIA_SRL_CONTROLLERS_API (\w+)", header)) == expected
    runtime = {
        name
        for name in ctl.__all__
        if inspect.isclass(getattr(ctl, name)) and hasattr(getattr(ctl, name), "step")
    }
    assert runtime == expected
    for name in expected:
        assert callable(getattr(ctl, name).requirements)


def test_every_equation_has_an_analytical_case_and_array_adapter():
    names = {entry["name"] for entry in INVENTORY["equations"]}
    assert names == set(CASES)
    assert names == {
        name for name in ctl.equations.__all__ if inspect.isfunction(getattr(ctl.equations, name))
    }
    header = (ROOT / "include/nvidia/srl/controllers/equations.hpp").read_text()
    raw_names = set(
        re.findall(
            r"NVIDIA_SRL_CONTROLLERS_HOST_DEVICE\s+(?:inline|constexpr)\s+[\w<>]+\s+(\w+)\(", header
        )
    )
    assert (
        raw_names - {"absolute_value", "square_root", "cosine", "arc_tangent2", "finite_value"}
        == names
    )
