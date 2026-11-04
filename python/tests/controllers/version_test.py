# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

"""Test version functionality.

@file version_test.py
@brief This file contains the tests for version functionality.
"""

# Standard Library
import unittest

# NVIDIA
from nvidia.srl.controllers import __version__


class VersionTest(unittest.TestCase):
    def test_full_version(self):
        """Test full version functionality."""
        # This test checks that the version attribute exists and returns a string
        self.assertIsInstance(__version__, str)
        self.assertGreater(len(__version__), 0)


if __name__ == "__main__":
    unittest.main()
