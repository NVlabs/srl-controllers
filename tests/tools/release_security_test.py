# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Offline regressions for CI HTTPS boundaries and bounded report parsing."""

import importlib.util
import ast
from pathlib import Path
import subprocess
import re
import tempfile
import textwrap
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HTTP = load("secure_http", "ci/secure_http.py")
XML = load("secure_xml", "tools/verify_cuda_python_tests.py")


class ReleaseSecurityTest(unittest.TestCase):
    def test_git_only_bootstraps_have_no_archive_variable_or_extraction(self):
        text = (ROOT / ".gitlab-ci.yml").read_text()
        count = 0
        pattern = r"python3 -I - <<'PY'\n(.*?)(?:^\s*PY$)"
        for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
            source = textwrap.dedent(match[1])
            if 'PROFILE_ID = os.environ.get("CONTROLLERS_OSS_PROFILE"' not in source:
                continue
            tree = ast.parse(source)
            names = {node.id for node in ast.walk(tree)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
            self.assertNotIn("archive", names)
            self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "extractall"
                                 for node in ast.walk(tree)))
            self.assertIn('"source_git_tree"', source)
            count += 1
        self.assertEqual(count, 14)

    def test_embedded_ci_security_helpers_are_current(self):
        expected = {
            "SECURE_HTTP_SOURCE": (ROOT / "ci/secure_http.py").read_text(),
            "RUNTIME_SECURITY_SOURCE": (ROOT / "ci/check_scan_runtime.py").read_text(),
            "XML_REPORT_SOURCE": (ROOT / "tools/verify_cuda_python_tests.py").read_text(),
        }
        counts = dict.fromkeys(expected, 0)
        for line in (ROOT / ".gitlab-ci.yml").read_text().splitlines():
            name, separator, value = line.strip().partition(" = ")
            if separator and name in expected:
                self.assertEqual(ast.literal_eval(value), expected[name], name)
                counts[name] += 1
        self.assertEqual(counts, {"SECURE_HTTP_SOURCE": 15, "RUNTIME_SECURITY_SOURCE": 15,
                                  "XML_REPORT_SOURCE": 1})

    def test_credentials_are_stdin_only_and_redirects_disabled(self):
        request = Request("https://example.invalid/token", data=b"grant_type=client_credentials",
                          headers={"Authorization": "Bearer example"})
        result = subprocess.CompletedProcess([], 0, b'{"example":true}', b"200")
        with patch.object(HTTP.subprocess, "run", return_value=result) as run:
            self.assertEqual(HTTP.urlopen(request).read(), result.stdout)
        args, kwargs = run.call_args
        self.assertNotIn("Bearer example", str(args))
        self.assertIn(b"Bearer example", kwargs["input"])
        self.assertNotIn(b"\nlocation\n", kwargs["input"])
        self.assertEqual(kwargs["env"]["LOCALDOMAIN"], ".")
        self.assertNotIn("SF_CLIENT_SECRET", kwargs["env"])

    def test_rejects_controls_and_insecure_destinations_before_execution(self):
        for url in ["http://example.invalid", "https://user:pass@example.invalid",
                    "https://example.invalid/%00", "https://example.invalid/\n"]:
            with self.subTest(url=url), patch.object(HTTP.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    HTTP.urlopen(url)
                run.assert_not_called()

    def test_http_failure_retains_status_without_logging_body(self):
        result = subprocess.CompletedProcess([], 0, b'{"private":"example"}', b"401")
        with patch.object(HTTP.subprocess, "run", return_value=result):
            with self.assertRaises(HTTPError) as failure:
                HTTP.urlopen("https://example.invalid/token")
        self.assertEqual(failure.exception.code, 401)
        self.assertNotIn("private", str(failure.exception))
        self.assertEqual(failure.exception.read(), result.stdout)

    def test_transport_failure_does_not_echo_diagnostics(self):
        result = subprocess.CompletedProcess([], 60, b"private response", b"private diagnostic000")
        with patch.object(HTTP.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError) as failure:
                HTTP.urlopen("https://example.invalid/token")
        self.assertNotIn("private", str(failure.exception))

    def test_xml_rejects_utf16_even_without_bom(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.xml"
            for encoding in ["utf-16-le", "utf-16-be"]:
                report.write_bytes("<testsuites/>".encode(encoding))
                with self.assertRaises(ValueError):
                    XML.read_report(report)

    def test_xml_timeout_does_not_parse_in_parent(self):
        with patch.object(XML.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 5)):
            with self.assertRaisesRegex(ValueError, "five-second"):
                XML.read_report(Path("unused.xml"))


if __name__ == "__main__":
    unittest.main()
