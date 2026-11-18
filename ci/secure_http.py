# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Bounded CI HTTPS reads without the affected CPython HTTP response parser."""

import io
import os
import subprocess
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit

MAX_RESPONSE_BYTES = 128 * 1024 * 1024


def _quoted(value: str) -> str:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Control character in HTTPS request")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def urlopen(request, timeout: int = 120) -> io.BytesIO:
    """Keep credentials in memory/stdin and validate HTTPS status separately."""
    if not isinstance(timeout, int) or not 0 < timeout <= 300:
        raise ValueError("A bounded HTTPS timeout is required")
    url = request if isinstance(request, str) else request.full_url
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or any(ord(char) < 32 or ord(char) == 127 for char in unquote(url))):
        raise ValueError("An HTTPS URL without user information or control characters is required")
    headers = [] if isinstance(request, str) else request.header_items()
    data = None if isinstance(request, str) else request.data
    config = ["url = " + _quoted(url), 'proto = "=https"', 'proto-redir = "=https"',
              "connect-timeout = " + str(min(timeout, 30)), "max-time = " + str(timeout),
              "max-filesize = " + str(MAX_RESPONSE_BYTES)]
    # Downloads may redirect over HTTPS. Credential-bearing requests may not.
    if not headers and data is None:
        config.extend(["location", "max-redirs = 5"])
    for key, value in headers:
        config.append("header = " + _quoted(key + ": " + value))
    if data is not None:
        if not isinstance(data, bytes):
            raise ValueError("CI request bodies must be encoded bytes")
        config.append("data-raw = " + _quoted(data.decode("ascii")))
    result = subprocess.run(
        ["/usr/bin/curl", "--silent", "--config", "-", "--write-out", "%{stderr}%{http_code}"],
        input=("\n".join(config) + "\n").encode(), capture_output=True,
        timeout=timeout + 5, check=False,
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "LANGUAGE": "C", "LOCALDOMAIN": "."},
    )
    # Neither response bodies nor curl diagnostics are included in exceptions.
    if result.returncode or len(result.stdout) > MAX_RESPONSE_BYTES:
        raise RuntimeError("CI HTTPS transport failed; response and diagnostics withheld")
    status = result.stderr[-3:]
    if len(status) != 3 or not status.isdigit():
        raise RuntimeError("CI HTTPS transport returned no valid status")
    code = int(status)
    body = io.BytesIO(result.stdout)
    if not 200 <= code < 300:
        raise HTTPError(url, code, "CI HTTPS request rejected", None, body)
    return body
