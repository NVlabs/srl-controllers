#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Inspect configured CA bundles offline; never perform a TLS connection."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess

# emSign ECC Root CA - G3, removed by certifi 2026.10.19 and Ubuntu's fix.
# https://ubuntu.com/blog/emudhra-ca-compromise-fixes-available
BLOCKED_ROOT = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"

def inspect_certificates(certificates, label):
    """Reject the affected root without changing any trust configuration."""
    certificates = list(certificates)
    if not certificates:
        raise ValueError("Empty CA store: " + label)
    if any(hashlib.sha256(cert).hexdigest() == BLOCKED_ROOT for cert in certificates):
        raise ValueError("Affected emSign trust anchor remains trusted: " + label)
    return {"store": label, "certificates": len(certificates), "blocked_root_absent": True}

def inspect_bundle(path):
    """Load only the supplied bundle, without implicitly adding system roots."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=str(path))
    return inspect_certificates(context.get_ca_certs(binary_form=True), str(path))

def version_at_least(name, minimum):
    version = importlib.metadata.version(name)
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}", version):
        raise ValueError("Unverified package version: " + name)
    parts = tuple(int(part) for part in version.split("."))
    if parts + (0,) * (3 - len(parts)) < minimum:
        raise ValueError("Certificate-security update required: " + name + " " + version)
    return version

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--pip", action="store_true")
    parser.add_argument("--certifi", action="store_true")
    parser.add_argument("--java-if-present", action="store_true")
    args = parser.parse_args()
    result = {"contract": "controllers-certificate-trust/1", "network_requests": 0,
              "blocked_root_sha256": BLOCKED_ROOT, "stores": [], "packages": {}}
    if args.bundle:
        result["stores"].append(inspect_bundle(args.bundle))
    else:
        result["stores"].append(inspect_certificates(
            ssl.create_default_context().get_ca_certs(binary_form=True), "Python default CA store"))
    for enabled, name, minimum, module in (
        (args.pip, "pip", (26, 2, 2), "pip._vendor.certifi"),
        (args.certifi, "certifi", (2026, 10, 19), "certifi"),
    ):
        if enabled:
            result["packages"][name] = version_at_least(name, minimum)
            bundle = importlib.import_module(module).where()
            result["stores"].append(inspect_bundle(bundle))
    if args.java_if_present:
        if shutil.which("keytool"):
            completed = subprocess.run(
                ["keytool", "-list", "-rfc", "-cacerts", "-storepass", "changeit"],
                check=True, capture_output=True, text=True, timeout=60,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
            certificates = re.findall(
                r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                completed.stdout, re.S)
            result["stores"].append(inspect_certificates(
                (ssl.PEM_cert_to_DER_cert(cert) for cert in certificates), "Java default CA store"))
        else:
            result["java"] = "not installed; inspect after installation"
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ssl.SSLError, subprocess.SubprocessError,
            importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error))

