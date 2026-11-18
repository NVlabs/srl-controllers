# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Read file locations from one existing core C++ scan; never submit inventory."""

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import ssl
import subprocess
import sys
import tempfile
import uuid
from urllib.parse import urlencode
from urllib.request import Request
import base64

ROOT = Path(__file__).resolve().parent
NSPECT = "NSPECT-BE6H-WKN4"
SOURCE = "ssh://git@gitlab-master.nvidia.com:12051/srl/cpp/controllers.git"
BD = "https://nvidia.app.blackduck.com"
INDEX = "https://artifactory.nvidia.com/artifactory/api/pypi/sw-sds-tools-pypi/simple"
ISSUER = "https://4ubglassowmtsi7ogqwarmut7msn1q5ynts62fwnr1i.ssa.nvidia.com/token"
PIP = "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
OTHER_MODES = ("CONTROLLERS_OSS_SCAN", "CONTROLLERS_MALWARE_SCAN",
               "CONTROLLERS_MALWARE_AUTH_DIAGNOSTIC", "CONTROLLERS_OSS_RUNTIME_CHECK",
               "CONTROLLERS_SCAN_IMAGE_REFRESH")


def selected_scan(environment):
    required = {"CI_PROJECT_ID": "160347", "CI_COMMIT_BRANCH": "main",
                "CI_COMMIT_REF_PROTECTED": "true", "CI_DISPOSABLE_ENVIRONMENT": "true",
                "CONTROLLERS_OSS_SCOPE_DIAGNOSTIC": "1"}
    if any(environment.get(k) != v for k, v in required.items()):
        raise ValueError("Expected the protected Controllers scope diagnostic")
    if environment.get("CI_PIPELINE_SOURCE") not in {"api", "web"}:
        raise ValueError("An explicitly requested web/API pipeline is required")
    if any(environment.get(k) == "1" for k in OTHER_MODES):
        raise ValueError("Scope reads cannot be combined with another execution mode")
    source_sha = environment.get("CONTROLLERS_SCOPE_SOURCE_SHA", "")
    reader_sha = environment.get("CI_COMMIT_SHA", "")
    pipeline = environment.get("CONTROLLERS_SCOPE_PIPELINE_ID", "")
    job = environment.get("CONTROLLERS_SCOPE_JOB_ID", "")
    if not all(re.fullmatch(r"[a-f0-9]{40}", s) for s in (source_sha, reader_sha)):
        raise ValueError("Full source and reader commit identities are required")
    if not all(re.fullmatch(r"[1-9][0-9]{0,15}", s) for s in (pipeline, job)):
        raise ValueError("Existing scan pipeline/job identities are required")
    return {"source_commit": source_sha, "reader_commit": reader_sha,
            "profile": "core_cpp", "pipeline_id": pipeline, "job_id": job,
            "project": f"[c-cpp-scan][oss][srl-controllers-{NSPECT}-core_cpp-{pipeline}-{job}]",
            "version": "0.12.0-" + source_sha}


def safe_write(path, value, secrets):
    data = json.dumps(value, indent=2) + "\n"
    if any(secret and secret in data for secret in secrets):
        raise ValueError("Credential detected; evidence write refused")
    path.write_text(data)


def collect_locations(data, identity, output, secrets):
    """Use only existing-project GET collections; preserve component and origin rows."""
    project = data.get_project_by_name(identity["project"])
    if project.get("name") != identity["project"]:
        raise ValueError("Existing backend project identity mismatch")
    version = data.get_project_version_by_name(project, identity["version"])
    if version.get("versionName") != identity["version"]:
        raise ValueError("Existing backend version identity mismatch")
    prefix = version.get("_meta", {}).get("href", "")
    if not re.fullmatch(re.escape(BD) + r"/api/projects/[a-f0-9-]{36}/versions/[a-f0-9-]{36}", prefix):
        raise ValueError("Unexpected backend version destination")
    components = list(data.get_project_components(
        identity["project"], identity["version"], confirm_unknown_versions=False))
    if not components:
        raise ValueError("Existing component inventory is empty")
    safe_write(output / "backend-components.json", {"complete": True, "items": components}, secrets)
    records = []
    for index, component in enumerate(components):
        for origin_index, resource in [(None, component), *enumerate(component.get("origins", []))]:
            links = resource.get("_meta", {}).get("links", [])
            url = next((x["href"] for x in links if x.get("rel") == "matched-files"), None)
            label = f"matched-files-{index}-" + ("component" if origin_index is None else str(origin_index))
            if url is None:
                records.append({"name": component.get("componentName"),
                                "version": component.get("componentVersionName"),
                                "origin": resource.get("externalId"), "complete": False,
                                "reason": "No matched-files relationship in the source response"})
                continue
            suffix = r"/components/[a-f0-9-]{36}/versions/[a-f0-9-]{36}(?:/origins/[a-f0-9-]{36})?/matched-files"
            if not re.fullmatch(re.escape(prefix) + suffix, url):
                raise ValueError("Unexpected matched-files destination")
            items = []
            for row in data.hub.get_items(url, headers={
                    "Accept": "application/vnd.blackducksoftware.bill-of-materials-6+json"}):
                items.append(row)
                if len(items) > 10000:
                    raise ValueError("Matched-files collection exceeds this diagnostic's bound")
            record = {"name": component.get("componentName"),
                      "version": component.get("componentVersionName"),
                      "origin": resource.get("externalId"), "source": url,
                      "complete": True, "items": items}
            safe_write(output / (label + ".json"), record, secrets)
            records.append({k: v for k, v in record.items() if k != "items"} | {"count": len(items)})
    safe_write(output / "location-index.json", {**identity, "collections": records,
               "complete": all(x["complete"] for x in records),
               "scan_started": False, "inventory_imported": False,
               "findings_changed": False, "exclusions_applied": False}, secrets)
    return len(records)


def bootstrap(identity):
    """Install the existing pinned SDK in an isolated, credential-free child environment."""
    if not Path("/.dockerenv").is_file() or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise ValueError("Expected the existing Linux amd64/Python 3.12 scan image")
    roots = ssl.create_default_context().get_ca_certs(binary_form=True)
    if not roots or any(hashlib.sha256(c).hexdigest() ==
            "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b" for c in roots):
        raise ValueError("Expected the reviewed system certificate trust")
    output = Path(os.environ["CI_PROJECT_DIR"]) / "controllers-scan-results/scope-diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="controllers-scope-read-") as directory:
        work = Path(directory)
        clean = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work),
                 "LANG": "C.UTF-8", "LANGUAGE": "C", "LOCALDOMAIN": ".",
                 "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                 "DD_TRACE_ENABLED": "false", "DD_INSTRUMENTATION_TELEMETRY_ENABLED": "false"}
        python = str(work / "venv/bin/python")
        def run(args):
            subprocess.run(args, env=clean, cwd=work, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        run([sys.executable, "-I", "-m", "venv", str(work / "venv")])
        run([python, "-I", "-m", "pip", "install", "--no-input", "--no-deps", PIP])
        run([python, "-I", "-m", "pip", "install", "--no-input", "--only-binary=:all:",
             "--require-hashes", "--index-url", INDEX, "-r", str(ROOT / "scope-reader.lock")])
        run([python, "-I", "-m", "pip", "check"])
        run([python, "-I", str(ROOT / "check_certificate_trust.py"), "--pip", "--certifi"])
        child = {**clean, "CONTROLLERS_SCOPE_READ_WORKER": "1"}
        for key in ("SF_CLIENT_ID", "SF_CLIENT_SECRET", "CI_PROJECT_DIR", "CI_PROJECT_ID",
                    "CI_COMMIT_SHA", "CI_COMMIT_BRANCH", "CI_COMMIT_REF_PROTECTED",
                    "CI_PIPELINE_SOURCE", "CI_DISPOSABLE_ENVIRONMENT",
                    "CONTROLLERS_OSS_SCOPE_DIAGNOSTIC", "CONTROLLERS_SCOPE_SOURCE_SHA",
                    "CONTROLLERS_SCOPE_PIPELINE_ID", "CONTROLLERS_SCOPE_JOB_ID", *OTHER_MODES):
            if key in os.environ:
                child[key] = os.environ[key]
        subprocess.run([python, "-I", str(Path(__file__).resolve())], env=child, cwd=work,
                       check=True, timeout=900)


def main():
    identity = selected_scan(os.environ)
    if os.environ.get("CONTROLLERS_SCOPE_READ_WORKER") != "1":
        bootstrap(identity)
        return
    output = Path(os.environ["CI_PROJECT_DIR"]) / "controllers-scan-results/scope-diagnostic"
    spec = importlib.util.spec_from_file_location("scope_https", ROOT / "secure_http.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    client_id = os.environ.pop("SF_CLIENT_ID", "")
    secret = os.environ.pop("SF_CLIENT_SECRET", "")
    if not client_id or not secret:
        raise ValueError("Existing protected SSA credentials are required")
    basic = base64.b64encode((client_id + ":" + secret).encode()).decode()
    secrets = [client_id, secret, basic]
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        request = Request(ISSUER, data=urlencode({"grant_type": "client_credentials",
                          "scope": "verify:nspectid sourcecode:blackduck"}).encode(),
                          headers={"Authorization": "Basic " + basic,
                                   "Content-Type": "application/x-www-form-urlencoded"})
        with transport.urlopen(request, timeout=60) as response:
            token = json.load(response).get("access_token")
        if not isinstance(token, str) or not token:
            raise ValueError("SSA did not return an access token")
        secrets.append(token)
        from pulse.utilities.api_requests import verify_nspect_id_and_retrive_info
        from pulse.oss.blackduck.api.api import BlackDuckDataHandler
        verified = verify_nspect_id_and_retrive_info(
            NSPECT, token, SOURCE, identity["source_commit"], str(uuid.uuid4()))
        key = verified.get("api-token")
        if not isinstance(key, str) or not key:
            raise ValueError("Existing scan read access is unavailable")
        secrets.append(key)
        data = BlackDuckDataHandler(bd_api_key=key, hub_url=BD)
        count = collect_locations(data, identity, output, secrets)
    print(json.dumps({"state": "existing_locations_saved", "collections": count,
                      "scan_started": False, "inventory_imported": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never expose SDK/HTTP exception strings, payloads or credentials.
        raise SystemExit("Existing-scan location read failed: " + type(error).__name__) from None
