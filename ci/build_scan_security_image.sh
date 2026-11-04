#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Invoked by the isolated manual CI job with a minimal environment.
set -eu
umask 077
test "$CI_REGISTRY" = gitlab-master.nvidia.com:5005
test "$CI_REGISTRY_IMAGE" = gitlab-master.nvidia.com:5005/srl/cpp/controllers
test "${#CI_COMMIT_SHA}" -eq 40
case "$CI_COMMIT_SHA" in *[!0-9a-f]*) exit 1 ;; esac
actual_commit=$(git -c safe.directory="$CI_PROJECT_DIR" -C "$CI_PROJECT_DIR" rev-parse HEAD)
test "$actual_commit" = "$CI_COMMIT_SHA"
tracked_changes=$(git -c safe.directory="$CI_PROJECT_DIR" -C "$CI_PROJECT_DIR" status --porcelain --untracked-files=no)
test -z "$tracked_changes"

results="$CI_PROJECT_DIR/controllers-scan-image"
mkdir -p "$results"
work=$(mktemp -d /tmp/controllers-scan-image.XXXXXXXX)
container=''
cleanup() {
    if [ -n "$container" ]; then docker_compat rm "$container" >/dev/null 2>&1 || :; fi
    sudo -n rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
mkdir -p "$work/context/ci" "$work/docker-auth"
cp "$CI_PROJECT_DIR/docker/controllers-scan-security.dockerfile" "$work/context/Dockerfile"
cp "$CI_PROJECT_DIR/ci/refresh_scan_system_packages.sh" "$work/context/ci/refresh_scan_system_packages.sh"

# Reuse the repository's previous successful Docker runner route and API fallback.
sudo -n apt-get update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install --yes docker.io
CI_DOCKER_API_VERSION=''
if ! sudo -n docker version >/dev/null 2>&1; then CI_DOCKER_API_VERSION=1.43; fi
docker_compat() {
    if [ -n "$CI_DOCKER_API_VERSION" ]; then
        sudo -n env DOCKER_CONFIG="$work/docker-auth" DOCKER_API_VERSION="$CI_DOCKER_API_VERSION" docker "$@"
    else
        sudo -n env DOCKER_CONFIG="$work/docker-auth" docker "$@"
    fi
}
docker_compat version > "$results/docker-version.txt"
printf '%s' "$CI_JOB_TOKEN" | docker_compat login --username gitlab-ci-token --password-stdin "$CI_REGISTRY"
unset CI_JOB_TOKEN
image="$CI_REGISTRY_IMAGE/ci:security-$CI_COMMIT_SHA"

# A tag from a prior/uncertain attempt must be reconciled, never overwritten.
if docker_compat manifest inspect "$image" > "$results/existing-manifest.json" 2> "$results/existing-manifest-error.txt"; then
    echo 'Candidate image tag already exists; inspect and reuse the prior build.' >&2
    exit 1
fi
if ! grep -Eq 'manifest unknown|no such manifest' "$results/existing-manifest-error.txt"; then
    echo 'Image absence could not be established; no build or push attempted.' >&2
    exit 1
fi
rm "$results/existing-manifest.json" "$results/existing-manifest-error.txt"
docker_compat build --pull --no-cache --platform linux/amd64 --target ci-security \
    --label "org.opencontainers.image.revision=$CI_COMMIT_SHA" \
    --label 'org.opencontainers.image.source=https://gitlab-master.nvidia.com/srl/cpp/controllers' \
    --iidfile "$results/image-id.txt" --tag "$image" "$work/context"
docker_compat image inspect "$image" > "$results/image-inspect.json"
container=$(docker_compat create --network none --entrypoint /bin/true "$image")
docker_compat cp "$container:/usr/local/share/controllers-scan-security/." "$results/"
docker_compat rm "$container" >/dev/null
container=''
test -s "$results/minimum-versions.tsv"
test -s "$results/host-package-versions.tsv"

# The build's full-version checks must succeed before the one registry push.
printf 'push_about_to_start\n' > "$results/push-state.txt"
docker_compat push "$image"
printf 'push_returned_success\n' > "$results/push-state.txt"
docker_compat image inspect --format '{{json .RepoDigests}}' "$image" > "$results/repo-digests.json"
printf '%s\n' "$image" > "$results/image-reference.txt"
printf '%s\n' "$CI_COMMIT_SHA" > "$results/source-commit.txt"
printf '%s\n' "$CI_PIPELINE_ID" > "$results/pipeline-id.txt"
printf '%s\n' "$CI_JOB_ID" > "$results/job-id.txt"
sha256sum "$CI_PROJECT_DIR/docker/controllers-scan-security.dockerfile" \
    "$CI_PROJECT_DIR/ci/refresh_scan_system_packages.sh" > "$results/build-input-sha256.txt"
python3 -I - "$results" "$CI_COMMIT_SHA" "$CI_REGISTRY_IMAGE/ci" <<'PYRECEIPT'
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

results = Path(sys.argv[1])
source_commit, repository = sys.argv[2:]
image = json.loads((results / "image-inspect.json").read_text())
assert len(image) == 1
image = image[0]
assert image["Os"] == "linux" and image["Architecture"] == "amd64"
assert image["Id"] == (results / "image-id.txt").read_text().strip()
assert image["Config"]["Labels"]["org.opencontainers.image.revision"] == source_commit
digests = json.loads((results / "repo-digests.json").read_text())
matches = [value for value in digests if re.fullmatch(re.escape(repository) + r"@sha256:[0-9a-f]{64}", value)]
assert len(matches) == 1, "Registry digest not uniquely observed; reconcile the pushed tag"
with (results / "minimum-versions.tsv").open() as handle:
    packages = list(csv.DictReader(handle, delimiter="\t"))
minimums = {
    "libc6": "2.39-0ubuntu8.9", "libc6-dev": "2.39-0ubuntu8.9",
    "python3.12": "3.12.3-1ubuntu0.15", "libpython3.12t64": "3.12.3-1ubuntu0.15",
    "libpython3.12-dev": "3.12.3-1ubuntu0.15",
    "libexpat1": "2.6.1-2ubuntu0.6", "libexpat1-dev": "2.6.1-2ubuntu0.6",
}
assert len(packages) == len(minimums) and {row["package"] for row in packages} == set(minimums)
for row in packages:
    assert row["architecture"] == "amd64" and row["minimum_version"] == minimums[row["package"]]
    subprocess.run(["dpkg", "--compare-versions", row["installed_version"], "ge", row["minimum_version"]], check=True)
receipt = {
    "contract": "controllers-scan-image-build/1", "source_commit": source_commit,
    "image_reference": (results / "image-reference.txt").read_text().strip(),
    "registry_reference": matches[0], "image_config_digest": image["Id"],
    "pipeline_id": (results / "pipeline-id.txt").read_text().strip(),
    "job_id": (results / "job-id.txt").read_text().strip(), "packages": packages,
    "package_inventory_sha256": hashlib.sha256((results / "host-package-versions.tsv").read_bytes()).hexdigest(),
    "library_build_or_scan_started": False, "official_completion": "not_inferred",
}
(results / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PYRECEIPT
