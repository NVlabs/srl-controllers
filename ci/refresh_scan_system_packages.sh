#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Run only in an explicitly authorized Linux CI image build.
set -eu
export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_CODENAME" = noble
test "$(dpkg --print-architecture)" = amd64

# These are Ubuntu package revisions, including backports, not upstream versions.
# Vendor references: CVE-2026-19542, USN-8509-1 and USN-8813-1.
required='libc6 libc6-dev python3.12 libpython3.12t64 libpython3.12-dev libexpat1 libexpat1-dev'
for package in $required; do
    test "$(dpkg-query -W -f='${Status}' "$package")" = 'install ok installed'
done

# Upgrade installed members together to retain each distribution package family's
# dependency consistency. Do not add a new distribution or downgrade packages.
packages=''
for package in libc6 libc6-dev libc-bin libc-dev-bin libc-devtools \
    python3.12 python3.12-minimal python3.12-dev libpython3.12t64 \
    libpython3.12-minimal libpython3.12-stdlib libpython3.12-dev \
    libexpat1 libexpat1-dev expat; do
    if [ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || :)" = 'install ok installed' ]; then
        packages="$packages $package"
    fi
done
apt-get update
# Intentional splitting of the literal package-name list above.
# shellcheck disable=SC2086
apt-get install --yes --no-install-recommends --only-upgrade --no-remove $packages

evidence=/usr/local/share/controllers-scan-security
mkdir -p "$evidence"
printf 'package\tinstalled_version\tminimum_version\tarchitecture\n' > "$evidence/minimum-versions.tsv"
for package in $required; do
    case "$package" in
        libc6|libc6-dev) minimum=2.39-0ubuntu8.9; family=2.39-0ubuntu8. ;;
        python3.12|libpython3.12t64|libpython3.12-dev) minimum=3.12.3-1ubuntu0.15; family=3.12.3-1ubuntu0. ;;
        libexpat1|libexpat1-dev) minimum=2.6.1-2ubuntu0.6; family=2.6.1-2ubuntu0. ;;
        *) exit 1 ;;
    esac
    version=$(dpkg-query -W -f='${Version}' "$package")
    architecture=$(dpkg-query -W -f='${Architecture}' "$package")
    test "$architecture" = amd64
    case "$version" in "$family"*) ;; *) echo "Unexpected Ubuntu package family for $package" >&2; exit 1 ;; esac
    dpkg --compare-versions "$version" ge "$minimum"
    printf '%s\t%s\t%s\t%s\n' "$package" "$version" "$minimum" "$architecture" >> "$evidence/minimum-versions.tsv"
done
dpkg --audit > "$evidence/dpkg-audit.txt"
test ! -s "$evidence/dpkg-audit.txt"
apt-get check
python3.12 --version > "$evidence/python-version.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' > "$evidence/host-package-versions.tsv"
cp /etc/os-release "$evidence/os-release.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$evidence/observed-at.txt"
rm -rf /var/lib/apt/lists/*
