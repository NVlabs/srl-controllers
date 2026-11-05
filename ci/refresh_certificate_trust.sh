#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Run only inside an explicitly authorized disposable CI image build.
set -eu
export DEBIAN_FRONTEND=noninteractive
checker=${1:?Pass the reviewed offline CA-store checker}
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_CODENAME" = noble
test "$(dpkg-query -W -f='${Status}' ca-certificates)" = 'install ok installed'
packages=ca-certificates
for package in libnss3 libnss3-dev; do
    if [ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || :)" = 'install ok installed' ]; then
        packages="$packages $package"
    fi
done
apt-get update
# Intentional splitting of this literal package list.
# shellcheck disable=SC2086
apt-get install --yes --no-install-recommends --only-upgrade --no-remove $packages
version=$(dpkg-query -W -f='${Version}' ca-certificates)
dpkg --compare-versions "$version" ge 20260816ubuntu0.24.04.1
update-ca-certificates --fresh
# Hash-pinned bootstrap is the Python incident response's secure upgrade route.
python3 -I -m pip install --break-system-packages --no-input --no-deps \
    'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
python3 -I -m pip install --break-system-packages --no-input --no-deps \
    'certifi @ https://files.pythonhosted.org/packages/c4/51/29046e48f693c54e5b7412216d447a1d8362c9653148d98c02cdf65fa733/certifi-2026.10.19-py3-none-any.whl#sha256=42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b'
evidence=/usr/local/share/controllers-scan-security
mkdir -p "$evidence"
python3 -I "$checker" --pip --certifi --java-if-present > "$evidence/certificate-trust.json"
printf '%s\n' "$version" > "$evidence/ca-certificates-version.txt"
# Refresh the inventory after the CA/NSS transaction, not only the earlier
# native-package refresh. The image receipt must describe the final image.
dpkg --audit > "$evidence/dpkg-audit.txt"
test ! -s "$evidence/dpkg-audit.txt"
apt-get check
dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' > "$evidence/host-package-versions.tsv"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$evidence/observed-at.txt"
rm -rf /var/lib/apt/lists/*

