# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Keep the reviewed toolchain; refresh only installed Ubuntu package families.
FROM gitlab-master.nvidia.com:5005/srl/cpp/controllers/ci@sha256:1db80666bcce101e61dd5da433d09e470bca8148a39ccadb6e89dd63c8e2099b AS ci-security
COPY ci/refresh_scan_system_packages.sh /tmp/controllers-refresh-system-packages.sh
COPY ci/refresh_certificate_trust.sh /tmp/controllers-refresh-certificate-trust.sh
COPY ci/check_certificate_trust.py /usr/local/libexec/controllers/check_certificate_trust.py
RUN sudo -n chmod 0644 /usr/local/libexec/controllers/check_certificate_trust.py \
    && sudo -n /bin/sh /tmp/controllers-refresh-system-packages.sh \
    && sudo -n /bin/sh /tmp/controllers-refresh-certificate-trust.sh /usr/local/libexec/controllers/check_certificate_trust.py \
    && sudo -n rm /tmp/controllers-refresh-system-packages.sh \
        /tmp/controllers-refresh-certificate-trust.sh
