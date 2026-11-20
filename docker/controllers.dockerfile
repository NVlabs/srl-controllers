# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

################################################################################
# Controllers project dockerfile.
#
# This dockerfile defines several docker images for the project.
#
# Build images from the project's root directory with Docker or BuildKit,
# selecting the base, ci, release, or dev target as needed.
#
################################################################################

# ----------------------------------------------------------------------------
# Base Image
# ----------------------------------------------------------------------------
# The base image is built from a basic OS image and has this project's
# necessary system dependencies installed into it.
FROM gitlab-master.nvidia.com:5005/srl/dockers/ubuntu24.04_py3.12:latest@sha256:c5a45b23824f22916b424989bf33b4ddefbd374ba47e3478e11fc82cfb055c4a AS base

# Set HOME environment variable
ENV HOME=/home/srl

# Copy in script to install apt packages
COPY install_apt_pkgs.sh $HOME/install_apt_pkgs.sh

# Copy list of apt package dependencies
COPY apt_pkg_list.txt $HOME/apt_pkg_list.txt

# Install apt package dependencies
RUN sudo apt-get update && \
    sudo $HOME/install_apt_pkgs.sh -r -s $HOME/apt_pkg_list.txt \
    && sudo rm -rf /var/lib/apt/lists/*

# Copy list of pip package dependencies files
COPY pyproject.toml $HOME/pyproject.toml
COPY extract_py_dependencies.py $HOME/extract_py_dependencies.py
RUN sudo chmod 0644 $HOME/pyproject.toml && \
    sudo chmod 0755 $HOME/extract_py_dependencies.py

# Generate requirements from pyproject.toml with Python 3.12's tomllib
RUN python3 $HOME/extract_py_dependencies.py

# Install runtime Python dependencies
# Update system and packaging trust before the first dependency download.
COPY ci/refresh_certificate_trust.sh /tmp/controllers-refresh-certificate-trust.sh
COPY ci/check_certificate_trust.py /usr/local/libexec/controllers/check_certificate_trust.py
RUN sudo -n chmod 0644 /usr/local/libexec/controllers/check_certificate_trust.py \
    && sudo -n /bin/sh /tmp/controllers-refresh-certificate-trust.sh /usr/local/libexec/controllers/check_certificate_trust.py
RUN \
    sudo pip3 install -U -r $HOME/pip_pkg_list.txt \
    --index-url https://pypi.org/simple

# ----------------------------------------------------------------------------
# CI Image
# ----------------------------------------------------------------------------
# The CI image is built from the project's base image and contains all the
# dependencies necessary to run the CI.
FROM base AS ci

# Copy list of apt package dependencies for CI
COPY apt_pkg_list_ci.txt $HOME/apt_pkg_list_ci.txt

# Install apt package dependencies for CI
RUN sudo apt-get update && \
    sudo $HOME/install_apt_pkgs.sh -r -s $HOME/apt_pkg_list_ci.txt \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Python build and CI dependencies
RUN \
    sudo pip3 install -U \
    -r $HOME/pip_pkg_list_build.txt \
    -r $HOME/pip_pkg_list_ci.txt \
    --index-url https://pypi.org/simple

# Keep full Ubuntu security revisions verified in future full CI image rebuilds.
COPY ci/refresh_scan_system_packages.sh /tmp/controllers-refresh-system-packages.sh
RUN sudo -n /bin/sh /tmp/controllers-refresh-system-packages.sh \
    && sudo -n /bin/sh /tmp/controllers-refresh-certificate-trust.sh /usr/local/libexec/controllers/check_certificate_trust.py \
    && sudo -n rm /tmp/controllers-refresh-system-packages.sh

# ----------------------------------------------------------------------------
# Release Image
# ----------------------------------------------------------------------------
# The release image installs this project's previously built Debian package from
# the trusted build context. Pass its relative path as DEB_PACKAGE and its exact
# package version as VERSION. Image publication requires separate authorization.
FROM base AS release

# Use the local build artifact rather than an unsigned network package source.
ARG DEB_PACKAGE
ARG VERSION
COPY ${DEB_PACKAGE} /tmp/nvidia-srl-controllers.deb
RUN test "$(dpkg-deb --field /tmp/nvidia-srl-controllers.deb Package)" = "nvidia-srl-controllers" \
    && test "$(dpkg-deb --field /tmp/nvidia-srl-controllers.deb Version)" = "${VERSION:?Pass the Debian package version}" \
    && sudo apt-get update \
    && sudo apt-get install -y /tmp/nvidia-srl-controllers.deb \
    && sudo rm -f /tmp/nvidia-srl-controllers.deb \
    && sudo rm -rf /var/lib/apt/lists/*


# ----------------------------------------------------------------------------
# Dev Image
# ----------------------------------------------------------------------------
# The dev image is built from the project's base image and is designed to be
# used for development. It has the project installed into it and should be run
# with the local repo linked in as volume.
FROM ci AS dev

# Get gitlab public keys
RUN mkdir /home/srl/.ssh
RUN touch /home/srl/.ssh/known_hosts
RUN ssh-keyscan gitlab-master.nvidia.com >> /home/srl/.ssh/known_hosts

# Set dev image arguments
ARG REPO_HOST_DIR=.

# Copy project to workspace
COPY ${REPO_HOST_DIR} /home/srl/controllers
