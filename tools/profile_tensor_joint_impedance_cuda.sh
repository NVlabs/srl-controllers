#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

set -eu

qualification_binary=${1:?usage: profile_tensor_joint_impedance_cuda.sh QUALIFICATION_BINARY OUTPUT_DIR}
output_dir=${2:?usage: profile_tensor_joint_impedance_cuda.sh QUALIFICATION_BINARY OUTPUT_DIR}
mkdir -p "$output_dir"

command -v ncu >/dev/null
command -v nsys >/dev/null

# --set full records registers, occupancy, shared-memory use, and available
# warp-stall metrics. Metric availability varies by SM and CUDA toolkit.
ncu --target-processes all --set full --export "$output_dir/qualification" \
  "$qualification_binary" --qualify
ncu --import "$output_dir/qualification.ncu-rep" --csv > "$output_dir/metrics.csv"
nsys profile --force-overwrite true --trace cuda,nvtx \
  --output "$output_dir/qualification" "$qualification_binary" --all-dimensions
