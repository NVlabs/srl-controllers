# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Replace the selected release tag's checksum placeholder with the immutable
# digest owned by this project. Flexiv replaced the v1.9.4 Linux x86_64 asset
# after creating the tag, so that version intentionally uses the replacement
# asset's verified digest.
if(NOT DEFINED RDK_HASH_FILE OR NOT DEFINED RDK_HASH OR
   "${RDK_HASH_FILE}" STREQUAL "" OR "${RDK_HASH}" STREQUAL "")
  message(FATAL_ERROR "RDK_HASH_FILE and RDK_HASH are required")
endif()

string(LENGTH "${RDK_HASH}" RDK_HASH_LENGTH)
if(NOT RDK_HASH_LENGTH EQUAL 64 OR NOT "${RDK_HASH}" MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "RDK_HASH must be a lowercase SHA-256 digest")
endif()

file(WRITE "${RDK_HASH_FILE}" "${RDK_HASH}")
