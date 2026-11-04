/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 * @brief Public status aggregation and final-output policy for custom controllers.
 */
#pragma once
#include "nvidia/srl/controllers/equations.hpp"
#include "nvidia/srl/controllers/types.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
/** Faults always establish safe mode; saturation, reset and held-output flags are informational. */
NVIDIA_SRL_CONTROLLERS_HOST_DEVICE inline constexpr std::uint64_t normalize_status(
    std::uint64_t flags) noexcept {
  constexpr std::uint64_t faults = static_cast<std::uint64_t>(StatusFlag::kUnsupportedMode) |
                                   static_cast<std::uint64_t>(StatusFlag::kInvalidParameter) |
                                   static_cast<std::uint64_t>(StatusFlag::kInvalidInput) |
                                   static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput) |
                                   static_cast<std::uint64_t>(StatusFlag::kInvalidDt) |
                                   static_cast<std::uint64_t>(StatusFlag::kNoValidOutput) |
                                   static_cast<std::uint64_t>(StatusFlag::kHoldExpired);
  return flags | ((flags & faults) ? static_cast<std::uint64_t>(StatusFlag::kSafeMode) : 0U);
}
/**
 * OR per-row flags, normalizing faults. Input descriptors live on the host even for CUDA.
 * Output may alias the first input exactly; it must not overlap any other input.
 * CUDA enqueues on context.cuda_stream; all underlying arrays must remain alive until completion.
 */
NVIDIA_SRL_CONTROLLERS_API Result merge_status(const StatusView* inputs,
                                               std::size_t count,
                                               MutableStatusView output,
                                               ExecutionContext context = {}) noexcept;
/** Reset safety history for all rows or a Boolean [batch] mask. */
NVIDIA_SRL_CONTROLLERS_API Result reset_safety_state(std::size_t batch_size,
                                                     SafetyStateView state,
                                                     TensorView reset_mask,
                                                     MutableStatusView status,
                                                     ExecutionContext context = {}) noexcept;
/**
 * Apply policy once per tick to this history. Use distinct history for a separately protected
 * final command; lifecycle blocks already apply their own policy. Structural errors abort the
 * tick and do not authorize reuse of previous outputs.
 */
NVIDIA_SRL_CONTROLLERS_API Result apply_safe_output_policy(MutableTensorView output,
                                                           MutableStatusView status,
                                                           SafetyStateView state,
                                                           StepOptions options = {},
                                                           ExecutionContext context = {}) noexcept;
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
