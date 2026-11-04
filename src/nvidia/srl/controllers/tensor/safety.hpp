/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file safety.hpp
 * @brief Bounded, caller-owned safe-output handling for tensor controllers.
 */

#pragma once

#include <cstddef>

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

/**
 * @brief Clear selected environments in a caller-owned safety state.
 *
 * reset_mask may be empty to reset all environments. This function performs no allocation,
 * synchronization, logging, or exception throwing. CUDA work is enqueued on the active caller
 * stream.
 */
Result reset_safety_state(std::size_t num_envs,
                          SafetyStateView safety_state,
                          const TensorView& reset_mask,
                          MutableStatusView status) noexcept;

/**
 * @brief Apply the configured safe-output policy to a completed tensor step.
 *
 * Environments without kSafeMode update last_valid_output and clear their hold counter. Faulted
 * environments are zeroed by default. kHoldLastValid restores a previously valid output for at
 * most max_hold_steps consecutive faulted steps; missing or expired history always produces zero.
 * Non-finite successful output is converted to kNonFiniteInput | kSafeMode before policy handling.
 *
 * This function performs no allocation, synchronization, logging, or exception throwing. CUDA
 * work is enqueued on the active caller stream.
 */
Result apply_safe_output_policy(const MutableTensorView& output,
                                MutableStatusView status,
                                SafetyStateView safety_state,
                                StepOptions options = {}) noexcept;

/**
 * @brief Run any existing tensor step and apply bounded safe-output handling.
 *
 * The callable must return Result and must capture all component-specific arguments. Structural
 * step failures are returned directly because the output/status views cannot be assumed usable.
 */
template <typename StepCallable>
Result step_with_safety(StepCallable&& step,
                        const MutableTensorView& output,
                        MutableStatusView status,
                        SafetyStateView safety_state,
                        StepOptions options = {}) noexcept {
  const Result result = step();
  if (!result.ok()) {
    return result;
  }
  return ::nvidia::srl::controllers::detail::tensor::apply_safe_output_policy(
      output, status, safety_state, options);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
