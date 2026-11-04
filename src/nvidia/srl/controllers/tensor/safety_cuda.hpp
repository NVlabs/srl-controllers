/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file safety_cuda.hpp
 * @brief Private CUDA backend declarations for tensor safe-output handling.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result reset_safety_state_cuda(std::size_t num_envs,
                               SafetyStateView safety_state,
                               const TensorView& reset_mask,
                               MutableStatusView status) noexcept;
Result apply_safe_output_policy_cuda(const MutableTensorView& output,
                                     MutableStatusView status,
                                     SafetyStateView safety_state,
                                     StepOptions options) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
