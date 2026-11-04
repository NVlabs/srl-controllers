/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file torque_filter.hpp
 * @brief Batched magnitude and slew-rate filter for joint-torque commands.
 */

#pragma once

#include <cstddef>

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

constexpr std::size_t kMaxTorqueFilterSignalDimension = 256;

PrepareResult prepare_torque_filter(const TensorView& command,
                                    const TensorView& dt,
                                    const ParameterSetView& parameters,
                                    StateView& state,
                                    const MutableTensorView& filtered_command);
Result apply_torque_filter_parameters(const ParameterSetView& parameters,
                                      std::size_t num_envs,
                                      std::size_t signal_dim,
                                      MutableStatusView parameter_status) noexcept;
Result reset_torque_filter(std::size_t num_envs,
                           StateView& state,
                           const TensorView& reset_mask,
                           MutableStatusView status) noexcept;
Result step_torque_filter(const TensorView& command,
                          const TensorView& dt,
                          const ParameterSetView& parameters,
                          StatusView parameter_status,
                          StateView& state,
                          const MutableTensorView& filtered_command,
                          MutableStatusView status) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
