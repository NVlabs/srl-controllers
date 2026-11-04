/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file torque_filter_cuda.hpp
 * @brief Private CUDA backend declarations for the tensor torque filter.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_torque_filter_parameters_cuda(const ParameterSetView& parameters,
                                           std::size_t num_envs,
                                           std::size_t signal_dim,
                                           MutableStatusView parameter_status) noexcept;
Result reset_torque_filter_cuda(std::size_t num_envs,
                                StateView& state,
                                const TensorView& reset_mask,
                                MutableStatusView status) noexcept;
Result step_torque_filter_cuda(const TensorView& command,
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
