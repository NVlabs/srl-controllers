/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia::srl::controllers::detail::tensor {

Result apply_static_impedance_parameters(ComponentMode mode,
                                         const ParameterSetView& parameters,
                                         std::size_t num_envs,
                                         std::size_t signal_dim,
                                         MutableStatusView parameter_status);
Result step_static_impedance(ComponentMode mode,
                             const TensorView& position_error,
                             const TensorView& velocity_error,
                             const ParameterSetView& parameters,
                             StatusView parameter_status,
                             const MutableTensorView& command,
                             MutableStatusView status);
Result reset_high_level_stateless(std::size_t num_envs,
                                  const TensorView& reset_mask,
                                  MutableStatusView status);

Result apply_unified_force_admittance_parameters(const ParameterSetView& parameters,
                                                 std::size_t num_envs,
                                                 MutableStatusView parameter_status);
Result reset_unified_force_admittance(std::size_t num_envs,
                                      StateView& state,
                                      const TensorView& reset_mask,
                                      MutableStatusView status);
Result step_unified_force_admittance(const TensorView& desired_wrench,
                                     const TensorView& external_wrench,
                                     const TensorView& feedforward_wrench,
                                     const TensorView& dt,
                                     const ParameterSetView& parameters,
                                     StatusView parameter_status,
                                     StateView& state,
                                     const MutableTensorView& commanded_acceleration,
                                     MutableStatusView status);

}  // namespace nvidia::srl::controllers::detail::tensor
