/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file energy_tank.hpp
 * @brief Batched tensor API for EnergyTank.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

PrepareResult prepare_energy_tank(const TensorView& power_signal,
                                  const TensorView& additional_power,
                                  const TensorView& dt,
                                  const ParameterSetView& parameters,
                                  StateView& state,
                                  const MutableTensorView& scale,
                                  const MutableTensorView& filtered_power);
Result apply_energy_tank_parameters(const ParameterSetView& parameters,
                                    std::size_t num_envs,
                                    MutableStatusView parameter_status);
Result reset_energy_tank(const ParameterSetView& parameters,
                         StatusView parameter_status,
                         StateView& state,
                         const TensorView& reset_mask,
                         MutableStatusView status);
Result step_energy_tank(const TensorView& power_signal,
                        const TensorView& additional_power,
                        const TensorView& dt,
                        const ParameterSetView& parameters,
                        StatusView parameter_status,
                        StateView& state,
                        const MutableTensorView& scale,
                        const MutableTensorView& filtered_power,
                        MutableStatusView status);

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
