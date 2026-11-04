/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */
#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia::srl::controllers::detail::tensor {
Result apply_static_impedance_parameters_cuda(
    ComponentMode, const ParameterSetView&, std::size_t, std::size_t, MutableStatusView);
Result step_static_impedance_cuda(ComponentMode,
                                  const TensorView&,
                                  const TensorView&,
                                  const ParameterSetView&,
                                  StatusView,
                                  const MutableTensorView&,
                                  MutableStatusView);
Result reset_high_level_stateless_cuda(std::size_t, const TensorView&, MutableStatusView);
Result apply_unified_force_admittance_parameters_cuda(const ParameterSetView&,
                                                      std::size_t,
                                                      MutableStatusView);
Result reset_unified_force_admittance_cuda(std::size_t,
                                           StateView&,
                                           const TensorView&,
                                           MutableStatusView);
Result step_unified_force_admittance_cuda(const TensorView&,
                                          const TensorView&,
                                          const TensorView&,
                                          const TensorView&,
                                          const ParameterSetView&,
                                          StatusView,
                                          StateView&,
                                          const MutableTensorView&,
                                          MutableStatusView);
}  // namespace nvidia::srl::controllers::detail::tensor
