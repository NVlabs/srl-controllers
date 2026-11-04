/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file operational_space_mass_cuda.hpp
 * @brief Private CUDA backend declarations for operational-space mass calculation.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_operational_space_mass_parameters_cuda(const ParameterSetView& parameters,
                                                    std::size_t num_envs,
                                                    MutableStatusView parameter_status) noexcept;
Result reset_operational_space_mass_cuda(std::size_t num_envs,
                                         const TensorView& reset_mask,
                                         MutableStatusView status) noexcept;
Result step_operational_space_mass_cuda(const TensorView& joint_mass_matrix,
                                        const TensorView& jacobian,
                                        const ParameterSetView& parameters,
                                        StatusView parameter_status,
                                        ScratchView& scratch,
                                        const MutableTensorView& cartesian_mass_matrix,
                                        MutableStatusView status) noexcept;

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
