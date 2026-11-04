/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file operational_space_mass.hpp
 * @brief Batched operational-space mass matrix calculation for CPU and CUDA tensors.
 */

#pragma once

#include <cstddef>

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

constexpr std::size_t kDefaultOperationalSpaceDimension = 6;
constexpr std::size_t kMaxOperationalSpaceDimension = 12;
constexpr std::size_t kMaxOperationalSpaceJointDimension = 64;

/**
 * Compute Lambda = (J M^-1 J^T + regularization I)^-1.
 *
 * joint_mass_matrix has shape [num_envs, joint_dim, joint_dim], jacobian has shape
 * [num_envs, task_dim, joint_dim], and cartesian_mass_matrix has shape
 * [num_envs, task_dim, task_dim], where task_dim is in [1, 12]. The required
 * scalar parameter `regularization` is shared or batched and has inverse-mass units. Optional
 * `minimum_pivot` controls SPD/singularity rejection and defaults by dtype.
 */
PrepareResult prepare_operational_space_mass(const TensorView& joint_mass_matrix,
                                             const TensorView& jacobian,
                                             const ParameterSetView& parameters,
                                             ScratchView& scratch,
                                             const MutableTensorView& cartesian_mass_matrix);
Result apply_operational_space_mass_parameters(const ParameterSetView& parameters,
                                               std::size_t num_envs,
                                               MutableStatusView parameter_status) noexcept;
Result reset_operational_space_mass(std::size_t num_envs,
                                    const TensorView& reset_mask,
                                    MutableStatusView status) noexcept;
Result step_operational_space_mass(const TensorView& joint_mass_matrix,
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
