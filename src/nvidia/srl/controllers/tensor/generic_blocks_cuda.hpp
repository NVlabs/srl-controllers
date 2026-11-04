/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file generic_blocks_cuda.hpp
 * @brief Private CUDA tensor backend declarations for generic controller blocks.
 */

#pragma once

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

Result apply_generic_proportional_block_parameters_cuda(const ParameterSetView& parameters,
                                                        std::size_t num_envs,
                                                        std::size_t signal_dim,
                                                        MutableStatusView parameter_status);
Result reset_generic_proportional_block_cuda(std::size_t num_envs,
                                             const TensorView& reset_mask,
                                             MutableStatusView status);
Result step_generic_proportional_block_cuda(const TensorView& input_signal,
                                            const ParameterSetView& parameters,
                                            StatusView parameter_status,
                                            const MutableTensorView& output_signal,
                                            MutableStatusView status);

Result apply_generic_integrator_block_parameters_cuda(const ParameterSetView& parameters,
                                                      std::size_t num_envs,
                                                      std::size_t signal_dim,
                                                      MutableStatusView parameter_status);
Result reset_generic_integrator_block_cuda(std::size_t num_envs,
                                           StateView& state,
                                           const TensorView& reset_mask,
                                           MutableStatusView status);
Result step_generic_integrator_block_cuda(const TensorView& input_signal,
                                          const TensorView& dt,
                                          const ParameterSetView& parameters,
                                          StatusView parameter_status,
                                          StateView& state,
                                          const MutableTensorView& output_signal,
                                          MutableStatusView status);

Result apply_generic_differentiator_block_parameters_cuda(std::size_t num_envs,
                                                          MutableStatusView parameter_status);
Result reset_generic_differentiator_block_cuda(std::size_t num_envs,
                                               StateView& state,
                                               const TensorView& reset_mask,
                                               MutableStatusView status);
Result step_generic_differentiator_block_cuda(const TensorView& input_signal,
                                              const TensorView& dt,
                                              StatusView parameter_status,
                                              StateView& state,
                                              const MutableTensorView& output_signal,
                                              MutableStatusView status);

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
