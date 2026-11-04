/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file execution_context.cpp
 * @brief Thread-local tensor execution context.
 */

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

thread_local TensorExecutionContext g_context{};

}  // namespace

TensorExecutionContext default_tensor_execution_context() noexcept {
  return TensorExecutionContext{};
}

TensorExecutionContext tensor_execution_context() noexcept { return g_context; }

void set_tensor_execution_context(const TensorExecutionContext context) noexcept {
  g_context = context;
}

ScopedTensorExecutionContext::ScopedTensorExecutionContext(
    const TensorExecutionContext context) noexcept
    : previous_(tensor_execution_context()) {
  set_tensor_execution_context(context);
}

ScopedTensorExecutionContext::~ScopedTensorExecutionContext() {
  set_tensor_execution_context(previous_);
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
