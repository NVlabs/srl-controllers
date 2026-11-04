/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file tensor/types.hpp
 * @brief Internal compatibility types used by the tensor kernels.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nvidia/srl/controllers/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

using ::nvidia::srl::controllers::ComponentMode;
using ::nvidia::srl::controllers::Device;
using ::nvidia::srl::controllers::DType;
using ::nvidia::srl::controllers::dtype_for;
using ::nvidia::srl::controllers::ExecutionContext;
using ::nvidia::srl::controllers::has_status_flag;
using ::nvidia::srl::controllers::kMaxTensorRank;
using ::nvidia::srl::controllers::Layout;
using ::nvidia::srl::controllers::make_mutable_tensor_view;
using ::nvidia::srl::controllers::make_tensor_view;
using ::nvidia::srl::controllers::MutableStatusView;
using ::nvidia::srl::controllers::MutableTensorView;
using ::nvidia::srl::controllers::Result;
using ::nvidia::srl::controllers::ResultCode;
using ::nvidia::srl::controllers::SafeOutputPolicy;
using ::nvidia::srl::controllers::SafetyStateView;
using ::nvidia::srl::controllers::set_status_flag;
using ::nvidia::srl::controllers::status_bit;
using ::nvidia::srl::controllers::StatusFlag;
using ::nvidia::srl::controllers::StatusView;
using ::nvidia::srl::controllers::StepOptions;
using ::nvidia::srl::controllers::TensorView;

using TensorExecutionContext = ExecutionContext;

TensorExecutionContext default_tensor_execution_context() noexcept;
TensorExecutionContext tensor_execution_context() noexcept;
void set_tensor_execution_context(TensorExecutionContext context) noexcept;

class ScopedTensorExecutionContext {
 public:
  explicit ScopedTensorExecutionContext(TensorExecutionContext context) noexcept;
  ScopedTensorExecutionContext(const ScopedTensorExecutionContext&) = delete;
  ScopedTensorExecutionContext& operator=(const ScopedTensorExecutionContext&) = delete;
  ~ScopedTensorExecutionContext();

 private:
  TensorExecutionContext previous_{};
};

struct ParameterView {
  const char* name{nullptr};
  TensorView values{};
  bool shared{false};
};

struct ParameterSetView {
  const ParameterView* parameters{nullptr};
  std::size_t parameter_count{0};
};

struct StateEntryView {
  const char* name{nullptr};
  MutableTensorView values{};
};

struct StateView {
  StateEntryView* entries{nullptr};
  std::size_t entry_count{0};
};

struct ScratchEntryView {
  const char* name{nullptr};
  MutableTensorView values{};
};

struct ScratchView {
  ScratchEntryView* entries{nullptr};
  std::size_t entry_count{0};
};

struct TensorDescriptor {
  std::string name;
  DType dtype{DType::kUnknown};
  Device device{Device::kCpu};
  Layout layout{Layout::kRowMajor};
  int rank{0};
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::array<std::int64_t, kMaxTensorRank> strides{};
  std::string units;
  std::string frame;
  std::string semantics;
};

struct PrepareResult {
  Result result{};
  std::size_t num_envs{0};
  std::size_t signal_dim{0};
  std::vector<TensorDescriptor> state;
  std::vector<TensorDescriptor> scratch;
  std::vector<TensorDescriptor> outputs;

  bool ok() const noexcept { return result.ok(); }
};

inline bool names_equal(const char* left, const char* right) noexcept {
  if (left == nullptr || right == nullptr) {
    return false;
  }
  while (*left != '\0' && *right != '\0' && *left == *right) {
    ++left;
    ++right;
  }
  return *left == '\0' && *right == '\0';
}

inline const ParameterView* find_parameter(const ParameterSetView& parameters,
                                           const char* name) noexcept {
  for (std::size_t index = 0; index < parameters.parameter_count; ++index) {
    if (names_equal(parameters.parameters[index].name, name)) {
      return &parameters.parameters[index];
    }
  }
  return nullptr;
}

inline StateEntryView* find_state(StateView& state, const char* name) noexcept {
  for (std::size_t index = 0; index < state.entry_count; ++index) {
    if (names_equal(state.entries[index].name, name)) {
      return &state.entries[index];
    }
  }
  return nullptr;
}

inline ScratchEntryView* find_scratch(ScratchView& scratch, const char* name) noexcept {
  for (std::size_t index = 0; index < scratch.entry_count; ++index) {
    if (names_equal(scratch.entries[index].name, name)) {
      return &scratch.entries[index];
    }
  }
  return nullptr;
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
