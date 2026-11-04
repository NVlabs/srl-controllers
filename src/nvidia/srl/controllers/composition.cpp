// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include "nvidia/srl/controllers/composition.hpp"
#include <initializer_list>
#include <limits>
#include "tensor/common.hpp"
#include "tensor/safety.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
namespace {
// Validate public utility views before private routines can access their storage.
bool span(const TensorView& view, std::uintptr_t* end) noexcept {
  if (!view.data || view.rank < 1 || view.rank > kMaxTensorRank ||
      view.layout != Layout::kRowMajor ||
      (view.device != Device::kCpu && view.device != Device::kCuda))
    return false;
  detail::tensor::internal::ByteSpan checked;
  if (!detail::tensor::internal::byte_span(view, &checked).ok()) return false;
  const auto width = view.dtype == DType::kFloat64 || view.dtype == DType::kUInt64 ? 8U
                     : view.dtype == DType::kFloat32                               ? 4U
                     : view.dtype == DType::kBool                                  ? 1U
                                                                                   : 0U;
  if (!width) return false;
  std::size_t elements = 1;
  for (int axis = view.rank - 1; axis >= 0; --axis) {
    if (view.shape[axis] <= 0 || view.strides[axis] <= 0 ||
        static_cast<std::size_t>(view.strides[axis]) != elements ||
        elements > std::numeric_limits<std::size_t>::max() / view.shape[axis])
      return false;
    elements *= static_cast<std::size_t>(view.shape[axis]);
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(view.data);
  if (elements > (std::numeric_limits<std::uintptr_t>::max() - begin) / width) return false;
  *end = begin + elements * width;
  return true;
}
Result disjoint(std::initializer_list<TensorView> views) noexcept {
  for (auto first = views.begin(); first != views.end(); ++first) {
    std::uintptr_t end = 0;
    if (!span(*first, &end))
      return {ResultCode::kInvalidView, "safety requires valid contiguous arrays"};
    for (auto second = first + 1; second != views.end(); ++second) {
      std::uintptr_t other_end = 0;
      if (!span(*second, &other_end) || first->device != second->device)
        return {ResultCode::kInvalidView, "safety arrays must match device and layout"};
      if (reinterpret_cast<std::uintptr_t>(first->data) < other_end &&
          reinterpret_cast<std::uintptr_t>(second->data) < end)
        return {ResultCode::kInvalidView, "safety arrays must not overlap"};
    }
  }
  return {};
}
}  // namespace
namespace detail {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
Result merge_status_cuda(const StatusView*,
                         std::size_t,
                         MutableStatusView,
                         ExecutionContext) noexcept;
#endif
}  // namespace detail
Result merge_status(const StatusView* inputs,
                    std::size_t count,
                    MutableStatusView output,
                    ExecutionContext context) noexcept {
  if (!inputs || !count || !output.flags || !output.size ||
      output.size > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) ||
      (output.device != Device::kCpu && output.device != Device::kCuda))
    return {ResultCode::kInvalidView, "status merge requires nonempty matching arrays"};
  const auto begin = reinterpret_cast<std::uintptr_t>(output.flags);
  if (begin % alignof(std::uint64_t) != 0)
    return {ResultCode::kInvalidView, "status address is not aligned"};
  const auto bytes = output.size * sizeof(std::uint64_t);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    return {ResultCode::kInvalidView, "status address overflow"};
  for (std::size_t i = 0; i < count; ++i) {
    const auto other = reinterpret_cast<std::uintptr_t>(inputs[i].flags);
    if (!other || other % alignof(std::uint64_t) != 0 || inputs[i].size != output.size ||
        inputs[i].device != output.device ||
        other > std::numeric_limits<std::uintptr_t>::max() - bytes)
      return {ResultCode::kInvalidView, "status arrays must match device and batch"};
    if (begin < other + bytes && other < begin + bytes && !(i == 0 && begin == other))
      return {ResultCode::kInvalidView, "only the first status input may alias the output exactly"};
  }
  if (output.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return detail::merge_status_cuda(inputs, count, output, context);
#else
    return {ResultCode::kBackendUnavailable, "CUDA requires ENABLE_CUDA=ON"};
#endif
  }
  (void)context;
  for (std::size_t row = 0; row < output.size; ++row) {
    std::uint64_t flags = 0;
    for (std::size_t i = 0; i < count; ++i) flags |= inputs[i].flags[row];
    output.flags[row] = normalize_status(flags);
  }
  return {};
}
Result reset_safety_state(std::size_t batch_size,
                          SafetyStateView state,
                          TensorView mask,
                          MutableStatusView status,
                          ExecutionContext context) noexcept {
  if (status.size != batch_size)
    return {ResultCode::kInvalidView, "safety status must match batch"};
  const auto flags = read_view(status.flags, status.device, status.size);
  const auto checked = mask.data ? disjoint({state.last_valid_output.as_const(),
                                             state.valid_mask.as_const(),
                                             state.hold_count.as_const(),
                                             flags,
                                             mask})
                                 : disjoint({state.last_valid_output.as_const(),
                                             state.valid_mask.as_const(),
                                             state.hold_count.as_const(),
                                             flags});
  if (!checked.ok()) return checked;
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::reset_safety_state(batch_size, state, mask, status);
}
Result apply_safe_output_policy(MutableTensorView output,
                                MutableStatusView status,
                                SafetyStateView state,
                                StepOptions options,
                                ExecutionContext context) noexcept {
  if (output.rank < 1 || status.size != static_cast<std::size_t>(output.shape[0]))
    return {ResultCode::kInvalidView, "safety status must match batch"};
  if (options.safe_output_policy != SafeOutputPolicy::kZero &&
      options.safe_output_policy != SafeOutputPolicy::kHoldLastValid)
    return {ResultCode::kInvalidArgument, "unknown safety policy"};
  const auto checked = disjoint({output.as_const(),
                                 state.last_valid_output.as_const(),
                                 state.valid_mask.as_const(),
                                 state.hold_count.as_const(),
                                 read_view(status.flags, status.device, status.size)});
  if (!checked.ok()) return checked;
  detail::tensor::ScopedTensorExecutionContext scope(context);
  return detail::tensor::apply_safe_output_policy(output, status, state, options);
}
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
