// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include "nvidia/srl/controllers/equation_adapter.hpp"
#include <limits>
#include "nvidia/srl/controllers/equation_evaluation.hpp"
#include "tensor/common.hpp"
namespace nvidia {
namespace srl {
namespace controllers {
namespace equation_adapter {
namespace {
Result invalid(const char* message) noexcept { return {ResultCode::kInvalidView, message}; }
Result validate_base(const Call& call) noexcept {
  if (!call.batch_size ||
      call.batch_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      (call.device != Device::kCpu && call.device != Device::kCuda) ||
      (call.dtype != DType::kFloat32 && call.dtype != DType::kFloat64) ||
      call.status.device != call.device || call.status.size != call.batch_size ||
      !call.status.flags)
    return invalid("equation requires float32/64, a positive batch and matching status storage");
  return {};
}
Result validate_view(TensorView view, Device device, DType dtype, int rank) noexcept {
  if (!view.data || view.device != device || view.dtype != dtype || view.rank != rank ||
      view.layout != Layout::kRowMajor)
    return invalid("equation array metadata mismatch");
  std::int64_t stride = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    if (view.shape[axis] <= 0 || view.strides[axis] != stride ||
        stride > std::numeric_limits<std::int64_t>::max() / view.shape[axis])
      return invalid("equation array must be nonempty and contiguous");
    stride *= view.shape[axis];
  }
  const std::size_t width = dtype == DType::kFloat64 || dtype == DType::kUInt64 ? 8
                            : dtype == DType::kFloat32                          ? 4
                                                                                : 1;
  const auto address = reinterpret_cast<std::uintptr_t>(view.data);
  if (static_cast<std::size_t>(stride) >
      (std::numeric_limits<std::uintptr_t>::max() - address) / width)
    return invalid("equation allocation range overflow");
  return detail::tensor::internal::validate_tensor_metadata(view);
}
Result validate_input(const Call& call, int index, int rank, DType dtype) noexcept {
  const auto& input = call.inputs[index];
  if (input.batching != ParameterBatching::kShared &&
      input.batching != ParameterBatching::kPerBatch)
    return invalid("equation arguments require explicit shared/per_batch");
  const bool batched = input.batching == ParameterBatching::kPerBatch;
  const Result result = validate_view(input.values, call.device, dtype, rank + (batched ? 1 : 0));
  if (!result.ok()) return result;
  if (batched && input.values.shape[0] != static_cast<std::int64_t>(call.batch_size))
    return invalid("equation argument batch mismatch");
  return {};
}
bool trailing_shape(const ParameterTensor& parameter,
                    std::int64_t rows,
                    std::int64_t cols = 0) noexcept {
  const auto& view = parameter.values;
  return cols ? view.shape[view.rank - 2] == rows && view.shape[view.rank - 1] == cols
              : view.shape[view.rank - 1] == rows;
}
Result validate_output(const Call& call, int index, std::int64_t dimension, DType dtype) noexcept {
  const auto& view = call.outputs[index];
  const Result result = validate_view(view.as_const(), call.device, dtype, dimension ? 2 : 1);
  if (!result.ok()) return result;
  if (view.shape[0] != static_cast<std::int64_t>(call.batch_size) ||
      (dimension && view.shape[1] != dimension))
    return invalid("equation output shape mismatch");
  return {};
}
bool overlap(TensorView a, TensorView b) noexcept {
  detail::tensor::internal::ByteSpan first;
  detail::tensor::internal::ByteSpan second;
  const Result first_result = detail::tensor::internal::byte_span(a, &first);
  const Result second_result = detail::tensor::internal::byte_span(b, &second);
  return !first_result.ok() || !second_result.ok() ||
         detail::tensor::internal::overlaps(first, second);
}
Result validate_disjoint(const Call& call, int inputs, int outputs) noexcept {
  const auto status = read_view(call.status.flags, call.device, call.batch_size);
  const Result result = validate_view(status, call.device, DType::kUInt64, 1);
  if (!result.ok()) return result;
  for (int i = 0; i < inputs; ++i)
    if (overlap(status, call.inputs[i].values))
      return invalid("status must not overlap equation inputs");
  for (int i = 0; i < outputs; ++i) {
    const auto output = call.outputs[i].as_const();
    if (overlap(output, status)) return invalid("status must not overlap equation outputs");
    for (int j = 0; j < inputs; ++j)
      if (overlap(output, call.inputs[j].values))
        return invalid("equation outputs must not overlap inputs");
    for (int j = 0; j < i; ++j)
      if (overlap(output, call.outputs[j].as_const()))
        return invalid("equation outputs must be disjoint");
  }
  return {};
}
}  // namespace
#include "equation_validation.inc"
Result execute(const Call& call, ExecutionContext context) noexcept {
  const Result checked = validate(call);
  if (!checked.ok()) return checked;
  if (call.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return execute_cuda(call, context);
#else
    return {ResultCode::kBackendUnavailable, "equation CUDA execution requires ENABLE_CUDA=ON"};
#endif
  }
  (void)context;
  for (std::size_t row = 0; row < call.batch_size; ++row) {
    if (call.dtype == DType::kFloat32)
      evaluate<float>(call, row);
    else
      evaluate<double>(call, row);
  }
  return {};
}
}  // namespace equation_adapter
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
