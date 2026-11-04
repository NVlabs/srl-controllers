/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file common.hpp
 * @brief Internal helpers for dependency-free tensor controller kernels.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "./types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace internal {

inline Result ok() noexcept { return Result{ResultCode::kOk, "ok"}; }

inline Result invalid_argument(const char* message) noexcept {
  return Result{ResultCode::kInvalidArgument, message};
}

inline Result invalid_view(const char* message) noexcept {
  return Result{ResultCode::kInvalidView, message};
}

inline Result unsupported(const char* message) noexcept {
  return Result{ResultCode::kUnsupported, message};
}

struct ByteSpan {
  std::uintptr_t begin{0};
  std::uintptr_t end{0};
  Device device{Device::kUnknown};
};

inline std::size_t dtype_size(const DType dtype) noexcept {
  switch (dtype) {
    case DType::kFloat32:
      return sizeof(float);
    case DType::kFloat64:
      return sizeof(double);
    case DType::kUInt64:
      return sizeof(std::uint64_t);
    case DType::kBool:
      return sizeof(std::uint8_t);
    default:
      return 0;
  }
}

inline Result byte_span(const TensorView& view, ByteSpan* output) noexcept {
  const std::size_t item_size = dtype_size(view.dtype);
  if (output == nullptr || item_size == 0 || view.rank < 0 || view.rank > kMaxTensorRank ||
      (view.device != Device::kCpu && view.device != Device::kCuda)) {
    return invalid_view("invalid tensor metadata for byte span");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(view.data);
  if (address % item_size != 0) {
    return invalid_view("tensor address is not aligned to its dtype");
  }
  bool empty = false;
  for (int dim = 0; dim < view.rank; ++dim) {
    if (view.shape[dim] < 0 || view.strides[dim] < 0) {
      return invalid_view("tensor extents and strides must be nonnegative");
    }
    empty = empty || view.shape[dim] == 0;
  }
  if (empty) {
    *output = ByteSpan{address, address, view.device};
    return ok();
  }
  if (view.data == nullptr) {
    return invalid_view("tensor data is null");
  }
  constexpr auto maximum = static_cast<std::uintptr_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const std::uintptr_t maximum_offset = maximum / item_size - 1;
  std::uintptr_t offset = 0;
  for (int dim = 0; dim < view.rank; ++dim) {
    const auto extent = static_cast<std::uintptr_t>(view.shape[dim] - 1);
    const auto stride = static_cast<std::uintptr_t>(view.strides[dim]);
    if (extent != 0 && stride > (maximum_offset - offset) / extent) {
      return invalid_view("tensor byte span overflows");
    }
    offset += extent * stride;
  }
  const std::uintptr_t bytes = (offset + 1) * item_size;
  if (address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return invalid_view("tensor address range overflows");
  }
  *output = ByteSpan{address, address + bytes, view.device};
  return ok();
}

inline bool overlaps(const ByteSpan& left, const ByteSpan& right) noexcept {
  return left.device == right.device && left.begin < left.end && right.begin < right.end &&
         left.begin < right.end && right.begin < left.end;
}

inline bool same_view(const TensorView& left, const TensorView& right) noexcept {
  return left.data == right.data && left.dtype == right.dtype && left.device == right.device &&
         left.rank == right.rank && left.shape == right.shape && left.strides == right.strides;
}

inline Result validate_disjoint(const TensorView& left,
                                const TensorView& right,
                                const char* message,
                                const bool allow_exact = false) noexcept {
  ByteSpan left_span;
  ByteSpan right_span;
  Result result = byte_span(left, &left_span);
  if (!result.ok()) {
    return result;
  }
  result = byte_span(right, &right_span);
  if (!result.ok()) {
    return result;
  }
  return overlaps(left_span, right_span) && !(allow_exact && same_view(left, right))
             ? invalid_view(message)
             : ok();
}

inline TensorView status_tensor(const MutableStatusView& status) noexcept {
  return read_view(status.flags, status.device, status.size);
}

inline TensorView status_tensor(const StatusView& status) noexcept {
  return read_view(status.flags, status.device, status.size);
}

class StorageValidation {
 public:
  void add(const TensorView& view, int first = 0, int last = 4, bool inplace = false) noexcept {
    if (!result.ok()) {
      return;
    }
    ByteSpan span;
    result = byte_span(view, &span);
    if (!result.ok()) {
      return;
    }
    if (count_ == entries_.size() || first < 0 || first > last || last > 4) {
      result = invalid_view("invalid storage lifetime or validation capacity exceeded");
      return;
    }
    for (std::size_t i = 0; i < count_; ++i) {
      const Entry& other = entries_[i];
      if (first <= other.last && other.first <= last && overlaps(span, other.span) &&
          !(inplace && other.inplace && same_view(view, other.view))) {
        result = invalid_view("independent storage buffers must not overlap");
        return;
      }
    }
    entries_[count_++] = Entry{view, span, first, last, inplace};
  }

  void history(std::initializer_list<TensorView> views) noexcept {
    for (const TensorView& view : views) {
      add(view);
    }
  }

  Result result{};

 private:
  struct Entry {
    TensorView view;
    ByteSpan span;
    int first;
    int last;
    bool inplace;
  };
  std::array<Entry,
             Requirements::kStateCapacity + Requirements::kScratchCapacity +
                 Requirements::kOutputCapacity>
      entries_{};
  std::size_t count_{0};
};

inline Result validate_tensor_metadata(const TensorView& view) noexcept {
  ByteSpan span;
  return byte_span(view, &span);
}

inline bool is_float_dtype(const DType dtype) noexcept {
  return dtype == DType::kFloat32 || dtype == DType::kFloat64;
}

inline bool is_cpu(const TensorView& view) noexcept { return view.device == Device::kCpu; }
inline bool is_cpu(const MutableTensorView& view) noexcept { return view.device == Device::kCpu; }

inline bool is_cuda(const TensorView& view) noexcept { return view.device == Device::kCuda; }
inline bool is_cuda(const MutableTensorView& view) noexcept { return view.device == Device::kCuda; }

inline bool is_row_major_matrix(const TensorView& view) noexcept {
  return validate_tensor_metadata(view).ok() && view.rank == 2 &&
         view.layout == Layout::kRowMajor && view.strides[1] == 1 &&
         view.strides[0] >= view.shape[1];
}

inline bool is_row_major_matrix(const MutableTensorView& view) noexcept {
  return is_row_major_matrix(view.as_const());
}

inline bool is_row_major_tensor3(const TensorView& view) noexcept {
  return validate_tensor_metadata(view).ok() && view.rank == 3 &&
         view.layout == Layout::kRowMajor && view.strides[2] == 1 &&
         view.strides[1] >= view.shape[2] && view.strides[1] > 0 &&
         view.shape[1] <= std::numeric_limits<std::int64_t>::max() / view.strides[1] &&
         view.strides[0] >= view.shape[1] * view.strides[1];
}

inline bool is_row_major_tensor3(const MutableTensorView& view) noexcept {
  return is_row_major_tensor3(view.as_const());
}

inline Result validate_status_view(const MutableStatusView& status,
                                   const std::size_t num_envs) noexcept {
  const Result metadata =
      validate_tensor_metadata(read_view(status.flags, status.device, status.size));
  if (!metadata.ok()) return metadata;
  if (status.flags == nullptr) {
    return invalid_view("status flags are null");
  }
  if (status.size < num_envs) {
    return invalid_view("status size is smaller than num_envs");
  }
  if (status.device == Device::kCuda) {
    return unsupported("CUDA status writes require CUDA tensor kernels");
  }
  if (status.device != Device::kCpu) {
    return invalid_view("status device is unsupported");
  }
  return ok();
}

inline Result validate_status_view(const StatusView& status,
                                   const std::size_t num_envs,
                                   const bool allow_empty) noexcept {
  const Result metadata =
      validate_tensor_metadata(read_view(status.flags, status.device, status.size));
  if (!metadata.ok()) return metadata;
  if (status.flags == nullptr) {
    return allow_empty ? ok() : invalid_view("status flags are null");
  }
  if (status.size < num_envs) {
    return invalid_view("status size is smaller than num_envs");
  }
  if (status.device == Device::kCuda) {
    return unsupported("CUDA status reads require CUDA tensor kernels");
  }
  if (status.device != Device::kCpu) {
    return invalid_view("status device is unsupported");
  }
  return ok();
}

inline void clear_status(MutableStatusView status, const std::size_t num_envs) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] = 0U;
  }
}

inline void set_all_status(MutableStatusView status,
                           const std::size_t num_envs,
                           const std::uint64_t flags) noexcept {
  for (std::size_t env = 0; env < num_envs; ++env) {
    status.flags[env] |= flags;
  }
}

inline Result validate_reset_mask(const TensorView& reset_mask,
                                  const std::size_t num_envs) noexcept {
  if (reset_mask.data == nullptr) {
    return ok();
  }
  if (reset_mask.device == Device::kCuda) {
    return unsupported("CUDA reset masks require CUDA tensor kernels");
  }
  if (reset_mask.device != Device::kCpu) {
    return invalid_view("reset mask device is unsupported");
  }
  const Result metadata = validate_tensor_metadata(reset_mask);
  if (!metadata.ok()) return metadata;
  if (reset_mask.dtype != DType::kBool) {
    return invalid_view("reset mask must use DType::kBool");
  }
  if (reset_mask.rank != 1 || reset_mask.shape[0] < static_cast<std::int64_t>(num_envs)) {
    return invalid_view("reset mask must have shape [num_envs]");
  }
  return ok();
}

inline bool reset_mask_value(const TensorView& reset_mask, const std::size_t env) noexcept {
  if (reset_mask.data == nullptr) {
    return true;
  }
  const auto* values = static_cast<const std::uint8_t*>(reset_mask.data);
  return values[env * static_cast<std::size_t>(reset_mask.strides[0])] != 0U;
}

inline Result validate_batched_signal(const TensorView& input,
                                      const char* name,
                                      std::size_t* num_envs,
                                      std::size_t* signal_dim) noexcept {
  if (input.data == nullptr) {
    return invalid_view(name);
  }
  if (is_cuda(input)) {
    return unsupported("CUDA tensor execution is schema-visible but not implemented in this build");
  }
  if (!is_cpu(input)) {
    return invalid_view("input tensor device is unsupported");
  }
  if (!is_float_dtype(input.dtype)) {
    return invalid_view("input tensor dtype must be float32 or float64");
  }
  if (!is_row_major_matrix(input)) {
    return invalid_view("input tensor must be row-major [num_envs, signal_dim]");
  }
  if (input.shape[0] <= 0 || input.shape[1] <= 0) {
    return invalid_view("input tensor shape must be positive");
  }
  *num_envs = static_cast<std::size_t>(input.shape[0]);
  *signal_dim = static_cast<std::size_t>(input.shape[1]);
  return ok();
}

inline Result validate_matching_output(const MutableTensorView& output,
                                       const DType dtype,
                                       const std::size_t num_envs,
                                       const std::size_t signal_dim) noexcept {
  if (output.data == nullptr) {
    return invalid_view("output tensor data is null");
  }
  if (is_cuda(output)) {
    return unsupported("CUDA tensor execution is schema-visible but not implemented in this build");
  }
  if (!is_cpu(output)) {
    return invalid_view("output tensor device is unsupported");
  }
  if (output.dtype != dtype) {
    return invalid_view("output tensor dtype must match input dtype");
  }
  if (!is_row_major_matrix(output) || output.shape[0] != static_cast<std::int64_t>(num_envs) ||
      output.shape[1] != static_cast<std::int64_t>(signal_dim)) {
    return invalid_view("output tensor must be row-major [num_envs, signal_dim]");
  }
  return ok();
}

inline Result validate_batched_matrix(const TensorView& input,
                                      const char* name,
                                      const DType dtype,
                                      const std::size_t num_envs,
                                      const std::size_t rows,
                                      const std::size_t cols) noexcept {
  if (input.data == nullptr) {
    return invalid_view(name);
  }
  if (is_cuda(input)) {
    return unsupported("CUDA tensor execution is schema-visible but not implemented in this build");
  }
  if (!is_cpu(input)) {
    return invalid_view("matrix tensor device is unsupported");
  }
  if (input.dtype != dtype) {
    return invalid_view("matrix tensor dtype must match the component dtype");
  }
  if (!is_row_major_tensor3(input) || input.shape[0] != static_cast<std::int64_t>(num_envs) ||
      input.shape[1] != static_cast<std::int64_t>(rows) ||
      input.shape[2] != static_cast<std::int64_t>(cols)) {
    return invalid_view("matrix tensor must be row-major [num_envs, rows, cols]");
  }
  return ok();
}

inline Result validate_scalar_batch(const TensorView& input,
                                    const char* name,
                                    const DType dtype,
                                    const std::size_t num_envs) noexcept {
  const Result metadata = validate_tensor_metadata(input);
  if (!metadata.ok()) return metadata;
  if (input.data == nullptr) {
    return invalid_view(name);
  }
  if (is_cuda(input)) {
    return unsupported("CUDA tensor execution is schema-visible but not implemented in this build");
  }
  if (!is_cpu(input)) {
    return invalid_view("scalar batch tensor device is unsupported");
  }
  if (input.dtype != dtype) {
    return invalid_view("scalar batch tensor dtype must match the component dtype");
  }
  if (input.rank == 0) {
    return ok();
  }
  if (input.rank == 1 && input.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      input.strides[0] >= 1) {
    return ok();
  }
  return invalid_view("scalar batch tensor must be scalar or shape [num_envs]");
}

inline Result validate_scalar_output(const MutableTensorView& output,
                                     const DType dtype,
                                     const std::size_t num_envs) noexcept {
  const Result metadata = validate_tensor_metadata(output.as_const());
  if (!metadata.ok()) return metadata;
  if (output.data == nullptr) {
    return invalid_view("scalar output data is null");
  }
  if (is_cuda(output)) {
    return unsupported("CUDA tensor execution is schema-visible but not implemented in this build");
  }
  if (!is_cpu(output)) {
    return invalid_view("scalar output device is unsupported");
  }
  if (output.dtype != dtype) {
    return invalid_view("scalar output dtype must match component dtype");
  }
  if (output.rank == 1 && output.shape[0] >= static_cast<std::int64_t>(num_envs) &&
      output.strides[0] >= 1) {
    return ok();
  }
  return invalid_view("scalar output must have shape [num_envs]");
}

inline Result validate_vector_parameter_metadata(const ParameterView* parameter,
                                                 const char* missing_message,
                                                 const DType dtype,
                                                 const std::size_t num_envs,
                                                 const std::size_t signal_dim,
                                                 const bool required) noexcept {
  if (parameter == nullptr) {
    return required ? invalid_view(missing_message) : ok();
  }
  const TensorView& values = parameter->values;
  const Result metadata = validate_tensor_metadata(values);
  if (!metadata.ok()) return metadata;
  if (values.data == nullptr) {
    return invalid_view("parameter data is null");
  }
  if (is_cuda(values)) {
    return unsupported("CUDA parameter tensors require CUDA tensor kernels");
  }
  if (!is_cpu(values)) {
    return invalid_view("parameter device is unsupported");
  }
  if (values.dtype != dtype) {
    return invalid_view("parameter dtype must match component dtype");
  }
  if (parameter->shared) {
    if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(signal_dim) &&
        values.strides[0] >= 1) {
      return ok();
    }
    return invalid_view("shared vector parameter must have shape [signal_dim]");
  }
  if (values.rank == 2 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(signal_dim) && values.strides[1] >= 1 &&
      values.strides[0] >= values.shape[1]) {
    return ok();
  }
  return invalid_view("batched vector parameter must have shape [num_envs, signal_dim]");
}

inline Result validate_matrix_parameter_metadata(const ParameterView* parameter,
                                                 const char* missing_message,
                                                 const DType dtype,
                                                 const std::size_t num_envs,
                                                 const std::size_t rows,
                                                 const std::size_t cols,
                                                 const bool required) noexcept {
  if (parameter == nullptr) {
    return required ? invalid_view(missing_message) : ok();
  }
  const TensorView& values = parameter->values;
  const Result metadata = validate_tensor_metadata(values);
  if (!metadata.ok()) return metadata;
  if (values.data == nullptr) {
    return invalid_view("matrix parameter data is null");
  }
  if (is_cuda(values)) {
    return unsupported("CUDA parameter tensors require CUDA tensor kernels");
  }
  if (!is_cpu(values)) {
    return invalid_view("matrix parameter device is unsupported");
  }
  if (values.dtype != dtype) {
    return invalid_view("matrix parameter dtype must match component dtype");
  }
  if (parameter->shared) {
    if (values.rank == 2 && values.layout == Layout::kRowMajor &&
        values.shape[0] == static_cast<std::int64_t>(rows) &&
        values.shape[1] == static_cast<std::int64_t>(cols) && values.strides[1] == 1 &&
        values.strides[0] >= values.shape[1]) {
      return ok();
    }
    return invalid_view("shared matrix parameter must have shape [rows, cols]");
  }
  if (is_row_major_tensor3(values) && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.shape[1] == static_cast<std::int64_t>(rows) &&
      values.shape[2] == static_cast<std::int64_t>(cols)) {
    return ok();
  }
  return invalid_view("batched matrix parameter must have shape [num_envs, rows, cols]");
}

inline Result validate_scalar_parameter_metadata(const ParameterView* parameter,
                                                 const char* missing_message,
                                                 const DType dtype,
                                                 const std::size_t num_envs,
                                                 const bool required) noexcept {
  if (parameter == nullptr) {
    return required ? invalid_view(missing_message) : ok();
  }
  const TensorView& values = parameter->values;
  const Result metadata = validate_tensor_metadata(values);
  if (!metadata.ok()) return metadata;
  if (values.data == nullptr) {
    return invalid_view("parameter data is null");
  }
  if (is_cuda(values)) {
    return unsupported("CUDA parameter tensors require CUDA tensor kernels");
  }
  if (!is_cpu(values)) {
    return invalid_view("parameter device is unsupported");
  }
  if (values.dtype != dtype) {
    return invalid_view("parameter dtype must match component dtype");
  }
  if (parameter->shared) {
    if (values.rank == 0 || (values.rank == 1 && values.shape[0] == 1)) {
      return ok();
    }
    return invalid_view("shared scalar parameter must be scalar or shape [1]");
  }
  if (values.rank == 1 && values.shape[0] == static_cast<std::int64_t>(num_envs) &&
      values.strides[0] >= 1) {
    return ok();
  }
  return invalid_view("batched scalar parameter must have shape [num_envs]");
}

template <typename Scalar>
inline Scalar read_matrix_value(const TensorView& view,
                                const std::size_t env,
                                const std::size_t dim) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             dim * static_cast<std::size_t>(view.strides[1]);
  return values[offset];
}

template <typename Scalar>
inline Scalar read_matrix_value(const MutableTensorView& view,
                                const std::size_t env,
                                const std::size_t dim) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             dim * static_cast<std::size_t>(view.strides[1]);
  return values[offset];
}

template <typename Scalar>
inline void write_matrix_value(const MutableTensorView& view,
                               const std::size_t env,
                               const std::size_t dim,
                               const Scalar value) noexcept {
  auto* values = static_cast<Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             dim * static_cast<std::size_t>(view.strides[1]);
  values[offset] = value;
}

template <typename Scalar>
inline Scalar read_scalar_value(const TensorView& view, const std::size_t env) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  if (view.rank == 0) {
    return values[0];
  }
  return values[env * static_cast<std::size_t>(view.strides[0])];
}

template <typename Scalar>
inline Scalar read_scalar_value(const MutableTensorView& view, const std::size_t env) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  if (view.rank == 0) {
    return values[0];
  }
  return values[env * static_cast<std::size_t>(view.strides[0])];
}

template <typename Scalar>
inline Scalar read_tensor3_value(const TensorView& view,
                                 const std::size_t env,
                                 const std::size_t row,
                                 const std::size_t col) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             row * static_cast<std::size_t>(view.strides[1]) +
                             col * static_cast<std::size_t>(view.strides[2]);
  return values[offset];
}

template <typename Scalar>
inline Scalar read_tensor3_value(const MutableTensorView& view,
                                 const std::size_t env,
                                 const std::size_t row,
                                 const std::size_t col) noexcept {
  const auto* values = static_cast<const Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             row * static_cast<std::size_t>(view.strides[1]) +
                             col * static_cast<std::size_t>(view.strides[2]);
  return values[offset];
}

template <typename Scalar>
inline void write_tensor3_value(const MutableTensorView& view,
                                const std::size_t env,
                                const std::size_t row,
                                const std::size_t col,
                                const Scalar value) noexcept {
  auto* values = static_cast<Scalar*>(view.data);
  const std::size_t offset = env * static_cast<std::size_t>(view.strides[0]) +
                             row * static_cast<std::size_t>(view.strides[1]) +
                             col * static_cast<std::size_t>(view.strides[2]);
  values[offset] = value;
}

template <typename Scalar>
inline void write_scalar_value(const MutableTensorView& view,
                               const std::size_t env,
                               const Scalar value) noexcept {
  auto* values = static_cast<Scalar*>(view.data);
  values[env * static_cast<std::size_t>(view.strides[0])] = value;
}

template <typename Scalar>
inline Scalar read_vector_parameter(const ParameterView* parameter,
                                    const std::size_t env,
                                    const std::size_t dim,
                                    const Scalar default_value) noexcept {
  if (parameter == nullptr) {
    return default_value;
  }
  const auto* values = static_cast<const Scalar*>(parameter->values.data);
  if (parameter->shared) {
    return values[dim * static_cast<std::size_t>(parameter->values.strides[0])];
  }
  const std::size_t offset = env * static_cast<std::size_t>(parameter->values.strides[0]) +
                             dim * static_cast<std::size_t>(parameter->values.strides[1]);
  return values[offset];
}

template <typename Scalar>
inline Scalar read_scalar_parameter(const ParameterView* parameter,
                                    const std::size_t env,
                                    const Scalar default_value) noexcept {
  if (parameter == nullptr) {
    return default_value;
  }
  const auto* values = static_cast<const Scalar*>(parameter->values.data);
  if (parameter->shared) {
    return values[0];
  }
  return values[env * static_cast<std::size_t>(parameter->values.strides[0])];
}

template <typename Scalar>
inline Scalar read_matrix_parameter(const ParameterView* parameter,
                                    const std::size_t env,
                                    const std::size_t row,
                                    const std::size_t col,
                                    const Scalar default_value) noexcept {
  if (parameter == nullptr) {
    return default_value;
  }
  const auto* values = static_cast<const Scalar*>(parameter->values.data);
  if (parameter->shared) {
    return values[row * static_cast<std::size_t>(parameter->values.strides[0]) +
                  col * static_cast<std::size_t>(parameter->values.strides[1])];
  }
  const std::size_t offset = env * static_cast<std::size_t>(parameter->values.strides[0]) +
                             row * static_cast<std::size_t>(parameter->values.strides[1]) +
                             col * static_cast<std::size_t>(parameter->values.strides[2]);
  return values[offset];
}

template <typename Scalar>
inline void zero_output_env(const MutableTensorView& output,
                            const std::size_t env,
                            const std::size_t signal_dim) noexcept {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    write_matrix_value(output, env, dim, static_cast<Scalar>(0));
  }
}

template <typename Scalar>
inline bool matrix_env_is_finite(const TensorView& input,
                                 const std::size_t env,
                                 const std::size_t signal_dim) noexcept {
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (!std::isfinite(read_matrix_value<Scalar>(input, env, dim))) {
      return false;
    }
  }
  return true;
}

template <typename Scalar>
inline bool tensor3_env_is_finite(const TensorView& input,
                                  const std::size_t env,
                                  const std::size_t rows,
                                  const std::size_t cols) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (!std::isfinite(read_tensor3_value<Scalar>(input, env, row, col))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
inline bool matrix_parameter_env_is_finite(const ParameterView* parameter,
                                           const std::size_t env,
                                           const std::size_t rows,
                                           const std::size_t cols) noexcept {
  if (parameter == nullptr) {
    return false;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (!std::isfinite(
              read_matrix_parameter<Scalar>(parameter, env, row, col, static_cast<Scalar>(0)))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Scalar>
inline bool vector_parameters_are_finite(const ParameterSetView& parameters,
                                         const char* first_name,
                                         const char* second_name,
                                         const char* third_name,
                                         const std::size_t env,
                                         const std::size_t signal_dim) noexcept {
  const ParameterView* first = find_parameter(parameters, first_name);
  const ParameterView* second = find_parameter(parameters, second_name);
  const ParameterView* third = find_parameter(parameters, third_name);
  for (std::size_t dim = 0; dim < signal_dim; ++dim) {
    if (first != nullptr &&
        !std::isfinite(read_vector_parameter<Scalar>(first, env, dim, static_cast<Scalar>(0)))) {
      return false;
    }
    if (second != nullptr &&
        !std::isfinite(read_vector_parameter<Scalar>(second, env, dim, static_cast<Scalar>(0)))) {
      return false;
    }
    if (third != nullptr &&
        !std::isfinite(read_vector_parameter<Scalar>(third, env, dim, static_cast<Scalar>(0)))) {
      return false;
    }
  }
  return true;
}

inline TensorDescriptor matrix_descriptor(const char* name,
                                          const DType dtype,
                                          const std::size_t num_envs,
                                          const std::size_t signal_dim,
                                          const char* units,
                                          const char* frame,
                                          const char* semantics) {
  TensorDescriptor descriptor;
  descriptor.name = name;
  descriptor.dtype = dtype;
  descriptor.rank = 2;
  descriptor.shape = {
      static_cast<std::int64_t>(num_envs), static_cast<std::int64_t>(signal_dim), 0, 0};
  descriptor.strides = {static_cast<std::int64_t>(signal_dim), 1, 0, 0};
  descriptor.units = units;
  descriptor.frame = frame;
  descriptor.semantics = semantics;
  return descriptor;
}

inline TensorDescriptor vector_descriptor(const char* name,
                                          const DType dtype,
                                          const std::size_t num_envs,
                                          const char* units,
                                          const char* frame,
                                          const char* semantics) {
  TensorDescriptor descriptor;
  descriptor.name = name;
  descriptor.dtype = dtype;
  descriptor.rank = 1;
  descriptor.shape = {static_cast<std::int64_t>(num_envs), 0, 0, 0};
  descriptor.strides = {1, 0, 0, 0};
  descriptor.units = units;
  descriptor.frame = frame;
  descriptor.semantics = semantics;
  return descriptor;
}

inline TensorDescriptor tensor3_descriptor(const char* name,
                                           const DType dtype,
                                           const std::size_t num_envs,
                                           const std::size_t rows,
                                           const std::size_t cols,
                                           const char* units,
                                           const char* frame,
                                           const char* semantics) {
  TensorDescriptor descriptor;
  descriptor.name = name;
  descriptor.dtype = dtype;
  descriptor.rank = 3;
  descriptor.shape = {static_cast<std::int64_t>(num_envs),
                      static_cast<std::int64_t>(rows),
                      static_cast<std::int64_t>(cols),
                      0};
  descriptor.strides = {
      static_cast<std::int64_t>(rows * cols), static_cast<std::int64_t>(cols), 1, 0};
  descriptor.units = units;
  descriptor.frame = frame;
  descriptor.semantics = semantics;
  return descriptor;
}

}  // namespace internal
}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
