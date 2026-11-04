// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
// Included inside the public implementation namespace; no installed dependencies.
namespace {
ControllerConfig block_config(const ControllerConfig& config) noexcept { return config; }
ControllerConfig block_config(const SignalBlockConfig& config) noexcept {
  ControllerConfig result;
  result.device = config.device;
  result.dtype = config.dtype;
  result.batch_size = config.batch_size;
  result.safety = config.safety;
  return result;
}
std::size_t block_dimension(const ControllerConfig& config) noexcept { return config.joint_dim; }
std::size_t block_dimension(const SignalBlockConfig& config) noexcept { return config.signal_dim; }

TensorRequirement block_requirement(const char* name,
                                    DType dtype,
                                    Device device,
                                    std::initializer_list<std::size_t> shape,
                                    const char* units,
                                    const char* frame) noexcept {
  TensorRequirement result;
  result.name = name;
  result.dtype = dtype;
  result.device = device;
  result.rank = static_cast<int>(shape.size());
  result.units = units;
  result.frame = frame;
  std::size_t index = 0;
  for (auto extent : shape) result.shape[index++] = static_cast<std::int64_t>(extent);
  return result;
}

template <std::size_t Capacity>
Result append_block_requirement(std::array<TensorRequirement, Capacity>* entries,
                                std::size_t* count,
                                const char* name,
                                DType dtype,
                                Device device,
                                std::initializer_list<std::size_t> shape,
                                const char* units,
                                const char* frame) noexcept {
  std::size_t elements = 1;
  const std::size_t width = detail::tensor::internal::dtype_size(dtype);
  if (shape.size() > static_cast<std::size_t>(kMaxTensorRank) || width == 0)
    return invalid_argument("block requirement rank or dtype is invalid");
  for (const std::size_t extent : shape) {
    if (extent == 0 ||
        extent > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) ||
        elements > static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()) / extent ||
        elements > std::numeric_limits<std::size_t>::max() / width / extent)
      return invalid_argument("block requirement dimensions overflow");
    elements *= extent;
  }
  return append_requirement(
             entries, count, block_requirement(name, dtype, device, shape, units, frame), name)
             ? Result{}
             : invalid_argument("block requirement capacity exceeded");
}

Result validate_block_array(const TensorView& view,
                            const ControllerConfig& config,
                            DType dtype,
                            std::initializer_list<std::size_t> shape) noexcept {
  if (view.rank != static_cast<int>(shape.size()) || view.data == nullptr || view.dtype != dtype ||
      view.device != config.device || view.layout != Layout::kRowMajor) {
    return invalid_view("block array must match the configured shape, dtype, device and layout");
  }
  std::int64_t stride = 1;
  int index = view.rank;
  for (auto extent = shape.end(); extent != shape.begin();) {
    --extent;
    --index;
    if (*extent == 0 ||
        *extent > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        view.shape[index] != static_cast<std::int64_t>(*extent) || view.strides[index] != stride ||
        stride > std::numeric_limits<std::int64_t>::max() / static_cast<std::int64_t>(*extent)) {
      return invalid_view("block array extent or stride is invalid");
    }
    stride *= static_cast<std::int64_t>(*extent);
  }
  return detail::tensor::internal::validate_tensor_metadata(view);
}
Result validate_block_array(const MutableTensorView& view,
                            const ControllerConfig& config,
                            DType dtype,
                            std::initializer_list<std::size_t> shape) noexcept {
  return validate_block_array(view.as_const(), config, dtype, shape);
}
Result validate_block_dt(const TensorView& view, const ControllerConfig& config) noexcept {
  return view.rank == 0 ? validate_block_array(view, config, config.dtype, {})
                        : validate_block_array(view, config, config.dtype, {config.batch_size});
}
Result validate_block_parameter(const ParameterTensor& parameter,
                                const ControllerConfig& config,
                                std::initializer_list<std::size_t> shape) noexcept {
  if (parameter.batching != ParameterBatching::kShared &&
      parameter.batching != ParameterBatching::kPerBatch) {
    return invalid_argument("parameter batching must be shared or per batch");
  }
  TensorView view = parameter.values;
  if (parameter.batching == ParameterBatching::kPerBatch) {
    if (view.rank < 1 || view.shape[0] != static_cast<std::int64_t>(config.batch_size)) {
      return invalid_view("per-batch parameter must start with the configured batch extent");
    }
    std::size_t width = 1;
    for (auto extent : shape) {
      if (extent == 0 || width > std::numeric_limits<std::size_t>::max() / extent)
        return invalid_view("parameter shape overflow");
      width *= extent;
    }
    if (view.strides[0] != static_cast<std::int64_t>(width))
      return invalid_view("parameter batch stride must be contiguous");
    --view.rank;
    for (int i = 0; i < view.rank; ++i) {
      view.shape[i] = view.shape[i + 1];
      view.strides[i] = view.strides[i + 1];
    }
  }
  return validate_block_array(view, config, config.dtype, shape);
}
Result validate_block_status(const ControllerConfig& c,
                             MutableStatusView parameter_status,
                             MutableStatusView status) noexcept {
  if (parameter_status.flags == nullptr || status.flags == nullptr ||
      parameter_status.size != c.batch_size || status.size != c.batch_size ||
      parameter_status.device != c.device || status.device != c.device ||
      parameter_status.flags == status.flags)
    return invalid_view("block status storage must be distinct arrays matching batch and device");
  return {};
}
bool block_arrays_overlap(std::initializer_list<MutableTensorView> views) noexcept {
  for (auto first = views.begin(); first != views.end(); ++first) {
    detail::tensor::internal::ByteSpan first_span;
    if (!detail::tensor::internal::byte_span(first->as_const(), &first_span).ok()) return true;
    for (auto second = first + 1; second != views.end(); ++second) {
      detail::tensor::internal::ByteSpan second_span;
      if (!detail::tensor::internal::byte_span(second->as_const(), &second_span).ok() ||
          detail::tensor::internal::overlaps(first_span, second_span))
        return true;
    }
  }
  return false;
}
bool block_input_overlaps(const TensorView& input,
                          std::initializer_list<MutableTensorView> storage) noexcept {
  MutableTensorView view{const_cast<void*>(input.data),
                         input.dtype,
                         input.device,
                         input.layout,
                         input.rank,
                         input.shape,
                         input.strides};
  for (const auto& target : storage)
    if (block_arrays_overlap({view, target})) return true;
  return false;
}
}  // namespace
