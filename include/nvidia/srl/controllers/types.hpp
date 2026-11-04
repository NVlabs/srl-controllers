/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file types.hpp
 * @brief Platform-neutral, non-owning tensor types for controllers.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#if defined(NVIDIA_SRL_CONTROLLERS_BUILDING_LIBRARY)
#define NVIDIA_SRL_CONTROLLERS_API __declspec(dllexport)
#else
#define NVIDIA_SRL_CONTROLLERS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define NVIDIA_SRL_CONTROLLERS_API __attribute__((visibility("default")))
#else
#define NVIDIA_SRL_CONTROLLERS_API
#endif

namespace nvidia {
namespace srl {
namespace controllers {

constexpr int kMaxTensorRank = 4;

enum class DType {
  kFloat32,
  kFloat64,
  kBool,
  kUInt64,
  kUnknown,
};

enum class Device {
  kCpu,
  kCuda,
  kUnknown,
};

enum class Layout {
  kRowMajor,
  kStrided,
  kUnknown,
};

enum class ResultCode {
  kOk = 0,
  kInvalidArgument = 1,
  kInvalidView = 2,
  kUnsupported = 3,
  kBackendUnavailable = 4,
};

enum class ComponentMode {
  kDiagonal,
  kDense,
  kUnsupported,
};

enum class StatusFlag : std::uint64_t {
  kNone = 0,
  kUnsupportedMode = 1ULL << 0,
  kInvalidParameter = 1ULL << 1,
  kInvalidInput = 1ULL << 2,
  kNonFiniteInput = 1ULL << 3,
  kInvalidDt = 1ULL << 4,
  kSaturation = 1ULL << 5,
  kResetApplied = 1ULL << 6,
  kSafeMode = 1ULL << 7,
  kOutputHeld = 1ULL << 8,
  kNoValidOutput = 1ULL << 9,
  kHoldExpired = 1ULL << 10,
};

enum class SafeOutputPolicy {
  kZero,
  kHoldLastValid,
};

enum class ParameterBatching {
  kShared,
  kPerBatch,
};

enum class CartesianFrame {
  kUnspecified,
  kRobotBase,
  kEndEffector,
  /** Task origins are unchanged; only task axes follow a configured main-frame rotation. */
  kMainFrameOrientation,
};

enum class BimanualBase {
  kArm1,
  kArm2,
};

enum class BimanualMainFrame {
  kArm1Base,
  kArm2Base,
  kExplicit,
};

/**
 * @brief Coordinate construction used by bimanual task kinematics.
 *
 * Decoupled cooperative coordinates express relative translation in the shared base axes.
 * Coupled rigid-transform coordinates express the complete relative pose in arm 1's TCP frame.
 */
enum class BimanualCoordinationMode {
  kDecoupledCooperative,
  kCoupledRigidTransform,
};

/**
 * @brief Fixed calibration used by bimanual task kinematics.
 *
 * Transforms are row-major homogeneous matrices with translation in metres. `arm1_T_arm2` maps
 * coordinates from arm 2's base into arm 1's base. For an explicit main frame,
 * `reference_T_main` locates the main frame relative to `explicit_reference`; only its rotation is
 * used to orient task coordinates and stiffness axes.
 */
struct BimanualFrameConfig {
  std::array<double, 16> arm1_T_arm2{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  BimanualMainFrame main_frame{BimanualMainFrame::kArm1Base};
  BimanualBase explicit_reference{BimanualBase::kArm1};
  std::array<double, 16> reference_T_main{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
};

struct StepOptions {
  SafeOutputPolicy safe_output_policy{SafeOutputPolicy::kZero};
  std::uint64_t max_hold_steps{0};
};

struct [[nodiscard]] Result {
  ResultCode code{ResultCode::kOk};
  const char* message{"ok"};

  bool ok() const noexcept { return code == ResultCode::kOk; }
};

/**
 * @brief Per-call execution context.
 *
 * CPU execution ignores cuda_stream. CUDA work is enqueued on the caller-owned stream and never
 * synchronizes implicitly. The context is deliberately not retained by controllers.
 */
struct ExecutionContext {
  void* cuda_stream{nullptr};
};

struct TensorView {
  const void* data{nullptr};
  DType dtype{DType::kUnknown};
  Device device{Device::kUnknown};
  Layout layout{Layout::kUnknown};
  int rank{0};
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::array<std::int64_t, kMaxTensorRank> strides{};
};

struct MutableTensorView {
  void* data{nullptr};
  DType dtype{DType::kUnknown};
  Device device{Device::kUnknown};
  Layout layout{Layout::kUnknown};
  int rank{0};
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::array<std::int64_t, kMaxTensorRank> strides{};

  TensorView as_const() const noexcept {
    return TensorView{data, dtype, device, layout, rank, shape, strides};
  }
};

struct StatusView {
  const std::uint64_t* flags{nullptr};
  std::size_t size{0};
  Device device{Device::kUnknown};
};

struct MutableStatusView {
  std::uint64_t* flags{nullptr};
  std::size_t size{0};
  Device device{Device::kUnknown};
};

struct SafetyStateView {
  MutableTensorView last_valid_output{};
  MutableTensorView valid_mask{};
  MutableTensorView hold_count{};
};

/** @brief A typed parameter tensor with explicit shared or per-environment batching. */
struct ParameterTensor {
  TensorView values{};
  ParameterBatching batching{ParameterBatching::kPerBatch};
};

struct ControllerConfig {
  Device device{Device::kCpu};
  DType dtype{DType::kFloat64};
  std::size_t batch_size{0};
  std::size_t joint_dim{0};
  StepOptions safety{};
  CartesianFrame cartesian_frame{CartesianFrame::kUnspecified};
  /** Task-space row count. Existing Cartesian controllers default to six. */
  std::size_t task_dim{6};
};

/** @brief Description of one caller-owned contiguous row-major array. */
struct TensorRequirement {
  const char* name{nullptr};
  DType dtype{DType::kUnknown};
  Device device{Device::kUnknown};
  int rank{0};
  std::array<std::int64_t, kMaxTensorRank> shape{};
  const char* units{nullptr};
  const char* frame{nullptr};
};

/**
 * @brief Storage description returned before preparation.
 *
 * outputs contains the command plus parameter and runtime status arrays. state contains
 * controller history followed by safety history when the controller emits a protected command.
 * scratch contains temporary arrays whose contents do not persist semantically between steps.
 */
struct Requirements {
  static constexpr std::size_t kStateCapacity = 32;
  static constexpr std::size_t kScratchCapacity = 32;
  static constexpr std::size_t kOutputCapacity = 4;
  std::size_t batch_size{0};
  std::size_t joint_dim{0};
  std::array<TensorRequirement, kStateCapacity> state{};
  std::size_t state_count{0};
  std::array<TensorRequirement, kScratchCapacity> scratch{};
  std::size_t scratch_count{0};
  std::array<TensorRequirement, kOutputCapacity> outputs{};
  std::size_t output_count{0};
};

inline std::uint64_t status_bit(const StatusFlag flag) noexcept {
  return static_cast<std::uint64_t>(flag);
}

inline bool has_status_flag(const std::uint64_t status, const StatusFlag flag) noexcept {
  return (status & status_bit(flag)) != 0U;
}

inline void set_status_flag(std::uint64_t* status, const StatusFlag flag) noexcept {
  *status |= status_bit(flag);
}

template <typename T>
DType dtype_for();

template <>
inline DType dtype_for<float>() {
  return DType::kFloat32;
}

template <>
inline DType dtype_for<double>() {
  return DType::kFloat64;
}

template <>
inline DType dtype_for<std::uint8_t>() {
  return DType::kBool;
}

template <>
inline DType dtype_for<std::uint64_t>() {
  return DType::kUInt64;
}

template <typename T>
TensorView make_tensor_view(const T* data,
                            const Device device,
                            const Layout layout,
                            const int rank,
                            const std::array<std::int64_t, kMaxTensorRank>& shape,
                            const std::array<std::int64_t, kMaxTensorRank>& strides) noexcept {
  return TensorView{data, dtype_for<T>(), device, layout, rank, shape, strides};
}

template <typename T>
MutableTensorView make_mutable_tensor_view(
    T* data,
    const Device device,
    const Layout layout,
    const int rank,
    const std::array<std::int64_t, kMaxTensorRank>& shape,
    const std::array<std::int64_t, kMaxTensorRank>& strides) noexcept {
  return MutableTensorView{data, dtype_for<T>(), device, layout, rank, shape, strides};
}

template <typename T, std::size_t Rank>
TensorView make_contiguous_tensor_view(const T* data,
                                       const Device device,
                                       const std::array<std::int64_t, Rank>& extents) noexcept {
  static_assert(Rank <= static_cast<std::size_t>(kMaxTensorRank), "tensor rank exceeds API limit");
  std::array<std::int64_t, kMaxTensorRank> shape{};
  std::array<std::int64_t, kMaxTensorRank> strides{};
  std::int64_t stride = 1;
  for (std::size_t reverse = 0; reverse < Rank; ++reverse) {
    const std::size_t dimension = Rank - reverse - 1;
    const std::int64_t extent = extents[dimension];
    if (extent < 0 || (extent != 0 && stride > std::numeric_limits<std::int64_t>::max() / extent)) {
      return TensorView{};
    }
    shape[dimension] = extent;
    strides[dimension] = stride;
    stride *= extent;
  }
  if (static_cast<std::uintmax_t>(stride) > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return TensorView{};
  }
  return make_tensor_view(data, device, Layout::kRowMajor, static_cast<int>(Rank), shape, strides);
}

template <typename T, std::size_t Rank>
MutableTensorView make_contiguous_tensor_view(
    T* data, const Device device, const std::array<std::int64_t, Rank>& extents) noexcept {
  const TensorView view = make_contiguous_tensor_view(static_cast<const T*>(data), device, extents);
  if (view.dtype == DType::kUnknown) {
    return MutableTensorView{};
  }
  return MutableTensorView{
      data, view.dtype, view.device, view.layout, view.rank, view.shape, view.strides};
}

namespace view_detail {

template <typename Extent>
std::int64_t checked_extent(const Extent extent) noexcept {
  if constexpr (std::is_signed<Extent>::value) {
    if (extent < 0) {
      return -1;
    }
  }
  if (static_cast<std::uintmax_t>(extent) >
      static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max())) {
    return -1;
  }
  return static_cast<std::int64_t>(extent);
}

template <typename... Extents>
std::array<std::int64_t, sizeof...(Extents)> extents_array(Extents... extents) noexcept {
  static_assert(sizeof...(Extents) <= static_cast<std::size_t>(kMaxTensorRank),
                "tensor rank exceeds API limit");
  static_assert((std::is_integral<Extents>::value && ...), "tensor extents must be integral");
  return {checked_extent(extents)...};
}

template <std::size_t Rank>
bool element_count(const std::array<std::int64_t, Rank>& extents, std::size_t* count) noexcept {
  std::size_t product = 1;
  for (std::size_t reverse = 0; reverse < Rank; ++reverse) {
    const std::int64_t extent = extents[Rank - reverse - 1];
    if (extent < 0 ||
        static_cast<std::uintmax_t>(extent) > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    const std::size_t size = static_cast<std::size_t>(extent);
    constexpr std::size_t kMaximumStride =
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    if (size != 0 && product > kMaximumStride / size) {
      return false;
    }
    product *= size;
  }
  *count = product;
  return true;
}

template <typename Container>
using ContainerElement = typename std::remove_cv<
    typename std::remove_pointer<decltype(std::declval<Container&>().data())>::type>::type;

}  // namespace view_detail

/** @brief Construct a read-only contiguous view over raw caller-owned memory. */
template <typename T, typename... Extents>
TensorView read_view(const T* data, const Device device, Extents... extents) noexcept {
  const auto shape = view_detail::extents_array(extents...);
  std::size_t ignored = 0;
  if (!view_detail::element_count(shape, &ignored)) {
    return TensorView{};
  }
  return make_contiguous_tensor_view(data, device, shape);
}

template <typename T, std::size_t Rank>
TensorView read_view(const T* data,
                     const Device device,
                     const std::array<std::int64_t, Rank>& extents) noexcept {
  std::size_t ignored = 0;
  if (!view_detail::element_count(extents, &ignored)) {
    return TensorView{};
  }
  return make_contiguous_tensor_view(data, device, extents);
}

/** @brief Construct a writable contiguous view over raw caller-owned memory. */
template <typename T, typename... Extents>
MutableTensorView write_view(T* data, const Device device, Extents... extents) noexcept {
  const auto shape = view_detail::extents_array(extents...);
  std::size_t ignored = 0;
  if (!view_detail::element_count(shape, &ignored)) {
    return MutableTensorView{};
  }
  return make_contiguous_tensor_view(data, device, shape);
}

template <typename T, std::size_t Rank>
MutableTensorView write_view(T* data,
                             const Device device,
                             const std::array<std::int64_t, Rank>& extents) noexcept {
  std::size_t ignored = 0;
  if (!view_detail::element_count(extents, &ignored)) {
    return MutableTensorView{};
  }
  return make_contiguous_tensor_view(data, device, extents);
}

/**
 * @brief Construct a checked read-only view over a contiguous container.
 *
 * A size mismatch returns an empty view so normal controller validation fails closed.
 */
template <typename Container, typename... Extents>
auto read_view(const Container& values, const Device device, Extents... extents) noexcept
    -> decltype(values.data(), values.size(), TensorView{}) {
  const auto shape = view_detail::extents_array(extents...);
  std::size_t count = 0;
  if (!view_detail::element_count(shape, &count) || count != values.size()) {
    return TensorView{};
  }
  return make_contiguous_tensor_view(values.data(), device, shape);
}

template <typename Container, typename... Extents>
auto read_view(const Container&& values, const Device, Extents...) noexcept
    -> decltype(values.data(), values.size(), TensorView{}) = delete;

template <typename Container, std::size_t Rank>
auto read_view(const Container& values,
               const Device device,
               const std::array<std::int64_t, Rank>& extents) noexcept -> decltype(values.data(),
                                                                                   values.size(),
                                                                                   TensorView{}) {
  std::size_t count = 0;
  if (!view_detail::element_count(extents, &count) || count != values.size()) {
    return TensorView{};
  }
  return make_contiguous_tensor_view(values.data(), device, extents);
}

template <typename Container, std::size_t Rank>
auto read_view(const Container&& values,
               const Device,
               const std::array<std::int64_t, Rank>&) noexcept -> decltype(values.data(),
                                                                           values.size(),
                                                                           TensorView{}) = delete;

/**
 * @brief Construct a checked writable view over a contiguous container.
 *
 * A size mismatch returns an empty view so normal controller validation fails closed.
 */
template <typename Container, typename... Extents>
auto write_view(Container& values, const Device device, Extents... extents) noexcept ->
    typename std::enable_if<!std::is_const<Container>::value,
                            decltype(values.data(), values.size(), MutableTensorView{})>::type {
  const auto shape = view_detail::extents_array(extents...);
  std::size_t count = 0;
  if (!view_detail::element_count(shape, &count) || count != values.size()) {
    return MutableTensorView{};
  }
  return make_contiguous_tensor_view(values.data(), device, shape);
}

template <typename Container, std::size_t Rank>
auto write_view(Container& values,
                const Device device,
                const std::array<std::int64_t, Rank>& extents) noexcept ->
    typename std::enable_if<!std::is_const<Container>::value,
                            decltype(values.data(), values.size(), MutableTensorView{})>::type {
  std::size_t count = 0;
  if (!view_detail::element_count(extents, &count) || count != values.size()) {
    return MutableTensorView{};
  }
  return make_contiguous_tensor_view(values.data(), device, extents);
}

/** @brief Construct a mutable per-row status view over raw caller-owned memory. */
inline MutableStatusView write_status(std::uint64_t* flags,
                                      const std::size_t size,
                                      const Device device) noexcept {
  return MutableStatusView{flags, size, device};
}

/** @brief Construct a mutable per-row status view over a contiguous container. */
template <typename Container>
MutableStatusView write_status(Container& flags, const Device device) noexcept {
  static_assert(
      std::is_same<typename view_detail::ContainerElement<Container>, std::uint64_t>::value,
      "status containers must contain uint64_t values");
  return MutableStatusView{flags.data(), flags.size(), device};
}

/** @brief Mark a parameter view as shared by every batch row. */
inline ParameterTensor shared(const TensorView values) noexcept {
  return ParameterTensor{values, ParameterBatching::kShared};
}

/** @brief Mark a parameter view as independently batched by row. */
inline ParameterTensor per_batch(const TensorView values) noexcept {
  return ParameterTensor{values, ParameterBatching::kPerBatch};
}

/** @brief Group the caller-owned bounded safe-output history. */
inline SafetyStateView make_safety_state(const MutableTensorView last_valid_output,
                                         const MutableTensorView valid_mask,
                                         const MutableTensorView hold_count) noexcept {
  return SafetyStateView{last_valid_output, valid_mask, hold_count};
}

}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
