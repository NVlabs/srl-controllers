/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(NVIDIASRLCONTROLLERS_ENABLE_CUDA)
#error Allocation coverage must use the CPU-only test build.
#endif

namespace allocation_test {

enum class Route : std::size_t {
  kMalloc,
  kCalloc,
  kRealloc,
  kAlignedAlloc,
  kPosixMemalign,
  kNew,
  kNewArray,
  kNothrowNew,
  kNothrowNewArray,
  kAlignedNew,
  kAlignedNewArray,
  kAlignedNothrowNew,
  kAlignedNothrowNewArray,
  kCount
};

constexpr std::size_t ROUTE_COUNT = static_cast<std::size_t>(Route::kCount);
constexpr std::size_t ALIGNMENT = 64;
void* allocate_for_calibration(Route route, std::size_t bytes, void* previous);
void release_calibration(Route route, void* allocation);

#if defined(NVIDIA_SRL_CONTROLLERS_ALLOCATION_CALIBRATION)

void* allocate_for_calibration(Route route, std::size_t bytes, void* previous) {
  switch (route) {
    case Route::kMalloc: {
      void* (*volatile allocate)(std::size_t) = std::malloc;
      return allocate(bytes);
    }
    case Route::kCalloc: {
      void* (*volatile allocate)(std::size_t, std::size_t) = std::calloc;
      return allocate(bytes, 1);
    }
    case Route::kRealloc: {
      void* (*volatile allocate)(void*, std::size_t) = std::realloc;
      return allocate(previous, bytes);
    }
    case Route::kAlignedAlloc: {
      void* (*volatile allocate)(std::size_t, std::size_t) = std::aligned_alloc;
      return allocate(ALIGNMENT, bytes);
    }
    case Route::kPosixMemalign: {
      int (*volatile allocate)(void**, std::size_t, std::size_t) = ::posix_memalign;
      void* allocation = nullptr;
      return allocate(&allocation, ALIGNMENT, bytes) == 0 ? allocation : nullptr;
    }
    case Route::kNew: {
      void* (*volatile allocate)(std::size_t) = ::operator new;
      return allocate(bytes);
    }
    case Route::kNewArray: {
      void* (*volatile allocate)(std::size_t) = ::operator new[];
      return allocate(bytes);
    }
    case Route::kNothrowNew: {
      void* (*volatile allocate)(std::size_t, const std::nothrow_t&) = ::operator new;
      return allocate(bytes, std::nothrow);
    }
    case Route::kNothrowNewArray: {
      void* (*volatile allocate)(std::size_t, const std::nothrow_t&) = ::operator new[];
      return allocate(bytes, std::nothrow);
    }
    case Route::kAlignedNew: {
      void* (*volatile allocate)(std::size_t, std::align_val_t) = ::operator new;
      return allocate(bytes, std::align_val_t{ALIGNMENT});
    }
    case Route::kAlignedNewArray: {
      void* (*volatile allocate)(std::size_t, std::align_val_t) = ::operator new[];
      return allocate(bytes, std::align_val_t{ALIGNMENT});
    }
    case Route::kAlignedNothrowNew: {
      void* (*volatile allocate)(std::size_t, std::align_val_t, const std::nothrow_t&) =
          ::operator new;
      return allocate(bytes, std::align_val_t{ALIGNMENT}, std::nothrow);
    }
    case Route::kAlignedNothrowNewArray: {
      void* (*volatile allocate)(std::size_t, std::align_val_t, const std::nothrow_t&) =
          ::operator new[];
      return allocate(bytes, std::align_val_t{ALIGNMENT}, std::nothrow);
    }
    case Route::kCount:
      return nullptr;
  }
  return nullptr;
}

void release_calibration(Route route, void* allocation) {
  switch (route) {
    case Route::kNew:
    case Route::kNothrowNew:
      ::operator delete(allocation);
      return;
    case Route::kNewArray:
    case Route::kNothrowNewArray:
      ::operator delete[](allocation);
      return;
    case Route::kAlignedNew:
    case Route::kAlignedNothrowNew:
      ::operator delete(allocation, std::align_val_t{ALIGNMENT});
      return;
    case Route::kAlignedNewArray:
    case Route::kAlignedNothrowNewArray:
      ::operator delete[](allocation, std::align_val_t{ALIGNMENT});
      return;
    default:
      std::free(allocation);
      return;
  }
}

#else

using Counts = std::array<std::size_t, ROUTE_COUNT>;
std::atomic<bool> is_guard_enabled{false};
std::array<std::atomic<std::size_t>, ROUTE_COUNT> allocation_counts{};

void record_allocation(Route route) noexcept {
  if (is_guard_enabled.load()) {
    allocation_counts[static_cast<std::size_t>(route)].fetch_add(1);
  }
}

class Guard {
 public:
  Guard() {
    for (auto& count : allocation_counts) {
      count.store(0);
    }
    is_guard_enabled.store(true);
  }
  ~Guard() { is_guard_enabled.store(false); }
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
  Counts finish() {
    is_guard_enabled.store(false);
    Counts counts{};
    for (std::size_t route = 0; route < ROUTE_COUNT; ++route) {
      counts[route] = allocation_counts[route].load();
    }
    return counts;
  }
};

#endif

}  // namespace allocation_test

#if !defined(NVIDIA_SRL_CONTROLLERS_ALLOCATION_CALIBRATION)

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

#include "./controller_test_support.hpp"
#include "nvidia/srl/controllers/controllers.hpp"

#define WRAP_ALLOCATION(NAME, SYMBOL, ROUTE, RETURN_TYPE, PARAMETERS, ARGUMENTS) \
  extern "C" RETURN_TYPE real_##NAME PARAMETERS asm("__real_" SYMBOL);           \
  extern "C" RETURN_TYPE wrap_##NAME PARAMETERS asm("__wrap_" SYMBOL);           \
  extern "C" RETURN_TYPE wrap_##NAME PARAMETERS {                                \
    allocation_test::record_allocation(allocation_test::Route::ROUTE);           \
    return real_##NAME ARGUMENTS;                                                \
  }

WRAP_ALLOCATION(malloc, "malloc", kMalloc, void*, (std::size_t bytes), (bytes))
WRAP_ALLOCATION(
    calloc, "calloc", kCalloc, void*, (std::size_t count, std::size_t bytes), (count, bytes))
WRAP_ALLOCATION(
    realloc, "realloc", kRealloc, void*, (void* previous, std::size_t bytes), (previous, bytes))
WRAP_ALLOCATION(aligned_alloc,
                "aligned_alloc",
                kAlignedAlloc,
                void*,
                (std::size_t alignment, std::size_t bytes),
                (alignment, bytes))
WRAP_ALLOCATION(posix_memalign,
                "posix_memalign",
                kPosixMemalign,
                int,
                (void** allocation, std::size_t alignment, std::size_t bytes),
                (allocation, alignment, bytes))

#if __SIZEOF_SIZE_T__ == 8
#define SIZE_MANGLE "m"
#else
#define SIZE_MANGLE "j"
#endif

WRAP_ALLOCATION(new, "_Znw" SIZE_MANGLE, kNew, void*, (std::size_t bytes), (bytes))
WRAP_ALLOCATION(new_array, "_Zna" SIZE_MANGLE, kNewArray, void*, (std::size_t bytes), (bytes))
WRAP_ALLOCATION(nothrow_new,
                "_Znw" SIZE_MANGLE "RKSt9nothrow_t",
                kNothrowNew,
                void*,
                (std::size_t bytes, const std::nothrow_t& tag),
                (bytes, tag))
WRAP_ALLOCATION(nothrow_new_array,
                "_Zna" SIZE_MANGLE "RKSt9nothrow_t",
                kNothrowNewArray,
                void*,
                (std::size_t bytes, const std::nothrow_t& tag),
                (bytes, tag))
WRAP_ALLOCATION(aligned_new,
                "_Znw" SIZE_MANGLE "St11align_val_t",
                kAlignedNew,
                void*,
                (std::size_t bytes, std::align_val_t alignment),
                (bytes, alignment))
WRAP_ALLOCATION(aligned_new_array,
                "_Zna" SIZE_MANGLE "St11align_val_t",
                kAlignedNewArray,
                void*,
                (std::size_t bytes, std::align_val_t alignment),
                (bytes, alignment))
WRAP_ALLOCATION(aligned_nothrow_new,
                "_Znw" SIZE_MANGLE "St11align_val_tRKSt9nothrow_t",
                kAlignedNothrowNew,
                void*,
                (std::size_t bytes, std::align_val_t alignment, const std::nothrow_t& tag),
                (bytes, alignment, tag))
WRAP_ALLOCATION(aligned_nothrow_new_array,
                "_Zna" SIZE_MANGLE "St11align_val_tRKSt9nothrow_t",
                kAlignedNothrowNewArray,
                void*,
                (std::size_t bytes, std::align_val_t alignment, const std::nothrow_t& tag),
                (bytes, alignment, tag))

#undef SIZE_MANGLE
#undef WRAP_ALLOCATION

namespace {

namespace controllers = nvidia::srl::controllers;
using allocation_test::Counts;
using allocation_test::Guard;
using allocation_test::Route;

class AllocationCalibrationTest : public ::testing::TestWithParam<Route> {};

TEST_P(AllocationCalibrationTest, DetectsObservableAllocationFromInstrumentedArchive) {
  constexpr std::size_t BYTES = 2 * allocation_test::ALIGNMENT;
  const Route route = GetParam();
  void* previous = route == Route::kRealloc ? std::malloc(BYTES / 2) : nullptr;
  ASSERT_TRUE(route != Route::kRealloc || previous != nullptr);
  Guard guard;
  void* allocation = allocation_test::allocate_for_calibration(route, BYTES, previous);
  if (allocation != nullptr) {
    static_cast<volatile unsigned char*>(allocation)[BYTES - 1] = 37;
  }
  const Counts counts = guard.finish();
  if (allocation == nullptr) {
    std::free(previous);
  }
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(static_cast<volatile unsigned char*>(allocation)[BYTES - 1], 37);
  allocation_test::release_calibration(route, allocation);
  EXPECT_EQ(counts[static_cast<std::size_t>(route)], 1U);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), std::size_t{0}), 1U);
}

INSTANTIATE_TEST_SUITE_P(CAndCppRoutes,
                         AllocationCalibrationTest,
                         ::testing::Values(Route::kMalloc,
                                           Route::kCalloc,
                                           Route::kRealloc,
                                           Route::kAlignedAlloc,
                                           Route::kPosixMemalign,
                                           Route::kNew,
                                           Route::kNewArray,
                                           Route::kNothrowNew,
                                           Route::kNothrowNewArray,
                                           Route::kAlignedNew,
                                           Route::kAlignedNewArray,
                                           Route::kAlignedNothrowNew,
                                           Route::kAlignedNothrowNewArray));

template <typename T, std::size_t Dimension = 2>
struct OutputFixture {
  std::array<T, 2 * Dimension> command{};
  std::array<T, 2 * Dimension> last_valid{};
  std::array<std::uint8_t, 2> valid{};
  std::array<std::uint64_t, 2> hold_count{};
  std::array<std::uint64_t, 2> parameter_status{};
  std::array<std::uint64_t, 2> status{};
  controllers::ControllerOutputStorage view() {
    return {controllers::write_view(command, controllers::Device::kCpu, 2, Dimension),
            controllers::write_status(parameter_status, controllers::Device::kCpu),
            controllers::write_status(status, controllers::Device::kCpu),
            controllers::make_safety_state(
                controllers::write_view(last_valid, controllers::Device::kCpu, 2, Dimension),
                controllers::write_view(valid, controllers::Device::kCpu, 2),
                controllers::write_view(hold_count, controllers::Device::kCpu, 2))};
  }
};

template <typename T>
controllers::ControllerConfig make_config(controllers::SafeOutputPolicy policy) {
  return {controllers::Device::kCpu,
          std::is_same_v<T, float> ? controllers::DType::kFloat32 : controllers::DType::kFloat64,
          2,
          2,
          {policy, 2}};
}

template <typename T>
struct JointFixture {
  OutputFixture<T> output;
  std::array<T, 4> stiffness{100, 2, 2, 120};
  std::array<T, 2> damping{1, 1};
  std::array<T, 4> position_error{T{0.1}, T{-0.1}, T{0.05}, T{0.2}};
  std::array<T, 4> velocity_error{T{0.2}, T{0.1}, T{-0.1}, T{0.05}};
  std::array<T, 4> acceleration{T{0.1}, T{0.2}, T{0.3}, T{0.4}};
  std::array<T, 8> mass{2, T{0.1}, T{0.1}, T{1.5}, 2, T{0.1}, T{0.1}, T{1.5}};
  controllers::JointImpedanceController controller;
  controllers::Result prepare(controllers::SafeOutputPolicy policy) {
    return controller.prepare(
        make_config<T>(policy),
        {controllers::shared(controllers::read_view(stiffness, controllers::Device::kCpu, 2, 2)),
         controllers::shared(controllers::read_view(damping, controllers::Device::kCpu, 2))},
        output.view());
  }
  void set_invalid(bool invalid) {
    position_error[0] = invalid ? std::numeric_limits<T>::quiet_NaN() : T{0.1};
  }
  controllers::Result step() {
    return controller.step({controllers::read_view(position_error, controllers::Device::kCpu, 2, 2),
                            controllers::read_view(velocity_error, controllers::Device::kCpu, 2, 2),
                            controllers::read_view(acceleration, controllers::Device::kCpu, 2, 2),
                            controllers::read_view(mass, controllers::Device::kCpu, 2, 2, 2)});
  }
};

template <typename T>
struct FilterFixture {
  OutputFixture<T> output;
  std::array<T, 2> lower{-100, -100};
  std::array<T, 2> upper{100, 100};
  std::array<T, 2> rate{10, 10};
  std::array<T, 4> command{1, -2, 3, -4};
  std::array<T, 2> dt{T{0.01}, T{0.01}};
  std::array<T, 4> previous_command{};
  std::array<std::uint8_t, 2> initialized{};
  controllers::TorqueFilter controller;
  controllers::Result prepare(controllers::SafeOutputPolicy policy) {
    controllers::TorqueFilter::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    storage.state = {controllers::write_view(previous_command, controllers::Device::kCpu, 2, 2),
                     controllers::write_view(initialized, controllers::Device::kCpu, 2)};
    return controller.prepare(
        make_config<T>(policy),
        {controllers::shared(controllers::read_view(lower, controllers::Device::kCpu, 2)),
         controllers::shared(controllers::read_view(upper, controllers::Device::kCpu, 2)),
         controllers::shared(controllers::read_view(rate, controllers::Device::kCpu, 2))},
        storage);
  }
  void set_invalid(bool invalid) {
    command[0] = invalid ? std::numeric_limits<T>::quiet_NaN() : T{1};
  }
  controllers::Result step() {
    return controller.step({controllers::read_view(command, controllers::Device::kCpu, 2, 2),
                            controllers::read_view(dt, controllers::Device::kCpu, 2)});
  }
};

template <typename T, std::size_t Dimension, typename Controller, typename Config>
struct StaticFixture {
  std::array<T, Dimension> gains;
  std::array<T, 2 * Dimension> error;
  std::array<T, 2 * Dimension> velocity{};
  std::array<T, Dimension * Dimension> stiffness{};
  std::array<T, Dimension * Dimension> damping{};
  std::array<T, 2 * Dimension> command{};
  std::array<T, 2 * Dimension> history{};
  std::array<std::uint64_t, 2> status{};
  std::array<std::uint64_t, 2> parameter_status{};
  std::array<std::uint64_t, 2> hold_count{};
  std::array<std::uint8_t, 2> valid{};
  Controller controller;
  controllers::Result prepare(controllers::SafeOutputPolicy policy) {
    gains.fill(T{1});
    error.fill(T{0.1});
    for (std::size_t i = 0; i < Dimension; ++i) stiffness[i * Dimension + i] = T{1};
    Config config;
    config.dtype = controllers::dtype_for<T>();
    config.batch_size = 2;
    config.joint_dim = Dimension;
    config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
    config.safety = {policy, 2};
    controllers::ControllerOutputStorage storage{
        controllers::write_view(command, controllers::Device::kCpu, 2, Dimension),
        {parameter_status.data(), 2, controllers::Device::kCpu},
        {status.data(), 2, controllers::Device::kCpu},
        {controllers::write_view(history, controllers::Device::kCpu, 2, Dimension),
         controllers::write_view(valid, controllers::Device::kCpu, 2),
         controllers::write_view(hold_count, controllers::Device::kCpu, 2)}};
    if constexpr (std::is_same<Controller, controllers::StaticJointImpedanceController>::value) {
      config.mode = controllers::ComponentMode::kDiagonal;
      const auto parameter =
          controllers::shared(controllers::read_view(gains, controllers::Device::kCpu, Dimension));
      return controller.prepare(config, {parameter, parameter}, storage);
    } else {
      return controller.prepare(config,
                                {controllers::shared(controllers::read_view(
                                     stiffness, controllers::Device::kCpu, Dimension, Dimension)),
                                 controllers::shared(controllers::read_view(
                                     damping, controllers::Device::kCpu, Dimension, Dimension))},
                                storage);
    }
  }
  void set_invalid(bool invalid) {
    error[0] = invalid ? std::numeric_limits<T>::quiet_NaN() : T{0.1};
  }
  controllers::Result step() {
    return controller.step(
        {controllers::read_view(error.data(), controllers::Device::kCpu, 2, Dimension),
         controllers::read_view(velocity.data(), controllers::Device::kCpu, 2, Dimension)});
  }
};

template <typename T, bool Unified>
struct AdmittanceFixture {
  using Controller = std::conditional_t<Unified,
                                        controllers::CartesianUnifiedForceAdmittanceController,
                                        controllers::CartesianAdmittancePoseController>;
  OutputFixture<T, 6> output;
  std::array<std::array<T, 12>, 7> state{};
  std::array<std::array<std::uint8_t, 2>, 4> initialized{};
  std::array<T, 36> zero_matrix{};
  std::array<T, 36> mass{};
  std::array<T, 6> zero{};
  std::array<T, 6> gain{};
  std::array<T, 12> external{};
  std::array<T, 12> desired{};
  std::array<T, 2> dt{T{0.01}, T{0.01}};
  Controller controller;
  controllers::Result prepare(controllers::SafeOutputPolicy policy) {
    external.fill(T{1});
    gain.fill(T{1});
    for (std::size_t i = 0; i < 6; ++i) mass[i * 6 + i] = T{1};
    controllers::ControllerConfig config = make_config<T>(policy);
    config.joint_dim = 7;
    config.cartesian_frame = controllers::CartesianFrame::kRobotBase;
    typename Controller::Storage storage;
    static_cast<controllers::ControllerOutputStorage&>(storage) = output.view();
    const auto signal = [this](std::size_t index) {
      return controllers::write_view(state[index], controllers::Device::kCpu, 2, 6);
    };
    const auto flag = [this](std::size_t index) {
      return controllers::write_view(initialized[index], controllers::Device::kCpu, 2);
    };
    const controllers::AdmittanceStateStorage admittance{
        signal(0), signal(1), signal(2), signal(3), flag(0), flag(1)};
    const auto zero_parameter =
        controllers::shared(controllers::read_view(zero, controllers::Device::kCpu, 6));
    const auto zero_matrix_parameter =
        controllers::shared(controllers::read_view(zero_matrix, controllers::Device::kCpu, 6, 6));
    const auto mass_parameter =
        controllers::shared(controllers::read_view(mass, controllers::Device::kCpu, 6, 6));
    if constexpr (Unified) {
      storage.state = {{signal(4), flag(2), signal(5), signal(6), flag(3)}, admittance};
      const auto gain_parameter =
          controllers::shared(controllers::read_view(gain, controllers::Device::kCpu, 6));
      return controller.prepare(config,
                                {zero_matrix_parameter,
                                 zero_matrix_parameter,
                                 mass_parameter,
                                 {},
                                 {},
                                 {},
                                 {gain_parameter,
                                  zero_parameter,
                                  zero_parameter,
                                  zero_parameter,
                                  {},
                                  {},
                                  zero_parameter}},
                                storage);
    } else {
      storage.state = admittance;
      return controller.prepare(config,
                                {zero_matrix_parameter,
                                 zero_matrix_parameter,
                                 mass_parameter,
                                 zero_parameter,
                                 {},
                                 {},
                                 {}},
                                storage);
    }
  }
  void set_invalid(bool invalid) {
    external[0] = invalid ? std::numeric_limits<T>::quiet_NaN() : T{1};
  }
  controllers::Result step() {
    const auto input = controllers::read_view(external, controllers::Device::kCpu, 2, 6);
    const auto zero_input = controllers::read_view(desired, controllers::Device::kCpu, 2, 6);
    const auto time = controllers::read_view(dt, controllers::Device::kCpu, 2);
    if constexpr (Unified)
      return controller.step({zero_input, input, zero_input, time});
    else
      return controller.step({input, zero_input, time});
  }
};

template <typename T>
struct DifferentiatorFixture {
  OutputFixture<T> output;
  std::array<T, 4> previous{};
  std::array<std::uint8_t, 2> initialized{};
  std::array<T, 4> input{1, 1, 1, 1};
  std::array<T, 2> dt{T{0.01}, T{0.01}};
  controllers::DifferentiatorBlock controller;
  controllers::Result prepare(controllers::SafeOutputPolicy policy) {
    controllers::SignalBlockConfig config;
    config.dtype = controllers::dtype_for<T>();
    config.batch_size = 2;
    config.signal_dim = 2;
    config.safety = {policy, 2};
    controllers::DifferentiatorBlock::Storage storage{
        output.view(),
        {controllers::write_view(previous, controllers::Device::kCpu, 2, 2),
         controllers::write_view(initialized, controllers::Device::kCpu, 2)},
        {}};
    return controller.prepare(config, {}, storage);
  }
  void set_invalid(bool invalid) {
    input[0] = invalid ? std::numeric_limits<T>::quiet_NaN() : T{1};
  }
  controllers::Result step() {
    input[0] += T{1};
    input[2] += T{1};
    return controller.step({controllers::read_view(input, controllers::Device::kCpu, 2, 2),
                            controllers::read_view(dt, controllers::Device::kCpu, 2)});
  }
};

template <typename Fixture>
void check_guarded_steps(Fixture& fixture) {
  Guard guard;
  bool all_steps_ok = true;
  for (int step = 0; step < 64; ++step) {
    all_steps_ok = fixture.step().ok() && all_steps_ok;
  }
  const Counts counts = guard.finish();
  EXPECT_TRUE(all_steps_ok);
  EXPECT_EQ(counts, Counts{});
}

template <typename Fixture>
void check_normal_safe_and_recovery(controllers::SafeOutputPolicy policy) {
  Fixture fixture;
  ASSERT_TRUE(fixture.prepare(policy).ok());
  ASSERT_TRUE(fixture.controller.reset().ok());
  check_guarded_steps(fixture);
  EXPECT_FALSE(
      controllers::has_status_flag(fixture.output.status[0], controllers::StatusFlag::kSafeMode));
  fixture.set_invalid(true);
  check_guarded_steps(fixture);
  EXPECT_TRUE(
      controllers::has_status_flag(fixture.output.status[0], controllers::StatusFlag::kSafeMode));
  EXPECT_FALSE(
      controllers::has_status_flag(fixture.output.status[1], controllers::StatusFlag::kSafeMode));
  fixture.set_invalid(false);
  check_guarded_steps(fixture);
  EXPECT_FALSE(
      controllers::has_status_flag(fixture.output.status[0], controllers::StatusFlag::kSafeMode));
  EXPECT_NE(fixture.output.command[0], 0);
  EXPECT_NE(fixture.output.command[2], 0);
}

template <typename T, typename Controller>
void check_composite_allocation() {
  for (const std::size_t batch : {1U, 3U}) {
    for (const auto policy :
         {controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid}) {
      controllers::test_support::CompositeFixture<T, Controller> fixture(batch);
      fixture.config.safety = {policy, 2};
      auto& status = fixture.buffers.template values<std::uint64_t>("status");
      auto& command = fixture.buffers.template values<T>("filtered_command");
      ASSERT_TRUE(fixture.prepare().ok());
      ASSERT_TRUE(fixture.reset().ok());
      check_guarded_steps(fixture);
      EXPECT_FALSE(controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode));
      fixture.velocity_error[0] = std::numeric_limits<T>::quiet_NaN();
      check_guarded_steps(fixture);
      EXPECT_TRUE(controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode));
      if (batch > 1) {
        EXPECT_FALSE(controllers::has_status_flag(status[1], controllers::StatusFlag::kSafeMode));
      }
      fixture.velocity_error[0] = T{0.01};
      check_guarded_steps(fixture);
      EXPECT_FALSE(controllers::has_status_flag(status[0], controllers::StatusFlag::kSafeMode));
      EXPECT_NE(command[0], 0);
    }
  }
}

template <typename T>
class AllocationGuardTest : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(AllocationGuardTest, FloatingTypes);

TYPED_TEST(AllocationGuardTest, JointImpedanceNormalSafeOutputAndRecovery) {
  for (const auto policy :
       {controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid}) {
    check_normal_safe_and_recovery<JointFixture<TypeParam>>(policy);
  }
}

TYPED_TEST(AllocationGuardTest, TorqueFilterNormalSafeOutputAndRecovery) {
  for (const auto policy :
       {controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid}) {
    check_normal_safe_and_recovery<FilterFixture<TypeParam>>(policy);
  }
}

TYPED_TEST(AllocationGuardTest, StaticControllersNormalSafeOutputAndRecovery) {
  for (const auto policy :
       {controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid}) {
    using Joint = StaticFixture<TypeParam,
                                64,
                                controllers::StaticJointImpedanceController,
                                controllers::JointBlockConfig>;
    using Cartesian = StaticFixture<TypeParam,
                                    6,
                                    controllers::StaticCartesianImpedanceController,
                                    controllers::ControllerConfig>;
    const auto check = [policy](auto& fixture) {
      ASSERT_TRUE(fixture.prepare(policy).ok());
      ASSERT_TRUE(fixture.controller.reset().ok());
      check_guarded_steps(fixture);
      fixture.set_invalid(true);
      check_guarded_steps(fixture);
      EXPECT_TRUE(
          controllers::has_status_flag(fixture.status[0], controllers::StatusFlag::kSafeMode));
      EXPECT_FALSE(
          controllers::has_status_flag(fixture.status[1], controllers::StatusFlag::kSafeMode));
      fixture.set_invalid(false);
      check_guarded_steps(fixture);
      EXPECT_FALSE(
          controllers::has_status_flag(fixture.status[0], controllers::StatusFlag::kSafeMode));
    };
    Joint joint;
    Cartesian cartesian;
    check(joint);
    check(cartesian);
  }
}

TYPED_TEST(AllocationGuardTest, AdmittanceAndGeneratedBlockFirstFaultAndRecovery) {
  for (const auto policy :
       {controllers::SafeOutputPolicy::kZero, controllers::SafeOutputPolicy::kHoldLastValid}) {
    check_normal_safe_and_recovery<AdmittanceFixture<TypeParam, false>>(policy);
    check_normal_safe_and_recovery<AdmittanceFixture<TypeParam, true>>(policy);
    check_normal_safe_and_recovery<DifferentiatorFixture<TypeParam>>(policy);
  }
}

TYPED_TEST(AllocationGuardTest, CartesianTorqueNormalSafeOutputAndRecovery) {
  check_composite_allocation<TypeParam, controllers::CartesianImpedanceTorqueController>();
}

TYPED_TEST(AllocationGuardTest, VariableCartesianTorqueNormalSafeOutputAndRecovery) {
  check_composite_allocation<TypeParam, controllers::VariableCartesianImpedanceTorqueController>();
}

TYPED_TEST(AllocationGuardTest, UnifiedForceTorqueNormalSafeOutputAndRecovery) {
  check_composite_allocation<TypeParam,
                             controllers::CartesianUnifiedForceImpedanceTorqueController>();
}

TYPED_TEST(AllocationGuardTest, BimanualTorqueNormalSafeOutputAndRecovery) {
  check_composite_allocation<TypeParam, controllers::BimanualCartesianImpedanceTorqueController>();
}

}  // namespace

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
