/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file rt_reference_loop.cpp
 * @brief Runs an argument-free 1 kHz CPU controller reference loop.
 *
 * This synthetic process validates one preallocated tensor hot path and reports observed timing.
 * It neither commands a robot nor qualifies the host for hard real-time operation.
 */

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <exception>

#include "nvidia/srl/controllers/controllers.hpp"
#include "nvidia/srl/controllers/rt_bootstrap.hpp"

namespace {
namespace controllers = nvidia::srl::controllers;

constexpr int kFifoPriority = 80;
constexpr int kCpuId = 1;
constexpr std::size_t kJointCount = 7;
constexpr std::chrono::nanoseconds kPeriod{1'000'000};
constexpr std::size_t kIterations = 5'000;

std::atomic<bool> g_stop{false};

void on_shutdown() { g_stop.store(true, std::memory_order_relaxed); }

struct Storage {
  std::array<double, kJointCount> command{10.0, -10.0, 20.0, -20.0, 5.0, -5.0, 0.0};
  std::array<double, kJointCount> filtered{};
  std::array<double, kJointCount> lower_limit{};
  std::array<double, kJointCount> upper_limit{};
  std::array<double, kJointCount> maximum_rate{};
  std::array<double, kJointCount> previous_command{};
  std::array<std::uint8_t, 1> initialized{};
  std::array<double, kJointCount> last_valid_command{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};
  std::array<std::int64_t, kIterations> lateness_ns{};
  double dt_seconds{0.001};
};

}  // namespace

int main() {
  using nvidia::srl::controllers::rt::assert_applied;
  using nvidia::srl::controllers::rt::block_signals;
  using nvidia::srl::controllers::rt::leave_fifo;
  using nvidia::srl::controllers::rt::lock_memory;
  using nvidia::srl::controllers::rt::Loop;
  using nvidia::srl::controllers::rt::prefault_region;
  using nvidia::srl::controllers::rt::set_fifo;
  using nvidia::srl::controllers::rt::tune_malloc;

  tune_malloc();
  bool fifo_applied = false;
  try {
    block_signals(&on_shutdown);
    static Storage storage;
    storage.lower_limit.fill(-80.0);
    storage.upper_limit.fill(80.0);
    storage.maximum_rate.fill(1000.0);

    constexpr controllers::Device kDevice = controllers::Device::kCpu;
    const controllers::TorqueFilter::Inputs inputs{
        controllers::read_view(storage.command, kDevice, 1, kJointCount),
        controllers::read_view(&storage.dt_seconds, kDevice)};
    const controllers::TorqueFilter::Parameters parameters{
        controllers::shared(controllers::read_view(storage.lower_limit, kDevice, kJointCount)),
        controllers::shared(controllers::read_view(storage.upper_limit, kDevice, kJointCount)),
        controllers::shared(controllers::read_view(storage.maximum_rate, kDevice, kJointCount))};
    const controllers::TorqueFilter::Storage controller_storage{
        {controllers::write_view(storage.filtered, kDevice, 1, kJointCount),
         controllers::write_status(storage.parameter_status, kDevice),
         controllers::write_status(storage.status, kDevice),
         controllers::make_safety_state(
             controllers::write_view(storage.last_valid_command, kDevice, 1, kJointCount),
             controllers::write_view(storage.valid, kDevice, 1),
             controllers::write_view(storage.hold_count, kDevice, 1))},
        {controllers::write_view(storage.previous_command, kDevice, 1, kJointCount),
         controllers::write_view(storage.initialized, kDevice, 1)}};
    controllers::TorqueFilter controller;
    const controllers::ControllerConfig config{
        controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointCount, {}};
    const controllers::Result prepare_result =
        controller.prepare(config, parameters, controller_storage);
    if (!prepare_result.ok()) {
      std::fprintf(stderr, "Controller preparation failed: %s\n", prepare_result.message);
      return 1;
    }
    const controllers::Result reset_result = controller.reset();
    if (!reset_result.ok()) {
      std::fprintf(stderr, "Controller reset failed: %s\n", reset_result.message);
      return 1;
    }

    std::fprintf(stdout,
                 "Starting synthetic reference: period_ns=%" PRId64
                 " iterations=%zu fifo_priority=%d cpu=%d\n",
                 kPeriod.count(),
                 kIterations,
                 kFifoPriority,
                 kCpuId);
    std::fflush(stdout);
    prefault_region(&storage, sizeof(storage));
    lock_memory(0);
    set_fifo(kFifoPriority, kCpuId);
    fifo_applied = true;
    assert_applied(kFifoPriority, kCpuId);

    std::size_t completed = 0;
    std::size_t runtime_faults = 0;
    std::uint64_t skipped_periods = 0;
    Loop loop(kPeriod);
    for (; completed < kIterations && !g_stop.load(std::memory_order_relaxed); ++completed) {
      storage.lateness_ns[completed] = loop.sleep_until_next();
      skipped_periods += loop.last_missed_periods();
      const controllers::Result result = controller.step(inputs);
      if (!result.ok() ||
          controllers::has_status_flag(storage.status[0], controllers::StatusFlag::kSafeMode) ||
          loop.last_sleep_error() != 0) {
        ++runtime_faults;
      }
    }

    leave_fifo();
    fifo_applied = false;
    std::int64_t maximum_lateness_ns = 0;
    std::int64_t positive_lateness_sum_ns = 0;
    std::size_t deadline_misses = 0;
    for (std::size_t index = 0; index < completed; ++index) {
      const std::int64_t lateness = storage.lateness_ns[index];
      maximum_lateness_ns = lateness > maximum_lateness_ns ? lateness : maximum_lateness_ns;
      positive_lateness_sum_ns += lateness > 0 ? lateness : 0;
      deadline_misses += lateness > kPeriod.count() ? 1U : 0U;
    }
    const std::int64_t mean_lateness_ns =
        completed == 0 ? 0 : positive_lateness_sum_ns / static_cast<std::int64_t>(completed);
    std::fprintf(stdout,
                 "classification=synthetic_reference iters=%zu max_late_ns=%" PRId64
                 " mean_late_ns=%" PRId64 " deadline_misses=%zu skipped_periods=%" PRIu64
                 " runtime_faults=%zu qualification=none\n",
                 completed,
                 maximum_lateness_ns,
                 mean_lateness_ns,
                 deadline_misses,
                 skipped_periods,
                 runtime_faults);
    return deadline_misses == 0 && runtime_faults == 0 && completed == kIterations ? 0 : 2;
  } catch (const std::exception& error) {
    if (fifo_applied) {
      try {
        leave_fifo();
      } catch (const std::exception&) {
        // The process exits immediately below; retain the original setup error.
      }
    }
    std::fprintf(stderr,
                 "Reference loop setup failed: %s. No real-time qualification was performed.\n",
                 error.what());
    return 1;
  }
}
