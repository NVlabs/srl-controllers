/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file rt_bootstrap.hpp
 * @brief Canonical real-time process bootstrap helpers for PREEMPT_RT + SCHED_FIFO.
 *
 * Call order in main():
 *   1. rt::tune_malloc()          -- MUST be first statements before any allocation
 *   2. Ensure LD_BIND_NOW=1 in env, or link with -Wl,-z,now
 *   3. rt::block_signals()
 *   4. preallocate pools / open sockets / construct objects
 *   5. rt::lock_memory(prefault_bytes)
 *   6. spawn non-RT helper threads
 *   7. On the RT thread only: rt::set_fifo(prio, cpu_id)
 *   8. rt::assert_applied(prio, cpu_id) -- one-shot verification + log
 *   9. Enter control loop using rt::Loop
 */

#pragma once

#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace nvidia {
namespace srl {
namespace controllers {
namespace rt {

/**
 * @brief Tell glibc never to release the heap back to the kernel after init.
 *
 * Must be called as the first statements in main(), before any allocation.
 * Pair with MALLOC_ARENA_MAX=1 in the environment to prevent per-CPU arena proliferation.
 */
inline void tune_malloc() noexcept {
  mallopt(M_TRIM_THRESHOLD, -1);
  mallopt(M_MMAP_MAX, 0);
}

/**
 * @brief Write one byte per page in a caller-owned region before entering real-time execution.
 *
 * The region must remain alive for the RT session. This helper is allocation-free and noexcept.
 */
inline void prefault_region(void* data, const std::size_t bytes) noexcept {
  if (data == nullptr || bytes == 0) {
    return;
  }
  const std::int64_t page_size = static_cast<std::int64_t>(sysconf(_SC_PAGESIZE));
  if (page_size <= 0) {
    return;
  }
  auto* memory = static_cast<volatile std::uint8_t*>(data);
  const std::size_t page = static_cast<std::size_t>(page_size);
  for (std::size_t offset = 0; offset < bytes; offset += page) {
    memory[offset] = memory[offset];
  }
  memory[bytes - 1] = memory[bytes - 1];
}

/**
 * @brief Lock all memory in RAM and pre-fault a working buffer to force actual page commit.
 *
 * mlockall(MCL_FUTURE) does NOT pre-fault - it only guarantees future mappings lock when touched.
 * Callers MUST write one byte per 4 KiB across their preallocated pools; this helper additionally
 * allocates prefault_bytes, writes one byte per page, and frees it so that the RT process's
 * top-of-stack / recently-used pages are resident.
 *
 * @param prefault_bytes Size of the throwaway prefault buffer (bytes).
 * @throws std::runtime_error if mlockall fails.
 */
inline void lock_memory(std::size_t prefault_bytes) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    throw std::runtime_error(std::string("mlockall failed: ") + std::strerror(errno));
  }
  if (prefault_bytes == 0) {
    return;
  }
  const std::int64_t page_size = static_cast<std::int64_t>(sysconf(_SC_PAGESIZE));
  if (page_size <= 0) {
    throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
  }
  const std::size_t page = static_cast<std::size_t>(page_size);
  char* buffer = static_cast<char*>(std::malloc(prefault_bytes));
  if (buffer == nullptr) {
    throw std::runtime_error("prefault buffer allocation failed");
  }
  for (std::size_t i = 0; i < prefault_bytes; i += page) {
    buffer[i] = 1;
  }
  std::free(buffer);
}

/**
 * @brief Block signals on the calling thread. Spawn one dedicated non-RT signal-handler thread.
 *
 * The handler thread calls shutdown_cb on the first caught signal. shutdown_cb is invoked from
 * the handler thread context, not the RT thread, so it can safely allocate / log / etc.
 *
 * @param shutdown_cb Callback invoked on SIGINT/SIGTERM. May be nullptr (then default is abort).
 * @throws std::runtime_error on pthread_sigmask failure.
 */
inline void block_signals(void (*shutdown_cb)() = nullptr) {
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGTERM);
  sigaddset(&mask, SIGCHLD);
  sigaddset(&mask, SIGPIPE);
  const int mask_error = pthread_sigmask(SIG_BLOCK, &mask, nullptr);
  if (mask_error != 0) {
    throw std::runtime_error(std::string("pthread_sigmask failed: ") + std::strerror(mask_error));
  }
  std::thread([mask, shutdown_cb]() {
    for (;;) {
      int sig = 0;
      const int wait_error = sigwait(&mask, &sig);
      if (wait_error != 0) {
        std::fprintf(stderr, "[rt] sigwait failed: %s\n", std::strerror(wait_error));
        std::exit(EXIT_FAILURE);
      }
      if (sig == SIGINT || sig == SIGTERM) {
        break;
      }
    }
    if (shutdown_cb != nullptr) {
      shutdown_cb();
    } else {
      std::exit(0);
    }
  }).detach();
}

/**
 * @brief Apply SCHED_FIFO at the given priority and pin the calling thread to cpu_id.
 *
 * @param priority FIFO priority (1..99). 80 is typical for motion control loops.
 * @param cpu_id Target logical CPU (must be in the isolated set).
 * @throws std::runtime_error on failure (likely RLIMIT_RTPRIO not raised or permissions).
 */
inline void set_fifo(int priority, int cpu_id) {
  if (priority < 1 || priority > 99 || cpu_id < 0 || cpu_id >= CPU_SETSIZE) {
    throw std::invalid_argument("invalid FIFO priority or CPU id");
  }
  cpu_set_t previous_set;
  CPU_ZERO(&previous_set);
  const int get_affinity_error =
      pthread_getaffinity_np(pthread_self(), sizeof(previous_set), &previous_set);
  if (get_affinity_error != 0) {
    throw std::runtime_error(std::string("pthread_getaffinity_np failed: ") +
                             std::strerror(get_affinity_error));
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_id, &set);
  const int affinity_error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (affinity_error != 0) {
    throw std::runtime_error(std::string("pthread_setaffinity_np failed: ") +
                             std::strerror(affinity_error));
  }
  sched_param param{};
  param.sched_priority = priority;
  const int scheduler_error = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
  if (scheduler_error != 0) {
    (void)pthread_setaffinity_np(pthread_self(), sizeof(previous_set), &previous_set);
    throw std::runtime_error(std::string("pthread_setschedparam failed: ") +
                             std::strerror(scheduler_error));
  }
}

/** @brief Return the calling thread to SCHED_OTHER before non-RT reporting or teardown. */
inline void leave_fifo() {
  sched_param param{};
  const int scheduler_error = pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
  if (scheduler_error != 0) {
    throw std::runtime_error(std::string("failed to leave SCHED_FIFO: ") +
                             std::strerror(scheduler_error));
  }
}

/**
 * @brief Verify that scheduler policy, priority, affinity, and VmLck match expectation.
 *
 * Logs the observed state once to stderr. Throws if policy or priority are wrong, or if no memory
 * is locked (VmLck == 0). Only called once at loop entry; never inside the hot path.
 */
inline void assert_applied(int expected_priority, int expected_cpu) {
  sched_param param{};
  int policy = 0;
  if (pthread_getschedparam(pthread_self(), &policy, &param) != 0) {
    throw std::runtime_error("pthread_getschedparam failed");
  }
  if (policy != SCHED_FIFO || param.sched_priority != expected_priority) {
    throw std::runtime_error("RT scheduler not applied as expected");
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    throw std::runtime_error("pthread_getaffinity_np failed");
  }
  if (!CPU_ISSET(expected_cpu, &set)) {
    throw std::runtime_error("RT CPU affinity not applied as expected");
  }
  std::size_t vm_lck_kb = 0;
  if (FILE* f = std::fopen("/proc/self/status", "r"); f != nullptr) {
    char line[256];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
      if (std::sscanf(line, "VmLck: %zu kB", &vm_lck_kb) == 1) {
        break;
      }
    }
    std::fclose(f);
  }
  if (vm_lck_kb == 0) {
    throw std::runtime_error("VmLck is 0 - mlockall not effective");
  }
  std::fprintf(stderr,
               "[rt] SCHED_FIFO prio=%d cpu=%d VmLck=%zu kB - OK\n",
               expected_priority,
               expected_cpu,
               vm_lck_kb);
}

/**
 * @brief Absolute-deadline sleep loop at a fixed period. Uses CLOCK_MONOTONIC + TIMER_ABSTIME.
 *
 * Constructed at the start of the control loop; call sleep_until_next() at the top of every
 * iteration. The deadline advances by `period` on each call, independent of iteration cost, so
 * jitter does not accumulate.
 */
class Loop {
 public:
  explicit Loop(std::chrono::nanoseconds period) noexcept
      : period_(period.count() > 0 ? period : std::chrono::nanoseconds{1}) {
    clock_gettime(CLOCK_MONOTONIC, &next_);
    advance();
  }

  /**
   * @brief Sleep until the next absolute deadline, then advance.
   *
   * Returns the ns past deadline if the sleep woke late (>= 0).
   */
  std::int64_t sleep_until_next() noexcept {
    int sleep_result = 0;
    do {
      sleep_result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_, nullptr);
    } while (sleep_result == EINTR);
    last_sleep_error_ = sleep_result;
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    const std::int64_t late_ns =
        static_cast<std::int64_t>(now.tv_sec - next_.tv_sec) * 1'000'000'000LL +
        static_cast<std::int64_t>(now.tv_nsec - next_.tv_nsec);
    last_missed_periods_ = 0;
    do {
      advance();
      if (deadline_after_now(now)) {
        break;
      }
      ++last_missed_periods_;
    } while (true);
    return late_ns;
  }

  std::uint64_t last_missed_periods() const noexcept { return last_missed_periods_; }
  int last_sleep_error() const noexcept { return last_sleep_error_; }

 private:
  bool deadline_after_now(const timespec& now) const noexcept {
    return next_.tv_sec > now.tv_sec || (next_.tv_sec == now.tv_sec && next_.tv_nsec > now.tv_nsec);
  }

  void advance() noexcept {
    const auto period_ns = period_.count();
    next_.tv_nsec += period_ns;
    while (next_.tv_nsec >= 1'000'000'000L) {
      next_.tv_nsec -= 1'000'000'000L;
      next_.tv_sec += 1;
    }
  }

  std::chrono::nanoseconds period_;
  timespec next_{};
  std::uint64_t last_missed_periods_{0};
  int last_sleep_error_{0};
};

}  // namespace rt
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
