/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 */

#if defined(__linux__)

#include <fcntl.h>
#include <gtest/gtest.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <tuple>

#include "nvidia/srl/controllers/rt_bootstrap.hpp"

namespace {

constexpr std::chrono::seconds CHILD_TIMEOUT{5};
constexpr std::chrono::seconds WAITER_TIMEOUT{1};
constexpr std::chrono::milliseconds POLL_INTERVAL{1};
constexpr std::chrono::milliseconds CALLBACK_QUIET_PERIOD{50};
constexpr int CHILD_CHECK_FAILED = 90;
constexpr int UNEXPECTED_CALLBACK = 91;
constexpr char READY_TO_SHUT_DOWN = 'R';

std::atomic<int> sigwait_call_count{0};
std::atomic<int> callback_count{0};
bool inject_sigwait_error = false;
pthread_t child_main_thread{};

}  // namespace

extern "C" int __real_sigwait(const sigset_t* mask, int* signal_number);

extern "C" int __wrap_sigwait(const sigset_t* mask, int* signal_number) {
  sigwait_call_count.fetch_add(1);
  if (inject_sigwait_error) {
    *signal_number = SIGTERM;
    return EINVAL;
  }
  return __real_sigwait(mask, signal_number);
}

namespace nvidia {
namespace srl {
namespace controllers {
namespace {

void record_shutdown() {
  if (inject_sigwait_error || pthread_equal(pthread_self(), child_main_thread) != 0) {
    _exit(UNEXPECTED_CALLBACK);
  }
  callback_count.fetch_add(1);
}

bool wait_for_count(const std::atomic<int>& count, int expected) {
  const auto deadline = std::chrono::steady_clock::now() + WAITER_TIMEOUT;
  while (std::chrono::steady_clock::now() < deadline) {
    if (count.load() == expected) {
      return true;
    }
    std::this_thread::sleep_for(POLL_INTERVAL);
  }
  return false;
}

void require_child_check(bool passed) {
  if (!passed) {
    _exit(CHILD_CHECK_FAILED);
  }
}

void send_signal(int signal_number) { require_child_check(kill(getpid(), signal_number) == 0); }

[[noreturn]] void wait_for_process_exit() {
  for (;;) {
    pause();
  }
}

[[noreturn]] void run_signal_child(int readiness_fd,
                                   int termination_signal,
                                   bool use_callback,
                                   bool send_ignored_signals,
                                   bool fail_sigwait) {
  child_main_thread = pthread_self();
  inject_sigwait_error = fail_sigwait;
  rt::block_signals(use_callback ? record_shutdown : nullptr);
  if (fail_sigwait) {
    wait_for_process_exit();
  }
  int expected_waits = 1;
  require_child_check(wait_for_count(sigwait_call_count, expected_waits));
  if (send_ignored_signals) {
    for (const int ignored_signal : std::array<int, 4>{SIGCHLD, SIGPIPE, SIGPIPE, SIGCHLD}) {
      send_signal(ignored_signal);
      require_child_check(wait_for_count(sigwait_call_count, ++expected_waits));
      require_child_check(callback_count.load() == 0);
    }
  }
  require_child_check(write(readiness_fd, &READY_TO_SHUT_DOWN, 1) == 1);
  send_signal(termination_signal);
  if (!use_callback) {
    wait_for_process_exit();
  }
  require_child_check(wait_for_count(callback_count, 1));
  send_signal(SIGINT);
  send_signal(SIGTERM);
  std::this_thread::sleep_for(CALLBACK_QUIET_PERIOD);
  require_child_check(callback_count.load() == 1);
  require_child_check(sigwait_call_count.load() == expected_waits);
  _exit(EXIT_SUCCESS);
}

int wait_for_child(pid_t child_pid) {
  const auto deadline = std::chrono::steady_clock::now() + CHILD_TIMEOUT;
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t wait_result = waitpid(child_pid, &status, WNOHANG);
    if (wait_result == child_pid) {
      return status;
    }
    if (wait_result < 0 && errno != EINTR) {
      return -1;
    }
    std::this_thread::sleep_for(POLL_INTERVAL);
  }
  kill(child_pid, SIGKILL);
  while (waitpid(child_pid, &status, 0) < 0 && errno == EINTR) {
  }
  return -1;
}

void expect_child_shutdown(int termination_signal,
                           bool use_callback,
                           bool send_ignored_signals,
                           bool fail_sigwait = false) {
  int readiness_pipe[2];
  ASSERT_EQ(pipe2(readiness_pipe, O_CLOEXEC | O_NONBLOCK), 0);
  const pid_t child_pid = fork();
  if (child_pid == 0) {
    close(readiness_pipe[0]);
    run_signal_child(
        readiness_pipe[1], termination_signal, use_callback, send_ignored_signals, fail_sigwait);
  }
  close(readiness_pipe[1]);
  if (child_pid < 0) {
    close(readiness_pipe[0]);
    FAIL() << "fork failed: " << errno;
  }
  const int status = wait_for_child(child_pid);
  char readiness = 0;
  const ssize_t readiness_bytes = read(readiness_pipe[0], &readiness, 1);
  close(readiness_pipe[0]);
  ASSERT_NE(status, -1) << "Child did not exit within timeout or waitpid failed";
  ASSERT_TRUE(WIFEXITED(status)) << "Child status: " << status;
  EXPECT_EQ(WEXITSTATUS(status), fail_sigwait ? EXIT_FAILURE : EXIT_SUCCESS);
  if (!fail_sigwait) {
    EXPECT_EQ(readiness_bytes, 1) << "Child exited before the termination signal was sent";
    EXPECT_EQ(readiness, READY_TO_SHUT_DOWN);
  }
}

class RtBootstrapSignalTest : public ::testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(RtBootstrapSignalTest, IgnoresChildAndPipeSignalsBeforeShutdown) {
  expect_child_shutdown(std::get<0>(GetParam()), std::get<1>(GetParam()), true);
}

TEST_P(RtBootstrapSignalTest, ShutsDownOnlyOnceOnTerminationSignal) {
  expect_child_shutdown(std::get<0>(GetParam()), std::get<1>(GetParam()), false);
}

INSTANTIATE_TEST_SUITE_P(InterruptAndTerminate,
                         RtBootstrapSignalTest,
                         ::testing::Combine(::testing::Values(SIGINT, SIGTERM),
                                            ::testing::Values(false, true)));

TEST(RtBootstrapSignalErrorTest, FailsWithoutDispatchingCallbackOnSigwaitError) {
  expect_child_shutdown(SIGTERM, true, false, true);
}

TEST(RtBootstrapSignalErrorTest, FailsInsteadOfDefaultSuccessfulExitOnSigwaitError) {
  expect_child_shutdown(SIGTERM, false, false, true);
}

}  // namespace
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
