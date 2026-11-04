// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <new>
#include "nvidia/srl/controllers/composition.hpp"
#include "nvidia/srl/controllers/controllers.hpp"
#include "nvidia/srl/controllers/equations.hpp"

namespace {
thread_local bool count_allocations = false;
thread_local std::size_t allocations = 0;
}  // namespace

void* operator new(std::size_t bytes) {
  if (count_allocations) ++allocations;
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

TEST(PublicCompositionTest, StatefulSignalLifecycleAndFinalPolicyDoNotAllocate) {
  namespace c = nvidia::srl::controllers;
  constexpr auto device = c::Device::kCpu;
  constexpr std::size_t batch = 2, width = 97;
  std::array<double, width> initial{}, lower{}, upper{};
  lower.fill(-10.);
  upper.fill(10.);
  std::array<double, batch * width> error{}, command{}, previous{}, integrated{}, last{},
      final_last{};
  error.fill(1.);
  std::array<std::uint8_t, batch> initialized{}, valid{}, final_valid{};
  std::array<std::uint64_t, batch> parameter_status{}, status{}, holds{}, final_holds{},
      final_flags{};
  double dt = .01;
  c::IntegratorBlock block;
  const c::SignalBlockConfig config{device, c::DType::kFloat64, batch, width};
  const c::IntegratorBlock::Parameters parameters{c::shared(c::read_view(initial, device, width)),
                                                  c::shared(c::read_view(lower, device, width)),
                                                  c::shared(c::read_view(upper, device, width))};
  const c::IntegratorBlock::Storage storage{{c::write_view(command, device, batch, width),
                                             c::write_status(parameter_status, device),
                                             c::write_status(status, device),
                                             {c::write_view(last, device, batch, width),
                                              c::write_view(valid, device, batch),
                                              c::write_view(holds, device, batch)}},
                                            {c::write_view(previous, device, batch, width),
                                             c::write_view(integrated, device, batch, width),
                                             c::write_view(initialized, device, batch)},
                                            {}};
  const c::SafetyStateView final_state{c::write_view(final_last, device, batch, width),
                                       c::write_view(final_valid, device, batch),
                                       c::write_view(final_holds, device, batch)};
  ASSERT_TRUE(block.prepare(config, parameters, storage).ok());
  ASSERT_TRUE(block.reset().ok());
  ASSERT_TRUE(
      c::reset_safety_state(batch, final_state, {}, c::write_status(final_flags, device)).ok());
  const c::IntegratorBlock::Inputs inputs{c::read_view(error, device, batch, width),
                                          c::read_view(&dt, device)};
  const c::StatusView upstream{status.data(), batch, device};
  bool success = true;
  allocations = 0;
  count_allocations = true;
  for (int tick = 0; tick < 200; ++tick) {
    success = success && block.step(inputs).ok();
    success = success && c::merge_status(&upstream, 1, c::write_status(final_flags, device)).ok();
    success = success && c::apply_safe_output_policy(c::write_view(command, device, batch, width),
                                                     c::write_status(final_flags, device),
                                                     final_state)
                             .ok();
  }
  count_allocations = false;
  EXPECT_TRUE(success);
  EXPECT_EQ(allocations, 0U);
  EXPECT_GT(command[0], 1.9);
  EXPECT_LT(command[0], 2.1);
  EXPECT_EQ(final_flags[0], 0U);
}

TEST(PublicCompositionTest, EnergyTankBudgetAndFailuresAreAllocationFree) {
  namespace c = nvidia::srl::controllers;
  const auto device = c::Device::kCpu;
  std::array<double, 1> upper{1}, lower{0.01}, initial{1}, width{0.1}, noise{0};
  std::array<double, 2> power{-200, -1}, recharge{0, -1}, dt{0.001, 0.001};
  std::array<double, 2> scale{}, filtered{}, state{std::sqrt(0.22), std::sqrt(2.0)};
  std::array<std::uint64_t, 2> parameters{}, status{};
  c::EnergyTank tank;
  const c::EnergyTank::Parameters p{c::shared(c::read_view(upper, device)),
                                    c::shared(c::read_view(lower, device)),
                                    c::shared(c::read_view(initial, device)),
                                    c::shared(c::read_view(width, device)),
                                    c::shared(c::read_view(noise, device))};
  const c::EnergyTank::Storage storage{c::write_view(scale, device, 2),
                                       c::write_view(filtered, device, 2),
                                       c::write_status(parameters, device),
                                       c::write_status(status, device),
                                       {c::write_view(state, device, 2)},
                                       {}};
  ASSERT_TRUE(tank.prepare({device, c::DType::kFloat64, 2, 1}, p, storage).ok());
  const c::EnergyTank::Inputs inputs{c::read_view(power, device, 2),
                                     c::read_view(recharge, device, 2),
                                     c::read_view(dt, device, 2)};
  const double held = state[1];
  allocations = 0;
  count_allocations = true;
  bool success = tank.step(inputs).ok();
  count_allocations = false;
  ASSERT_TRUE(success);
  EXPECT_EQ(allocations, 0U);
  EXPECT_NEAR(scale[0], 0.5, 1e-14);
  EXPECT_NEAR(0.5 * state[0] * state[0], 0.01, 1e-14);
  EXPECT_TRUE(c::has_status_flag(status[0], c::StatusFlag::kSaturation));
  EXPECT_TRUE(c::has_status_flag(status[1], c::StatusFlag::kSafeMode));
  EXPECT_EQ(scale[1], 0.0);
  EXPECT_EQ(state[1], held);
  count_allocations = true;
  for (int step = 0; step < 200; ++step) success = tank.step(inputs).ok() && success;
  count_allocations = false;
  EXPECT_TRUE(success);
  EXPECT_EQ(allocations, 0U);
  EXPECT_NEAR(0.5 * state[0] * state[0], 0.01, 1e-14);
}

TEST(PublicCompositionTest, EnergyTankReversalKeepsTaperAndFloorRecoversFromDissipation) {
  namespace eq = nvidia::srl::controllers::equations;
  const auto step = [](double energy, double power, double recharge, double dt) {
    return eq::energy_tank_step(std::sqrt(2 * energy), power, recharge, dt, 1.0, 0.01, 0.1, 0.0);
  };
  const auto draining = step(0.06, -1.0, 0, 0.001);
  const auto absorbing = step(0.06, 1.0, 0, 0.001);
  EXPECT_NEAR(draining.scale, 0.5, 1e-14);
  EXPECT_DOUBLE_EQ(absorbing.scale, draining.scale);
  EXPECT_NEAR(0.5 * absorbing.state_variable * absorbing.state_variable, 0.0605, 1e-14);
  const auto depleted = step(0.11, -200, 0, 0.001);
  const double floor = 0.5 * depleted.state_variable * depleted.state_variable;
  EXPECT_NEAR(floor, 0.01, 1e-14);
  const auto refill = step(floor, 1.0, 0.5, 0.01);
  EXPECT_NEAR(refill.scale, 0.0, 1e-14);
  const double recovered = 0.5 * refill.state_variable * refill.state_variable;
  EXPECT_NEAR(recovered, 0.015, 1e-14);
  const auto resumed = step(recovered, -1.0, 0, 0.001);
  EXPECT_GT(resumed.scale, 0);
  EXPECT_LT(resumed.scale, 1);
  EXPECT_NEAR(0.5 * resumed.state_variable * resumed.state_variable,
              recovered - resumed.scale * 0.001,
              1e-14);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(PublicCompositionTest, RejectsUnalignedStatusBeforeMutation) {
  namespace c = nvidia::srl::controllers;
  alignas(std::uint64_t) std::array<unsigned char, 16> unaligned{};
  std::array<std::uint64_t, 1> flags{static_cast<std::uint64_t>(c::StatusFlag::kInvalidInput)};
  auto* bad = reinterpret_cast<std::uint64_t*>(unaligned.data() + 1);
  const c::StatusView input{bad, 1, c::Device::kCpu};
  EXPECT_EQ(c::merge_status(&input, 1, c::write_status(flags, c::Device::kCpu)).code,
            c::ResultCode::kInvalidView);
  EXPECT_EQ(flags[0], static_cast<std::uint64_t>(c::StatusFlag::kInvalidInput));
  const c::StatusView good_input{flags.data(), flags.size(), c::Device::kCpu};
  const c::MutableStatusView output{bad, 1, c::Device::kCpu};
  EXPECT_EQ(c::merge_status(&good_input, 1, output).code, c::ResultCode::kInvalidView);
  for (auto byte : unaligned) EXPECT_EQ(byte, 0);
}
