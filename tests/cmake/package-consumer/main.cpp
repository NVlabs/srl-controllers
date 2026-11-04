// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

#include "nvidia/srl/controllers/controllers.hpp"
#include "public_surface.hpp"  // NOLINT(build/include_subdir): standalone installed consumer.

int main(int argc, char** argv) {
  if (!public_surface_is_available()) return 1;
  namespace controllers = nvidia::srl::controllers;
#if defined(__linux__)
  Dl_info library_info{};
  struct stat loaded_library {};
  struct stat expected_library {};
  if (argc != 2 ||
      dladdr(reinterpret_cast<const void*>(&controllers::get_full_version), &library_info) == 0 ||
      library_info.dli_fname == nullptr || stat(library_info.dli_fname, &loaded_library) != 0 ||
      stat(argv[1], &expected_library) != 0 || loaded_library.st_dev != expected_library.st_dev ||
      loaded_library.st_ino != expected_library.st_ino) {
    std::cerr << "The consumer did not load the expected relocated Controllers library.\n";
    return 1;
  }
#else
  static_cast<void>(argc);
  static_cast<void>(argv);
#endif
  const std::string version = controllers::get_full_version();
  if (version.empty()) {
    std::cerr << "The installed Controllers package returned an empty version.\n";
    return 1;
  }

  constexpr std::size_t kJointDim = 2;
  std::array<double, 4> stiffness{100.0, 0.0, 0.0, 200.0};
  std::array<double, 2> damping{1.0, 1.0};
  std::array<double, 2> position_error{0.1, -0.1};
  std::array<double, 2> zeros{};
  std::array<double, 4> mass{1.0, 0.0, 0.0, 1.0};
  std::array<double, 2> torque{};
  std::array<double, 2> last_valid_torque{};
  std::array<std::uint8_t, 1> valid{};
  std::array<std::uint64_t, 1> hold_count{};
  std::array<std::uint64_t, 1> parameter_status{};
  std::array<std::uint64_t, 1> status{};

  controllers::JointImpedanceController controller;
  const controllers::JointImpedanceController::Parameters parameters{
      controllers::per_batch(
          controllers::read_view(stiffness, controllers::Device::kCpu, 1, kJointDim, kJointDim)),
      controllers::per_batch(
          controllers::read_view(damping, controllers::Device::kCpu, 1, kJointDim))};
  const controllers::JointImpedanceController::Storage storage{
      controllers::write_view(torque, controllers::Device::kCpu, 1, kJointDim),
      controllers::write_status(parameter_status, controllers::Device::kCpu),
      controllers::write_status(status, controllers::Device::kCpu),
      controllers::make_safety_state(
          controllers::write_view(last_valid_torque, controllers::Device::kCpu, 1, kJointDim),
          controllers::write_view(valid, controllers::Device::kCpu, 1),
          controllers::write_view(hold_count, controllers::Device::kCpu, 1))};
  const controllers::JointImpedanceController::Inputs inputs{
      controllers::read_view(position_error, controllers::Device::kCpu, 1, kJointDim),
      controllers::read_view(zeros, controllers::Device::kCpu, 1, kJointDim),
      controllers::read_view(zeros, controllers::Device::kCpu, 1, kJointDim),
      controllers::read_view(mass, controllers::Device::kCpu, 1, kJointDim, kJointDim)};
  const controllers::Result prepare = controller.prepare(
      {controllers::Device::kCpu, controllers::DType::kFloat64, 1, kJointDim, {}},
      parameters,
      storage);
  const controllers::Result reset = controller.reset();
  const controllers::Result step = controller.step(inputs);
  if (!prepare.ok() || !reset.ok() || !step.ok() || status[0] != 0U || torque[0] != 10.0 ||
      torque[1] != -20.0) {
    std::cerr << "The installed controller API failed.\n";
    return 1;
  }

  std::cout << version << '\n';
  return 0;
}
