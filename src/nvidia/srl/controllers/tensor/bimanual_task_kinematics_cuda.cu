/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file bimanual_task_kinematics_cuda.cu
 * @brief CUDA backend for bimanual task kinematics.
 */

#include "tensor/bimanual_task_kinematics_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/cuda_stream.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr int kThreads = 128;
constexpr std::uint64_t kInvalidInput = static_cast<std::uint64_t>(StatusFlag::kInvalidInput);
constexpr std::uint64_t kNonFiniteInput = static_cast<std::uint64_t>(StatusFlag::kNonFiniteInput);
constexpr std::uint64_t kResetApplied = static_cast<std::uint64_t>(StatusFlag::kResetApplied);
constexpr std::uint64_t kSafeMode = static_cast<std::uint64_t>(StatusFlag::kSafeMode);

template <typename Scalar>
struct ConstView {
  const Scalar* data;
  std::int64_t strides[3];
};

template <typename Scalar>
struct View {
  Scalar* data;
  std::int64_t strides[3];
};

struct FrameData {
  double arm1_T_arm2[16];
  double reference_T_main[16];
  int main_frame;
  int explicit_reference;
};

template <typename Scalar>
ConstView<Scalar> const_view(const TensorView& view) noexcept {
  return ConstView<Scalar>{static_cast<const Scalar*>(view.data),
                           {view.strides[0], view.strides[1], view.strides[2]}};
}

template <typename Scalar>
View<Scalar> mutable_view(const MutableTensorView& view) noexcept {
  return View<Scalar>{static_cast<Scalar*>(view.data),
                      {view.strides[0], view.strides[1], view.strides[2]}};
}

template <typename Scalar>
__device__ Scalar read3(ConstView<Scalar> view, std::size_t env, std::size_t row, std::size_t col) {
  return view.data[env * static_cast<std::size_t>(view.strides[0]) +
                   row * static_cast<std::size_t>(view.strides[1]) +
                   col * static_cast<std::size_t>(view.strides[2])];
}

template <typename Scalar>
__device__ void write2(View<Scalar> view, std::size_t env, std::size_t col, Scalar value) {
  view.data[env * static_cast<std::size_t>(view.strides[0]) +
            col * static_cast<std::size_t>(view.strides[1])] = value;
}

template <typename Scalar>
__device__ void write3(
    View<Scalar> view, std::size_t env, std::size_t row, std::size_t col, Scalar value) {
  view.data[env * static_cast<std::size_t>(view.strides[0]) +
            row * static_cast<std::size_t>(view.strides[1]) +
            col * static_cast<std::size_t>(view.strides[2])] = value;
}

template <typename Scalar>
__device__ void load_pose(ConstView<Scalar> view, std::size_t env, Scalar output[4][4]) {
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      output[row][col] = read3(view, env, row, col);
    }
  }
}

template <typename Scalar>
__device__ bool rigid_valid(const Scalar transform[4][4]) {
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      if (!isfinite(transform[row][col])) {
        return false;
      }
    }
  }
  return detail::rotation_is_valid(detail::ConstMatrixView<Scalar>{&transform[0][0], 4, 4, 4, 1});
}

template <typename Scalar>
__device__ void multiply_pose(const Scalar left[4][4],
                              const Scalar right[4][4],
                              Scalar output[4][4]) {
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      Scalar value = 0;
      for (std::size_t inner = 0; inner < 4; ++inner) {
        value += left[row][inner] * right[inner][col];
      }
      output[row][col] = value;
    }
  }
}

template <typename Scalar>
__device__ void multiply_rotation(const Scalar left[3][3],
                                  const Scalar right[3][3],
                                  Scalar output[3][3]) {
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      output[row][col] = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < 3; ++inner) {
        output[row][col] += left[row][inner] * right[inner][col];
      }
    }
  }
}

template <typename Scalar>
__device__ void rotation_exp(const Scalar vector[3], Scalar rotation[3][3]) {
  const Scalar angle = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  const Scalar first = angle < static_cast<Scalar>(1.0e-7)
                           ? static_cast<Scalar>(1) - angle * angle / static_cast<Scalar>(6)
                           : sin(angle) / angle;
  const Scalar second = angle < static_cast<Scalar>(1.0e-7)
                            ? static_cast<Scalar>(0.5) - angle * angle / static_cast<Scalar>(24)
                            : (static_cast<Scalar>(1) - cos(angle)) / (angle * angle);
  const Scalar skew[3][3]{
      {0, -vector[2], vector[1]}, {vector[2], 0, -vector[0]}, {-vector[1], vector[0], 0}};
  Scalar skew_squared[3][3]{};
  multiply_rotation(skew, skew, skew_squared);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      rotation[row][col] = (row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0)) +
                           first * skew[row][col] + second * skew_squared[row][col];
    }
  }
}

template <typename Scalar>
__device__ void calculate_midpoint_pose(const Scalar arm1_tcp[4][4],
                                        const Scalar arm2_tcp[4][4],
                                        Scalar midpoint[4][4]) {
  Scalar relative_pose[4][4]{};
  relative_pose[3][3] = static_cast<Scalar>(1);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        relative_pose[row][col] += arm1_tcp[inner][row] * arm2_tcp[inner][col];
      }
    }
  }
  Scalar identity[4][4]{};
  identity[0][0] = identity[1][1] = identity[2][2] = identity[3][3] = static_cast<Scalar>(1);
  Scalar rotation_vector[3]{};
  (void)detail::so3_log_error(detail::ConstMatrixView<Scalar>{&identity[0][0], 4, 4, 4, 1},
                              detail::ConstMatrixView<Scalar>{&relative_pose[0][0], 4, 4, 4, 1},
                              detail::MutableVectorView<Scalar>{rotation_vector, 3, 1});
  for (std::size_t row = 0; row < 3; ++row) {
    rotation_vector[row] *= static_cast<Scalar>(0.5);
  }
  Scalar half_rotation[3][3]{};
  rotation_exp(rotation_vector, half_rotation);
  Scalar arm1_rotation[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_rotation[row][col] = arm1_tcp[row][col];
    }
  }
  Scalar midpoint_rotation[3][3]{};
  multiply_rotation(arm1_rotation, half_rotation, midpoint_rotation);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      midpoint[row][col] = midpoint_rotation[row][col];
    }
    midpoint[row][3] = static_cast<Scalar>(0.5) * (arm1_tcp[row][3] + arm2_tcp[row][3]);
  }
  midpoint[3][3] = static_cast<Scalar>(1);
}

template <typename Scalar>
__device__ void multiply_rotation_vector(const Scalar rotation[3][3],
                                         const Scalar input[3],
                                         Scalar output[3]) {
  for (std::size_t row = 0; row < 3; ++row) {
    output[row] =
        rotation[row][0] * input[0] + rotation[row][1] * input[1] + rotation[row][2] * input[2];
  }
}

template <typename Scalar>
__device__ void so3_right_jacobian(const Scalar vector[3], bool inverse, Scalar jacobian[3][3]) {
  const Scalar angle = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  const Scalar skew[3][3]{
      {0, -vector[2], vector[1]}, {vector[2], 0, -vector[0]}, {-vector[1], vector[0], 0}};
  Scalar skew_squared[3][3]{};
  multiply_rotation(skew, skew, skew_squared);
  Scalar first = static_cast<Scalar>(-0.5);
  Scalar second = static_cast<Scalar>(1.0 / 6.0);
  if (inverse) {
    first = static_cast<Scalar>(0.5);
    second = angle < static_cast<Scalar>(1.0e-6)
                 ? static_cast<Scalar>(1.0 / 12.0)
                 : static_cast<Scalar>(1) / (angle * angle) -
                       (static_cast<Scalar>(1) + cos(angle)) /
                           (static_cast<Scalar>(2) * angle * sin(angle));
  } else if (angle >= static_cast<Scalar>(1.0e-6)) {
    first = -(static_cast<Scalar>(1) - cos(angle)) / (angle * angle);
    second = (angle - sin(angle)) / (angle * angle * angle);
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      jacobian[row][col] = (row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0)) +
                           first * skew[row][col] + second * skew_squared[row][col];
    }
  }
}

template <typename Scalar>
__device__ void calculate_midpoint_angular_velocity(const Scalar arm1_tcp[4][4],
                                                    const Scalar arm2_tcp[4][4],
                                                    const Scalar arm1_angular[3],
                                                    const Scalar arm2_angular[3],
                                                    Scalar midpoint_angular[3]) {
  Scalar relative_pose[4][4]{};
  relative_pose[3][3] = static_cast<Scalar>(1);
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        relative_pose[row][col] += arm1_tcp[inner][row] * arm2_tcp[inner][col];
      }
    }
  }
  Scalar identity[4][4]{};
  identity[0][0] = identity[1][1] = identity[2][2] = identity[3][3] = static_cast<Scalar>(1);
  Scalar phi[3]{};
  (void)detail::so3_log_error(detail::ConstMatrixView<Scalar>{&identity[0][0], 4, 4, 4, 1},
                              detail::ConstMatrixView<Scalar>{&relative_pose[0][0], 4, 4, 4, 1},
                              detail::MutableVectorView<Scalar>{phi, 3, 1});
  Scalar half_phi[3]{static_cast<Scalar>(0.5) * phi[0],
                     static_cast<Scalar>(0.5) * phi[1],
                     static_cast<Scalar>(0.5) * phi[2]};
  Scalar half_rotation[3][3]{};
  rotation_exp(half_phi, half_rotation);
  Scalar arm1_body[3]{};
  Scalar arm2_body[3]{};
  Scalar transported_arm1_body[3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_body[row] += arm1_tcp[col][row] * arm1_angular[col];
      arm2_body[row] += arm2_tcp[col][row] * arm2_angular[col];
    }
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      transported_arm1_body[row] += relative_pose[col][row] * arm1_body[col];
    }
  }
  Scalar relative_body[3]{arm2_body[0] - transported_arm1_body[0],
                          arm2_body[1] - transported_arm1_body[1],
                          arm2_body[2] - transported_arm1_body[2]};
  Scalar inverse_relative_jacobian[3][3]{};
  Scalar half_jacobian[3][3]{};
  so3_right_jacobian(phi, true, inverse_relative_jacobian);
  so3_right_jacobian(half_phi, false, half_jacobian);
  Scalar phi_rate[3]{};
  multiply_rotation_vector(inverse_relative_jacobian, relative_body, phi_rate);
  for (std::size_t row = 0; row < 3; ++row) {
    phi_rate[row] *= static_cast<Scalar>(0.5);
  }
  Scalar half_body_rate[3]{};
  multiply_rotation_vector(half_jacobian, phi_rate, half_body_rate);
  Scalar transported_arm1[3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      transported_arm1[row] += half_rotation[col][row] * arm1_body[col];
    }
  }
  Scalar midpoint_body[3]{transported_arm1[0] + half_body_rate[0],
                          transported_arm1[1] + half_body_rate[1],
                          transported_arm1[2] + half_body_rate[2]};
  Scalar arm1_rotation[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_rotation[row][col] = arm1_tcp[row][col];
    }
  }
  Scalar midpoint_rotation[3][3]{};
  multiply_rotation(arm1_rotation, half_rotation, midpoint_rotation);
  multiply_rotation_vector(midpoint_rotation, midpoint_body, midpoint_angular);
}

template <typename Scalar>
__device__ void pose_error(const Scalar desired[4][4],
                           const Scalar current[4][4],
                           Scalar error[6]) {
  for (std::size_t row = 0; row < 3; ++row) {
    error[row] = desired[row][3] - current[row][3];
  }
  (void)detail::so3_log_error(detail::ConstMatrixView<Scalar>{&current[0][0], 4, 4, 4, 1},
                              detail::ConstMatrixView<Scalar>{&desired[0][0], 4, 4, 4, 1},
                              detail::MutableVectorView<Scalar>{error + 3, 3, 1});
}

template <typename Scalar>
__device__ void rotate_spatial(const Scalar rotation[3][3],
                               const Scalar input[6],
                               Scalar output[6]) {
  for (std::size_t part = 0; part < 2; ++part) {
    for (std::size_t row = 0; row < 3; ++row) {
      output[part * 3 + row] = rotation[row][0] * input[part * 3] +
                               rotation[row][1] * input[part * 3 + 1] +
                               rotation[row][2] * input[part * 3 + 2];
    }
  }
}

template <typename Scalar>
__device__ void load_frames(const FrameData frames,
                            Scalar arm1_T_arm2[4][4],
                            Scalar main_R_arm1[3][3]) {
  Scalar reference_T_main[4][4]{};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      arm1_T_arm2[row][col] = static_cast<Scalar>(frames.arm1_T_arm2[row * 4 + col]);
      reference_T_main[row][col] = static_cast<Scalar>(frames.reference_T_main[row * 4 + col]);
    }
  }
  Scalar arm1_R_main[3][3]{};
  if (frames.main_frame == static_cast<int>(BimanualMainFrame::kArm1Base)) {
    for (std::size_t row = 0; row < 3; ++row) {
      arm1_R_main[row][row] = static_cast<Scalar>(1);
    }
  } else if (frames.main_frame == static_cast<int>(BimanualMainFrame::kArm2Base)) {
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        arm1_R_main[row][col] = arm1_T_arm2[row][col];
      }
    }
  } else if (frames.explicit_reference == static_cast<int>(BimanualBase::kArm1)) {
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        arm1_R_main[row][col] = reference_T_main[row][col];
      }
    }
  } else {
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          arm1_R_main[row][col] += arm1_T_arm2[row][inner] * reference_T_main[inner][col];
        }
      }
    }
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      main_R_arm1[row][col] = arm1_R_main[col][row];
    }
  }
}

template <typename Scalar>
__device__ void zero_environment(View<Scalar> error, View<Scalar> jacobian, std::size_t env) {
  for (std::size_t row = 0; row < 12; ++row) {
    write2(error, env, row, static_cast<Scalar>(0));
    for (std::size_t col = 0; col < 14; ++col) {
      write3(jacobian, env, row, col, static_cast<Scalar>(0));
    }
  }
}

template <typename Scalar>
__global__ void step_kernel(int coordination_mode,
                            FrameData frames,
                            ConstView<Scalar> desired_absolute_pose,
                            ConstView<Scalar> arm1_tcp_pose,
                            ConstView<Scalar> arm2_tcp_pose,
                            ConstView<Scalar> desired_relative_pose,
                            ConstView<Scalar> arm1_jacobian,
                            ConstView<Scalar> arm2_jacobian,
                            View<Scalar> task_pose_error,
                            View<Scalar> task_jacobian,
                            std::size_t num_envs,
                            std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  Scalar arm1_T_arm2[4][4]{};
  Scalar main_R_arm1[3][3]{};
  load_frames(frames, arm1_T_arm2, main_R_arm1);
  Scalar arm1_R_arm2[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_R_arm2[row][col] = arm1_T_arm2[row][col];
    }
  }
  Scalar current_absolute[4][4]{};
  Scalar target_absolute[4][4]{};
  Scalar arm1_tcp[4][4]{};
  Scalar arm2_tcp_local[4][4]{};
  Scalar target_relative[4][4]{};
  load_pose(desired_absolute_pose, env, target_absolute);
  load_pose(arm1_tcp_pose, env, arm1_tcp);
  load_pose(arm2_tcp_pose, env, arm2_tcp_local);
  load_pose(desired_relative_pose, env, target_relative);
  bool valid = rigid_valid(target_absolute) && rigid_valid(arm1_tcp) &&
               rigid_valid(arm2_tcp_local) && rigid_valid(target_relative);
  for (std::size_t row = 0; row < 6 && valid; ++row) {
    for (std::size_t col = 0; col < 7; ++col) {
      valid = valid && isfinite(read3(arm1_jacobian, env, row, col)) &&
              isfinite(read3(arm2_jacobian, env, row, col));
    }
  }
  if (!valid) {
    zero_environment(task_pose_error, task_jacobian, env);
    status[env] = kInvalidInput | kNonFiniteInput | kSafeMode;
    return;
  }
  Scalar arm2_tcp[4][4]{};
  Scalar desired_arm2_tcp[4][4]{};
  multiply_pose(arm1_T_arm2, arm2_tcp_local, arm2_tcp);
  calculate_midpoint_pose(arm1_tcp, arm2_tcp, current_absolute);
  multiply_pose(arm1_tcp, target_relative, desired_arm2_tcp);
  Scalar absolute_error[6]{};
  Scalar relative_error[6]{};
  Scalar absolute_main[6]{};
  Scalar relative_main[6]{};
  pose_error(target_absolute, current_absolute, absolute_error);
  pose_error(desired_arm2_tcp, arm2_tcp, relative_error);
  rotate_spatial(main_R_arm1, absolute_error, absolute_main);
  if (coordination_mode == static_cast<int>(BimanualCoordinationMode::kDecoupledCooperative)) {
    for (std::size_t row = 0; row < 3; ++row) {
      relative_error[row] = target_relative[row][3] - (arm2_tcp[row][3] - arm1_tcp[row][3]);
    }
  }
  rotate_spatial(main_R_arm1, relative_error, relative_main);
  for (std::size_t row = 0; row < 6; ++row) {
    write2(task_pose_error, env, row, absolute_main[row]);
    write2(task_pose_error, env, 6 + row, relative_main[row]);
  }
  const Scalar relative_position[3]{arm2_tcp[0][3] - arm1_tcp[0][3],
                                    arm2_tcp[1][3] - arm1_tcp[1][3],
                                    arm2_tcp[2][3] - arm1_tcp[2][3]};
  for (std::size_t joint = 0; joint < 7; ++joint) {
    Scalar arm1_column[6]{};
    Scalar arm2_local[6]{};
    Scalar arm2_arm1[6]{};
    for (std::size_t row = 0; row < 6; ++row) {
      arm1_column[row] = read3(arm1_jacobian, env, row, joint);
      arm2_local[row] = read3(arm2_jacobian, env, row, joint);
    }
    rotate_spatial(arm1_R_arm2, arm2_local, arm2_arm1);
    Scalar absolute_arm1[6]{};
    Scalar absolute_arm2[6]{};
    Scalar relative_arm1[6]{};
    Scalar relative_arm2[6]{};
    for (std::size_t row = 0; row < 6; ++row) {
      absolute_arm1[row] = static_cast<Scalar>(0.5) * arm1_column[row];
      absolute_arm2[row] = static_cast<Scalar>(0.5) * arm2_arm1[row];
      relative_arm1[row] = -arm1_column[row];
      relative_arm2[row] = arm2_arm1[row];
    }
    const Scalar zero_angular[3]{};
    calculate_midpoint_angular_velocity(
        arm1_tcp, arm2_tcp, arm1_column + 3, zero_angular, absolute_arm1 + 3);
    calculate_midpoint_angular_velocity(
        arm1_tcp, arm2_tcp, zero_angular, arm2_arm1 + 3, absolute_arm2 + 3);
    if (coordination_mode == static_cast<int>(BimanualCoordinationMode::kCoupledRigidTransform)) {
      relative_arm1[0] +=
          relative_position[1] * arm1_column[5] - relative_position[2] * arm1_column[4];
      relative_arm1[1] +=
          relative_position[2] * arm1_column[3] - relative_position[0] * arm1_column[5];
      relative_arm1[2] +=
          relative_position[0] * arm1_column[4] - relative_position[1] * arm1_column[3];
    }
    Scalar absolute_arm1_main[6]{};
    Scalar absolute_arm2_main[6]{};
    Scalar arm1_main[6]{};
    Scalar arm2_main[6]{};
    rotate_spatial(main_R_arm1, absolute_arm1, absolute_arm1_main);
    rotate_spatial(main_R_arm1, absolute_arm2, absolute_arm2_main);
    rotate_spatial(main_R_arm1, relative_arm1, arm1_main);
    rotate_spatial(main_R_arm1, relative_arm2, arm2_main);
    for (std::size_t row = 0; row < 6; ++row) {
      write3(task_jacobian, env, row, joint, absolute_arm1_main[row]);
      write3(task_jacobian, env, row, 7 + joint, absolute_arm2_main[row]);
      write3(task_jacobian, env, 6 + row, joint, arm1_main[row]);
      write3(task_jacobian, env, 6 + row, 7 + joint, arm2_main[row]);
    }
  }
  status[env] = 0U;
}

template <typename Scalar>
__global__ void reset_kernel(const std::uint8_t* mask,
                             std::int64_t mask_stride,
                             View<Scalar> error,
                             View<Scalar> jacobian,
                             std::size_t num_envs,
                             std::uint64_t* status) {
  const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= num_envs) {
    return;
  }
  const bool selected = mask == nullptr || mask[env * static_cast<std::size_t>(mask_stride)] != 0;
  if (selected) {
    zero_environment(error, jacobian, env);
  }
  status[env] = selected ? kResetApplied : 0U;
}

template <typename Scalar>
__global__ void task_velocity_error_kernel(ConstView<Scalar> desired_task_velocity,
                                           ConstView<Scalar> joint_velocity,
                                           ConstView<Scalar> task_jacobian,
                                           View<Scalar> task_velocity_error,
                                           std::size_t num_envs) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_envs * 12) {
    return;
  }
  const std::size_t env = index / 12;
  const std::size_t row = index % 12;
  Scalar value =
      desired_task_velocity
          .data[env * desired_task_velocity.strides[0] + row * desired_task_velocity.strides[1]];
  for (std::size_t joint = 0; joint < 14; ++joint) {
    value -=
        read3(task_jacobian, env, row, joint) *
        joint_velocity.data[env * joint_velocity.strides[0] + joint * joint_velocity.strides[1]];
  }
  write2(task_velocity_error, env, row, value);
}

Result launch_result() noexcept {
  return cudaPeekAtLastError() == cudaSuccess
             ? Result{ResultCode::kOk, "ok"}
             : Result{ResultCode::kInvalidArgument, "CUDA bimanual kinematics launch failed"};
}

FrameData frame_data(const BimanualFrameConfig& config) noexcept {
  FrameData frames{};
  for (std::size_t index = 0; index < 16; ++index) {
    frames.arm1_T_arm2[index] = config.arm1_T_arm2[index];
    frames.reference_T_main[index] = config.reference_T_main[index];
  }
  frames.main_frame = static_cast<int>(config.main_frame);
  frames.explicit_reference = static_cast<int>(config.explicit_reference);
  return frames;
}

template <typename Scalar>
Result launch_step(const BimanualCoordinationMode coordination_mode,
                   const BimanualFrameConfig& config,
                   const TensorView& desired_absolute_pose,
                   const TensorView& arm1_tcp_pose,
                   const TensorView& arm2_tcp_pose,
                   const TensorView& desired_relative_pose,
                   const TensorView& arm1_jacobian,
                   const TensorView& arm2_jacobian,
                   const MutableTensorView& task_pose_error,
                   const MutableTensorView& task_jacobian,
                   const MutableStatusView status) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(arm1_tcp_pose.shape[0]);
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  step_kernel<Scalar>
      <<<blocks, kThreads, 0, active_cuda_stream()>>>(static_cast<int>(coordination_mode),
                                                      frame_data(config),
                                                      const_view<Scalar>(desired_absolute_pose),
                                                      const_view<Scalar>(arm1_tcp_pose),
                                                      const_view<Scalar>(arm2_tcp_pose),
                                                      const_view<Scalar>(desired_relative_pose),
                                                      const_view<Scalar>(arm1_jacobian),
                                                      const_view<Scalar>(arm2_jacobian),
                                                      mutable_view<Scalar>(task_pose_error),
                                                      mutable_view<Scalar>(task_jacobian),
                                                      num_envs,
                                                      status.flags);
  return launch_result();
}

}  // namespace

Result reset_bimanual_task_kinematics_cuda(const std::size_t num_envs,
                                           const TensorView& reset_mask,
                                           const MutableTensorView task_pose_error,
                                           const MutableTensorView task_jacobian,
                                           const MutableStatusView status) noexcept {
  const int blocks = static_cast<int>((num_envs + kThreads - 1) / kThreads);
  if (task_pose_error.dtype == DType::kFloat32) {
    reset_kernel<float><<<blocks, kThreads, 0, active_cuda_stream()>>>(
        static_cast<const std::uint8_t*>(reset_mask.data),
        reset_mask.strides[0],
        mutable_view<float>(task_pose_error),
        mutable_view<float>(task_jacobian),
        num_envs,
        status.flags);
  } else {
    reset_kernel<double><<<blocks, kThreads, 0, active_cuda_stream()>>>(
        static_cast<const std::uint8_t*>(reset_mask.data),
        reset_mask.strides[0],
        mutable_view<double>(task_pose_error),
        mutable_view<double>(task_jacobian),
        num_envs,
        status.flags);
  }
  return launch_result();
}

Result step_bimanual_task_kinematics_cuda(const BimanualCoordinationMode coordination_mode,
                                          const BimanualFrameConfig& config,
                                          const TensorView& desired_absolute_pose,
                                          const TensorView& arm1_tcp_pose,
                                          const TensorView& arm2_tcp_pose,
                                          const TensorView& desired_relative_pose,
                                          const TensorView& arm1_jacobian,
                                          const TensorView& arm2_jacobian,
                                          const MutableTensorView& task_pose_error,
                                          const MutableTensorView& task_jacobian,
                                          const MutableStatusView status) noexcept {
  return arm1_tcp_pose.dtype == DType::kFloat32 ? launch_step<float>(coordination_mode,
                                                                     config,
                                                                     desired_absolute_pose,
                                                                     arm1_tcp_pose,
                                                                     arm2_tcp_pose,
                                                                     desired_relative_pose,
                                                                     arm1_jacobian,
                                                                     arm2_jacobian,
                                                                     task_pose_error,
                                                                     task_jacobian,
                                                                     status)
                                                : launch_step<double>(coordination_mode,
                                                                      config,
                                                                      desired_absolute_pose,
                                                                      arm1_tcp_pose,
                                                                      arm2_tcp_pose,
                                                                      desired_relative_pose,
                                                                      arm1_jacobian,
                                                                      arm2_jacobian,
                                                                      task_pose_error,
                                                                      task_jacobian,
                                                                      status);
}

Result step_bimanual_task_velocity_error_cuda(
    const TensorView& desired_task_velocity,
    const TensorView& joint_velocity,
    const TensorView& task_jacobian,
    const MutableTensorView& task_velocity_error) noexcept {
  const std::size_t num_envs = static_cast<std::size_t>(desired_task_velocity.shape[0]);
  const int blocks = static_cast<int>((num_envs * 12 + kThreads - 1) / kThreads);
  if (desired_task_velocity.dtype == DType::kFloat32) {
    task_velocity_error_kernel<float>
        <<<blocks, kThreads, 0, active_cuda_stream()>>>(const_view<float>(desired_task_velocity),
                                                        const_view<float>(joint_velocity),
                                                        const_view<float>(task_jacobian),
                                                        mutable_view<float>(task_velocity_error),
                                                        num_envs);
  } else {
    task_velocity_error_kernel<double>
        <<<blocks, kThreads, 0, active_cuda_stream()>>>(const_view<double>(desired_task_velocity),
                                                        const_view<double>(joint_velocity),
                                                        const_view<double>(task_jacobian),
                                                        mutable_view<double>(task_velocity_error),
                                                        num_envs);
  }
  return launch_result();
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
