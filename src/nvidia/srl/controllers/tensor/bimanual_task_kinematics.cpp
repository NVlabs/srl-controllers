/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file bimanual_task_kinematics.cpp
 * @brief CPU implementation and dispatch for bimanual task kinematics.
 */

#include "tensor/bimanual_task_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nvidia/srl/controllers/detail/component_core.hpp"

#include "tensor/common.hpp"
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
#include "tensor/bimanual_task_kinematics_cuda.hpp"
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace {

constexpr std::size_t kPoseDim = 4;
constexpr std::size_t kSpatialDim = 6;
constexpr std::size_t kArmJointDim = 7;
constexpr std::size_t kJointDim = 14;
constexpr std::size_t kTaskDim = 12;

template <typename Scalar>
Scalar read_pose(const TensorView& view,
                 const std::size_t env,
                 const std::size_t row,
                 const std::size_t col) noexcept {
  return internal::read_tensor3_value<Scalar>(view, env, row, col);
}

template <typename Scalar>
void read_pose_matrix(const TensorView& view, const std::size_t env, Scalar pose[4][4]) noexcept {
  for (std::size_t row = 0; row < kPoseDim; ++row) {
    for (std::size_t col = 0; col < kPoseDim; ++col) {
      pose[row][col] = read_pose<Scalar>(view, env, row, col);
    }
  }
}

template <typename Scalar>
bool rigid_transform_is_valid(const Scalar transform[4][4]) noexcept {
  for (std::size_t row = 0; row < kPoseDim; ++row) {
    for (std::size_t col = 0; col < kPoseDim; ++col) {
      if (!std::isfinite(transform[row][col])) {
        return false;
      }
    }
  }
  return detail::rotation_is_valid(detail::ConstMatrixView<Scalar>{&transform[0][0], 4, 4, 4, 1});
}

template <typename Scalar>
void multiply_pose(const Scalar left[4][4],
                   const Scalar right[4][4],
                   Scalar output[4][4]) noexcept {
  for (std::size_t row = 0; row < kPoseDim; ++row) {
    for (std::size_t col = 0; col < kPoseDim; ++col) {
      Scalar value = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < kPoseDim; ++inner) {
        value += left[row][inner] * right[inner][col];
      }
      output[row][col] = value;
    }
  }
}

template <typename Scalar>
void transpose_multiply_rotation(const Scalar left[4][4],
                                 const Scalar right[4][4],
                                 Scalar output[3][3]) noexcept {
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      output[row][col] = static_cast<Scalar>(0);
      for (std::size_t inner = 0; inner < 3; ++inner) {
        output[row][col] += left[inner][row] * right[inner][col];
      }
    }
  }
}

template <typename Scalar>
void multiply_rotation(const Scalar left[3][3],
                       const Scalar right[3][3],
                       Scalar output[3][3]) noexcept {
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
void rotation_log(const Scalar rotation[3][3], Scalar vector[3]) noexcept {
  Scalar current[4][4]{};
  Scalar desired[4][4]{};
  current[3][3] = static_cast<Scalar>(1);
  desired[3][3] = static_cast<Scalar>(1);
  for (std::size_t row = 0; row < 3; ++row) {
    current[row][row] = static_cast<Scalar>(1);
    for (std::size_t col = 0; col < 3; ++col) {
      desired[row][col] = rotation[row][col];
    }
  }
  (void)detail::so3_log_error(detail::ConstMatrixView<Scalar>{&current[0][0], 4, 4, 4, 1},
                              detail::ConstMatrixView<Scalar>{&desired[0][0], 4, 4, 4, 1},
                              detail::MutableVectorView<Scalar>{vector, 3, 1});
}

template <typename Scalar>
void rotation_exp(const Scalar vector[3], Scalar rotation[3][3]) noexcept {
  const Scalar angle =
      std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  const Scalar first = angle < static_cast<Scalar>(1.0e-7)
                           ? static_cast<Scalar>(1) - angle * angle / static_cast<Scalar>(6)
                           : std::sin(angle) / angle;
  const Scalar second = angle < static_cast<Scalar>(1.0e-7)
                            ? static_cast<Scalar>(0.5) - angle * angle / static_cast<Scalar>(24)
                            : (static_cast<Scalar>(1) - std::cos(angle)) / (angle * angle);
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
void calculate_midpoint_pose(const Scalar arm1_tcp[4][4],
                             const Scalar arm2_tcp[4][4],
                             Scalar midpoint[4][4]) noexcept {
  Scalar relative_rotation[3][3]{};
  transpose_multiply_rotation(arm1_tcp, arm2_tcp, relative_rotation);
  Scalar rotation_vector[3]{};
  rotation_log(relative_rotation, rotation_vector);
  for (Scalar& value : rotation_vector) {
    value *= static_cast<Scalar>(0.5);
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
void multiply_rotation_vector(const Scalar rotation[3][3],
                              const Scalar input[3],
                              Scalar output[3]) noexcept {
  for (std::size_t row = 0; row < 3; ++row) {
    output[row] =
        rotation[row][0] * input[0] + rotation[row][1] * input[1] + rotation[row][2] * input[2];
  }
}

template <typename Scalar>
void so3_right_jacobian(const Scalar vector[3],
                        const bool inverse,
                        Scalar jacobian[3][3]) noexcept {
  const Scalar angle =
      std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
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
                       (static_cast<Scalar>(1) + std::cos(angle)) /
                           (static_cast<Scalar>(2) * angle * std::sin(angle));
  } else if (angle >= static_cast<Scalar>(1.0e-6)) {
    first = -(static_cast<Scalar>(1) - std::cos(angle)) / (angle * angle);
    second = (angle - std::sin(angle)) / (angle * angle * angle);
  }
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      jacobian[row][col] = (row == col ? static_cast<Scalar>(1) : static_cast<Scalar>(0)) +
                           first * skew[row][col] + second * skew_squared[row][col];
    }
  }
}

template <typename Scalar>
void calculate_midpoint_angular_velocity(const Scalar arm1_tcp[4][4],
                                         const Scalar arm2_tcp[4][4],
                                         const Scalar arm1_angular[3],
                                         const Scalar arm2_angular[3],
                                         Scalar midpoint_angular[3]) noexcept {
  Scalar relative_rotation[3][3]{};
  transpose_multiply_rotation(arm1_tcp, arm2_tcp, relative_rotation);
  Scalar phi[3]{};
  rotation_log(relative_rotation, phi);
  Scalar half_phi[3]{static_cast<Scalar>(0.5) * phi[0],
                     static_cast<Scalar>(0.5) * phi[1],
                     static_cast<Scalar>(0.5) * phi[2]};
  Scalar half_rotation[3][3]{};
  rotation_exp(half_phi, half_rotation);
  Scalar arm1_rotation[3][3]{};
  Scalar arm2_rotation[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_rotation[row][col] = arm1_tcp[row][col];
      arm2_rotation[row][col] = arm2_tcp[row][col];
    }
  }
  Scalar arm1_rotation_transpose[3][3]{};
  Scalar arm2_rotation_transpose[3][3]{};
  Scalar relative_rotation_transpose[3][3]{};
  Scalar half_rotation_transpose[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_rotation_transpose[row][col] = arm1_rotation[col][row];
      arm2_rotation_transpose[row][col] = arm2_rotation[col][row];
      relative_rotation_transpose[row][col] = relative_rotation[col][row];
      half_rotation_transpose[row][col] = half_rotation[col][row];
    }
  }
  Scalar arm1_body[3]{};
  Scalar arm2_body[3]{};
  Scalar transported_arm1_body[3]{};
  multiply_rotation_vector(arm1_rotation_transpose, arm1_angular, arm1_body);
  multiply_rotation_vector(arm2_rotation_transpose, arm2_angular, arm2_body);
  multiply_rotation_vector(relative_rotation_transpose, arm1_body, transported_arm1_body);
  Scalar relative_body[3]{arm2_body[0] - transported_arm1_body[0],
                          arm2_body[1] - transported_arm1_body[1],
                          arm2_body[2] - transported_arm1_body[2]};
  Scalar inverse_relative_jacobian[3][3]{};
  Scalar half_jacobian[3][3]{};
  so3_right_jacobian(phi, true, inverse_relative_jacobian);
  so3_right_jacobian(half_phi, false, half_jacobian);
  Scalar phi_rate[3]{};
  Scalar half_body_rate[3]{};
  multiply_rotation_vector(inverse_relative_jacobian, relative_body, phi_rate);
  for (Scalar& value : phi_rate) {
    value *= static_cast<Scalar>(0.5);
  }
  multiply_rotation_vector(half_jacobian, phi_rate, half_body_rate);
  Scalar transported_arm1[3]{};
  multiply_rotation_vector(half_rotation_transpose, arm1_body, transported_arm1);
  Scalar midpoint_body[3]{transported_arm1[0] + half_body_rate[0],
                          transported_arm1[1] + half_body_rate[1],
                          transported_arm1[2] + half_body_rate[2]};
  Scalar midpoint_rotation[3][3]{};
  multiply_rotation(arm1_rotation, half_rotation, midpoint_rotation);
  multiply_rotation_vector(midpoint_rotation, midpoint_body, midpoint_angular);
}

template <typename Scalar>
void pose_error(const Scalar desired[4][4], const Scalar current[4][4], Scalar error[6]) noexcept {
  for (std::size_t row = 0; row < 3; ++row) {
    error[row] = desired[row][3] - current[row][3];
  }
  (void)detail::so3_log_error(detail::ConstMatrixView<Scalar>{&current[0][0], 4, 4, 4, 1},
                              detail::ConstMatrixView<Scalar>{&desired[0][0], 4, 4, 4, 1},
                              detail::MutableVectorView<Scalar>{error + 3, 3, 1});
}

template <typename Scalar>
void rotate_spatial(const Scalar rotation[3][3], const Scalar input[6], Scalar output[6]) noexcept {
  for (std::size_t part = 0; part < 2; ++part) {
    for (std::size_t row = 0; row < 3; ++row) {
      output[part * 3 + row] = rotation[row][0] * input[part * 3] +
                               rotation[row][1] * input[part * 3 + 1] +
                               rotation[row][2] * input[part * 3 + 2];
    }
  }
}

template <typename Scalar>
void config_transform(const std::array<double, 16>& source, Scalar output[4][4]) noexcept {
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      output[row][col] = static_cast<Scalar>(source[row * 4 + col]);
    }
  }
}

template <typename Scalar>
void main_rotation(const BimanualFrameConfig& config, Scalar main_R_arm1[3][3]) noexcept {
  Scalar arm1_T_arm2[4][4]{};
  Scalar reference_T_main[4][4]{};
  config_transform(config.arm1_T_arm2, arm1_T_arm2);
  config_transform(config.reference_T_main, reference_T_main);
  Scalar arm1_R_main[3][3]{};
  if (config.main_frame == BimanualMainFrame::kArm1Base) {
    for (std::size_t row = 0; row < 3; ++row) {
      arm1_R_main[row][row] = static_cast<Scalar>(1);
    }
  } else if (config.main_frame == BimanualMainFrame::kArm2Base) {
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        arm1_R_main[row][col] = arm1_T_arm2[row][col];
      }
    }
  } else if (config.explicit_reference == BimanualBase::kArm1) {
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

Result validate_pose_view(const TensorView& view,
                          const DType dtype,
                          const Device device,
                          const std::size_t num_envs) noexcept {
  return view.data != nullptr && view.dtype == dtype && view.device == device &&
                 internal::is_row_major_tensor3(view) &&
                 view.shape[0] == static_cast<std::int64_t>(num_envs) && view.shape[1] == 4 &&
                 view.shape[2] == 4
             ? internal::ok()
             : internal::invalid_view("bimanual poses must be row-major [batch_size,4,4]");
}

Result validate_matrix_view(const TensorView& view,
                            const DType dtype,
                            const Device device,
                            const std::size_t num_envs,
                            const std::size_t rows,
                            const std::size_t cols) noexcept {
  return view.data != nullptr && view.dtype == dtype && view.device == device &&
                 internal::is_row_major_tensor3(view) &&
                 view.shape[0] == static_cast<std::int64_t>(num_envs) &&
                 view.shape[1] == static_cast<std::int64_t>(rows) &&
                 view.shape[2] == static_cast<std::int64_t>(cols)
             ? internal::ok()
             : internal::invalid_view("bimanual Jacobian has an invalid shape");
}

Result validate_output_view(const MutableTensorView& error,
                            const MutableTensorView& jacobian,
                            const MutableStatusView status,
                            const DType dtype,
                            const Device device,
                            const std::size_t num_envs) noexcept {
  if (error.data == nullptr || error.dtype != dtype || error.device != device ||
      !internal::is_row_major_matrix(error) ||
      error.shape[0] != static_cast<std::int64_t>(num_envs) || error.shape[1] != 12) {
    return internal::invalid_view("task_pose_error must be row-major [batch_size,12]");
  }
  if (jacobian.data == nullptr || jacobian.dtype != dtype || jacobian.device != device ||
      !internal::is_row_major_tensor3(jacobian) ||
      jacobian.shape[0] != static_cast<std::int64_t>(num_envs) || jacobian.shape[1] != 12 ||
      jacobian.shape[2] != 14) {
    return internal::invalid_view("task_jacobian must be row-major [batch_size,12,14]");
  }
  return status.flags != nullptr && status.device == device && status.size >= num_envs
             ? internal::ok()
             : internal::invalid_view("status must match the bimanual batch and device");
}

template <typename Scalar>
void zero_environment(const MutableTensorView& error,
                      const MutableTensorView& jacobian,
                      const std::size_t env) noexcept {
  for (std::size_t row = 0; row < kTaskDim; ++row) {
    internal::write_matrix_value(error, env, row, static_cast<Scalar>(0));
    for (std::size_t col = 0; col < kJointDim; ++col) {
      internal::write_tensor3_value(jacobian, env, row, col, static_cast<Scalar>(0));
    }
  }
}

template <typename Scalar>
Result step_cpu(const BimanualCoordinationMode coordination_mode,
                const BimanualFrameConfig& config,
                const TensorView& desired_absolute_pose,
                const TensorView& arm1_tcp_pose,
                const TensorView& arm2_tcp_pose,
                const TensorView& desired_relative_pose,
                const TensorView& arm1_jacobian,
                const TensorView& arm2_jacobian,
                const MutableTensorView& task_pose_error,
                const MutableTensorView& task_jacobian,
                const MutableStatusView status,
                const std::size_t num_envs) noexcept {
  Scalar arm1_T_arm2[4][4]{};
  config_transform(config.arm1_T_arm2, arm1_T_arm2);
  Scalar main_R_arm1[3][3]{};
  main_rotation(config, main_R_arm1);
  Scalar arm1_R_arm2[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      arm1_R_arm2[row][col] = arm1_T_arm2[row][col];
    }
  }

  for (std::size_t env = 0; env < num_envs; ++env) {
    Scalar current_absolute[4][4]{};
    Scalar target_absolute[4][4]{};
    Scalar arm1_tcp[4][4]{};
    Scalar arm2_tcp_local[4][4]{};
    Scalar target_relative[4][4]{};
    read_pose_matrix(desired_absolute_pose, env, target_absolute);
    read_pose_matrix(arm1_tcp_pose, env, arm1_tcp);
    read_pose_matrix(arm2_tcp_pose, env, arm2_tcp_local);
    read_pose_matrix(desired_relative_pose, env, target_relative);
    if (!rigid_transform_is_valid(target_absolute) || !rigid_transform_is_valid(arm1_tcp) ||
        !rigid_transform_is_valid(arm2_tcp_local) || !rigid_transform_is_valid(target_relative) ||
        !internal::tensor3_env_is_finite<Scalar>(arm1_jacobian, env, 6, 7) ||
        !internal::tensor3_env_is_finite<Scalar>(arm2_jacobian, env, 6, 7)) {
      status.flags[env] = status_bit(StatusFlag::kNonFiniteInput) |
                          status_bit(StatusFlag::kInvalidInput) | status_bit(StatusFlag::kSafeMode);
      zero_environment<Scalar>(task_pose_error, task_jacobian, env);
      continue;
    }

    Scalar arm2_tcp[4][4]{};
    Scalar desired_arm2_tcp[4][4]{};
    multiply_pose(arm1_T_arm2, arm2_tcp_local, arm2_tcp);
    calculate_midpoint_pose(arm1_tcp, arm2_tcp, current_absolute);
    multiply_pose(arm1_tcp, target_relative, desired_arm2_tcp);
    Scalar absolute_error_arm1[6]{};
    Scalar relative_error_arm1[6]{};
    pose_error(target_absolute, current_absolute, absolute_error_arm1);
    pose_error(desired_arm2_tcp, arm2_tcp, relative_error_arm1);
    Scalar absolute_error_main[6]{};
    Scalar relative_error_main[6]{};
    rotate_spatial(main_R_arm1, absolute_error_arm1, absolute_error_main);
    if (coordination_mode == BimanualCoordinationMode::kDecoupledCooperative) {
      for (std::size_t row = 0; row < 3; ++row) {
        relative_error_arm1[row] = target_relative[row][3] - (arm2_tcp[row][3] - arm1_tcp[row][3]);
      }
    }
    rotate_spatial(main_R_arm1, relative_error_arm1, relative_error_main);
    for (std::size_t row = 0; row < 6; ++row) {
      internal::write_matrix_value(task_pose_error, env, row, absolute_error_main[row]);
      internal::write_matrix_value(task_pose_error, env, 6 + row, relative_error_main[row]);
    }

    const Scalar relative_position[3]{arm2_tcp[0][3] - arm1_tcp[0][3],
                                      arm2_tcp[1][3] - arm1_tcp[1][3],
                                      arm2_tcp[2][3] - arm1_tcp[2][3]};
    for (std::size_t joint = 0; joint < kArmJointDim; ++joint) {
      Scalar arm1_column[6]{};
      Scalar arm2_column_local[6]{};
      for (std::size_t row = 0; row < 6; ++row) {
        arm1_column[row] = internal::read_tensor3_value<Scalar>(arm1_jacobian, env, row, joint);
        arm2_column_local[row] =
            internal::read_tensor3_value<Scalar>(arm2_jacobian, env, row, joint);
      }
      Scalar arm2_column_arm1[6]{};
      rotate_spatial(arm1_R_arm2, arm2_column_local, arm2_column_arm1);
      Scalar absolute_arm1_column[6]{};
      Scalar absolute_arm2_column[6]{};
      Scalar relative_arm1_column[6]{};
      Scalar relative_arm2_column[6]{};
      for (std::size_t row = 0; row < 6; ++row) {
        absolute_arm1_column[row] = static_cast<Scalar>(0.5) * arm1_column[row];
        absolute_arm2_column[row] = static_cast<Scalar>(0.5) * arm2_column_arm1[row];
        relative_arm1_column[row] = -arm1_column[row];
        relative_arm2_column[row] = arm2_column_arm1[row];
      }
      const Scalar zero_angular[3]{};
      calculate_midpoint_angular_velocity(
          arm1_tcp, arm2_tcp, arm1_column + 3, zero_angular, absolute_arm1_column + 3);
      calculate_midpoint_angular_velocity(
          arm1_tcp, arm2_tcp, zero_angular, arm2_column_arm1 + 3, absolute_arm2_column + 3);
      if (coordination_mode == BimanualCoordinationMode::kCoupledRigidTransform) {
        relative_arm1_column[0] +=
            relative_position[1] * arm1_column[5] - relative_position[2] * arm1_column[4];
        relative_arm1_column[1] +=
            relative_position[2] * arm1_column[3] - relative_position[0] * arm1_column[5];
        relative_arm1_column[2] +=
            relative_position[0] * arm1_column[4] - relative_position[1] * arm1_column[3];
      }
      Scalar absolute_arm1_main[6]{};
      Scalar absolute_arm2_main[6]{};
      Scalar relative_arm1_main[6]{};
      Scalar relative_arm2_main[6]{};
      rotate_spatial(main_R_arm1, absolute_arm1_column, absolute_arm1_main);
      rotate_spatial(main_R_arm1, absolute_arm2_column, absolute_arm2_main);
      rotate_spatial(main_R_arm1, relative_arm1_column, relative_arm1_main);
      rotate_spatial(main_R_arm1, relative_arm2_column, relative_arm2_main);
      for (std::size_t row = 0; row < 6; ++row) {
        internal::write_tensor3_value(task_jacobian, env, row, joint, absolute_arm1_main[row]);
        internal::write_tensor3_value(task_jacobian, env, row, 7 + joint, absolute_arm2_main[row]);
        internal::write_tensor3_value(task_jacobian, env, 6 + row, joint, relative_arm1_main[row]);
        internal::write_tensor3_value(
            task_jacobian, env, 6 + row, 7 + joint, relative_arm2_main[row]);
      }
    }
    status.flags[env] = 0U;
  }
  return internal::ok();
}

}  // namespace

Result validate_bimanual_frame_config(const BimanualFrameConfig& config) noexcept {
  if (config.main_frame != BimanualMainFrame::kArm1Base &&
      config.main_frame != BimanualMainFrame::kArm2Base &&
      config.main_frame != BimanualMainFrame::kExplicit) {
    return internal::invalid_argument("main_frame is not a supported bimanual frame");
  }
  if (config.main_frame == BimanualMainFrame::kExplicit &&
      config.explicit_reference != BimanualBase::kArm1 &&
      config.explicit_reference != BimanualBase::kArm2) {
    return internal::invalid_argument("explicit_reference must select arm 1 or arm 2");
  }
  double arm1_T_arm2[4][4]{};
  double reference_T_main[4][4]{};
  config_transform(config.arm1_T_arm2, arm1_T_arm2);
  config_transform(config.reference_T_main, reference_T_main);
  if (!rigid_transform_is_valid(arm1_T_arm2)) {
    return internal::invalid_argument("arm1_T_arm2 must be a finite rigid transform");
  }
  if (config.main_frame == BimanualMainFrame::kExplicit &&
      !rigid_transform_is_valid(reference_T_main)) {
    return internal::invalid_argument("reference_T_main must be a finite rigid transform");
  }
  return internal::ok();
}

Result reset_bimanual_task_kinematics(const std::size_t num_envs,
                                      const TensorView& reset_mask,
                                      const MutableTensorView task_pose_error,
                                      const MutableTensorView task_jacobian,
                                      const MutableStatusView status) noexcept {
  if (task_pose_error.device == Device::kCuda || task_jacobian.device == Device::kCuda ||
      status.device == Device::kCuda || reset_mask.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return reset_bimanual_task_kinematics_cuda(
        num_envs, reset_mask, task_pose_error, task_jacobian, status);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (num_envs == 0 || status.flags == nullptr || status.size < num_envs ||
      status.device != Device::kCpu) {
    return internal::invalid_view("invalid bimanual reset storage");
  }
  for (std::size_t env = 0; env < num_envs; ++env) {
    const bool selected = internal::reset_mask_value(reset_mask, env);
    if (selected) {
      if (task_pose_error.dtype == DType::kFloat32) {
        zero_environment<float>(task_pose_error, task_jacobian, env);
      } else {
        zero_environment<double>(task_pose_error, task_jacobian, env);
      }
    }
    status.flags[env] = selected ? status_bit(StatusFlag::kResetApplied) : 0U;
  }
  return internal::ok();
}

Result step_bimanual_task_kinematics(const BimanualCoordinationMode coordination_mode,
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
  if (arm1_tcp_pose.data == nullptr || !internal::is_float_dtype(arm1_tcp_pose.dtype) ||
      !internal::is_row_major_tensor3(arm1_tcp_pose) || arm1_tcp_pose.shape[0] <= 0) {
    return internal::invalid_view("arm1_tcp_pose must be row-major [batch_size,4,4]");
  }
  if (coordination_mode != BimanualCoordinationMode::kDecoupledCooperative &&
      coordination_mode != BimanualCoordinationMode::kCoupledRigidTransform) {
    return internal::invalid_argument("unsupported bimanual coordination mode");
  }
  const std::size_t num_envs = static_cast<std::size_t>(arm1_tcp_pose.shape[0]);
  const Device device = arm1_tcp_pose.device;
  const TensorView poses[]{desired_absolute_pose, arm2_tcp_pose, desired_relative_pose};
  Result result = validate_pose_view(arm1_tcp_pose, arm1_tcp_pose.dtype, device, num_envs);
  for (const TensorView& pose : poses) {
    if (result.ok()) {
      result = validate_pose_view(pose, arm1_tcp_pose.dtype, device, num_envs);
    }
  }
  if (result.ok()) {
    result = validate_matrix_view(arm1_jacobian, arm1_tcp_pose.dtype, device, num_envs, 6, 7);
  }
  if (result.ok()) {
    result = validate_matrix_view(arm2_jacobian, arm1_tcp_pose.dtype, device, num_envs, 6, 7);
  }
  if (result.ok()) {
    result = validate_output_view(
        task_pose_error, task_jacobian, status, arm1_tcp_pose.dtype, device, num_envs);
  }
  if (!result.ok()) {
    return result;
  }
  if (device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_bimanual_task_kinematics_cuda(coordination_mode,
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
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (device != Device::kCpu) {
    return internal::invalid_view("bimanual task device is unsupported");
  }
  return arm1_tcp_pose.dtype == DType::kFloat32 ? step_cpu<float>(coordination_mode,
                                                                  config,
                                                                  desired_absolute_pose,
                                                                  arm1_tcp_pose,
                                                                  arm2_tcp_pose,
                                                                  desired_relative_pose,
                                                                  arm1_jacobian,
                                                                  arm2_jacobian,
                                                                  task_pose_error,
                                                                  task_jacobian,
                                                                  status,
                                                                  num_envs)
                                                : step_cpu<double>(coordination_mode,
                                                                   config,
                                                                   desired_absolute_pose,
                                                                   arm1_tcp_pose,
                                                                   arm2_tcp_pose,
                                                                   desired_relative_pose,
                                                                   arm1_jacobian,
                                                                   arm2_jacobian,
                                                                   task_pose_error,
                                                                   task_jacobian,
                                                                   status,
                                                                   num_envs);
}

Result step_bimanual_task_velocity_error(const TensorView& desired_task_velocity,
                                         const TensorView& joint_velocity,
                                         const TensorView& task_jacobian,
                                         const MutableTensorView& task_velocity_error) noexcept {
  if (desired_task_velocity.data == nullptr ||
      !internal::is_float_dtype(desired_task_velocity.dtype) ||
      !internal::is_row_major_matrix(desired_task_velocity) ||
      desired_task_velocity.shape[0] <= 0 || desired_task_velocity.shape[1] != 12) {
    return internal::invalid_view("desired_task_velocity must be row-major [batch_size,12]");
  }
  const std::int64_t batch_size = desired_task_velocity.shape[0];
  const bool has_matching_velocity =
      joint_velocity.data != nullptr && joint_velocity.dtype == desired_task_velocity.dtype &&
      joint_velocity.device == desired_task_velocity.device &&
      internal::is_row_major_matrix(joint_velocity) && joint_velocity.shape[0] == batch_size &&
      joint_velocity.shape[1] == 14;
  const bool has_matching_jacobian =
      task_jacobian.data != nullptr && task_jacobian.dtype == desired_task_velocity.dtype &&
      task_jacobian.device == desired_task_velocity.device &&
      internal::is_row_major_tensor3(task_jacobian) && task_jacobian.shape[0] == batch_size &&
      task_jacobian.shape[1] == 12 && task_jacobian.shape[2] == 14;
  const bool has_matching_output = task_velocity_error.data != nullptr &&
                                   task_velocity_error.dtype == desired_task_velocity.dtype &&
                                   task_velocity_error.device == desired_task_velocity.device &&
                                   internal::is_row_major_matrix(task_velocity_error) &&
                                   task_velocity_error.shape[0] == batch_size &&
                                   task_velocity_error.shape[1] == 12;
  if (!has_matching_velocity || !has_matching_jacobian || !has_matching_output) {
    return internal::invalid_view("bimanual task velocity tensors have incompatible shapes");
  }
  if (desired_task_velocity.device == Device::kCuda) {
#ifdef NVIDIASRLCONTROLLERS_ENABLE_CUDA
    return step_bimanual_task_velocity_error_cuda(
        desired_task_velocity, joint_velocity, task_jacobian, task_velocity_error);
#else
    return internal::unsupported("CUDA tensor execution requires ENABLE_CUDA=ON");
#endif
  }
  if (desired_task_velocity.device != Device::kCpu) {
    return internal::invalid_view("bimanual task velocity device is unsupported");
  }
  const auto execute = [&](auto scalar_tag) {
    using Scalar = decltype(scalar_tag);
    for (std::size_t env = 0; env < static_cast<std::size_t>(batch_size); ++env) {
      for (std::size_t row = 0; row < kTaskDim; ++row) {
        Scalar value = internal::read_matrix_value<Scalar>(desired_task_velocity, env, row);
        for (std::size_t joint = 0; joint < kJointDim; ++joint) {
          value -= internal::read_tensor3_value<Scalar>(task_jacobian, env, row, joint) *
                   internal::read_matrix_value<Scalar>(joint_velocity, env, joint);
        }
        internal::write_matrix_value(task_velocity_error, env, row, value);
      }
    }
  };
  if (desired_task_velocity.dtype == DType::kFloat32) {
    execute(float{});
  } else {
    execute(double{});
  }
  return internal::ok();
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
