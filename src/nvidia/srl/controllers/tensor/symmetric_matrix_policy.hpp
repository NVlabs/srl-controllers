/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file symmetric_matrix_policy.hpp
 * @brief Allocation-free numerical policy for small self-adjoint matrices.
 */

#pragma once

#include <cfloat>
#include <cstddef>

#ifdef __CUDACC__
#define NVIDIA_SRL_HOST_DEVICE __host__ __device__
#else
#define NVIDIA_SRL_HOST_DEVICE
#endif

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {
namespace internal {

template <typename Scalar>
struct MachineEpsilon;

template <>
struct MachineEpsilon<float> {
  NVIDIA_SRL_HOST_DEVICE static constexpr float value() { return FLT_EPSILON; }
};

template <>
struct MachineEpsilon<double> {
  NVIDIA_SRL_HOST_DEVICE static constexpr double value() { return DBL_EPSILON; }
};

template <typename Scalar>
struct SymmetricMatrixPolicy {
  NVIDIA_SRL_HOST_DEVICE static constexpr Scalar epsilon() {
    return MachineEpsilon<Scalar>::value();
  }

  NVIDIA_SRL_HOST_DEVICE static constexpr Scalar gamma(const std::size_t dim) {
    const Scalar scaled_epsilon = static_cast<Scalar>(dim) * epsilon();
    return scaled_epsilon / (static_cast<Scalar>(1) - scaled_epsilon);
  }

  NVIDIA_SRL_HOST_DEVICE static constexpr Scalar symmetry_relative_tolerance(
      const std::size_t dim) {
    return static_cast<Scalar>(4) * gamma(dim);
  }

  NVIDIA_SRL_HOST_DEVICE static constexpr Scalar jacobi_relative_tolerance(const std::size_t dim) {
    return gamma(dim) / static_cast<Scalar>(dim > static_cast<std::size_t>(1) ? dim - 1 : 1);
  }

  NVIDIA_SRL_HOST_DEVICE static constexpr Scalar spectral_relative_tolerance(
      const std::size_t dim) {
    // Eight gamma_n covers eigenvalue perturbation; the Jacobi residual contributes gamma_n.
    return static_cast<Scalar>(9) * gamma(dim);
  }
};

}  // namespace internal
}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia

#undef NVIDIA_SRL_HOST_DEVICE
