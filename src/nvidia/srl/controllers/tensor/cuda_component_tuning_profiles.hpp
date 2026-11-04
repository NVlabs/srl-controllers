/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file cuda_component_tuning_profiles.hpp
 * @brief Per-launch-family CUDA geometry profiles for tensor controllers.
 */

#pragma once

#include <type_traits>

#include "nvidia/srl/controllers/tensor/types.hpp"

namespace nvidia {
namespace srl {
namespace controllers {
namespace detail {
namespace tensor {

enum class CudaTensorLaunchFamily {
  kCartesianApplyMatrix6,
  kCartesianApplyShaping,
  kCartesianApplyDampingDesign,
  kCartesianApplyFlow,
  kCartesianApplyInverseApf,
  kCartesianApplyImpedance,
  kCartesianApplyOperationalSpace,
  kCartesianApplyAdmittance,
  kCartesianApplyVariableStiffness,
  kCartesianApplyForce,
  kCartesianApplyUnifiedForceImpedance,
  kCartesianApplyVariableImpedance,
  kCartesianApplyVariableUnified,
  kCartesianStepShaping,
  kCartesianStepCommandObserver,
  kCartesianStepFlow,
  kCartesianStepInverseApf,
  kCartesianStepDampingDesign,
  kCartesianStepMatrix6,
  kCartesianStepVirtualMass,
  kCartesianResetInertial,
  kCartesianResetAdmittance,
  kCartesianResetForce,
  kCartesianResetVariableStiffness,
  kCartesianStepOperationalSpace,
  kCartesianStepInertial,
  kCartesianStepImpedance,
  kCartesianStepAdmittance,
  kCartesianStepVariableStiffness,
  kCartesianStepForce,
  kCartesianCombineSum,
  kCartesianCombineVariableImpedance,
  kCartesianResetStateless,
  kCartesianClearCommandObserver,
  kCartesianClearInertial,
  kEnergyApply,
  kEnergyReset,
  kEnergyStep,
  kGenericApplyProportional,
  kGenericClearStatus,
  kGenericApplyIntegrator,
  kGenericStepProportional,
  kGenericResetIntegrator,
  kGenericResetDifferentiator,
  kGenericStepIntegrator,
  kGenericStepDifferentiator,
  kGenericResetProportional,
  kJointApplyDiagonal,
  kJointApplyDense,
  kJointApplyImpedance,
  kJointApplyNullspace,
  kJointStepDiagonal,
  kJointStepDense,
  kJointStepDoubleDiagonalization,
  kJointStepImpedanceGeneric,
  kJointStepNullspace,
  kJointResetStateless,
  kHighLevelApplyStaticImpedance,
  kHighLevelStepStaticImpedance,
  kHighLevelApplyUnifiedForceAdmittance,
  kHighLevelStepUnifiedForceAdmittance,
};

struct CudaTensorLaunchTuningProfile {
  int sm{0};
  CudaTensorLaunchFamily family{};
  DType dtype{DType::kUnknown};
  int threads_per_block{0};
};

#ifndef NVIDIASRLCONTROLLERS_CUDA_BUILD_SM
#error "NVIDIASRLCONTROLLERS_CUDA_BUILD_SM must be provided by the CUDA build"
#endif

#ifndef NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE
#define NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE 0
#endif

constexpr int kCudaComponentTuningBuildSm = NVIDIASRLCONTROLLERS_CUDA_BUILD_SM;

#define NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE(SM, FAMILY, DTYPE, THREADS) \
  CudaTensorLaunchTuningProfile{SM, CudaTensorLaunchFamily::FAMILY, DType::DTYPE, THREADS},
#define NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_SCALARS(SM, FAMILY, THREADS) \
  NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE(SM, FAMILY, kFloat32, THREADS)     \
  NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE(SM, FAMILY, kFloat64, THREADS)
inline constexpr CudaTensorLaunchTuningProfile kCudaTensorLaunchTuningProfiles[] = {
#include "tensor/cuda_component_tuning_profiles.inc"
};
#undef NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_SCALARS
#undef NVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE

constexpr int cuda_tensor_launch_threads(const CudaTensorLaunchFamily family, const DType dtype) {
  for (const CudaTensorLaunchTuningProfile& profile : kCudaTensorLaunchTuningProfiles) {
    if (profile.sm == kCudaComponentTuningBuildSm && profile.family == family &&
        profile.dtype == dtype) {
      return profile.threads_per_block;
    }
  }
  return 0;
}

constexpr int cuda_tensor_safe_fallback_threads(const CudaTensorLaunchFamily family) {
  // These kernels use one dynamically allocated shared-memory workspace for
  // the environment handled by a block. Launching more than one thread per
  // block would alias that workspace between environments.
  return family == CudaTensorLaunchFamily::kJointStepDoubleDiagonalization ||
                 family == CudaTensorLaunchFamily::kJointStepImpedanceGeneric ||
                 family == CudaTensorLaunchFamily::kJointStepNullspace
             ? 1
             : 128;
}

template <CudaTensorLaunchFamily Family, DType Dtype>
constexpr int cuda_tensor_tuned_threads() {
  constexpr int threads = cuda_tensor_launch_threads(Family, Dtype);
#if NVIDIASRLCONTROLLERS_CUDA_REQUIRE_COMPONENT_TUNING_PROFILE
  static_assert(threads > 0,
                "every CUDA tensor launch family must have a committed profile for this SM");
  return threads;
#else
  return threads == 0 ? cuda_tensor_safe_fallback_threads(Family) : threads;
#endif
}

template <CudaTensorLaunchFamily Family, typename Scalar>
constexpr int cuda_tensor_tuned_threads() {
  static_assert(std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value,
                "CUDA tensor tuning profiles support float and double scalar types");
  constexpr DType dtype = std::is_same<Scalar, float>::value ? DType::kFloat32 : DType::kFloat64;
  return cuda_tensor_tuned_threads<Family, dtype>();
}

}  // namespace tensor
}  // namespace detail
}  // namespace controllers
}  // namespace srl
}  // namespace nvidia
