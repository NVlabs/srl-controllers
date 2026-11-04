# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)

foreach(REQUIRED_VARIABLE LIBRARY_FILE NM_TOOL)
  if(NOT DEFINED ${REQUIRED_VARIABLE} OR "${${REQUIRED_VARIABLE}}" STREQUAL "")
    message(FATAL_ERROR "${REQUIRED_VARIABLE} is required")
  endif()
endforeach()
if(NOT EXISTS "${LIBRARY_FILE}")
  message(FATAL_ERROR "Controllers library does not exist: ${LIBRARY_FILE}")
endif()

find_program(CXXFILT_TOOL c++filt REQUIRED)
execute_process(
  COMMAND "${NM_TOOL}" -D --defined-only "${LIBRARY_FILE}"
  COMMAND "${CXXFILT_TOOL}"
  RESULT_VARIABLE SYMBOL_RESULT
  OUTPUT_VARIABLE SYMBOLS
  ERROR_VARIABLE SYMBOL_ERROR
)
if(NOT SYMBOL_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not inspect exported symbols: ${SYMBOL_ERROR}")
endif()

set(PUBLIC_TYPES
  ProportionalBlock
  IntegratorBlock
  DifferentiatorBlock
  JointStiffness
  JointDamping
  JointDampingDoubleDiagonalization
  CartesianStiffness
  VariableCartesianStiffness
  CartesianDamping
  CartesianDampingDoubleDiagonalization
  CartesianDampingFactorizationDesign
  CartesianVirtualMass
  ShapingFunction
  CommandObserver
  FlowFunnel
  InverseAPF
  InertialCoupling
  EnergyTank
  VariableCartesianUnifiedForceImpedanceController
  TorqueFilter
  JointImpedanceController
  NullspaceController
  OperationalSpaceMass
  CartesianImpedanceController
  BimanualCartesianImpedanceController
  BimanualTaskKinematics
  CartesianAdmittancePoseController
  OperationalSpaceController
  OperationalSpaceInverseDynamicsController
  CartesianForceController
  CartesianUnifiedForceImpedanceController
  StaticJointImpedanceController
  StaticCartesianImpedanceController
  CartesianUnifiedForceAdmittanceController
  VariableCartesianImpedanceController
  TaskWrenchToJointTorqueController
  CartesianImpedanceTorqueController
  VariableCartesianImpedanceTorqueController
  CartesianUnifiedForceImpedanceTorqueController
  BimanualCartesianImpedanceTorqueController
)
foreach(PUBLIC_TYPE IN LISTS PUBLIC_TYPES)
  string(FIND "${SYMBOLS}" "nvidia::srl::controllers::${PUBLIC_TYPE}::" TYPE_INDEX)
  if(TYPE_INDEX EQUAL -1)
    message(FATAL_ERROR "The public ${PUBLIC_TYPE} lifecycle is not exported")
  endif()
endforeach()
string(FIND "${SYMBOLS}" "nvidia::srl::controllers::get_full_version" VERSION_INDEX)
if(VERSION_INDEX EQUAL -1)
  message(FATAL_ERROR "get_full_version is not exported")
endif()

string(JOIN "|" PUBLIC_TYPES_REGEX ${PUBLIC_TYPES})
string(REPLACE "\n" ";" SYMBOL_LINES "${SYMBOLS}")
foreach(SYMBOL_LINE IN LISTS SYMBOL_LINES)
  if(SYMBOL_LINE MATCHES "nvidia::srl::controllers::")
    if(NOT SYMBOL_LINE MATCHES "nvidia::srl::controllers::(${PUBLIC_TYPES_REGEX})::" AND
       NOT SYMBOL_LINE MATCHES "nvidia::srl::controllers::(get_full_version|all_controller_example_specs|controller_example_specs|find_controller_example_spec|merge_status|reset_safety_state|apply_safe_output_policy)")
      message(FATAL_ERROR "Non-public controller symbol is exported: ${SYMBOL_LINE}")
    endif()
  endif()
endforeach()
