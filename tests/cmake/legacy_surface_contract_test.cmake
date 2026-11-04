# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

cmake_minimum_required(VERSION 3.22.1)

if(NOT DEFINED CONTROLLERS_SOURCE_DIR OR "${CONTROLLERS_SOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR "CONTROLLERS_SOURCE_DIR is required")
endif()

set(PUBLIC_SOURCE_DIR "${CONTROLLERS_SOURCE_DIR}/include/nvidia/srl/controllers")
set(CONTROLLER_SOURCE_DIR "${CONTROLLERS_SOURCE_DIR}/src/nvidia/srl/controllers")

file(GLOB ROOT_HEADERS RELATIVE "${PUBLIC_SOURCE_DIR}" "${PUBLIC_SOURCE_DIR}/*.hpp")
set(ALLOWED_ROOT_HEADERS
  controllers.hpp
  equations.hpp
  composition.hpp
  example_specs.hpp
  rt_bootstrap.hpp
  types.hpp
)
foreach(HEADER IN LISTS ROOT_HEADERS)
  if(NOT HEADER IN_LIST ALLOWED_ROOT_HEADERS)
    message(FATAL_ERROR "Retired controller header remains: ${HEADER}")
  endif()
endforeach()

if(EXISTS "${PUBLIC_SOURCE_DIR}/tensor")
  message(FATAL_ERROR "The retired public tensor header tree must not exist")
endif()
if(EXISTS "${PUBLIC_SOURCE_DIR}/prepared_controllers.hpp" OR
   EXISTS "${CONTROLLERS_SOURCE_DIR}/prepared_controllers.cpp" OR
   EXISTS "${CONTROLLERS_SOURCE_DIR}/prepared_python.hpp" OR
   EXISTS "${CONTROLLERS_SOURCE_DIR}/python/src/nvidia/srl/controllers/prepared.py")
  message(FATAL_ERROR "The retired Prepared-named public surface must not exist")
endif()
file(READ "${PUBLIC_SOURCE_DIR}/controllers.hpp" PUBLIC_CONTROLLERS_TEXT)
if(PUBLIC_CONTROLLERS_TEXT MATCHES "Prepared[A-Z]")
  message(FATAL_ERROR "A Prepared-prefixed public C++ type remains")
endif()
file(READ
  "${CONTROLLERS_SOURCE_DIR}/python/src/nvidia/srl/controllers/_api.py"
  PUBLIC_PYTHON_TEXT)
if(PUBLIC_PYTHON_TEXT MATCHES "class Prepared[A-Z]" OR
   PUBLIC_PYTHON_TEXT MATCHES "\"Prepared[A-Z]")
  message(FATAL_ERROR "A Prepared-prefixed public Python type remains")
endif()
if(EXISTS "${CONTROLLERS_SOURCE_DIR}/include/nvidia/srl/controllers_python")
  message(FATAL_ERROR "The retired per-controller Python wrapper tree must not exist")
endif()

file(GLOB ROOT_SOURCES RELATIVE "${CONTROLLER_SOURCE_DIR}" "${CONTROLLER_SOURCE_DIR}/*.cpp")
set(ALLOWED_ROOT_SOURCES composition.cpp equation_adapter.cpp controllers.cpp controllers_python.cpp example_specs.cpp version.cpp)
foreach(SOURCE IN LISTS ROOT_SOURCES)
  if(NOT SOURCE IN_LIST ALLOWED_ROOT_SOURCES)
    message(FATAL_ERROR "Retired scalar controller source remains: ${SOURCE}")
  endif()
endforeach()

foreach(FORBIDDEN_BACKEND_FILE schema.cpp unsupported_controllers.cpp)
  if(EXISTS "${CONTROLLER_SOURCE_DIR}/tensor/${FORBIDDEN_BACKEND_FILE}")
    message(FATAL_ERROR "Retired backend compatibility source remains: ${FORBIDDEN_BACKEND_FILE}")
  endif()
endforeach()

file(GLOB PYTHON_TESTS RELATIVE "${CONTROLLERS_SOURCE_DIR}/python/tests/controllers"
  "${CONTROLLERS_SOURCE_DIR}/python/tests/controllers/*_test.py")
set(ALLOWED_PYTHON_TESTS hardware_profiles_test.py isaac_adapters_test.py documentation_test.py controller_api_test.py version_test.py public_blocks_test.py equations_test.py composition_test.py public_inventory_test.py energy_tank_test.py cartesian_fields_test.py)
foreach(TEST_FILE IN LISTS PYTHON_TESTS)
  if(NOT TEST_FILE IN_LIST ALLOWED_PYTHON_TESTS)
    message(FATAL_ERROR "Retired Python regression test remains: ${TEST_FILE}")
  endif()
endforeach()

set(FORBIDDEN_TEXT
  "nvidia/srl/controllers/block.hpp"
  "nvidia/srl/controllers/utils.hpp"
  "nvidia/srl/controllers/tensor/controllers.hpp"
  "nvidia/srl/controllers/tensor/safe_controllers.hpp"
  "nvidia/srl/controllers/tensor/schema.hpp"
  "nvidia/srl/controllers/tensor/unsupported_controllers.hpp"
  "from nvidia.srl._controllers.nvidia_srl_controllers_cpp import tensor"
  "pybind_tensor"
  "kComponentCatalog"
)
file(GLOB_RECURSE ACTIVE_FILES
  "${CONTROLLERS_SOURCE_DIR}/README.md"
  "${CONTROLLERS_SOURCE_DIR}/include/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/src/*.cpp"
  "${CONTROLLERS_SOURCE_DIR}/src/*.cu"
  "${CONTROLLERS_SOURCE_DIR}/src/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/python/src/*.py"
  "${CONTROLLERS_SOURCE_DIR}/python/tests/*.py"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.md"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.py"
)
foreach(SOURCE_FILE IN LISTS ACTIVE_FILES)
  file(READ "${SOURCE_FILE}" SOURCE_TEXT)
  foreach(PATTERN IN LISTS FORBIDDEN_TEXT)
    string(FIND "${SOURCE_TEXT}" "${PATTERN}" FORBIDDEN_INDEX)
    if(NOT FORBIDDEN_INDEX EQUAL -1)
      message(FATAL_ERROR "Retired surface '${PATTERN}' remains in ${SOURCE_FILE}")
    endif()
  endforeach()
endforeach()

set(FORBIDDEN_DUPLICATE_EQUATIONS
  "tensor/energy_tank_cuda.cu|const Scalar tank_energy"
  "tensor/generic_blocks_cuda.cu|(input_value - previous_value) / dt_value"
  "tensor/torque_filter_cuda.cu|fmax(lower_value, fmin(raw, upper_value))"
  "tensor/cartesian_blocks_cuda.cu|potential_force ="
  "tensor/cartesian_blocks_cuda.cu|static_cast<Scalar>(0.5) * mass_derivative_term - jacobian_term"
  "tensor/cartesian_blocks.cpp|std::atan2(sine"
  "tensor/cartesian_blocks_cuda.cu|atan2(sine"
  "tensor/joint_blocks_cuda.cu|stiffness_torque + damping_torque[row] + acceleration_torque"
  "tensor/joint_blocks_cuda.cu|desired_torque[row] - correction"
)
foreach(ENTRY IN LISTS FORBIDDEN_DUPLICATE_EQUATIONS)
  string(REPLACE "|" ";" PARTS "${ENTRY}")
  list(GET PARTS 0 RELATIVE_SOURCE)
  list(GET PARTS 1 FORBIDDEN_EQUATION)
  file(READ "${CONTROLLER_SOURCE_DIR}/${RELATIVE_SOURCE}" SOURCE_TEXT)
  string(FIND "${SOURCE_TEXT}" "${FORBIDDEN_EQUATION}" FORBIDDEN_INDEX)
  if(NOT FORBIDDEN_INDEX EQUAL -1)
    message(FATAL_ERROR
      "Duplicated controller equation '${FORBIDDEN_EQUATION}' found in ${RELATIVE_SOURCE}")
  endif()
endforeach()
