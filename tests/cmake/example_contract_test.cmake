# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

function(read_example_source SOURCE_FILE OUTPUT_NAME)
  file(READ "${SOURCE_FILE}" SOURCE_TEXT)
  if(SOURCE_FILE MATCHES "/franka/(franka_virtual_walls_controller|franka_artificial_potential_field_controller)\\.cpp$")
    if(NOT SOURCE_TEXT MATCHES "run_cartesian_field_example")
      message(FATAL_ERROR "Field launcher must use the common prepared runner: ${SOURCE_FILE}")
    endif()
    file(READ "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_cartesian_field.hpp" FIELD_RUNNER_TEXT)
    string(APPEND SOURCE_TEXT "${FIELD_RUNNER_TEXT}")
  endif()
  set(${OUTPUT_NAME} "${SOURCE_TEXT}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED CONTROLLERS_SOURCE_DIR OR "${CONTROLLERS_SOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR "CONTROLLERS_SOURCE_DIR must be set")
endif()

file(GLOB FRANKA_SOURCES
  "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_*.cpp")
list(LENGTH FRANKA_SOURCES FRANKA_SOURCE_COUNT)
if(NOT FRANKA_SOURCE_COUNT EQUAL 9)
  message(FATAL_ERROR "Expected nine Franka controller examples, found ${FRANKA_SOURCE_COUNT}")
endif()

set(RIZON_CMAKE "${CONTROLLERS_SOURCE_DIR}/examples/rizon/CMakeLists.txt")
set(RIZON_SOURCE "${CONTROLLERS_SOURCE_DIR}/examples/rizon/rizon_controller_example.cpp")
set(RIZON_UTILS "${CONTROLLERS_SOURCE_DIR}/examples/rizon/rizon_utils.hpp")
set(RIZON_WORKSPACES "${CONTROLLERS_SOURCE_DIR}/examples/rizon/rizon_workspaces.hpp")
set(RIZON_VERSIONS "${CONTROLLERS_SOURCE_DIR}/examples/rizon/FlexivRdkVersions.cmake")
set(RIZON_DEPENDENCY_PINS
    "${CONTROLLERS_SOURCE_DIR}/examples/rizon/FlexivRdkDependencyPins.cmake")
set(RIZON_DEPENDENCY_PATCH
    "${CONTROLLERS_SOURCE_DIR}/examples/rizon/PatchFlexivRdkDependencies.cmake")
set(RIZON_CONSUMER "${CONTROLLERS_SOURCE_DIR}/examples/rizon/consumer/CMakeLists.txt")
foreach(RIZON_FILE IN ITEMS
    ${RIZON_CMAKE}
    ${RIZON_SOURCE}
    ${RIZON_UTILS}
    ${RIZON_WORKSPACES}
    ${RIZON_VERSIONS}
    ${RIZON_DEPENDENCY_PINS}
    ${RIZON_DEPENDENCY_PATCH}
    ${RIZON_CONSUMER})
  if(NOT EXISTS "${RIZON_FILE}")
    message(FATAL_ERROR "Missing Rizon example contract file: ${RIZON_FILE}")
  endif()
endforeach()
file(READ "${RIZON_CMAKE}" RIZON_CMAKE_TEXT)
file(READ "${RIZON_SOURCE}" RIZON_SOURCE_TEXT)
file(READ "${RIZON_UTILS}" RIZON_UTILS_TEXT)
file(READ "${RIZON_WORKSPACES}" RIZON_WORKSPACES_TEXT)
file(READ "${RIZON_VERSIONS}" RIZON_VERSIONS_TEXT)
file(READ "${RIZON_DEPENDENCY_PINS}" RIZON_DEPENDENCY_PINS_TEXT)
file(READ "${RIZON_DEPENDENCY_PATCH}" RIZON_DEPENDENCY_PATCH_TEXT)
file(READ "${RIZON_CONSUMER}" RIZON_CONSUMER_TEXT)
set(RIZON_EXAMPLE_NAMES
  rizon_cartesian_admittance_controller
  rizon_cartesian_force_impedance_controller
  rizon_cartesian_force_impedance_controller_in_ee
  rizon_cartesian_impedance_controller
  rizon_joint_impedance_controller
  rizon_variable_cartesian_impedance_controller)
foreach(RIZON_EXAMPLE IN LISTS RIZON_EXAMPLE_NAMES)
  if(NOT RIZON_CMAKE_TEXT MATCHES "${RIZON_EXAMPLE}" OR
     NOT RIZON_SOURCE_TEXT MATCHES "${RIZON_EXAMPLE}")
    message(FATAL_ERROR "Missing configured Rizon counterpart: ${RIZON_EXAMPLE}")
  endif()
endforeach()
foreach(RIZON_VERSION IN ITEMS 1.9.1 1.9.2 1.9.3 1.9.4)
  string(FIND "${RIZON_VERSIONS_TEXT}" "${RIZON_VERSION}" VERSION_INDEX)
  if(VERSION_INDEX EQUAL -1)
    message(FATAL_ERROR "Missing supported RDK version metadata for ${RIZON_VERSION}")
  endif()
endforeach()
if(NOT RIZON_DEPENDENCY_PINS_TEXT MATCHES "FLEXIV_RDK_FAST_DDS_COMMIT" OR
   NOT RIZON_DEPENDENCY_PINS_TEXT MATCHES "FLEXIV_RDK_BOOST_SHA256" OR
   NOT RIZON_DEPENDENCY_PATCH_TEXT MATCHES "THIRDPARTY_Asio=FORCE" OR
   NOT RIZON_DEPENDENCY_PATCH_TEXT MATCHES
       "CMAKE_DISABLE_FIND_PACKAGE_foonathan_memory=ON")
  message(FATAL_ERROR "Static RDK dependencies must use controller-owned immutable pins")
endif()
string(FIND "${RIZON_CONSUMER_TEXT}"
  "find_package(flexiv_rdk \${FLEXIV_RDK_EXPECTED_VERSION} EXACT REQUIRED CONFIG)"
  RIZON_FIND_PACKAGE_INDEX)
string(FIND "${RIZON_CONSUMER_TEXT}"
  "\$ORIGIN/../\${CMAKE_INSTALL_LIBDIR}" RIZON_INSTALL_RPATH_INDEX)
if(RIZON_CMAKE_TEXT MATCHES "/usr/local" OR
   NOT RIZON_CMAKE_TEXT MATCHES "install\\(DIRECTORY" OR
   NOT RIZON_CMAKE_TEXT MATCHES "rizon_examples_project" OR
   RIZON_FIND_PACKAGE_INDEX EQUAL -1 OR
   RIZON_INSTALL_RPATH_INDEX EQUAL -1)
  message(FATAL_ERROR "Rizon examples must use only the relocatable local RDK installation")
endif()
string(FIND "${RIZON_UTILS_TEXT}" "arguments.run_duration_seconds, arguments.non_rt_requested"
  RIZON_CONFIRM_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "rdk::Robot robot" RIZON_CLIENT_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "!response.empty()" RIZON_CONFIRMATION_INPUT_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "description.initial_joint_position, startup_failure"
  RIZON_INITIAL_MOTION_INDEX)
string(FIND "${RIZON_UTILS_TEXT}"
  "if (!confirm_controller_start(description.initial_joint_position))"
  RIZON_CONTROLLER_CONFIRM_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "rdk::Model model" RIZON_MODEL_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "arguments.non_rt_requested ? 0 : scheduler.max_priority()"
  RIZON_SCHEDULER_MODE_INDEX)
string(FIND "${RIZON_UTILS_TEXT}" "        -1);" RIZON_UNBOUND_AFFINITY_INDEX)
if(RIZON_CONFIRM_INDEX EQUAL -1 OR RIZON_CLIENT_INDEX EQUAL -1 OR
   RIZON_CONFIRM_INDEX GREATER RIZON_CLIENT_INDEX)
  message(FATAL_ERROR "Rizon examples must confirm before constructing the RDK client")
endif()
if(RIZON_INITIAL_MOTION_INDEX EQUAL -1 OR RIZON_CONTROLLER_CONFIRM_INDEX EQUAL -1 OR
   RIZON_MODEL_INDEX EQUAL -1 OR RIZON_INITIAL_MOTION_INDEX GREATER RIZON_CONTROLLER_CONFIRM_INDEX OR
   RIZON_CONTROLLER_CONFIRM_INDEX GREATER RIZON_MODEL_INDEX)
  message(FATAL_ERROR
    "Rizon examples must move, confirm again, then capture controller initial state")
endif()
if(NOT RIZON_UTILS_TEXT MATCHES "--non-rt" OR
   NOT RIZON_UTILS_TEXT MATCHES "--duration-seconds=" OR
   NOT RIZON_UTILS_TEXT MATCHES "rdk::Mode::NRT_JOINT_POSITION" OR
   NOT RIZON_UTILS_TEXT MATCHES "robot\\.SendJointPosition" OR
   NOT RIZON_UTILS_TEXT MATCHES "kInitialJointPosition" OR
   RIZON_CONFIRMATION_INPUT_INDEX EQUAL -1 OR
   RIZON_SCHEDULER_MODE_INDEX EQUAL -1 OR RIZON_UNBOUND_AFFINITY_INDEX EQUAL -1 OR
   NOT RIZON_UTILS_TEXT MATCHES "model\\.Update\\(state\\.q, state\\.dq\\)" OR
   NOT RIZON_UTILS_TEXT MATCHES "model\\.c\\(\\)" OR
   NOT RIZON_UTILS_TEXT MATCHES "desired_torques \\+= cycle_data\\.coriolis_torque" OR
   NOT RIZON_UTILS_TEXT MATCHES "description\\.friction_compensation_percent" OR
   NOT RIZON_UTILS_TEXT MATCHES "description\\.torque_magnitude_limit_factor" OR
   NOT RIZON_UTILS_TEXT MATCHES "kTorqueMagnitudeLimitFactor = 1\\.0" OR
   NOT RIZON_UTILS_TEXT MATCHES "friction_compensation_percent\\{100\\.0\\}" OR
   NOT RIZON_UTILS_TEXT MATCHES "kTorqueSlewRateNmPerSecond = 1000\\.0" OR
   NOT RIZON_UTILS_TEXT MATCHES "filter_torques\\(desired_torques" OR
   NOT RIZON_UTILS_TEXT MATCHES "state_timestamp_nanoseconds < previous_state_timestamp_nanoseconds" OR
   NOT RIZON_UTILS_TEXT MATCHES "state_timestamp_nanoseconds == previous_state_timestamp_nanoseconds" OR
   NOT RIZON_UTILS_TEXT MATCHES "robot\\.SwitchMode\\(rdk::Mode::RT_JOINT_TORQUE\\)" OR
   NOT RIZON_UTILS_TEXT MATCHES "robot_guard\\.arm\\(\\)")
  message(FATAL_ERROR "Rizon lifecycle, dynamics, or torque safety contract is incomplete")
endif()
if(RIZON_UTILS_TEXT MATCHES "Run summary:|Torque joint |print_torque_diagnostics|print_summary")
  message(FATAL_ERROR "Rizon examples must not print runtime diagnostic summaries")
endif()
if(RIZON_WORKSPACES_TEXT MATCHES
   "kMaximumPoseError|final_measured_force_newtons_|maximum_absolute_force_newtons_")
  message(FATAL_ERROR "Rizon force examples must not impose outcome assumptions or summaries")
endif()
if(NOT RIZON_SOURCE_TEXT MATCHES
   "\"Rizon joint impedance hold\",[\n ]+\"Command:[^\"]+\"[\n ]+\"Feedback:[^\"]+\",[\n ]+1\\.0,[\n ]+100\\.0")
  message(FATAL_ERROR
    "Rizon joint impedance must use 100% torque magnitude and friction compensation")
endif()
if(RIZON_UTILS_TEXT MATCHES "ClearFault|RunAutoRecovery|ExecutePlan" OR
   RIZON_SOURCE_TEXT MATCHES "ClearFault|RunAutoRecovery|ExecutePlan" OR
   RIZON_WORKSPACES_TEXT MATCHES "ClearFault|RunAutoRecovery|ExecutePlan|has_FT_sensor" OR
   RIZON_WORKSPACES_TEXT MATCHES "\\.init\\(|\\.compute\\(")
  message(FATAL_ERROR "Rizon examples contain a forbidden lifecycle or retired API behavior")
endif()
if(NOT RIZON_WORKSPACES_TEXT MATCHES "JointImpedanceController" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "CartesianAdmittancePoseController" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "CartesianUnifiedForceImpedanceController" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "VariableCartesianImpedanceController")
  message(FATAL_ERROR "Rizon examples must use the root controller API")
endif()
if(NOT RIZON_SOURCE_TEXT MATCHES "run_cartesian_motion_example" OR
   NOT RIZON_UTILS_TEXT MATCHES "RT_CARTESIAN_MOTION_FORCE" OR
   NOT RIZON_UTILS_TEXT MATCHES "StreamCartesianMotionForce" OR
   NOT RIZON_UTILS_TEXT MATCHES "SetForceControlAxis" OR
   RIZON_WORKSPACES_TEXT MATCHES "acceleration_limit_|twist_limit_|pose_limit_|0\\.031|0\\.201")
  message(FATAL_ERROR
    "Rizon admittance must use unbounded output through the RDK Cartesian motion interface")
endif()
if(NOT RIZON_WORKSPACES_TEXT MATCHES "NullspaceController" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES
     "kNullspaceStiffness\\{120, 80, 60, 40, 30, 20, 10\\}" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "kNullspaceDamping\\{12, 8, 6, 4, 3, 2, 1\\}" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "desired_joint_position_ = initial\\.joint_position")
  message(FATAL_ERROR "Rizon Cartesian impedance nullspace contract is incomplete")
endif()
if(NOT RIZON_WORKSPACES_TEXT MATCHES
     "maximum_stiffness_ = \\{2000, 2000, 2000, 200, 200, 200\\}" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "kMinimumStiffness = 1\\.0" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "kMaximumStiffness = 2000\\.0" OR
   NOT RIZON_WORKSPACES_TEXT MATCHES "kFrequencyHertz = 0\\.1")
  message(FATAL_ERROR "Rizon variable stiffness range or frequency is incorrect")
endif()

set(PHYSICAL_SOURCES ${FRANKA_SOURCES}
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_admittance_controller.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_unified_force_admittance_controller.cpp")
foreach(SOURCE_FILE IN LISTS PHYSICAL_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(NOT SOURCE_TEXT MATCHES "parse_arguments\\(argc, argv" OR
     NOT SOURCE_TEXT MATCHES "--non-rt")
    message(FATAL_ERROR
      "${SOURCE_FILE} must accept one positional robot host/IP plus optional --non-rt")
  endif()
  if(SOURCE_TEXT MATCHES "controllers/tensor|tensor::")
    message(FATAL_ERROR
      "${SOURCE_FILE} must use the root controller API, not its internal tensor kernels")
  endif()
  if(NOT SOURCE_TEXT MATCHES "nvidia/srl/controllers/controllers\\.hpp")
    message(FATAL_ERROR "${SOURCE_FILE} must include the root controller API")
  endif()
  if(SOURCE_TEXT MATCHES "\\.init\\(|\\.compute\\(")
    message(FATAL_ERROR
      "${SOURCE_FILE} must not call the retired scalar controller lifecycle")
  endif()
  if(SOURCE_TEXT MATCHES "setCollisionBehavior|system_clock|log\\.csv")
    message(FATAL_ERROR "${SOURCE_FILE} contains a forbidden hardware-example behavior")
  endif()
  if(SOURCE_FILE MATCHES "/franka/")
    string(FIND "${SOURCE_TEXT}" "franka_example::confirm_execution" CONFIRM_INDEX)
    string(FIND "${SOURCE_TEXT}" "franka::Robot robot" CLIENT_INDEX)
    string(FIND "${SOURCE_TEXT}" "franka_example::move_to_initial_configuration" MOVE_INDEX)
    string(FIND "${SOURCE_TEXT}" "if (!franka_example::confirm_controller_start())"
      CONTROLLER_CONFIRM_INDEX)
    string(FIND "${SOURCE_TEXT}" "robot.readOnce" INITIAL_STATE_INDEX)
    string(FIND "${SOURCE_TEXT}" "robot.control(" CONTROLLER_INDEX)
    if(MOVE_INDEX EQUAL -1 OR CONTROLLER_CONFIRM_INDEX EQUAL -1 OR INITIAL_STATE_INDEX EQUAL -1 OR
       CLIENT_INDEX GREATER MOVE_INDEX OR MOVE_INDEX GREATER CONTROLLER_CONFIRM_INDEX OR
       CONTROLLER_CONFIRM_INDEX GREATER INITIAL_STATE_INDEX)
      message(FATAL_ERROR
        "${SOURCE_FILE} must move, confirm again, then capture controller state")
    endif()
    if(CONTROLLER_CONFIRM_INDEX EQUAL -1 OR CONTROLLER_INDEX EQUAL -1 OR
       MOVE_INDEX GREATER CONTROLLER_CONFIRM_INDEX OR
       CONTROLLER_CONFIRM_INDEX GREATER INITIAL_STATE_INDEX OR
       INITIAL_STATE_INDEX GREATER CONTROLLER_INDEX)
      message(FATAL_ERROR
        "${SOURCE_FILE} must move, confirm again, capture its reference, then activate control")
    endif()
    if(SOURCE_FILE MATCHES "/franka_cartesian_force_impedance_controller_with_tank\\.cpp$" AND
       SOURCE_TEXT MATCHES "--trace|TraceBuffer|write_csv|fstream")
      message(FATAL_ERROR "The Franka energy-tank example must use the standard example CLI")
    endif()
  else()
    if(SOURCE_FILE MATCHES "/cartesian_admittance_controller\\.cpp$" AND
       (NOT SOURCE_TEXT MATCHES "apply_cartesian_pose_offset" OR
        SOURCE_TEXT MATCHES "integrated_twist"))
      message(FATAL_ERROR
        "${SOURCE_FILE} must command the controller's pose offset without separate twist integration")
    endif()
    string(FIND "${SOURCE_TEXT}" "if (!confirm_execution(" CONFIRM_INDEX)
    string(FIND "${SOURCE_TEXT}" "urcl::ExampleRobotWrapper robot" CLIENT_INDEX)
    if(NOT SOURCE_TEXT MATCHES "--duration-seconds=SECONDS" OR
       NOT SOURCE_TEXT MATCHES
         "elapsed_seconds >= arguments\\.run_duration_seconds")
      message(FATAL_ERROR "${SOURCE_FILE} must accept and apply a configurable run duration")
    endif()
    if(SOURCE_TEXT MATCHES
       "pose_limit|kTranslationEnvelope|kRotationEnvelope|out-of-envelope")
      message(FATAL_ERROR "${SOURCE_FILE} must not apply a software pose envelope")
    endif()
    if(NOT SOURCE_TEXT MATCHES "Press Enter to connect" OR
       NOT SOURCE_TEXT MATCHES "!response\\.empty\\(\\)")
      message(FATAL_ERROR "${SOURCE_FILE} must require an empty-line confirmation")
    endif()
    if(NOT SOURCE_TEXT MATCHES "warning-only FIFO" OR
       NOT SOURCE_TEXT MATCHES "accepted here without changing that upstream")
      message(FATAL_ERROR "${SOURCE_FILE} must document the UR --non-rt no-op boundary")
    endif()
  endif()
  if(CONFIRM_INDEX EQUAL -1 OR CLIENT_INDEX EQUAL -1 OR CONFIRM_INDEX GREATER CLIENT_INDEX)
    message(FATAL_ERROR "${SOURCE_FILE} must require confirmation before constructing a robot client")
  endif()
endforeach()

set(UR_INITIAL_MOTION
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/initial_motion.hpp")
file(READ "${UR_INITIAL_MOTION}" UR_INITIAL_MOTION_TEXT)
if(NOT UR_INITIAL_MOTION_TEXT MATCHES "kInitialJointPosition" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "kEndEffectorForceInitialJointPosition" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "0\\.0" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "90\\.0 \\* kDegreesToRadians" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "-90\\.0 \\* kDegreesToRadians" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "135\\.0 \\* kDegreesToRadians" OR
   NOT UR_INITIAL_MOTION_TEXT MATCHES "executor\\.moveJ")
  message(FATAL_ERROR "UR initial motion must use moveJ to the documented joint target")
endif()
foreach(SOURCE_FILE IN ITEMS
    "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_admittance_controller.cpp"
    "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_unified_force_admittance_controller.cpp"
    "${CONTROLLERS_SOURCE_DIR}/examples/ur/joint_impedance_controller.cpp"
    "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_impedance_controller.cpp")
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(NOT SOURCE_TEXT MATCHES "move_to_initial_joint_position")
    message(FATAL_ERROR "${SOURCE_FILE} must execute the shared UR initial motion")
  endif()
  string(FIND "${SOURCE_TEXT}" "move_to_initial_joint_position(" MOVE_INDEX)
  string(FIND "${SOURCE_TEXT}"
    "if (!nvidia::srl::controllers::examples::ur::confirm_controller_start())"
    CONTROLLER_CONFIRM_INDEX)
  string(FIND "${SOURCE_TEXT}" "while (!completed)" CONTROL_INDEX)
  if(CONTROLLER_CONFIRM_INDEX EQUAL -1 OR CONTROL_INDEX EQUAL -1 OR
     MOVE_INDEX GREATER CONTROLLER_CONFIRM_INDEX OR CONTROLLER_CONFIRM_INDEX GREATER CONTROL_INDEX)
    message(FATAL_ERROR "${SOURCE_FILE} must move, confirm again, then start its controller loop")
  endif()
endforeach()

file(READ
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_unified_force_admittance_controller.cpp"
  UR_FORCE_ADMITTANCE_TEXT)
file(READ
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_unified_force_admittance_controller_in_ee.cpp"
  UR_FORCE_ADMITTANCE_EE_TEXT)
file(READ
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_admittance_controller.cpp"
  UR_ADMITTANCE_TEXT)
file(READ "${CONTROLLERS_SOURCE_DIR}/examples/ur/consumer/CMakeLists.txt"
  UR_CONSUMER_CMAKE_TEXT)
if(NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "uses_end_effector_force_target \\? kTargetForceMagnitude : -kTargetForceMagnitude" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "\\(\\*desired_external_wrench\\)\\[kForceAxis\\] = -commanded_robot_force" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "rotation_base_from_controller = measured_pose" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "controller_wrench = rotate_cartesian_vector\\(rotation_controller_from_base" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "controller_pose_offset =" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "update_cartesian_matrix\\(rotation_base_from_controller, stiffness_diagonal" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "update_cartesian_matrix\\(rotation_base_from_controller, damping_diagonal" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "virtual_mass_diagonal, &virtual_mass" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "external_wrench_base = rotate_cartesian_vector" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "desired_wrench_base = rotate_cartesian_vector" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "stiffness_diagonal\\{1000, 1000, 10, 30, 30, 30\\}" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "mass_diagonal\\{15, 15, 1\\.5, 0\\.5, 0\\.5, 0\\.5\\}" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "kForceAxisDampingScale = 1\\.5" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "kEndEffectorForceAxisDampingScale = 3\\.0" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "uses_end_effector_force_target \\? kEndEffectorForceAxisDampingScale" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "controllers::ShapingFunction shaping_function" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "kTargetForceMagnitude / \\(kMaximumShapingDistance - kFullForceDistance\\)" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "maximum_distance" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES
     "shaping_distance" OR
   NOT UR_ADMITTANCE_TEXT MATCHES
     "external_wrench\\[axis\\] = filtered_force" OR
   NOT UR_FORCE_ADMITTANCE_TEXT MATCHES "kEndEffector" OR
   NOT UR_FORCE_ADMITTANCE_EE_TEXT MATCHES "ForceTargetFrame::kEndEffector" OR
   NOT UR_CONSUMER_CMAKE_TEXT MATCHES
     "cartesian_unified_force_admittance_controller_in_ee" OR
   NOT UR_CONSUMER_CMAKE_TEXT MATCHES "UR_UNIFIED_FORCE_ADMITTANCE_NO_MAIN=1")
  message(FATAL_ERROR "UR base/EE force-admittance targets are incomplete")
endif()

set(ADMITTANCE_SOURCES
  "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_admittance.hpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_cartesian_admittance_controller.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_admittance_controller.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_unified_force_admittance_controller.cpp")
foreach(SOURCE_FILE IN LISTS ADMITTANCE_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(SOURCE_FILE MATCHES "/franka_admittance\\.hpp$")
    string(REPLACE " acceleration_limit=unset twist_limit=unset pose_limit=unset" ""
      SOURCE_TEXT "${SOURCE_TEXT}")
  endif()
  if(SOURCE_TEXT MATCHES
     "acceleration_limit|twist_limit|pose_limit|maximum_pose_displacement|integral_lower_limit|integral_upper_limit|out-of-envelope")
    message(FATAL_ERROR "${SOURCE_FILE} must leave admittance and force-integral limits unset")
  endif()
endforeach()
foreach(SOURCE_FILE IN ITEMS
    "${CONTROLLERS_SOURCE_DIR}/examples/isaaclab/_ur5e_admittance_parallel.py"
    "${CONTROLLERS_SOURCE_DIR}/examples/isaaclab/_ur5e_unified_force_admittance_parallel.py")
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(SOURCE_TEXT MATCHES "maximum_joint_step|soft_joint_pos_limits|soft_joint_pos_limit_factor")
    message(FATAL_ERROR "${SOURCE_FILE} must not clip or slew-limit admittance joint targets")
  endif()
endforeach()

set(UR_IMPEDANCE_SOURCES
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/joint_impedance_controller.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_impedance_controller.cpp")
foreach(SOURCE_FILE IN LISTS UR_IMPEDANCE_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  string(FIND "${SOURCE_TEXT}" "confirm_connection(" CONFIRM_INDEX)
  string(FIND "${SOURCE_TEXT}" "urcl::ExampleRobotWrapper robot" CLIENT_INDEX)
  if(CONFIRM_INDEX EQUAL -1 OR CLIENT_INDEX EQUAL -1 OR CONFIRM_INDEX GREATER CLIENT_INDEX OR
     NOT SOURCE_TEXT MATCHES "parse_arguments\\(argc, argv" OR
     NOT SOURCE_TEXT MATCHES "--duration-seconds=SECONDS" OR
     NOT SOURCE_TEXT MATCHES "--non-rt" OR
     NOT SOURCE_TEXT MATCHES "MODE_TORQUE" OR
     NOT SOURCE_TEXT MATCHES "TorqueFilterWorkspace" OR
     NOT SOURCE_TEXT MATCHES "best_effort_zero_torque" OR
     NOT SOURCE_TEXT MATCHES "elapsed_seconds >= arguments\\.run_duration_seconds" OR
     NOT SOURCE_TEXT MATCHES "nvidia/srl/controllers/controllers\\.hpp")
    message(FATAL_ERROR "${SOURCE_FILE} does not satisfy the UR torque-example contract")
  endif()
endforeach()
file(READ "${CONTROLLERS_SOURCE_DIR}/examples/ur/joint_impedance_controller.cpp"
  UR_JOINT_IMPEDANCE_TEXT)
file(READ "${CONTROLLERS_SOURCE_DIR}/examples/ur/cartesian_impedance_controller.cpp"
  UR_CARTESIAN_IMPEDANCE_TEXT)
if(NOT UR_JOINT_IMPEDANCE_TEXT MATCHES
     "kStiffness\\{1000\\.0, 800\\.0, 600\\.0," OR
   NOT UR_JOINT_IMPEDANCE_TEXT MATCHES
     "parse_arguments\\(argc, argv, true, &arguments\\)" OR
   NOT UR_JOINT_IMPEDANCE_TEXT MATCHES
     "torque_limits_for_model\\(arguments\\.robot_model\\)" OR
   NOT UR_JOINT_IMPEDANCE_TEXT MATCHES
     "56\\.5685424948, 101\\.1928851256, 37\\.9473319220" OR
   NOT UR_CARTESIAN_IMPEDANCE_TEXT MATCHES
     "1000\\.0, 1000\\.0, 1000\\.0, 100\\.0, 100\\.0, 100\\.0" OR
   NOT UR_CARTESIAN_IMPEDANCE_TEXT MATCHES
     "kDamping\\{178\\.8854382000" OR
   NOT UR_CARTESIAN_IMPEDANCE_TEXT MATCHES
     "18\\.9736659612\\};")
  message(FATAL_ERROR "UR static impedance gains do not match the documented values")
endif()
file(READ "${CONTROLLERS_SOURCE_DIR}/examples/ur/impedance_common.hpp" UR_IMPEDANCE_COMMON_TEXT)
if(NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "Press Enter to connect" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "announce_torque_start" OR
   UR_IMPEDANCE_COMMON_TEXT MATCHES "response != \"START\"" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "5\\.23\\.0" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "10\\.10\\.0" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "kUr5eTorqueLimitsNm" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES
     "150\\.0, 150\\.0, 150\\.0, 28\\.0, 28\\.0, 28\\.0" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "kUr10eTorqueLimitsNm" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES
     "330\\.0, 330\\.0, 150\\.0, 54\\.0, 54\\.0, 54\\.0" OR
   NOT UR_IMPEDANCE_COMMON_TEXT MATCHES "kTorqueRateLimitNmPerSecond = 500\\.0")
  message(FATAL_ERROR "UR impedance confirmation, version, or torque-limit contract is incomplete")
endif()
file(READ "${CONTROLLERS_SOURCE_DIR}/examples/ur/rtde_torque_output_recipe.txt"
  UR_TORQUE_RECIPE_TEXT)
foreach(REQUIRED_FIELD IN ITEMS timestamp actual_q actual_qd actual_TCP_pose tcp_offset)
  if(NOT UR_TORQUE_RECIPE_TEXT MATCHES "(^|\n)${REQUIRED_FIELD}(\n|$)")
    message(FATAL_ERROR "UR torque RTDE recipe is missing ${REQUIRED_FIELD}")
  endif()
endforeach()
foreach(ROBOT_MODEL IN ITEMS ur5e ur10e)
  if(NOT EXISTS "${CONTROLLERS_SOURCE_DIR}/examples/ur/models/${ROBOT_MODEL}.urdf")
    message(FATAL_ERROR "Missing generated ${ROBOT_MODEL} URDF")
  endif()
endforeach()

foreach(SOURCE_FILE IN LISTS FRANKA_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(SOURCE_FILE MATCHES "/franka_cartesian_admittance_controller\\.cpp$")
    if(NOT SOURCE_TEXT MATCHES "NVIDIA_SRL_FRANKA_FER" OR
       NOT SOURCE_TEXT MATCHES "franka::CartesianPose" OR
       NOT SOURCE_TEXT MATCHES "state\\.O_T_EE_c" OR
       NOT SOURCE_TEXT MATCHES "return franka::CartesianPose\\(state\\.O_T_EE_c\\)" OR
       NOT SOURCE_TEXT MATCHES "reference_initialized = true" OR
       NOT SOURCE_TEXT MATCHES "franka::ControllerMode::kJointImpedance" OR
       NOT SOURCE_TEXT MATCHES "franka::Torques" OR
       NOT SOURCE_TEXT MATCHES "apply_cartesian_pose_offset" OR
       NOT SOURCE_TEXT MATCHES "impedance\\.step" OR
       NOT SOURCE_TEXT MATCHES "franka::MotionFinished\\(command\\)" OR
       SOURCE_TEXT MATCHES "franka::CartesianVelocities|integrated_twist|AdmittanceTracePrinter|std::cout")
      message(FATAL_ERROR "${SOURCE_FILE} must use Cartesian pose for FER and Cartesian impedance torques for FR3 without callback printing")
    endif()
  endif()
  if(NOT SOURCE_TEXT MATCHES "model\\.coriolis")
    message(FATAL_ERROR "${SOURCE_FILE} must add model Coriolis torque")
  endif()
  if(NOT SOURCE_TEXT MATCHES "filtered_torques")
    message(FATAL_ERROR "${SOURCE_FILE} must magnitude- and rate-filter torque commands")
  endif()
endforeach()

file(READ "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_utils.hpp" FRANKA_UTILS_TEXT)
if(NOT FRANKA_UTILS_TEXT MATCHES
     "kMaximumJointTorqueNm\\{87\\.0, 87\\.0, 87\\.0, 87\\.0,[\n ]+12\\.0, 12\\.0, 12\\.0\\}" OR
   NOT FRANKA_UTILS_TEXT MATCHES "franka::kMaxTorqueRate" OR
   NOT FRANKA_UTILS_TEXT MATCHES "franka::RealtimeConfig::kIgnore" OR
   NOT FRANKA_UTILS_TEXT MATCHES "non_rt_requested=")
  message(FATAL_ERROR "Franka torque-filter or non-RT contract is incomplete")
endif()

set(FLEXIV_DUAL_SOURCE
  "${CONTROLLERS_SOURCE_DIR}/examples/flexiv_dual/consumer/flexiv_dual_controller_example.cpp")
file(READ "${FLEXIV_DUAL_SOURCE}" FLEXIV_DUAL_SOURCE_TEXT)
if(FLEXIV_DUAL_SOURCE_TEXT MATCHES "Run summary:")
  message(FATAL_ERROR "Flexiv dual-arm example must not print a runtime diagnostic summary")
endif()
foreach(REQUIRED_SNIPPET IN ITEMS
    "--non-rt"
    "--duration-seconds="
    "response.empty()"
    "config.run_duration_seconds"
    "config.non_rt_requested ? 0 : scheduler.max_priority()"
    "arm1_robot.reached_timeliness_failure_limit()"
    "arm2_robot.reached_timeliness_failure_limit()"
    "arm1_robot.SetTimelinessFailureLimit(2.0)"
    "arm2_robot.SetTimelinessFailureLimit(2.0)"
    "rdk::Mode::NRT_JOINT_POSITION"
    "robot.SendJointPosition"
    "confirm_controller_start()"
    "kInitialJointPosition"
    "arm1_robot.SwitchMode(rdk::Mode::RT_JOINT_TORQUE)"
    "arm2_robot.SwitchMode(rdk::Mode::RT_JOINT_TORQUE)"
    "kTorqueMagnitudeLimitFactor = 1.0"
    "kTorqueSlewRateNmPerSecond = 1000.0"
    "kCircleDiameterMeters = 0.2"
    "kCircleRadiusMeters = 0.5 * kCircleDiameterMeters"
    "kCircleFrequencyHertz = 0.2"
    "std::cos(phase) - 1.0"
    "kCircleAbsoluteStiffness{"
    "1500, 1500, 1500, 150, 150, 150"
    "kCircleRelativeStiffness{500, 500, 500, 50, 50, 50}"
    "kHoldAbsoluteStiffness{300, 300, 300, 30, 30, 30}"
    "kHoldRelativeStiffness{750, 750, 750, 75, 75, 75}"
    "kNullspaceStiffness{40, 30, 20, 16, 10, 6, 2}"
    "kNullspaceDamping{5, 4, 3, 2, 1.5, 1, 0.5}"
    "damping_factors_[axis] = 0.7"
    "BimanualCartesianImpedanceTorqueController"
    "se3_right_jacobian(relative_log).inverse()"
    "initial.arm1_tcp_pose * se3_exp(0.5 * se3_log(relative_transform))"
    "state.q.size() == kArmJointDim && state.theta.size() == kArmJointDim"
    "kEnableGravityCompensation = true"
    "kEnableSoftLimits = true"
    "kFrictionCompensationPercent = 100.0"
    "arm1_robot.StreamJointTorque(arm1_command,"
    "arm2_robot.StreamJointTorque(arm2_command,"
    "non_rt_requested=")
  string(FIND "${FLEXIV_DUAL_SOURCE_TEXT}" "${REQUIRED_SNIPPET}" REQUIRED_SNIPPET_INDEX)
  if(REQUIRED_SNIPPET_INDEX EQUAL -1)
    message(FATAL_ERROR "Flexiv dual-arm non-RT or safety contract is missing: ${REQUIRED_SNIPPET}")
  endif()
endforeach()
string(FIND "${FLEXIV_DUAL_SOURCE_TEXT}"
  "move_to_initial_position(" FLEXIV_DUAL_MOTION_DEFINITION_INDEX)
string(FIND "${FLEXIV_DUAL_SOURCE_TEXT}"
  "if (!move_to_initial_position(" FLEXIV_DUAL_MOTION_CALL_INDEX)
string(FIND "${FLEXIV_DUAL_SOURCE_TEXT}"
  "if (!confirm_controller_start())" FLEXIV_DUAL_CONTROLLER_CONFIRM_INDEX)
string(FIND "${FLEXIV_DUAL_SOURCE_TEXT}"
  "rdk::Model arm1_model" FLEXIV_DUAL_MODEL_INDEX)
if(FLEXIV_DUAL_MOTION_DEFINITION_INDEX EQUAL -1 OR FLEXIV_DUAL_MOTION_CALL_INDEX EQUAL -1 OR
   FLEXIV_DUAL_CONTROLLER_CONFIRM_INDEX EQUAL -1 OR FLEXIV_DUAL_MODEL_INDEX EQUAL -1 OR
   FLEXIV_DUAL_MOTION_CALL_INDEX GREATER FLEXIV_DUAL_CONTROLLER_CONFIRM_INDEX OR
   FLEXIV_DUAL_CONTROLLER_CONFIRM_INDEX GREATER FLEXIV_DUAL_MODEL_INDEX)
  message(FATAL_ERROR
    "Flexiv dual-arm must move both arms, confirm again, then capture task targets")
endif()
set(SYNTHETIC_SOURCES
  "${CONTROLLERS_SOURCE_DIR}/examples/synthetic/rt_reference_loop.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/synthetic/joint_impedance_step.cpp")
foreach(SOURCE_FILE IN LISTS SYNTHETIC_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(NOT SOURCE_TEXT MATCHES "int main\\(\\)")
    message(FATAL_ERROR "${SOURCE_FILE} must remain argument-free")
  endif()
endforeach()

set(ISAACLAB_HARDWARE_TARGETS
  franka_virtual_walls_controller
  franka_artificial_potential_field_controller
  franka_cartesian_admittance_controller
  franka_cartesian_force_impedance_controller
  franka_cartesian_force_impedance_controller_with_tank
  franka_cartesian_force_impedance_controller_in_ee
  franka_cartesian_impedance_controller
  franka_joint_impedance_controller
  franka_variable_cartesian_impedance_controller
  rizon_cartesian_admittance_controller
  rizon_cartesian_force_impedance_controller
  rizon_cartesian_force_impedance_controller_in_ee
  rizon_cartesian_impedance_controller
  rizon_joint_impedance_controller
  rizon_variable_cartesian_impedance_controller
  flexiv_dual_bimanual_cartesian_impedance
  flexiv_dual_bimanual_compliant_hold
  joint_impedance_controller
  cartesian_impedance_controller
  cartesian_admittance_controller
  cartesian_unified_force_admittance_controller
  cartesian_unified_force_admittance_controller_in_ee)
list(LENGTH ISAACLAB_HARDWARE_TARGETS ISAACLAB_TARGET_COUNT)
if(NOT ISAACLAB_TARGET_COUNT EQUAL 22)
  message(FATAL_ERROR "Expected exactly 22 hardware example targets")
endif()
foreach(HARDWARE_TARGET IN LISTS ISAACLAB_HARDWARE_TARGETS)
  set(SOURCE_FILE
      "${CONTROLLERS_SOURCE_DIR}/examples/isaaclab/${HARDWARE_TARGET}_parallel.py")
  if(NOT EXISTS "${SOURCE_FILE}")
    message(FATAL_ERROR "Missing canonical Isaac Lab companion: ${SOURCE_FILE}")
  endif()
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(NOT SOURCE_TEXT MATCHES "main\\(\\\"${HARDWARE_TARGET}\\\"\\)")
    message(FATAL_ERROR "${SOURCE_FILE} must select its matching hardware example spec")
  endif()
endforeach()

file(GLOB_RECURSE CONTROLLER_USER_FACING_SOURCES
  "${CONTROLLERS_SOURCE_DIR}/include/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/src/*.cpp"
  "${CONTROLLERS_SOURCE_DIR}/src/*.cu"
  "${CONTROLLERS_SOURCE_DIR}/src/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.cpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.hpp"
  "${CONTROLLERS_SOURCE_DIR}/examples/*.py"
  "${CONTROLLERS_SOURCE_DIR}/python/src/*.py")
string(CONCAT RETIRED_CONTROLLER_ADJECTIVE "Guard" "ed")
string(CONCAT RETIRED_COMMAND_TYPE "Command" "Guard")
string(CONCAT RETIRED_BACKEND_NAME "command_" "guard")
foreach(SOURCE_FILE IN LISTS CONTROLLER_USER_FACING_SOURCES)
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(SOURCE_TEXT MATCHES
     "${RETIRED_CONTROLLER_ADJECTIVE}|${RETIRED_COMMAND_TYPE}|${RETIRED_BACKEND_NAME}")
    message(FATAL_ERROR "${SOURCE_FILE} reintroduces a retired guard controller API")
  endif()
endforeach()

foreach(ISAACLAB_IMPLEMENTATION IN ITEMS
    _native_components.py)
  set(SOURCE_FILE
      "${CONTROLLERS_SOURCE_DIR}/examples/isaaclab/${ISAACLAB_IMPLEMENTATION}")
  read_example_source("${SOURCE_FILE}" SOURCE_TEXT)
  if(NOT SOURCE_TEXT MATCHES "TorqueFilter" OR
     SOURCE_TEXT MATCHES "TaskWrenchToJointTorqueController")
    message(FATAL_ERROR
      "${SOURCE_FILE} must connect normal controllers to an explicit TorqueFilter")
  endif()
endforeach()

file(READ "${CONTROLLERS_SOURCE_DIR}/examples/isaaclab/_canonical_launcher.py"
     ISAACLAB_LAUNCHER_TEXT)
foreach(REQUIRED_ARGUMENT --num_envs --duration --validate)
  string(FIND "${ISAACLAB_LAUNCHER_TEXT}" "${REQUIRED_ARGUMENT}" ARGUMENT_INDEX)
  if(ARGUMENT_INDEX EQUAL -1)
    message(FATAL_ERROR "Canonical Isaac Lab launcher is missing ${REQUIRED_ARGUMENT}")
  endif()
endforeach()

if(EXISTS "${CONTROLLERS_SOURCE_DIR}/examples/franka/franka_apf_with_tank.cpp")
  message(FATAL_ERROR "Retired Franka tank examples must not be present")
endif()
