# Flexiv dual-arm Cartesian impedance examples

These examples use Flexiv RDK 1.9.1 with two independent 7-DoF Rizon arms.

The two bimanual controller examples are:

- `flexiv_dual_bimanual_cartesian_impedance`: one 12D controller for the Cartesian position
  midpoint, symmetric SO(3) orientation midpoint, relative Cartesian displacement, and relative
  orientation. Its midpoint follows a 20 cm diameter circle in the arm-1-base y/z plane, with the
  captured starting pose at the top of the circle.
- `flexiv_dual_bimanual_compliant_hold`: the same 12D controller with a static reference, absolute
  and relative poses captured at startup.

The circular example uses absolute midpoint stiffness `[1500,1500,1500,150,150,150]` and relative
stiffness `[500,500,500,50,50,50]`. The compliant-hold example uses absolute midpoint stiffness
`[300,300,300,30,30,30]` and relative stiffness `[750,750,750,75,75,75]`.

All task vectors and Jacobians use `[linear; angular]` order, metres/radians, and damping ratio
`0.7`. The bimanual executables select the default decoupled-cooperative coordinates. The public
controller equally supports coupled rigid-transform coordinates, including their expected
translation/rotation lever-arm coupling. The two serial numbers must be distinct.

For the circular bimanual and independent baseline examples, the captured initial reference is the
top point of a `0.2 m` diameter circle in the arm-1-base y-z plane. Their speed ramps smoothly from
rest to `0.2 Hz` over 2 seconds, after which it continues at constant frequency. The
compliant-hold reference remains fixed.

## Build and install

The integration is opt-in and qualified only for Linux x86-64. Enabling it clones exact Flexiv RDK
1.9.1 commit `5707121ffadc5a1577c5e0eabf2afa20b9842acb`, verifies the Linux RDK archive digest
`41875278fd89af2db0834458d72f518e8916c26b34fdadb5fa85244caf2cd370`, and builds the upstream
dependency set into a private build-tree prefix. It does not use a system Flexiv RDK installation.
When the single-arm Rizon examples are enabled in the same build, both integrations reuse this one
exact RDK package.

```bash
cmake -S . -B _build \
  -DENABLE_EXAMPLES=ON -DENABLE_PYBIND=OFF \
  -DENABLE_FLEXIV_DUAL_ARM_RDK_EXAMPLE=ON
cmake --build _build --parallel
cmake --install _build --prefix "$PWD/_install"
```

Installing the selected examples copies all three executables, the RDK headers and static library,
and all locally built shared dependencies and CMake packages into the chosen prefix. The executable
uses `$ORIGIN/../lib`, so no build-tree library path is required after installation. The first build
requires network access for the pinned RDK release and its upstream version-pinned dependencies.

## Run

The relative base calibration is a row-major rigid transform `ARM1_BASE_T_ARM2_BASE`, with
translation in metres. It maps coordinates from arm 2's base frame into arm 1's base frame, so the
arms can have any fixed relative translation and rotation. RDK 1.9.1 reports each arm's TCP pose and
Jacobian in that arm's own WORLD/base frame; the public bimanual controller applies the calibration
when it combines the two TCP states. The two model link names must identify each arm's
flange in its respective
`Model::link_names()` result.

```bash
./_install/bin/flexiv_dual_bimanual_cartesian_impedance \
  Rizon4s-123456 Rizon4s-654321 flange flange \
  --arm1-t-arm2=1,0,0,0.8,0,1,0,0,0,0,1,0,0,0,0,1 \
  --duration-seconds=60
```

The run duration defaults to 10 seconds and accepts any finite positive value through
`--duration-seconds=SECONDS`. Append `--non-rt` to select the non-real-time scheduler mode.

After the first empty-line confirmation, both arms switch to `NRT_JOINT_POSITION` and use the RDK
internal motion generator to move to `[0, -15, 0, 75, 0, 0, 0]` degrees. Motion limits are
`0.5 rad/s` and `1 rad/s²` per joint, and both arms must converge within 60 seconds. The program
then stops both arms and requires a second empty-line confirmation before capturing either the two
arm-specific targets or the midpoint/relative targets, then entering torque mode.

The program validates both 7-DoF arms, finite state, calibrated transforms, torque limits,
controller status, and RDK timeliness. Every example uses an independent dynamically consistent
nullspace projection for each arm. The bimanual examples use one full 12×14 task Jacobian and one
complete public bimanual impedance torque controller. Its mass-aware damping, operational-space
mass derivative, `Jdot*qdot`, and desired-acceleration feedforward terms remain active. The per-arm
projected nullspace impedances hold the joint postures captured immediately before torque mode,
using per-joint
stiffness `[40, 30, 20, 16, 10, 6, 2]` and damping `[5, 4, 3, 2, 1.5, 1, 0.5]`. They add no
admittance or force-control layer.
The translational task coordinates are the Cartesian midpoint `0.5 * (p1 + p2)` and displacement
`p2 - p1`, both expressed in the configured main-frame axes. They intentionally contain no
TCP-separation lever-arm term. Orientation uses a symmetric SO(3) midpoint and canonical relative
SO(3) error. Before connecting, each bimanual executable runs a synthetic finite-difference check
that all 12 task-error rates agree with the 12×14 task Jacobian.
Model kinematics and Coriolis torque use link-side `q` with motor-side `dtheta`, matching the RDK
dynamics-engine convention. The bimanual task and nullspace damping also use `dtheta`; noisy
link-side `dq` is retained only in runtime diagnostics. Magnitude is clamped to 100% of each
connected robot's official `RobotInfo::tau_max`; the
example-defined slew limit is `1 Nm/ms` (`1000 Nm/s`) per joint. Each arm has its own RDK model, and
the combined mass matrix is block diagonal because there is no rigid dynamic coupling between the
robot bases. It requests RDK gravity compensation and soft limits with `100%` friction compensation,
and stops both arms after the configured duration or any rejection.

By default, the 1 ms RDK task requests the scheduler's maximum real-time priority. With
`--non-rt`, it instead uses priority 0 and CPU affinity -1, selecting non-real-time scheduling with
no CPU binding. RDK timeliness, robot-state, mode, and torque checks remain enabled.

## Safety and qualification boundary

The program requires an operator to confirm before connection and initial motion, then confirm
again before controller startup. It may enable a healthy disabled robot, but it does not clear
faults, run recovery, home, execute a plan, change collision thresholds, or establish
collision-free motion. Confirm the TCP/payload models, base calibration, link names, free workspace,
robot safety settings, and emergency-stop access before execution.

The controller hot path is allocation-free, but Flexiv RDK state/model calls can allocate and throw.
The two state snapshots and torque sends are sequential API calls within one scheduler callback;
RDK 1.9.1 does not make them one atomic network transaction. A successful build or synthetic/CUDA
test is not physical-robot validation or hard-real-time qualification.


## Complete CLI and evidence

The required positional arguments are two distinct serial numbers and two model flange link names.
`--arm1-t-arm2` is mandatory. The parser also accepts `--main-frame=arm1|arm2|explicit-arm1|explicit-arm2`
(default arm1); explicit selections require `--reference-t-main` with another row-major rigid
transform. These frame descriptors are validated, but the shared circular setpoint remains defined
in arm-1 base and both native controllers remain independent in their own bases. Duration must be
finite and positive; repeated/malformed options are rejected before connection.

Both robots must satisfy the single-arm [Rizon preparation](../rizon/README.md) requirements.
Cancellation at either Enter gate reports failure. Normal completion or a rejected arm stops both
arms; sequential command sends do not prove simultaneous observed completion. Capture both robot
limits/TCP/payload configurations, calibration, source and RDK revisions, console output and stop
reason. No automatic log file is written.

The [Isaac companion](../isaaclab/README.md) uses the same two independent controller stacks and
reference equations. Supply both captured torque-limit vectors and the same base calibration.
Its scene transport does not reproduce RDK communication, friction compensation or readiness gates.
