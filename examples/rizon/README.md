# Rizon examples

These six examples reproduce selected fixed behaviors of the Franka hardware examples on a Flexiv
Rizon through a selectable Flexiv RDK 1.9 release. They use the public controller API with
caller-owned CPU arrays. CUDA is not required. Each performs the documented initial joint motion,
requires two Enter confirmations, and accepts duration and scheduling options.

## Build and install

The integration is qualified for Ubuntu 24.04 on x86-64. `FLEXIV_RDK_VERSION` accepts the exact
versions `1.9.1`, `1.9.2`, `1.9.3`, and `1.9.4`, and defaults to `1.9.1`. Enabling it downloads the
selected immutable RDK source revision and checksum-verified Linux release library into the CMake
build tree. It never searches `/usr/local` or another RDK checkout.

```bash
cmake -S . -B _build \
  -DENABLE_EXAMPLES=ON -DENABLE_PYBIND=OFF \
  -DENABLE_RIZON_EXAMPLES=ON \
  -DFLEXIV_RDK_VERSION=1.9.1
cmake --build _build --parallel
cmake --install _build --prefix "$PWD/_install"
```

The initial build requires network access to GitHub and, for a static RDK, GitLab and the Boost
source archive. The selected install prefix receives:

- all six executables under `bin/`;
- Flexiv RDK headers under `include/flexiv/`;
- `libflexiv_rdk.a` for versions 1.9.1-1.9.3 or `libflexiv_rdk.so` for version 1.9.4 under `lib/`;
- the Flexiv RDK CMake package under `lib/cmake/flexiv_rdk/`.

Versions 1.9.1-1.9.3 provide a static RDK archive, so their exact, transitive Fast DDS, RBDyn, and
support-library revisions are built into the same local prefix and installed with the examples.
Those upstream static-dependency installers require the default `CMAKE_INSTALL_LIBDIR=lib`.
Version 1.9.4 provides a self-contained shared RDK library. Installed executables use a relative
runtime path to the bundled libraries; POSIX Threads, glibc, and the C++ runtime remain ordinary
host dependencies. The upstream v1.9.4 tag predates a replacement of its Linux release asset; this
integration pins and verifies the replacement asset's published SHA-256 in addition to pinning the
source commit. The selector metadata is checked for every supported version. End-to-end build and
relocatable-install qualification covers the default 1.9.1 static path and the distinct 1.9.4
shared path.

## Run

Each executable accepts one Flexiv robot serial number, an optional positive duration in seconds,
and an optional non-RT request:

```bash
./_install/bin/rizon_joint_impedance_controller Rizon4s-123456
./_install/bin/rizon_cartesian_impedance_controller Rizon4s-123456 \
  --duration-seconds=60 --non-rt
```

The duration defaults to 10 seconds. With `--non-rt`, the RDK scheduler uses priority 0 and CPU
affinity -1, selecting non-real-time scheduling without binding the task to a CPU. The RDK
timeliness failure limit, robot readiness checks, torque limits, and all other runtime safety gates
remain enabled.

The controller workspace is prepared before any connection. The program prints its fixed command,
frames, units, safety assumptions, and initial joint target, then constructs the RDK client only
after the operator confirms by pressing Enter on an empty line. Every executable switches to
`NRT_JOINT_POSITION` and uses `SendJointPosition()` to move to
`[0, -15, 0, 75, 0, 0, 0]` degrees, except that the TCP-frame force/impedance example moves to
`[0, -15, 0, 75, 45, 0, 0]` degrees. The RDK internal motion generator is limited to `0.5 rad/s`
and `1 rad/s²`; the program validates each target against the connected robot's software limits
and requires convergence within 60 seconds. After the move stops, a second empty-line confirmation
is required before the controller captures its initial measured state. Five examples enter
joint-torque mode; the admittance example enters RDK real-time Cartesian motion mode.

| Executable | Fixed behavior |
|---|---|
| `rizon_joint_impedance_controller` | Hold the measured post-motion seven-joint configuration. |
| `rizon_cartesian_impedance_controller` | Hold post-motion TCP pose and joint posture. |
| `rizon_variable_cartesian_impedance_controller` | Hold post-motion TCP pose; vary WORLD-z stiffness from 1 to 2000 N/m at 0.1 Hz. |
| `rizon_cartesian_admittance_controller` | Convert estimated WORLD-frame wrench to an unbounded pose command streamed through RDK Cartesian motion control. |
| `rizon_cartesian_force_impedance_controller` | Ramp WORLD-z force from 0 to -15 N in unloaded compliant contact. |
| `rizon_cartesian_force_impedance_controller_in_ee` | Ramp TCP-z force from 0 to +10 N with TCP-frame control. |

The Cartesian impedance hold uses the full `CartesianImpedanceController` wrench law, including
automatic damping and the inertial-coupling terms based on numerical operational-space mass and
Jacobian derivatives. Its diagonal stiffness is
`[2000, 2000, 2000, 200, 200, 200]` in `[N/m, Nm/rad]`. Double diagonalization automatically designs
the damping matrix from the current operational-space mass `Lambda`, stiffness, and damping ratio
`zeta = 0.7` on every axis. Desired acceleration is zero. The dynamically consistent nullspace
controller holds the measured post-motion joint posture with diagonal stiffness
`[120, 80, 60, 40, 30, 20, 10] Nm/rad` and explicit diagonal damping
`[12, 8, 6, 4, 3, 2, 1] Nms/rad`.

WORLD is the fixed robot-base frame. TCP is the configured tool-center frame. Translation is
metres, rotations are radians, joint angles are radians, torque is Nm, and wrench is ordered force
then moment in N/Nm. RDK poses are interpreted as `[x, y, z, qw, qx, qy, qz]`. WORLD-frame
controllers use the TCP Jacobian expressed in WORLD; the EE force example uses the same Jacobian
expressed in TCP. The configured TCP and payload must match the physical tool.

The admittance and force examples use RDK's estimated `ext_wrench_in_world` and
`ext_wrench_in_tcp`, not the raw F/T sensor field. They therefore do not require a physical F/T
sensor and do not automatically zero one. Both force/impedance examples use diagonal stiffness
`[2000, 2000, 50, 200, 200, 200]` in `[N/m, Nm/rad]` with damping ratio `0.7`. The admittance
example applies an additional first-order
7 Hz low-pass filter and a `0.5 N/Nm` deadband. Its virtual stiffness is
`[1000, 1000, 1000, 100, 100, 100]`, damping is
`[100, 100, 100, 20, 20, 20]`, and virtual mass is identity. It does not apply acceleration,
twist, or pose limits. The controller's translation and rotation offsets are applied independently
to the measured start pose and streamed without velocity or acceleration feedforward through
`RT_CARTESIAN_MOTION_FORCE`. All force control axes are disabled, and RDK Cartesian impedance is
`[2000, 2000, 2000, 200, 200, 200]` with damping ratio `0.7`.

## Safety and timing boundary

The robot must have compatible robot software, a suitable RDK license, released emergency stop,
normal Auto Remote state, and valid seven-joint limits. The examples enable an otherwise healthy
disabled robot, but fail closed on faults, recovery, reduced/manual/regular-auto states, external
axes, invalid data, or startup timeout. They never clear faults, run
recovery, home, execute plans, or alter robot safety settings.

After switching successfully to its real-time control mode, every exit stops the scheduler and
calls blocking `Robot::Stop()`. The five torque-output examples use link-side `q` and `dq`, add model
Coriolis feedforward exactly once, retain RDK gravity compensation and soft joint limits, use 100%
friction compensation, clamp commands to 100% of `RobotInfo::tau_max`, and apply a `1 Nm/ms`
(`1000 Nm/s`) per-joint slew limit. The admittance example instead relies on RDK's pure Cartesian
motion controller and intentionally has no software acceleration, twist, or pose envelope.
Repeated RDK robot-state timestamps skip command generation; a backward timestamp or more than two
consecutive repeats stops control.

By default, the RDK scheduler requests a 1 ms interval and maximum priority, which requires
appropriate Linux real-time privileges. `--non-rt` changes only the scheduler priority/affinity as
described above. RDK state/model calls return allocating dynamic objects and can throw, so these
examples are functional 1 kHz integrations, not allocation-free hard-real-time code or hardware
qualification. Force readings and a clean stop after the configured duration do not establish task
success.


## Preparation, results and companion

Before launching, verify the robot's Auto Remote state, RDK-compatible software/license, configured
TCP and payload, dedicated reachable link and seven joint limits. Use the robot's supported vendor
workflow to resolve faults; this executable does not perform recovery. Check the printed target and
force direction before the first Enter. Text or EOF at either gate cancels. The second gate follows
blocking completion of the initial move; it does not authorize a later unattended restart.

Normal completion and detected faults stop the scheduler and call blocking `Robot::Stop()` after
entering the control mode. A rejected connection or initialization reports an error; this does not
establish that a remote robot observed a stop. Capture console output, selected RDK version,
`RobotInfo::tau_max`, TCP/payload and stop reason. The program does not write a trace file.

Every target has a canonical [Isaac companion](../isaaclab/README.md). Native configurations match
this source, including the distinct pose-output admittance path and 1000 Nm/s rate limit. Supply
captured robot torque limits to the simulator; nominal asset tables do not establish a connected
robot's limits. Rizon4M hardware support is separate from its unavailable pinned simulator asset.
