# Universal Robots example

`joint_impedance_controller`, `cartesian_impedance_controller`,
`cartesian_admittance_controller`, `cartesian_unified_force_admittance_controller`, and
`cartesian_unified_force_admittance_controller_in_ee` use the public controller API on caller-owned
CPU arrays and the `ur_client_library` RTDE/external-control interface. CUDA is not required.
The opt-in build fetches and builds the exact pinned `ur_client_library` 2.14.1 revision in the
build tree. It also builds Pinocchio 3.9.0 privately for the Cartesian impedance Jacobian. It does
not search for or modify either system installation. The first build requires network access to
GitHub. Host prerequisites are Eigen >=3.4, Boost >=1.71 (filesystem/serialization),
urdfdom >=1.0, console_bridge, TinyXML, pkg-config and a C++17 compiler.

The corresponding joint and Cartesian impedance **Isaac companions are non-functional** for
UR5e and UR10e and retained for diagnosis. This simulation classification does not change these
physical examples, their controller settings, or their role as configuration ground truth. See
[the simulator status and evidence](../../docs/public-api-validation.md#non-functional-ur-isaac-impedance-companions).

```bash
cmake -S . -B _build \
  -DENABLE_EXAMPLES=ON -DENABLE_PYBIND=OFF \
  -DENABLE_UR_EXAMPLES=ON
cmake --build _build --parallel
cmake --install _build --prefix "$PWD/_install"
./_install/bin/cartesian_admittance_controller ROBOT_IP
./_install/bin/cartesian_admittance_controller ROBOT_IP \
  --duration-seconds=60 --non-rt
./_install/bin/cartesian_unified_force_admittance_controller ROBOT_IP
./_install/bin/cartesian_unified_force_admittance_controller ROBOT_IP \
  --duration-seconds=60 --non-rt
./_install/bin/cartesian_unified_force_admittance_controller_in_ee ROBOT_IP
./_install/bin/cartesian_unified_force_admittance_controller_in_ee ROBOT_IP \
  --duration-seconds=60 --non-rt
./_install/bin/joint_impedance_controller ROBOT_IP --robot-model=ur5e
./_install/bin/joint_impedance_controller ROBOT_IP --robot-model=ur10e
./_install/bin/cartesian_impedance_controller ROBOT_IP --robot-model=ur5e
./_install/bin/cartesian_impedance_controller ROBOT_IP --robot-model=ur10e \
  --duration-seconds=60 --non-rt
```

Each executable accepts one IP address, an optional positive `--duration-seconds=SECONDS`, and
optional `--non-rt` in either order. Both impedance examples additionally require exactly one
`--robot-model=ur5e|ur10e`. The duration defaults to 10 seconds. Each requires an empty-line
confirmation (press Enter) before constructing `ExampleRobotWrapper`; text or EOF cancels.
After connecting, every example moves through URCL `moveJ` to
`[0, -90, -90, -90, 90, 0]` degrees at 0.5 rad/s and 0.5 rad/s^2 before capturing its controller
target. The end-effector-frame force/admittance example sets joint 5 to 135 degrees, so its target is
`[0, -90, -90, -90, 135, 0]` degrees. After the move completes, every example requires a second
empty Enter confirmation before starting the controller; text or EOF cancels. The direct-torque
examples also validate the connected PolyScope version before the initial motion.
The install includes URCL and the RTDE/script resources and uses relative runtime paths, so
invocation does not depend on the build or source directory.

The Cartesian impedance URDF resources were generated with `xacro` from
`UniversalRobots/Universal_Robots_ROS2_Description` 3.5.1 commit
`710505166d0399b506838d55663f634880d235fe`:

```bash
xacro ur.urdf.xacro -o models/ur5e.urdf name:=ur ur_type:=ur5e
xacro ur.urdf.xacro -o models/ur10e.urdf name:=ur ur_type:=ur10e
```

They provide nominal kinematics only. The configured RTDE `tcp_offset` is appended to the URDF
flange, Pinocchio supplies a `LOCAL_WORLD_ALIGNED` TCP Jacobian, and measured RTDE TCP pose supplies
the hold error. The URDF is not used for torque limits, gravity compensation, or robot identity.

The current `ur_client_library` API does not expose a switch to disable its warning-only FIFO
scheduling attempt. Therefore `--non-rt` is an explicit, documented no-op for this backend: it
allows a uniform physical-example command line, but the upstream client may still acquire FIFO
scheduling when permitted. All runtime sample, timing, command, and cleanup checks are unchanged.

`joint_impedance_controller` captures the first measured `actual_q` after the initial motion,
computes
`K(q_desired-q) + D(-qd)`, and sends gravity-compensated direct torque. Its diagonal stiffness is
`[1000, 800, 600, 200, 100, 50]` Nm/rad and damping is
`[56.569, 101.193, 37.947, 17.889, 8.944, 4.899]` Nms/rad.

`cartesian_impedance_controller` captures the first measured TCP pose after the initial motion.
Translation and rotation-vector errors are expressed in the robot base frame; TCP velocity is
`J*qd`, and the command is `J^T*(K*pose_error + D*velocity_error)`. Its diagonal Cartesian
stiffness is `[1000, 1000, 1000, 100, 100, 100]` in N/m and Nm/rad, and damping is
`[178.885, 178.885, 178.885, 18.974, 18.974, 18.974]` in Ns/m and Nms/rad.
The kinematics explicitly transforms the URDF's REP-103 `base_link` result through the fixed
180-degree z rotation into the UR controller's `base` frame before using the pose or Jacobian.

The joint damping values are four times their original stiffness-scaled values, with another factor
of two on joint 2; Cartesian damping is four times its original values. The static-controller
structure is unchanged.

Both impedance examples apply the published joint maxima for the selected model: UR5e uses
`[150, 150, 150, 28, 28, 28]` Nm and UR10e uses `[330, 330, 150, 54, 54, 54]` Nm. Their example-level
slew-rate limit is `500 Nm/s` on every joint. Universal Robots notes that published joint maxima
cannot necessarily be used fully in an application; configured force, speed, payload, and safety
limits remain authoritative. The examples send a best-effort zero torque before `stopControl()` on
every normal or detected-fault exit. Direct torque requires PolyScope 5.23.0 / PolyScope X 10.10.0
or newer and assumes the payload and TCP are configured correctly on the robot.

The first valid RTDE `actual_TCP_pose` is the equilibrium transform. Translation is meters and
rotation is an axis-angle vector in radians, expressed in the robot base frame. Measured
`actual_TCP_force` is N/Nm. Successive RTDE timestamps supply `dt`; missing, non-finite,
non-monotonic, or excessively delayed samples stop the run.

All UR admittance variants scale their nominal virtual mass by 1.3. They use twice critical damping
except for the force-controlled z axis described below.

In `cartesian_admittance_controller`, a persistent 5 Hz low-pass filter feeds the controller. Its
caller-owned pose offset is composed with the measured initial transform, as in the Rizon and
FER/FR3 examples: `p_des = p_initial + offset_translation` and
`R_des = Exp(offset_rotation) * R_initial`. The controller integrates its own pose state; the
example does not separately integrate its twist. The filtered UR wrench is passed directly as the
environment-on-robot external wrench, so a manually applied disturbance produces compliant motion
in the disturbance direction. Acceleration, twist, and pose limits are unset. Parameter validation,
per-step status, non-finite rejection, and a zero-command safety policy remain active. Cleanup calls
`stopControl()` on every loop exit path.

In `cartesian_unified_force_admittance_controller`, the filter feeds the measured base-frame wrench
to the fused force-error PID and admittance controller. The robot-applied base-z force ramps from 0
to -10 N over two seconds; all other desired wrench channels remain zero.
`cartesian_unified_force_admittance_controller_in_ee` applies +10 N along the live end-effector z
axis. By action-reaction, these become desired environment-on-robot controller inputs of +10 N and
-10 N, respectively. UR RTDE `actual_TCP_force` is a TCP wrench with base-oriented components; the
UR adapter passes it directly to both admittance controllers. The EE example rotates both force and
torque into end-effector axes without a moment-arm translation because the reference point remains
the TCP. It evaluates shaping in live end-effector coordinates and rotates the shaped measured and
desired wrenches back to the base frame. The integrated pose, twist, PID history, and safety history
stay in the inertial base representation. Each cycle rotates the complete EE-frame virtual mass,
damping, and stiffness matrices into that representation. This produces live EE-oriented dynamics
without differentiating state in a rotating frame or introducing unmodeled transport terms. The
base example uses fixed base-frame matrices and signals.

Both force examples use proportional force feedback with no integral term. The controlled z axis
uses 10 N/m stiffness, 1.95 kg virtual mass, and 1.5 times critical damping of approximately
13.2 Ns/m in the base-frame example. EE z uses three times critical damping of approximately
135 Ns/m
because its damping design includes the shaping transition's additional 250 N/m effective
stiffness. The remaining axes retain 1000 N/m and 30 Nm/rad stiffness, their previous virtual
masses, and twice critical damping. The public shaping block leaves the force-loop contribution
unchanged through 0.06 m/rad of pose offset, then linearly reduces it to zero at 0.1 m/rad. This
gives a bounded static equilibrium when no surface is present instead of accumulating
force-integral windup. Against an environment, the achieved steady force still depends on its
compliance. Acceleration, twist, pose, and force-integral limits are unset. Parameter validation,
per-step status, non-finite rejection, and zero-command safety policies remain active. Cleanup calls
`stopControl()` on every loop exit path.

The operator must verify the RTDE wrench sign and selected base-z or end-effector-z force direction
for the workcell before confirming execution. Reaching the commanded force is not inferred as task
completion.

The configured robot safety limits remain authoritative. These examples perform no motion planning
or additional collision checking and are not physical or hard-real-time qualification evidence.


## Preparation, output and simulator pairing

Prepare the robot in Remote Control with the installed payload/TCP and compatible PolyScope
version. These examples construct URCL in headless mode using their installed external-control
script and RTDE recipes. Verify robot-to-host reverse communication as well as host-to-robot RTDE
and script connectivity; use the pinned URCL network requirements and the configured firewall
rather than assuming that ping alone proves readiness. The direct-torque feature/version check
occurs before the initial move. All five programs require a first empty Enter before connection
and another after `moveJ` completes, before controller execution.

A duration exit reports the configured controller run ending. Input/timestamp/command rejection
reports a stop reason and stops control; torque examples first attempt zero torque. Pose examples
retain their last pose until `stopControl()`. A network failure can prevent delivery; observe the
robot's stopped state. Capture stdout/stderr, model, TCP/payload, PolyScope version and source/URCL
revision. No automatic trace file or inferred task-success result is produced.

The [Isaac companions](../isaaclab/README.md) use the exact static gains and magnitude/rate limits
above. Admittance outputs a native absolute pose before the simulator's DLS/joint-servo adapter.
The live EE force variant keeps native state in base coordinates and rotates parameters each tick;
its registry frame is therefore base, while its force target is tool-oriented. Contact sensors,
nominal assets and simulated servos do not reproduce RTDE wrench estimation or robot firmware.


For the pinned URCL default configuration, allow robot-to-host TCP **50001** (reverse control),
**50002** (script sender), **50003** (trajectory forwarding) and **50004** (script commands).
RTDE connects from host to robot TCP **30004**. These are the defaults in the pinned
[driver configuration](https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/a854de920d602300ce7c0fe3bd79655cb85bd678/include/ur_client_library/ur/ur_driver.h)
and [RTDE client](https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/a854de920d602300ce7c0fe3bd79655cb85bd678/include/ur_client_library/rtde/rtde_client.h).
The reverse host address must be reachable from the robot; a different deployment configuration
requires its corresponding ports. Additional dashboard/script connections used by the wrapper
also require the vendor's documented access. No network or firewall settings are changed here.
