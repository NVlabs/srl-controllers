# Franka examples

The FER/Panda and FR3 suites each have nine examples. They use the public controller API
on caller-owned CPU arrays and libfranka. CUDA is not required.

Configure one supported libfranka version. The build checks out the selected
libfranka source and its pinned `common` submodule, builds it under `_build/_deps`,
and links the examples only against that local installation. FR3 builds a local
Pinocchio as well. Eigen, Poco, and the other ordinary build prerequisites are
provided by the host. Install Eigen >=3.4, Poco and a C++17 toolchain; FR3 also
requires Eigen >=3.4, Boost >=1.71 (filesystem and serialization), urdfdom >=1.0,
console_bridge, tinyxml, and their runtime libraries for Pinocchio.

```bash
cmake -S . -B _build \
  -DENABLE_EXAMPLES=ON -DENABLE_PYBIND=OFF \
  -DENABLE_FRANKA_EXAMPLES=ON \
  -DLIBFRANKA_VERSION=0.9.2
cmake --build _build --parallel
```

Use libfranka `0.9.2` for FER/Panda or `0.10.0` and newer for FR3. The default is
`0.21.3` for FR3. A successful build establishes source/SDK
compatibility only; it does not validate a robot, workcell, network, kernel, or tuning.
The nine directly runnable binaries are in
`_build/_deps/libfranka-<version>/example-install/bin/`. The local SDK, controller,
and model libraries are copied into that installation's `lib/` directory.

Each executable accepts one robot host, an optional positive duration in seconds, and an optional
non-RT request:

```bash
./_build/_deps/libfranka-0.9.2/example-install/bin/franka_joint_impedance_controller ROBOT_HOST
./_build/_deps/libfranka-0.9.2/example-install/bin/franka_joint_impedance_controller ROBOT_HOST \
  --duration-seconds=60 --non-rt
```

All examples support `--duration-seconds=SECONDS`, with a default of 10 seconds. Fractional
seconds are accepted, and the option can appear before or after `--non-rt`. The duration applies to
the controller run after the initial positioning move, and is printed before confirmation.

By default, libfranka enforces its real-time setup. `--non-rt` passes
`franka::RealtimeConfig::kIgnore`, so failure to obtain real-time scheduling and the absence of a
PREEMPT_RT kernel are nonfatal. This does not force ordinary scheduling: libfranka still attempts
its normal setup and may acquire real-time scheduling when permitted. The flag does not disable any
state, command, torque, duration, or operator-confirmation check.

Before connecting, it prints fixed frames, units, commanded behavior, the initial joint target,
and safety assumptions. It constructs a robot client only after the operator confirms by pressing
Enter on an empty line; text or EOF cancels. Each example then performs a synchronized joint
motion at speed factor 0.5 to its predefined initial configuration. After the move completes,
a second empty Enter confirmation is required before controller startup; text or EOF cancels.
The measured `readOnce()` state after that second confirmation supplies the pose or joint reference.
The admittance example starts with zero
internal pose offset and velocity. FER sends the pose offset through Franka's Cartesian position
interface; FR3 keeps the Cartesian impedance cascade and torque interface.
Before the initial motion, each example sets the joint collision torque thresholds to
`[60, 60, 60, 60, 10, 10, 10]` Nm and the Cartesian collision wrench thresholds to
`[40, 40, 40, 10, 10, 10]` in order Fx, Fy, Fz (N) and Tx, Ty, Tz (Nm). Lower and upper
thresholds are equal, and the same limits apply during acceleration and constant-velocity motion.

Joint and Cartesian impedance use `[0, 0, 0, -3π/4, 0, 3π/4, π/4]` rad. Both use the full
impedance controllers with damping factors of 0.7 and model-derived inertia. Cartesian impedance
also includes the inertial compensation terms. Desired velocity and acceleration are zero for
the measured-state hold. Virtual walls, APF, variable impedance, admittance and base-frame force
examples use `[0, 0, 0, -2.3π/4, 0, 2.3π/4, π/4]` rad. The EE-frame force example uses that
configuration with an additional `π/4` rad (45°) on joint 5.

| Example | Fixed behavior |
|---|---|
| `franka_joint_impedance_controller` | Hold the measured seven-joint configuration with mass-dependent damping and joint-inertia feedforward. |
| `franka_cartesian_impedance_controller` | Hold measured `O_T_EE` with operational-space inertia, mass-dependent damping, and inertial compensation; measured joints are the nullspace target. |
| `franka_variable_cartesian_impedance_controller` | Hold measured `O_T_EE`; vary base-z stiffness at 0.1 Hz. |
| `franka_cartesian_admittance_controller` | Low-pass filter base-frame wrench at 7 Hz. FER sends the admittance pose offset through the Cartesian position interface; FR3 sends joint torques from the full Cartesian impedance controller. |
| `franka_cartesian_force_impedance_controller` | Ramp base-z force from 0 to -10 N in unloaded compliant contact. |
| `franka_cartesian_force_impedance_controller_in_ee` | Ramp tool-z force from 0 to +10 N with EE-frame control. |
| `franka_cartesian_force_impedance_controller_with_tank` | Ramp base-z force from 0 to -10 N; scale the force contribution with a 1 J energy tank and disable distance shaping. |

The two force examples use the measured start pose as their impedance reference. Their force
contribution tapers from full strength at 0.06 m/rad pose error to zero at 0.10 m/rad on each axis.
This shaping does not impose a hard travel limit; neither example has a pose-error abort threshold.

`O` is the Franka base frame and `EE` is the configured end-effector frame. Translation is meters,
rotation errors are SO(3) logarithms in the selected base or EE axes, in radians, wrench is N/Nm, joint angles are
radians, and torques are Nm. Force examples assume unloaded initial contact with a compliant
surface and report observed force only; they do not infer task success.

The admittance example leaves acceleration, twist, and pose limits unset. FER returns
`franka::CartesianPose` commands with libfranka's optional rate limiter disabled; libfranka's
default command low-pass filter remains active. FR3 returns `franka::Torques` commands, including
model Coriolis torque exactly once and passing through the shared joint-torque magnitude and rate
filter before reaching libfranka. Admittance stiffness is 500 N/m for translation and 50 Nm/rad
for rotation. FER virtual mass is 29.25 kg for translation and 3.9 kg·m² for rotation; its
damping is about 483.74 N·s/m and 55.86 Nm·s/rad. FR3 virtual mass is 1.95 kg for translation and
1.95 kg·m² for rotation, with damping about 124.90 N·s/m and 39.50 Nm·s/rad. Damping is twice
critical, `D = 4 * sqrt(M * K)` per axis. The wrench
deadband is zero. It preserves the existing negation of `O_F_ext_hat_K` at
the wrench adapter. The 7 Hz first-order filter uses the measured callback period and starts at
zero, so it filters the first wrench sample too.

The admittance controller integrates acceleration into twist and twist into its internal pose
offset. FER captures `O_T_EE_c` on its first callback and returns it unchanged as the first pose
command. It then applies the pose offset directly to that reference pose. Invalid input or state
finishes with the previous commanded pose; duration completion returns the current pose with
`MotionFinished`. FR3 applies the offset to the measured start pose and supplies admittance twist
and acceleration to the full Cartesian impedance controller. Its output wrench is mapped through
the measured Jacobian to joint torques. FR3 starts with the previous commanded torques and
finishes with the filtered torque.

The admittance executable does not print controller configuration or per-callback diagnostics.
It retains the normal confirmation, errors, and run summary outside the control callback.

Torque examples add model Coriolis torque exactly once, except virtual walls, which command
only the wall torque. Every torque example filters the command to the
official joint maxima
`[87, 87, 87, 87, 12, 12, 12]` Nm and libfranka's exact `999.999 Nm/s` per-joint rate limit.
They reject non-finite state, model, wrench, status, and command values. Controller parameter/state/
scratch/output memory is prepared before connection; steady-state steps use fixed caller-owned
storage. These properties do not by themselves establish a hard-real-time or physical safety case.


## Robot preparation, stopping and companion

Enable FCI through the supported Franka interface, verify the selected libfranka/robot software
combination, configured TCP/end-effector frames and payload, released stop and reachable robot
network. Check the printed joint destination and free workspace before the first Enter; the second
Enter authorizes controller startup after the move. Neither gate is a collision-free trajectory
certificate. Text or EOF cancels the current stage.

The collision thresholds above are written before initial motion. These examples do not restore
previous thresholds on exit; verify the robot's settings before subsequent operation. Duration
completion returns `MotionFinished` with the current filtered command. Rejection finishes with the
previous torque or pose according to the executable; libfranka communication/control exceptions
are reported outside the callback. Confirm the robot's observed stopped state independently.

Console summaries and observed force do not prove contact-task success. Preserve source revision,
selected SDK/model, stdout/stderr, robot settings and the stop reason. No automatic controller trace
file is created. [Isaac companions](../isaaclab/README.md) use the same native parameters and class
composition; physics, wrench sensing and vendor position-servo behavior remain separate adapters.

The energy-tank example composes separate public mass, impedance, force, and tank blocks.
It retains the base-frame force example's stiffness and z-axis gains P=1, D=0.01, I=0.
Its tank starts at 1 J, has a maximum of 1 J, a minimum of 0.01 J, and a 0.1 J
cosine depletion transition. Its scale depends on energy in both motion directions;
velocity reversal does not restore full force. The power deadband is zero. Half the
existing impedance damping dissipation supplies additional recharge, leaving half
the damping power dissipative. Spring and inertial work are not credited as damping.
Measured twist is `J * dq`; the force contribution supplies tank power `-force_wrench.dot(twist)`
in watts. It uses the callback period in seconds, limits spending before issuing the scale,
and combines the scaled force wrench with the impedance wrench before joint mapping.
The tank covers only this sampled force contribution. Torque filtering, impedance dynamics,
and sampling prevent interpreting it as a whole-robot passivity guarantee. A tank does not
limit force at zero velocity or impose a travel limit. No second impedance tank is used.
After the run, the example prints final/minimum energy, minimum scale, limiting sample count,
and observed force. Duration completion establishes completion of the configured run only.

## Virtual walls and artificial potential fields

Run `franka_virtual_walls_controller ROBOT_HOST` or
`franka_artificial_potential_field_controller ROBOT_HOST` from the local example installation.
Virtual walls and APF use the same higher initial configuration as admittance, variable impedance and
base-frame force examples: `[0, 0, 0, -2.3π/4, 0, 2.3π/4, π/4]` rad. On the nominal Panda model,
this raises the task point by about 0.20 m compared with the impedance starting configuration.
The measured `O_T_EE` and base-frame `zeroJacobian(kEndEffector)` refer to the same configured EE
point. Geometry is relative to that starting point in base axes. A custom EE transform changes
that point; these examples do not protect robot links or the tool volume.

The walls form a box with half extents `[0.1, 0.1, 0.1]` m. Inside the box, the requested joint
torques are zero: libfranka provides gravity compensation, with no translation damping,
orientation hold, posture hold, or Coriolis feedforward. Outside each face, an inward spring uses
1000 N/m stiffness. Normal damping (40 Ns/m) activates continuously with a cubic smoothstep over
0.005 m penetration;
its unilateral force is clamped to zero when moving inward sufficiently fast. These are compliant
walls, so penetration is possible. Wall forces have no Cartesian norm cap and map directly through
`Jᵀ` to joint torque. The existing joint torque magnitude/rate filter and libfranka command filter
still apply, so torques can briefly decay toward zero when returning inside the box. There is no
commanded wall contact or traversal. The fixed wall damping is a tuning choice, not a guarantee of
critical damping for every robot configuration or payload.

The APF goal is offset `[0, 0.12, 0]` m along base-frame +y; the virtual obstacle center is offset
`[-0.025, 0.06, 0]` m, with radius 0.02 m. Attraction uses 50 N/m and damping 20 Ns/m.
For surface clearance `d < d0 = 0.035` m, the repulsive force magnitude is
`eta * (1/d - 1/d0) / d²`, with `eta = 0.00005 N m³`, directed away from the sphere.
Both conservative forces share a one-second quintic ramp. There is no minimum-clearance cutoff
and no Cartesian force or moment cap. The repulsion equation requires positive surface clearance;
contact/penetration is an invalid equation input, and non-finite inputs or outputs end control.
APFs can have local minima and do not perform
collision checking or guarantee a collision-free path.

The APF example holds measured orientation with 10 Nm/rad stiffness and 3 Nm s/rad damping, and
holds posture through the mass-weighted nullspace controller. Wrenches map through `Jᵀ`, then nullspace
and Coriolis torques are added before the existing joint torque magnitude/rate filter.
Gravity is compensated by libfranka in both examples. Preparation and storage allocation happen
before control; the callback uses fixed arrays and emits no logs.

Duration completion is reported separately from goal attainment. The APF reports observed minimum
clearance, goal error, repulsion activity, and whether error stayed at most 0.005 m with TCP speed
at most 0.01 m/s for 0.5 s. Invalid-input termination uses the existing filtered last-command finish helper;
this is a request to end control, not a certified immediate stop.

The legacy public `InverseAPF` block retains its repulsive pose/deadzone semantics. Its prepare/apply
parameter status now rejects negative gains, negative deadzone/damping, and reversed damping limits
on CPU and CUDA, consistently with its existing step checks. Its six-component norm
combines translation in meters and SO(3) error in radians. With a nonzero deadzone, the
potential term jumps at the boundary; these inherited semantics remain unchanged.
