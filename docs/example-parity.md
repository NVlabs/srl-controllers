# Hardware and simulator parity contract

The current physical example source is ground truth. Simulator alignment does not rewrite,
wrap, launch or emulate entire hardware executables. It selects the identical native public
controller implementations and source configurations. Native storage, reset state, gains,
parameter layout, frame convention and composition are audited separately from transport.
The immutable reviewed source hashes are in
[`hardware_profile_contract.json`](../examples/isaaclab/hardware_profile_contract.json).
Update an audited hash only after reviewing the corresponding profile against the changed source.

## Native boundary and composition

| Hardware family / target suffix | Native composition | Output before transport |
|---|---|---|
| Franka / Rizon `joint_impedance_controller` | Full joint impedance, model Coriolis, torque filter | Seven torques, Nm |
| Franka / Rizon `cartesian_impedance_controller` | Operational-space mass, full Cartesian impedance, nullspace, Coriolis, filter | Seven torques, Nm |
| Franka / Rizon `variable_cartesian_impedance_controller` | Mass, variable Cartesian impedance, Coriolis, filter; no extra nullspace | Seven torques, Nm |
| Franka / Rizon force impedance, base or EE | Mass, unified force/impedance with shaping, Coriolis, filter; no extra nullspace | Seven torques, Nm |
| Franka FR3 admittance | Filtered/negated wrench, native admittance, mass and full impedance tracker, Coriolis, filter | Seven torques, Nm |
| Franka Panda / Rizon admittance | Native admittance and absolute-pose composition | Base pose, metres and orientation |
| Dual Flexiv | Complete 12D midpoint/relative impedance with per-arm nullspaces, Coriolis, combined filter | Fourteen torques, arm 1 then arm 2 |
| UR joint / Cartesian impedance | Hardware static impedance; Cartesian maps wrench through TCP Jacobian; filter | Six torques, Nm |
| UR ordinary / unified admittance | Native admittance; unified adds standalone shaping and force-error PID | Absolute base pose before DLS conversion |

The UR EE force target is expressed along live tool z, but its native integrator/PID state and
rotated parameter matrices remain in the inertial base frame. Dual Flexiv captures midpoint and
relative displacement poses in arm-1-base coordinates. Task vectors use the configured main-frame
axes; arm 1 is the default, with arm 2 and explicit reference frames also supported. Circular
position references stay in arm-1-base coordinates, while their derivatives rotate into main axes.

`_hardware_profiles.py` contains the resolved immutable configuration;
`_native_components.py` is the reusable native CPU/CUDA factory. Simulator runners supply robot
state/model observations and orchestrate the same stages. No NumPy/Eigen reimplementation of a
controller law is used by the simulator. Shared native code does not make PhysX dynamics,
observations or vendor servo implementation identical to a physical robot.

**Current functionality:** the UR joint and Cartesian impedance Isaac companions are
**non-functional** on both UR5e and UR10e. Native controller parity and an equilibrium-only
simulator pass do not qualify disturbance stability. Their registry entries remain available
for diagnosis; the hardware examples and official asset parameters remain unchanged. See
[status and evidence](public-api-validation.md#non-functional-ur-isaac-impedance-companions).

## Configuration details that must remain equal

| Item | Source configuration |
|---|---|
| Dtype | float64 for hardware and simulator native arrays |
| Franka/Rizon damping ratio | 0.7 for full impedance |
| Variable stiffness | Franka `1000*(sin(2*pi*0.1*t)+1.001)`; Rizon `1000.5+999.5*sin(2*pi*0.1*t)` |
| Franka admittance | K `[500,500,500,50,50,50]`; FR3 M `[1.95]*6`, Panda M `[29.25]*3+[3.9]*3`; D `4*sqrt(M*K)`; zero deadzone; 7 Hz wrench filter initialized at zero; negate filtered wrench |
| Rizon admittance | K `[1000]*3+[100]*3`, M identity, D `[100]*3+[20]*3`; 0.5 deadzone; 7 Hz filter seeded from the first measurement; vendor pose output |
| UR admittance | M is source mass times 1.3; 5 Hz measured-wrench filter; source-specific deadzone and damping |
| UR unified force | Environment-on-robot z target +10 N in base variant, -10 N in EE variant, 2 s ramp; proportional gains one, I/D zero; shaping 0.06 to 0.1 m/rad; no integration envelopes |
| UR z damping | Base `1.5*2*sqrt(10*1.95)`; EE `3*2*sqrt((10+10/(0.1-0.06))*1.95)` |
| Franka torque filter | Magnitudes `[87]*4+[12]*3`; rate `[999.999]*7` Nm/s; fixed SDK filter period 1 ms; previous commanded torque |
| Rizon torque filter | Connected `RobotInfo::tau_max`; rate `[1000]*7` Nm/s. Dual concatenates both robot limit vectors. |
| UR torque filter | UR5e `[150]*3+[28]*3`, UR10e `[330,330,150,54,54,54]`; rate `[500]*6`; reset history without an invented zero seed |

Initial joint destinations, force gains, matrix regularization/pivots and nullspace gains are
resolved per source in the profile and printed in rollout JSON. Hardware holds capture their
measured post-motion target. Synthetic offline
Rizon test fixtures use nominal model limit tables; a simulator run requires explicitly supplied
limits copied from the paired robot. The simulator never connects to obtain them.

## Runtime model support

| Family | Configuration support | Simulator assets / model boundary |
|---|---|---|
| Franka | FR3 and Panda/FER | Isaac-provided FR3/Panda assets; model/firmware and TCP differences remain transport concerns |
| Rizon | 4, 4s, 4M, 10, 10s | Pinned official checkout contains 4, 4s, 10, 10s; **Rizon4M unavailable**, fails explicitly |
| Dual Flexiv | Two selections from 4, 4s, 10, 10s | Pinned assets; calibrated relative-base transform supplied separately |
| UR | Impedance UR5e/UR10e; admittance UR5e | Hardware URDF `base`/flange frames and configured TCP offset are applied explicitly. NVIDIA's official USD mass/inertia and joint armature are preserved; simulator actuation and contact input remain separate from RTDE/Pinocchio. |

All Isaac link Jacobians are shifted from their PhysX centre-of-mass evaluation point to
the controller wrist/TCP origin. For UR this includes the configured flange-to-TCP lever arm.
Pose, Jacobian, twist and force coordinates follow the same base convention, including the
URDF's 180-degree `base_link`-to-`base` rotation. UR mass/inertia comes from the official
Isaac Sim USD without a companion inertia override. Native parameters and torque filtering
remain those of the hardware example. See the
[frame adapter contract](../examples/isaaclab/README.md#frame-and-plant-adapters).

## Verification layers

1. Hash and independent source-value audits detect physical-source/configuration drift, including
   branch-specific force damping and variable frequency. Hash equality alone does not prove parity.
2. Hardware-free C++ trace oracles compile the actual unchanged Franka admittance helper for FR3
   and FER, with Eigen but no vendor SDK/client. Python compares filtered input, output, integration
   history and statuses with that oracle. No hardware executable is run.
3. Native CPU/CUDA comparisons cover all registered compositions and supported model configurations
   (including dual-arm model pairs) with batches 1, 16, 256 and 1024,
   plus a bad row with healthy rows compared independently. Transfers occur at test boundaries.
4. Opt-in Isaac tests assess adapters, nominal tracking, fault latching, stable buffers and
   actual actuator saturation. A simulator failure remains a separate qualification failure;
   changing source gains to make it pass would violate this contract.
5. Physical timing, initial motion, both Enter gates, vendor controller readiness, safe stops,
   robot/workcell behavior and observed task completion require attended hardware validation.

The fixed-size C++ `controller_example_specs()` accessor retains its 20-entry type for existing
consumers. New C++ code uses `all_controller_example_specs()` to enumerate all 22; lookup finds
all entries. Python `controller_example_specs()` returns the full 22-entry registry.

The [current validation record](public-api-validation.md) and [structured results](example-validation.json)
retain motion failures, exact source provenance and verification limits.
