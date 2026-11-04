# Isaac Lab companions

There are **22 canonical companions**: nine Franka, six Rizon, two dual Flexiv,
and five UR. Each canonical file is named `<hardware-target>_parallel.py`. The
[complete catalog](../README.md) links each hardware family and lists every target.

**Non-functional simulation companions:** `joint_impedance_controller_parallel.py` and
`cartesian_impedance_controller_parallel.py`, for both **UR5e and UR10e**. They are retained
for diagnosis and print this status in CLI help and before simulator startup. Do not use an
equilibrium-only pass as evidence of stable torque control. See
[UR impedance status and evidence](../../docs/public-api-validation.md#non-functional-ur-isaac-impedance-companions).
This classification applies to these two Isaac companions; the paired physical examples remain
the configuration authority.

The physical example sources are the configuration authority. `_hardware_profiles.py` resolves
those configurations, and `_native_components.py` prepares the same public native controller
classes for CPU or CUDA. Gains, damping, force signs, filters, limits, frame choices, integrator
initialization and controller topology follow the paired hardware source; there are no simulator
gain presets. The [parity contract](../../docs/example-parity.md) identifies the native boundary,
source audits, model limitations and the separate simulator adapters.

Franka virtual walls command only the translation wall field, with zero requested torque inside
the box and no Cartesian force cap. The APF example composes static orientation hold with its
translation field and mass-weighted nullspace control. Both follow their paired hardware sources.

## Build with the simulator interpreter

Use the Python interpreter belonging to your Isaac Lab installation. The extension ABI must match
that interpreter; a Python 3.12 extension cannot be imported by Isaac's Python 3.11 runtime.
The distributable Python package requires Python >=3.12. An Isaac environment using 3.11 can build
the CMake extension and import the checkout through `PYTHONPATH`, without installing that package.
Install the binding build prerequisite in that environment first:

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaac-environment/bin/python
"$ISAACLAB_PYTHON" -m pip install 'pybind11>=3.1,<4'
cmake -S . -B _build_isaac \
  -DENABLE_CUDA=ON -DENABLE_PYBIND=ON -DENABLE_EXAMPLES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DNVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF \
  -DPython3_EXECUTABLE="$ISAACLAB_PYTHON"
cmake --build _build_isaac --parallel
```

This is a **candidate functional build** for SM89. Select the actual GPU architecture. SM89 has
no committed deployment tuning profile here; the default profile gate would reject this build
without the explicit override. Deployment requires a reviewed matching profile and separate
[qualification](../../docs/public-api-validation.md). CMake creates an ABI-specific extension
symlink in `python/src/nvidia/srl/_controllers`; rebuilding another Python ABI uses another suffix.
Confirm the loaded module path when multiple build trees exist.

## Run and CLI

From the repository root:

```bash
PYTHONPATH="$PWD/python/src:$PYTHONPATH" "$ISAACLAB_ROOT/isaaclab.sh" -p \
  "$PWD/examples/isaaclab/franka_cartesian_impedance_controller_parallel.py" \
  --headless --num_envs 16 --duration 10 --validate
```

List canonical launchers with the same interpreter and extension:

```bash
PYTHONPATH="$PWD/python/src" "$ISAACLAB_PYTHON" -c \
  'import nvidia.srl.controllers as c; print(*(s.isaac_launcher for s in c.controller_example_specs()), sep="\n")'
```

| Option | Contract |
|---|---|
| `--num_envs N` | Positive batch size; default 16. Controller buffers are allocated for the selected batch. |
| `--duration SECONDS` | Finite positive rollout duration; default 10 s. |
| `--device cuda:0` | CUDA simulation device; CPU simulator execution is rejected. Offline controller tests cover CPU. |
| `--headless` | Isaac AppLauncher option for execution without a viewer. Other AppLauncher options remain available. |
| `--validate` | Inject a non-finite input in row 0 and check latched stop, healthy-row isolation, tracking where applicable, saturation and buffer addresses. |
| `--robot-model` | Franka `fr3`/`panda`; Rizon `rizon4`/`rizon4s`/`rizon4m`/`rizon10`/`rizon10s`; UR impedance `ur5e`/`ur10e`. UR admittance is UR5e only. |
| `--left-robot-model`, `--right-robot-model` | Dual Flexiv model selection; default `rizon4s` on each arm. |
| `--arm1-t-arm2` | Dual Flexiv row-major 4x4 rigid transform, comma-separated, metres; maps arm-2 base coordinates into arm-1 base. Default scene has aligned bases 1.2 m apart along y. |
| `--torque-limits-nm` | Required for Rizon and dual Flexiv: copy the paired robot's `RobotInfo::tau_max` in joint order, comma-separated; concatenate arm 1 then arm 2 for dual. No hardware connection occurs. |
| `--tcp-offset` | UR only: flange-to-TCP translation in metres followed by rotation vector in radians, six comma-separated values. Default all zero, matching the hardware default. Copy the offset configured on the paired robot. |

Rizon simulation requires `FLEXIV_ISAAC_SIM_WS` pointing to an official
`flexivrobotics/isaac_sim_ws` checkout at `2d07d67887aff718ea93250e68e9e8eff4b0a62d`.
That revision supplies Rizon4, Rizon4s, Rizon10 and Rizon10s. **Rizon4M is a supported hardware
profile but has no asset at this revision**; selecting it fails explicitly. No other robot is
substituted. See [runtime model support](../../docs/example-parity.md#runtime-model-support).

### Dual Rizon bimanual companions

`flexiv_dual_bimanual_cartesian_impedance_parallel.py` follows the hardware midpoint circle:
0.1 m radius, 0.2 Hz cruise frequency, and a 2 s speed ramp, starting at the top of the circle.
`flexiv_dual_bimanual_compliant_hold_parallel.py` holds the midpoint and relative pose captured
after writing the hardware initial joint configuration. Both use the complete public 12D
bimanual torque controller, the paired hardware gains, independent per-arm nullspaces, Coriolis
compensation, and a 1000 Nm/s torque-rate limit. Neither companion changes the hardware source.

Both expose the hardware frame options: `--arm1-t-arm2`,
`--main-frame=arm1|arm2|explicit-arm1|explicit-arm2`, and `--reference-t-main`.
Transforms contain 16 row-major values with translation in metres; explicit main frames require
`--reference-t-main`. The base transform maps arm-2-base coordinates into arm-1-base coordinates.
With no base transform, the demonstration places arm 2 at `[0, 1.2, 0]` m with parallel axes.
The circular pose reference remains in arm-1-base axes, while velocity and acceleration are
rotated into the selected main-frame axes, matching hardware. Task vectors use
`[absolute linear; absolute angular; relative linear; relative angular]` order.

For either launcher, supply both arms' captured `RobotInfo.tau_max` values in joint order:

```bash
PYTHONPATH="$PWD/python/src:$PYTHONPATH" "$ISAACLAB_PYTHON" \
  examples/isaaclab/flexiv_dual_bimanual_cartesian_impedance_parallel.py \
  --headless --num_envs 16 --duration 10 \
  --torque-limits-nm="$DUAL_RIZON_TORQUE_LIMITS_NM" --main-frame=arm1
```

Replace the launcher with `flexiv_dual_bimanual_compliant_hold_parallel.py` for the static hold.
Run nominal and `--validate` fault cases separately. Returned metrics include the selected frame
configuration, midpoint/relative tracking error, inter-wrist distance error, latched faults,
saturation, and caller-owned buffer stability. Successful offline tests do not qualify PhysX
dynamics or observed hardware completion.

Historical aliases remain: `fr3_cartesian_impedance_parallel.py` selects the Franka Cartesian hold, and
`ur5e_cartesian_admittance_parallel.py` selects ordinary UR admittance. Use canonical names to make
the intended hardware pairing explicit.

## Controller and simulator boundaries

Controller arrays are float64 on CUDA, matching hardware dtype. Separate persistent float32 PhysX
buffers supply observations and receive actuator commands; conversion stays on the GPU. Each native
call uses Torch's current stream. Numerical CPU/CUDA results are compared with tolerances; different
floating-point execution orders need not be bitwise identical.

Franka and Rizon full impedance use model-dependent damping and operational-space inertia. Their
force variants use the paired shaping and force gains. Both variable-stiffness profiles use 0.1 Hz,
with their distinct amplitudes and limits. FR3 admittance feeds the full impedance torque tracker;
Panda/FER and Rizon admittance output absolute poses. UR impedance uses the hardware static gains
and model-specific torque limits. UR admittance uses the native integrator; the unified EE variant
rotates its matrices and shaped signals into the inertial base frame each tick.

Dual Flexiv uses the complete native 12D midpoint/relative impedance torque controller with
per-arm nullspaces, Coriolis compensation and a combined torque filter. The circle and hold
profiles use their paired hardware gains and decoupled cooperative kinematics.

Pose outputs are recorded before the Isaac adapter converts them to DLS IK joint targets. The
simulator's joint position servo (`kp=2500`, `kd=125`) approximates the vendor pose interface; it is
not libfranka, RDK, or URCL servo code. Gravity-compensated torque interfaces are represented by
gravity-disabled single-arm plants.
Asset dynamics, contacts, vendor friction compensation and robot firmware remain separate from
native controller parity. There is no initial-motion or Enter prompt in simulation.

Contact input is a TCP-body contact sensor's net force, including friction. Torque channels are zero
because this adapter has no qualified contact-moment estimate. It does not replace the hardware's
wrench estimator or F/T sensor. No synthetic wrench or phase offset is added to ordinary admittance.
Cartesian vectors use `[linear; angular]`, wrench uses `[force_N; torque_Nm]`, translations are metres,
rotation vectors radians, and Isaac quaternions `wxyz`.

## Frame and plant adapters

PhysX link Jacobians describe linear velocity at the link centre of mass. The adapters shift
the linear rows to the wrist/TCP origin using `J_v(TCP) = J_v(COM) + J_omega × (TCP - COM)`,
with both terms expressed in the controller frame. Pose, Jacobian and `J*q_dot` therefore
describe the same point. Initial joint-state writes are
followed by a forward-kinematics refresh before capturing pose references; dual-arm tracking
metrics include both arms.

UR controllers use the URDF frame named `base`, rather than Isaac's articulation root
`base_link`. `_ur_adapter.py` reads the unchanged hardware URDF's fixed 180-degree rotation
about z, and the `wrist_3_link`-to-`flange` transform. It applies the configured TCP offset
after that flange transform. Poses, both Jacobian blocks, twists, contact forces and absolute
pose commands consistently use that hardware base/TCP convention. Rotating every task
quantity consistently does not itself change an isotropic static impedance torque, so this
frame correction alone does not resolve an unstable actuator plant.

UR5e and UR10e load NVIDIA's official Isaac Sim USD assets at
`{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/<model>/<model>.usd`. Their link masses,
centres of mass and inertia tensors come from those assets. Joint armature remains unspecified
in the companion actuator configuration, so Isaac Lab uses the USD joint's authored value.
Zero joint armature does not mean zero link inertia. No inertia-capture override is exposed.
The direct-effort wrapper retains zero drive stiffness/damping for native torque output and
the existing gravity-compensated interface representation described above.

### Non-functional UR torque companions

Both UR joint and Cartesian impedance companions are **non-functional** for UR5e and UR10e.
The joint hold's earlier nominal/fault passes started exactly at their captured target with zero
velocity and zero torque; those passes did not exercise disturbance rejection. A wrist velocity
perturbation of only `0.001 rad/s` causes oscillation in both controllers. The UR5e tests also
fail with a separate `0.001 rad` wrist-position perturbation. Every diagnostic run used 16 environments.

The hardware's pinned URCL rounds torque commands to a `1e-6 Nm` grid. Isaac currently sends
unquantized floating-point effort. Applying hardware-equivalent rounding only in the diagnostic
wrapper eliminates the spontaneous Cartesian equilibrium oscillation for 10 seconds, but both
controllers still oscillate after a disturbance. That rounding has **not** been applied to the
companions. Increasing solver iterations, using 4 or 16 physics substeps with unchanged 500 Hz
feedback, and enabling external forces every solver iteration also fail to stabilize the UR5e tests.

Fresh UR5e/UR10e frame and Jacobian checks pass against the hardware URDF. No remaining 180-degree
frame flip or effort-delivery sign/scale error was found. Instability does not establish incorrect
official asset inertia. The remaining difference from physical operation is unresolved; hardware
motor/feedback dynamics, compensation settings and timing were not measured. Hardware source,
shared controller code/configuration and official asset parameters remain unchanged. The three UR
admittance companions retain their separate historical validation results.

The [current validation record](../../docs/public-api-validation.md#non-functional-ur-isaac-impedance-companions)
documents the experiments, limitations and criteria for withdrawing this classification.

## Results and validation

The CLI prints JSON containing the resolved configuration, final native command for row 0 before
actuator conversion, simulation steps, maximum tracking error, minimum accumulated displacement,
healthy-row spread, distinct unsafe rows, saturated row-steps, longest saturation run, fault isolation,
and stable-buffer result. Redirect stdout/stderr to preserve evidence; no automatic log file is
created by these scripts. Isaac itself can write runtime logs.

Faulted rows remain stopped until the rollout ends: Franka retains the preceding torque, other
torque adapters command zero, and pose adapters retain the last pose. Healthy rows continue.
A native block's reset/recovery capability does not automatically resume a stopped example.
`--validate` requires exactly one stopped row, isolation, stable native buffers, at most ten
consecutive actuator-limiting steps, and healthy-row spread <=1 mm. Hold/tracking paths
also require <=20 mm positional error. Stiffness-rate limiting is
separate from actuator saturation.
Validation failure prints its cause and raises; it is not a successful rollout. Check both
application metrics and process status: Kit shutdown can obscure a failed process's exit
status. Missing rollout metrics never establish successful completion.

Run nominal rollouts separately from injected-fault rollouts. Motion thresholds are simulator
regression criteria, not physical safety gates. Preserve qualification failures with the unchanged
hardware gains; see the [current validation record](../../docs/public-api-validation.md).

Offline parity checks do not import Isaac:

```bash
CONTROLLERS_TEST_CUDA=1 CONTROLLERS_ORACLE_BUILD="$PWD/_build" \
  PYTHONPATH="$PWD/python/src" pytest -q python/tests/controllers/hardware_profiles_test.py
```

Reference capture, frame rotation, full-controller posture/fault behavior and CUDA graph replay
have a focused Torch suite using the simulator interpreter without starting Kit:

```bash
CONTROLLERS_TEST_BIMANUAL_TORCH=1 \
  PYTHONPATH="$PWD/python/src:$PWD/examples/isaaclab:$PYTHONPATH" \
  "$ISAACLAB_PYTHON" -m pytest -q python/optional_tests/flexiv_dual_torch_test.py
```

The optional full simulator matrix lives in
`python/optional_tests/isaaclab_motion_examples_test.py`; its environment options are documented
in that file. Physical execution, vendor startup/stop behavior and hard real-time qualification
require separate attended evidence.

The optional simulator matrix starts a fresh process for each target, batch size and fault mode.
Isaac Sim 5.1 retains GPU allocations when many scenes replace one another in a single
application; process isolation bounds that memory and keeps controller thresholds unchanged.
The full 132-case default motion matrix is slow; select targets/counts for focused adapter checks.
Separate tests in the same file compare UR5e/UR10e poses and Jacobians against the hardware
URDF at 16 configurations with rotated roots and nonzero TCP offsets.

Run just those two adapter cases with the simulator's ABI-matching extension:

```bash
NVIDIA_SRL_CONTROLLERS_ENABLE_ISAACLAB_MOTION_TEST=1 \
  PYTHONPATH="$PWD/python/src:$PWD/examples/isaaclab:$PYTHONPATH" \
  "$ISAACLAB_PYTHON" -m pytest -q python/optional_tests/isaaclab_motion_examples_test.py \
  -k 'frames_match'
```

`franka_cartesian_force_impedance_controller_with_tank_parallel.py` runs the paired
base-frame force/impedance composition on FR3 or Panda. It uses the same -10 N ramp,
PD gains, impedance tuning, and 1 J tank settings as hardware, with shaping disabled
and no added nullspace controller. Each environment owns independent tank state.
The tank is stepped on Torch's current CUDA stream and uses `-force_wrench.dot(J * dq)`.
Invalid upstream rows retain their energy and receive safe effort commands.
Its optional `energy_tank` JSON diagnostics report minimum/maximum energy in joules, minimum
scale, limiting row-samples, and maximum per-step balance residual in joules.
Normal tank depletion is counted separately from actuator saturation. `--validate`
checks tank bounds/accounting, one-row fault isolation and persistent stopping, and stable controller
storage. The contact sensor supplies force but no qualified contact moment.
These checks establish simulator behavior, not physical safety or actuation-level passivity.

The shared tank now uses the cosine energy taper in both power directions and
credits scaled absorbed work. Unlike the Franka hardware example, this simulation
does not yet harvest impedance damping dissipation; at the energy floor the force
contribution cannot recover without additional recharge. `--validate` continues
to reject persistent actuator saturation. Tank work-balance tests establish the
force-channel budget, but do not qualify the complete filtered robot command or
contact-task performance.

## Franka point fields

`franka_virtual_walls_controller_parallel.py` and
`franka_artificial_potential_field_controller_parallel.py` support `--robot-model fr3` and `panda`.
They match the hardware initial joint configuration, field geometry, gains and torque filtering
described in [the Franka guide](../franka/README.md). Walls use 1000 N/m stiffness and 40 Ns/m
normal damping, with no interior damping, orientation/posture hold, Coriolis feedforward or Cartesian
force cap. The APF retains its orientation/posture hold with no Cartesian force/moment caps or
minimum-clearance cutoff. Its repulsion equation requires positive surface clearance and finite results.
Both start in the higher admittance configuration. The APF goal is 0.12 m along base-frame +y,
with the virtual sphere centered at offset `[-0.025, 0.06, 0]` m.
The simulator point is the resolved hand body origin, with its matching PhysX Jacobian, measured
in robot-root axes. It is not assumed identical to a hardware EE with a custom tool transform.
The rendered box/sphere are virtual shapes with no collision properties. The wall example holds
inside the box; all six face directions are exercised by numerical tests.

APF `--validate --duration 10` additionally requires repulsion activity, finite measured clearance,
final goal error at most 0.005 m, and TCP speed at most 0.01 m/s for at least 0.5 s in every healthy
environment. Injected faults and invalid equation inputs remain stopped and are checked independently.
Duration completion
alone does not establish goal attainment. These checks concern the task point only.

The full C++ registry view and Python registry enumerate 22 entries. The fixed-size
C++ accessor retains its 20-entry return type. Remaining examples retain their relative
order and robot-family enum values.
