# Public controller composition

All controller computations in the library are available to downstream users. C++ provides
36 typed lifecycle components in `controllers.hpp`, canonical host/device equations in
`equations.hpp`, and status/safety utilities in `composition.hpp`. Python exposes the same
components from `nvidia.srl.controllers` and checked equations from its `equations` module.
The [inventory](public_api_inventory.json) records every component and checked equation.

Users own their controller topology, state machines, feedback paths, and additional algorithms.
There is no registry of permitted controller combinations. Applications can supply custom C++
functions, CUDA kernels, or Python array operations between library blocks. Existing solver
and dimension limits still apply to individual blocks; publishing a block does not generalize
its mathematical assumptions or turn the package into an implementation of every control law.

## Lifecycle and ownership

For C++, construct a component, query `requirements(config, &description)`, allocate typed
`Parameters` and `Storage`, then call `prepare`, `reset`, and `step`. `apply_parameters` updates
parameters without allocating controller buffers. Check every returned `Result`.

Python construction performs preparation. Each new block has public `<Name>Parameters`,
`<Name>State`, `<Name>Scratch`, and `<Name>Storage` dataclasses. Its `requirements(config)`
returns immutable typed descriptors with names, shapes, dtypes, units and frames. The new block
descriptor names are storage field paths, for example `state.previous_signal` or
`output.safety.last_valid_output`. The executable Python example allocates these trees solely
from this metadata.

All arrays are caller-owned and contiguous row-major storage. New blocks reject overlapping
mutable storage, parameter/storage aliases and input/storage aliases before launching work.
Parameters explicitly use `shared(array)` or `per_batch(array)`. Constructors retain Python
parameter/storage owners; asynchronous callers must additionally retain each input and any
replaced parameters until its stream work completes. Never mutate an allocation in use.

Generic signal blocks use `SignalBlockConfig(device, dtype, batch_size, signal_dim)` and do not
inherit the robot joint/task limits. EnergyTank has one scalar state per row and uses
`signal_dim=1`. JointStiffness and JointDamping use `JointBlockConfig` with explicit `DENSE` or
`DIAGONAL` modes. Other blocks retain their existing configuration and dimension constraints:
joint solvers that use dense factorization are limited to 32 joints; ControllerConfig itself
accepts up to 64, fixed Cartesian blocks have six axes, and the bimanual specialization uses
14 joints and 12 task axes. These constraints are checked during storage queries/preparation.

`component_requirements()` on complete torque pipelines describes each stage: `mass`, `task`,
`nullspace`, `output`, and bimanual `kinematics` and `velocity`. The torque-output stage describes
`unfiltered_joint_torque` and nested `filter` requirements. This mirrors the composite storage
hierarchy. The flat `requirements()` query describes the complete pipeline, including intermediate state,
scratch and persistent status arrays. Intermediate descriptors use qualified stage names; final
output descriptors and descriptors within individual stages retain their established semantic
names. The associated typed Storage defines their field placement.

`Requirements` has capacities of 32 state, 32 scratch and 4 output descriptors. This changes the
C++ ABI: rebuild the shared library, C++ consumers and Python native extension together. Failed
queries clear their output; generic signal extents and descriptor sizes are checked for overflow.
Static joint impedance supports up to 64 joints in both diagonal and dense modes.

## Bimanual coordinates

`BimanualTaskConfig::coordination_mode` supports two formulations. The default
`kDecoupledCooperative` mode uses the Cartesian/SO(3) midpoint as its absolute task, shared-base
`p2 - p1` displacement as relative translation, and relative SO(3) orientation. The
`kCoupledRigidTransform` mode uses the moving-frame rigid transform from arm-1 TCP to arm-2 TCP and
therefore includes the expected translation/rotation lever-arm coupling.

Callers provide each robot's authoritative `6x7` TCP Jacobian in its own base frame. The library
applies the configured base calibration and main-frame axis rotation, then constructs the combined
`12x14` task Jacobian. The complete torque controller accepts desired 12D task velocity and computes
`desired_task_velocity - J_task * joint_velocity` internally. Its 14-joint nullspace gains must have
zero cross-arm blocks; each dense `7x7` diagonal block is projected through only that arm's TCP
Jacobian.

## Fault propagation in a custom controller

A structural error (wrong shape, backend, dtype, layout, mode or alias) returns a failed C++
Result or raises a Python exception. Abort the dependent tick immediately. An old array value
left behind by such a failure is not a newly computed command.

Numerical faults are row-local status flags. `merge_status` ORs upstream flags and normalizes
UnsupportedMode, InvalidParameter, InvalidInput, NonFiniteInput, InvalidDt, NoValidOutput and
HoldExpired into SafeMode. Saturation, ResetApplied and OutputHeld alone do not create a fault.
The existing wrench-to-torque composition and safety policy use the same normalization.

After computing a custom command, merge every participating stage's status and call
`apply_safe_output_policy`. Use separately allocated `SafetyState` for the final command;
each lifecycle block already advances its own safety history. Applying a policy twice to the
same history would consume two hold steps. Reset with `reset_safety_state`; optional reset masks
are Boolean `[B]` arrays. Hold-last-valid requires a positive `max_hold_steps` and falls back to
zero when there is no valid command or that budget expires.

Status inputs to C++ `merge_status` are host descriptors, even for CUDA; their underlying arrays
must be on the chosen device. Its output may alias only the first input exactly. All other
partial or exact overlaps are rejected. Python exposes the equivalent `merge_status(*inputs,
output=..., context=...)` operation.

## Canonical equations and checked arrays

Raw C++ equations are templates in `nvidia::srl::controllers::equations`. They are usable inside
user CUDA kernels, perform no allocation and own no state. Strided vector/matrix views support
user-owned storage. Callers must provide valid pointers and sufficient extents, finite operands,
positive time increments, ordered clamp limits, nonnegative rates, and valid physical parameters.
Dense feedback requires a matrix row within bounds and a matching vector. Rotation operations
require valid homogeneous 4x4 transforms; their Boolean result must be checked. Flow geometry
uses three-element vectors with distinct endpoints. Energy-tank state is positive, its energy
bounds are positive and ordered, and the transition width fits between the bounds.

The 45 checked Python equation adapters use `EquationConfig(device, dtype, batch_size)`.
All numeric operands, including scalar values and Booleans, are explicit shared/per-batch arrays:

| Operand | Shared | Per batch |
|---|---|---|
| Scalar | `[]` | `[B]` |
| Vector | `[N]` | `[B,N]` |
| Matrix | `[M,N]` | `[B,M,N]` |

Indices such as `row` and `col` remain ordinary host integers. All output arrays are supplied
by the caller: scalar results use `[B]`, vector results `[B,N]`, and Boolean results Boolean
`[B]`. Functions with multiple results have named `<FunctionName>Outputs` dataclasses. Every
call receives writable uint64 `[B]` `status`. No implicit broadcasting, dtype conversion,
allocation, device transfer or stream synchronization occurs in the native adapter.

Structural errors abort before writing. Non-finite operands/results and checked domain errors
zero the affected output row and set a fault status. Positive-dt, ordered-limit and geometry
checks are included; the adapters do not prove physical consistency among independently supplied
operands. Predicate equations return false for a failed predicate; callers decide how that
predicate affects the larger controller. Saturation/validity fields in structured results remain
explicit data that callers can incorporate into their policy.

## Backend and timing contract

Every lifecycle component and checked equation supports float32/float64 on CPU and CUDA.
CUDA requires a CUDA-enabled library; a CPU-only build reports BackendUnavailable for CUDA.
All device arrays and streams belong to the caller. Calls enqueue on `ExecutionContext`'s stream
and never silently synchronize another stream. Order producers and consumers on the same stream,
or use explicit application-owned CUDA events. Synchronize before reading results on the host.
Python object handling is not a hard-real-time interface.

The C++ lifecycle uses fixed-size descriptors and caller storage in its hot path. Canonical
C++ equations can be fused in a custom CUDA kernel to avoid one launch per scalar equation.
Graph-capture tests cover all newly public lifecycle blocks and checked equations. Such tests
establish execution/ownership behavior, not a hard-real-time latency guarantee.

## Packaging and verification

`python -m pip wheel . --no-deps -w dist` builds an ordinary platform/ABI-specific native wheel.
The extension and controller shared library are colocated with relative runtime lookup. Set
`CONTROLLERS_CMAKE_ARGS` for optional CUDA flags and `CMAKE_BUILD_PARALLEL_LEVEL` for build
parallelism. CUDA still requires a compatible toolkit/runtime and the project's per-SM launch
profile contract. Candidate functional builds may explicitly set
`NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF`; this is not deployment qualification.

C++ consumers use `find_package(nvidia-srl-controllers CONFIG REQUIRED)` and link
`nvidia::srl::controllers`. Package tests relocate the installed prefix and compile all 36 public
types, plus host/device canonical equations, without private include paths. Public symbol tests
continue to reject leaked raw backend entrypoints.

Run `CONTROLLERS_TEST_CUDA=1 PYTHONPATH=python/src pytest -q python/tests/controllers` to require
real GPU tests; CUDA errors fail instead of skipping. Run both API generators with `--check`
after changes. The committed inventory is intentionally separate so adding a declaration without
updating coverage and documentation fails its consistency tests.

## Component inventory

Every entry supports shared/per-batch parameters, CPU/CUDA execution and float32/float64.

| Public lifecycle type | Configuration |
|---|---|
| `TorqueFilter` | documented ControllerConfig bounds |
| `JointImpedanceController` | documented ControllerConfig bounds |
| `NullspaceController` | documented ControllerConfig bounds |
| `OperationalSpaceMass` | documented ControllerConfig bounds |
| `CartesianImpedanceController` | documented ControllerConfig bounds |
| `BimanualCartesianImpedanceController` | 14 joints, 12 task axes |
| `BimanualTaskKinematics` | 14 joints, 12 task axes |
| `CartesianAdmittancePoseController` | documented ControllerConfig bounds |
| `OperationalSpaceInverseDynamicsController` | documented ControllerConfig bounds; preferred name for acceleration-level inverse dynamics |
| `OperationalSpaceController` | compatibility name with the same behavior |
| `CartesianForceController` | documented ControllerConfig bounds |
| `CartesianUnifiedForceImpedanceController` | documented ControllerConfig bounds |
| `StaticJointImpedanceController` | diagonal or dense joint gains |
| `StaticCartesianImpedanceController` | 6 task axes in an explicit Cartesian frame |
| `CartesianUnifiedForceAdmittanceController` | 6 task axes in an explicit Cartesian frame |
| `VariableCartesianImpedanceController` | documented ControllerConfig bounds |
| `TaskWrenchToJointTorqueController` | documented ControllerConfig bounds |
| `CartesianImpedanceTorqueController` | documented ControllerConfig bounds |
| `VariableCartesianImpedanceTorqueController` | documented ControllerConfig bounds |
| `CartesianUnifiedForceImpedanceTorqueController` | documented ControllerConfig bounds |
| `BimanualCartesianImpedanceTorqueController` | 14 joints, 12 task axes |
| `ProportionalBlock` | positive signal_dim |
| `IntegratorBlock` | positive signal_dim |
| `DifferentiatorBlock` | positive signal_dim |
| `JointStiffness` | documented ControllerConfig bounds |
| `JointDamping` | documented ControllerConfig bounds |
| `JointDampingDoubleDiagonalization` | documented ControllerConfig bounds |
| `CartesianStiffness` | documented ControllerConfig bounds |
| `VariableCartesianStiffness` | documented ControllerConfig bounds |
| `CartesianDamping` | documented ControllerConfig bounds |
| `CartesianDampingDoubleDiagonalization` | documented ControllerConfig bounds |
| `CartesianDampingFactorizationDesign` | documented ControllerConfig bounds |
| `CartesianVirtualMass` | documented ControllerConfig bounds |
| `ShapingFunction` | positive signal_dim |
| `CommandObserver` | positive signal_dim |
| `FlowFunnel` | documented ControllerConfig bounds |
| `InverseAPF` | documented ControllerConfig bounds |
| `InertialCoupling` | documented ControllerConfig bounds |
| `EnergyTank` | signal_dim = 1 |
| `VariableCartesianUnifiedForceImpedanceController` | documented ControllerConfig bounds |

## Selecting a controller

| Application input / intent | Public component |
|---|---|
| Joint error with model mass and automatic damping | `JointImpedanceController` |
| Joint error with explicit constant K/D | `StaticJointImpedanceController` |
| Cartesian pose/twist error with operational-space mass | `CartesianImpedanceController`, plus mass and torque projection stages |
| Cartesian pose/twist error with explicit constant K/D | `StaticCartesianImpedanceController` |
| Measured wrench producing integrated pose offsets | `CartesianAdmittancePoseController` |
| Force error plus pose admittance | `CartesianUnifiedForceAdmittanceController`; add standalone shaping when required by the application |
| Impedance with force feedback | `CartesianUnifiedForceImpedanceController` |
| Two-arm absolute/relative task | `BimanualTaskKinematics` and the bimanual controller/pipeline |
| Two independent arms sharing a reference | Two separate single-arm controllers; a 12D coupled task changes the control law |

Follow the selected component's `requirements` and typed parameters/storage; do not substitute
static and mass-dependent controllers based only on the robot's joint count. The
[example parity guide](example-parity.md) records complete real-world compositions and distinguishes
native controller output from vendor/simulator actuation.

## Reset, updates and concurrency

Allocate all storage before the control loop, construct/prepare, inspect parameter status, then
reset. A reset mask selects rows whose history should be cleared; it is not a robot motion,
recovery or target-capture operation. For integrators it resets pose/twist/previous-signal history;
for filters it resets previous-command initialization. Application-level stopped rows remain
stopped until the application explicitly permits a restart.

`apply_parameters` preserves the existing storage contract and updates parameters. Retain both old
and new owners until earlier asynchronous work completes; validate the resulting parameter status
before admitting commands. A configuration/dimension change needs appropriately allocated storage
and another preparation. Each mutable state/storage tree belongs to one ordered execution path.
Do not step/reset/update the same instance concurrently from two CPU threads or unrelated CUDA
streams. Separate instances and storage support independent execution.

CPU Python callers can inspect `storage.status` and `storage.parameter_status` directly as NumPy
arrays. CUDA callers keep status processing on-device or explicitly synchronize at a non-real-time
observation boundary. Structural exceptions stop the dependent tick; numerical status must be
checked even when the call returns normally. Zero/hold-last is an output policy, not an instruction
to automatically restart an example after a fault.

## Migration and example registry

Replace retired Eigen `init`/`compute` calls and `Prepared*` interfaces with the single typed
`requirements`/`prepare`/`reset`/`step` lifecycle. Move vendor/Eigen conversion into the application
adapter and pass array views; keep units, frames and input order explicit. Use installed
`controllers.hpp`, `types.hpp`, `composition.hpp`, `equations.hpp` and `example_specs.hpp` instead
of private backend headers. Python uses `nvidia.srl.controllers`, with array providers owning CPU
or CUDA memory.

C++ `controller_example_specs()` preserves the original fixed `std::array<...,20>` return type.
Use `all_controller_example_specs()` for the full iterable 22-entry view and
`find_controller_example_spec()` for lookup. Python's
`controller_example_specs()` returns all 22 entries. Registry model support identifies a native
configuration; [asset availability](example-parity.md#runtime-model-support) is a separate check.

## Energy-tank accounting

`EnergyTank` uses `signal_dim=1`, one state variable `x` per row in sqrt(J), and stored
energy `E = x*x/2`. Energy limits, initial energy, and the depletion transition width
are joules; power inputs and `power_noise` are watts; `dt` is positive seconds.
Positive scaled `power_signal` charges the tank. For a raw wrench to be scaled, supply
`power_signal = -wrench.dot(measured_twist)` with power-conjugate frames and points.
`additional_power` is independent, nonnegative recovered power. Callers are responsible
for its physical provenance and avoiding double-counting it with the main power.

The cosine depletion scale depends only on energy, in both power directions;
velocity reversal does not restore full scale. Positive work is credited after
scaling, so absorbing a raw command that is attenuated cannot overfill the tank.
The scale is capped by start-of-step usable energy divided by the
requested draining work. Energy is integrated directly in joules; recharge is credited
after determining this step's command scale. Excess recharge is discarded at the upper
limit. The deadband affects `filtered_power` and positive recharge only: negative raw
power is always debited. This changes the historical square-root Euler integration,
deadband accounting, and gated/signed `additional_power` behavior without changing
EnergyTank signatures or storage shapes. Budget capping and discarded recharge set
saturation status; ordinary depletion is not safe mode.

Invalid input/state or unrepresentable arithmetic makes lifecycle output zero and
preserves state until a valid step or reset. Checked equation adapters zero their
invalid outputs under their existing status policy. The unchecked equation returns
an invalid-state sentinel (-1) for invalid domains/arithmetic. Inspect status before
reusing checked outputs. Boundary reconstruction roundoff is normalized only within
eight scalar machine epsilons relative to the affected bound. Floating-point energy
balance is checked on reconstructed stored energy with a cumulative tolerance
`32 * epsilon * step_count * upper_energy_limit`.

The energy guarantee concerns supplied sampled power and period. It does not qualify
hardware, enforce a contact-force/travel limit, or account for downstream actuator
filtering and additional controller contributions.

The paired tank example appends `kCartesianForceImpedanceWithTankTorque` to the example
kind enum, exposed as `cartesian_force_impedance_with_tank_torque` in Python.
The fixed-size `controller_example_specs()` accessor remains a 20-entry legacy view.
`all_controller_example_specs()` includes the tank, virtual walls, and goal/obstacle APF examples,
for 21 entries. Non-removed entries retain their relative order and enum values.
Virtual walls and goal/obstacle APF append
`kCartesianVirtualWallsTorque` and `kCartesianPotentialFieldTorque`, exposed as
`cartesian_virtual_walls_torque` and `cartesian_potential_field_torque` in Python.

At the floor, external recovered power is needed to resume the force contribution;
the tank remains operational. The Franka hardware example harvests half its existing
impedance damping dissipation without crediting spring or inertial work.
This sampled energy bound is not a smoothness guarantee or a passivity
qualification of the impedance, Coriolis compensation, or filtered actuator
command.
