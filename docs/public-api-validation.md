# Public API implementation validation

## Non-functional UR Isaac impedance companions

**Current status, 2026-10-01: non-functional.** This applies to
`joint_impedance_controller_parallel.py` and `cartesian_impedance_controller_parallel.py`
for both **UR5e and UR10e**. The launchers remain available for diagnosis and display this
classification in help and before simulator startup. The classification is specific to these
simulation companions; the physical examples remain the controller/configuration ground truth.
The three UR admittance companions retain their separate historical results.

### What failed

The earlier joint-impedance nominal and injected-fault passes started exactly at the captured
target with zero velocity and zero torque. They establish equilibrium holding and fault
isolation for that state, not disturbance stability. Those historical measurements are retained,
but their joint companion qualification is **superseded by the non-functional classification**.
The historical 17/18 aggregate below must not be read as a current functionality claim.

The follow-up comprised **26 complete headless rollouts with 16 environments each**: 22 ran
for 1 s and four for 10 s. Both hardware-model profiles, shared native controller code, 2 ms
controller period, torque/rate limits and official assets were preserved. After capturing each
hold reference, a temporary invocation wrapper applied an identical `0.001 rad/s` velocity
perturbation to `wrist_3_joint` in every environment. Separate UR5e cases applied a `0.001 rad`
position perturbation instead. Both controllers oscillate on both models.

| Diagnostic | Joint impedance | Cartesian impedance |
|---|---|---|
| UR5e, wrist velocity perturbation, 1 s | Rate limiting for 498/500 consecutive ticks | Rate limiting for 499/500 consecutive ticks |
| UR10e, same perturbation, 1 s | Rate limiting for 498/500 consecutive ticks | Rate limiting for 499/500 consecutive ticks |
| UR5e, hardware-equivalent torque rounding, perturbation, 10 s | Rate limiting for 4,998/5,000 consecutive ticks | Rate limiting for 4,999/5,000 consecutive ticks |
| UR5e, hardware-equivalent rounding, equilibrium only, 10 s | Zero limiting; remains at equilibrium | Zero limiting; remains at equilibrium |
| UR5e, solver position/velocity iterations 8/1 to 64/8 | Instability persists | Instability persists |
| UR5e, 4 or 16 physics-only substeps, unchanged 500 Hz feedback | Instability persists | Instability persists |
| UR5e, external forces every solver iteration | Instability persists | Instability persists |

These counters include the native **torque-rate limiter**, not just maximum-torque saturation.
Small TCP translation error can hide wrist rotation; neither a translation-only check nor an
equilibrium-only `--validate` pass overrides the classification. The deliberate perturbations
are diagnostic experiments, not options exposed by the stock launcher.

### What was confirmed and what remains unresolved

Fresh frame/Jacobian tests passed for both models, each using 16 configurations, rotated roots,
and zero/nonzero TCP offsets against independent hardware-URDF kinematics. The current adapter
handles the 180-degree base rotation. No remaining frame flip or effort-delivery sign/scale
error was found: sampled PhysX actuation forces equal the expected float32 effort command.
Sampled native Cartesian outputs agree with the configured law to approximately `2.85e-14 Nm`.

The pinned hardware URCL rounds torque commands to a `1e-6 Nm` grid; the current Isaac adapter
sends unquantized floating-point effort. A diagnostic wrapper matching
[URCL's pinned encoder](https://github.com/UniversalRobots/Universal_Robots_Client_Library/blob/a854de920d602300ce7c0fe3bd79655cb85bd678/src/control/reverse_interface.cpp)
suppresses the spontaneous Cartesian equilibrium trigger, but fails to stabilize either
controller after a disturbance. That rounding change has **not been applied** to the examples.
It would correct transport parity without establishing functional torque control.

The initial velocity response agrees with the recorded PhysX mass matrix and commanded torque.
The configured simulation shows sampled damping feedback instability at the 2 ms controller
interval. More physics steps while holding the same torque do not shorten that feedback interval.
This does not establish an official asset inertia defect. Official masses, centres of mass,
inertias and authored armature remain unchanged, as do the physical examples and native settings.

[UR direct torque](https://www.universal-robots.com/manuals/EN/HTML/SW5_24/Content/prod-scriptmanual/all_scripts/direct_torque.htm)
includes gravity and default friction compensation. Simulation gravity is disabled and joint
friction is zero; this is an interface approximation, not a measured reconstruction of firmware
and motor behavior. Physical RTDE timing, velocity estimation, residual friction, motor dynamics
and active compensation settings were not observed. **The remaining cause of the difference
from stable physical operation is not confirmed.** No physical robot was contacted.

### Reproduction and status removal

The existing launchers can still reproduce the default hold with the simulator's ABI-matching
extension and interpreter, as described in the [Isaac guide](../examples/isaaclab/README.md):

```bash
PYTHONPATH="$PWD/python/src:$PYTHONPATH" "$ISAACLAB_PYTHON" \
  examples/isaaclab/joint_impedance_controller_parallel.py \
  --headless --num_envs 16 --duration 10 --robot-model ur5e
PYTHONPATH="$PWD/python/src:$PYTHONPATH" "$ISAACLAB_PYTHON" \
  examples/isaaclab/cartesian_impedance_controller_parallel.py \
  --headless --num_envs 16 --duration 10 --robot-model ur5e
```

Replace `ur5e` with `ur10e` for the second model. The stock joint hold can still return successful
equilibrium metrics. To reproduce the disturbance diagnosis, capture the reference first, then
write the same small wrist position or velocity perturbation through the simulator state API,
refresh kinematics, and retain the unchanged native controller/filter and 500 Hz feedback.
Record requested effort, PhysX actuation forces, joint position/velocity and limiter status.

Remove the non-functional status only after an evidenced simulator/interface correction preserves
the hardware controller/configuration and official asset parameters, and repeatable equilibrium,
disturbed-hold and fault-isolation runs pass on **both UR5e and UR10e with 16 environments**.
Keep torque-rate limiting and rotational behavior in those checks. A successful equilibrium hold
alone is insufficient.

The `ur_impedance_functionality` section of [structured evidence](example-validation.json)
records the current classification, all 26 case summaries and provenance. Historical snapshots
below and their raw measurements are preserved. Full local traces, probe scripts, the unapplied
rounding proposal and report are retained in `/tmp/controllers-ur-interface-check-tmsdbl_t`;
that temporary directory is not required to read the repository evidence. The initial incorrectly
timed substep probes are explicitly excluded; corrected probes verify 2 ms physical time per
controller update. This was simulation diagnosis, not physical or hard-real-time qualification.

## UR asset reversion, 2026-10-01

UR5e and UR10e use NVIDIA's official Isaac Sim USD assets. Their link masses, centres of mass
and inertia tensors were never edited. The optional companion armature override, its CLI
argument and capture loader have been removed. Joint armature is again taken only from the
USD joint's authored value. The asset factory and existing direct-effort configuration are
unchanged; the base/TCP frame and COM-to-origin Jacobian corrections remain in place.

The earlier claim that these assets lack proper inertia or require a paired inertia capture
is withdrawn. Zero joint armature does not mean zero link inertia. The observed Cartesian
instability does not establish an asset inertia defect; its remaining cause is unresolved.

| Check after reversion | Result |
|---|---|
| Public Python suite, CPU and mandatory CUDA | 663 passed |
| UR5e/UR10e hardware-frame checks, 16 configurations each | Both passed |
| Repository checks and explicit adapter lint | Passed |
| UR Cartesian nominal rollout, headless, 16 environments, 10 s | 5,000 ticks completed; numerical metrics and final native command identical to the previous nominal run |

That rollout still fails the unchanged motion criteria. Removing the unused override does
not change default actuation behavior. The earlier 36-case retained qualification record remains
historical; it did not use the override. All 39 retained audited physical files and 81 checked native
controller/configuration files remain unchanged. The `ur_asset_reversion` section of
[structured evidence](example-validation.json) records current source hashes and checks.

## Historical companion adapter fixes, 2026-09-30

The fixes preserve all 39 retained audited physical source/header/model/CMake files and all 81 checked
native/controller-configuration files from the start of this work. Only simulator adapters,
their tests, packaging and documentation changed. The earlier working-tree changes, including
hardware Enter gates, remain intact. No physical executable was run or robot commanded.

The dual-arm adapter refreshes forward kinematics before capturing both pose references and
reports tracking error for both arms. All link Jacobians now shift from PhysX centre of mass
to the controller wrist/TCP origin. UR adapters read the hardware URDF's 180-degree base
rotation and flange transform, apply the configured TCP offset, and consistently transform
poses, Jacobians, twists, contact forces and pose commands. An optional UR inertia override
was added at this snapshot and subsequently removed on 2026-10-01. The 36 retained qualification
rollouts did not use that override. See the current
[frame adapter contract](../examples/isaaclab/README.md#frame-and-plant-adapters).

| Check | Post-fix result |
|---|---|
| Public Python suite, CPU and mandatory CUDA | 672 passed |
| Dedicated Isaac adapter tests, 16 environments each | Three retained cases passed: UR5e, UR10e, supplied-inertia feature |
| Repository checks | Passed generation, licenses, cpplint, Black, flake8 and mypy |
| Relocated CMake installation | New helper loads both installed URDF models |
| Nominal rollouts, all canonical companions | 18/18 retained completed; 17/18 qualified |
| Separate injected-fault rollouts | 17/18 retained qualified |

Every retained rollout used 16 environments and a fresh application process and lasted
10 seconds. The original tracking, spread and actuator-limiting
thresholds remain unchanged. Rizon limits are nominal unmeasured fixtures. The three retained adapter
tests are separate from the 36 retained motion cases; the supplied-inertia test uses explicitly synthetic
values and does not qualify a calibrated UR plant.

Dual Flexiv's nominal tracking error is 0.011879 mm,
spread 0.002030 mm, and longest actuator-limiting run
0 ticks. Its stale right-arm reference failure is resolved.

Remaining nominal failures:

| Companion | Observed qualification failure |
|---|---|
| `cartesian_impedance_controller` | actuator limiting persisted 4994 ticks >10; spread 0.245649842 m >0.001 m; tracking error 0.13125002699570612 m >0.02 m |

The earlier inference that UR requires a measured rotor-inertia override is withdrawn.
These runs used the official USD asset's mass/inertia and authored joint armature. The observed
Cartesian instability does not establish an asset inertia defect. The optional override was
removed on 2026-10-01; the remaining failure's cause is unresolved. Vendor actuation, payload
calibration and simulator integration remain distinct from native controller parity.

[Structured evidence](example-validation.json), under `adapter_fix_validation`, preserves all
36 retained outcomes, raw final native commands and resolved configurations when metrics were returned,
and source hashes. Earlier results remain under their original keys. Local full
logs and scripts are in `/tmp/controllers-companion-fixes-ue1b4ec0`; temporary paths may be
removed independently of the repository documentation. The full 126-case model/batch motion
matrix, attended hardware and hard real-time qualification remain unvalidated.

## Earlier companion alignment, 2026-09-30

These results apply to the dirty working tree based on
`ec3a384d62c7f4c003002bee97c80b038d29ccfd`, including the pre-existing hardware Enter gates.
All 39 retained audited physical source/header/model/CMake files retain their initial working-tree hashes.
[Structured evidence](example-validation.json) records companion source hashes, model variants,
observed simulator metrics and qualification failures. The [parity contract](example-parity.md)
defines the native boundary and separates plant/transport differences.

| Check | Result |
|---|---|
| Fresh CPU CMake/CTest | 47 passed |
| Fresh CUDA CMake/CTest, SM89 | 55 passed |
| Public Python suite, CPU and mandatory CUDA | 657 passed |
| Native profile/composition parity | All 18 retained targets and supported model variants, batches 1/16/256/1024 |
| Actual unchanged Franka admittance helper traces, Eigen-only FR3/FER oracles | Passed filtered wrench, pose/twist, history and status comparisons |
| CPU wheel, fresh isolated Python 3.12 environment | 321 passed, 10 source-only cases deselected |
| Repository `make check`, with `TMPDIR=/tmp` | Passed generation, licenses, cpplint, Black, flake8 and mypy |
| Doxygen and relocated guide installation | Passed |
| Default-model nominal simulator runs, batch 16 | All 18 retained completed; 16 meet motion criteria |
| Default-model one-row fault simulator qualification, batch 16 | 16 retained passed; 2 failed |
| Fresh-process nominal/fault smoke for FR3 Cartesian impedance, batch 16 | 2 passed |

CPU/CUDA builds used GCC 13.3, Python 3.12 and CUDA 13.1 on RTX 4090. Simulator builds used
its separate Python 3.11.15 ABI, Isaac Sim 5.1, Isaac Lab 0.54.3 and Torch 2.7.0+cu128.
SM89 has no committed deployment tuning profile, so candidate builds explicitly disabled that
gate. This validates functionality; it does not qualify deployment timing.

Each retained default-model simulator case ran for 1 s.
Rizon cases used explicitly supplied **nominal test torque-limit fixtures**, not limits obtained
from a connected robot. Nominal completion means the run returned metrics; qualification also
checks stable buffers, stopped rows, spread, tracking where applicable and actuator limiting.
The fault sweep injects one non-finite row and checks isolation plus a latched stop.

The following failures remain with the unchanged hardware configurations:

| Companion | Observed qualification failure |
|---|---|
| Dual Flexiv | Healthy-row spread 8.766 mm versus 1 mm; actuator limiting persisted 390 ticks |
| UR Cartesian impedance | Spread 9.116 mm versus 1 mm; actuator limiting persisted 494 ticks |
| Panda/FER admittance, additional 256-row fault run | Spread 17.938 mm versus 1 mm |

Four representative FR3/Panda/UR10e/Rizon companions were also checked with batches 1 and
256 for 0.6 s: all four retained batch-one runs passed; three batch-256 runs passed, with the Panda failure
above. Those are adapter checks, not full model/batch qualification. Native comparisons cover
all supported configuration variants, including asset-unavailable Rizon4M, without importing Isaac.

A first combined 42-scene run exhausted GPU memory after about 26 scenes. Short lifecycle probes
showed retained 640 MB allocations per replaced PhysX scene while Torch cache size stayed stable;
reset, garbage collection and stage detachment did not bound that growth. The exact retained SDK
owner remains unconfirmed. Separate fresh application cohorts produced the nominal/fault results
above; the opt-in regression harness now uses a fresh process for each target/batch/fault case.
The focused isolated-process tests passed. The full default 126-case opt-in matrix has **not** run.

The full simulator model matrix, actual captured Rizon limits, attended hardware execution,
initial-motion/Enter behavior on robots, workcell safety, vendor servo behavior and hard real-time
qualification remain unvalidated. Rizon4M has no asset in the pinned official checkout and fails
explicitly. No physical robot was connected or commanded for this work.

Reproduce the checkout checks using the ABI-matching interpreter and fresh build directories:

```sh
TMPDIR=/tmp make check
ctest --test-dir <fresh-cpu-or-cuda-build> --output-on-failure
CONTROLLERS_TEST_CUDA=1 CONTROLLERS_ORACLE_BUILD=<fresh-cuda-build> \
  PYTHONPATH="$PWD/python/src" pytest -q python/tests/controllers
```

Use the [Isaac guide](../examples/isaaclab/README.md) for the CUDA/interpreter build and optional
simulator matrix. Temporary logs and build artifacts for this local run are under
`/tmp/controllers-companion-evidence`; the structured report remains in this repository when
those temporary files are removed.

## Historical public API validation, 2026-09-22

The record below describes an earlier working tree and is not the current qualification result.

Validated from the working tree based on `0cea75330f27bd8691c289a4b721c46e0ac9fc8c` on
2026-09-22, with Python 3.12, GCC 13.3, CUDA 13.1 and an NVIDIA RTX 4090 (SM89).

| Check | Result |
|---|---|
| Fresh CPU CMake build and complete CTest suite | 36 passed |
| Fresh CUDA CMake build and complete CTest suite | 45 passed |
| Public Python suite, CPU and mandatory real CUDA | 557 passed |
| CPU wheel in a fresh virtual environment, isolated imports | 287 passed, 2 source-only tests deselected |
| CUDA wheel in a fresh virtual environment, isolated imports | 552 passed, 2 source-only tests deselected |
| `make check` | Passed generators, license checks, cpplint, Black, flake8 and mypy |
| Python synthetic PID/nonlinear/energy-tank example | Passed |
| Standalone custom-state CUDA example against relocated installed prefix | Passed, command 0.393 |
| Representative C++ stateful composition, 200 ticks | Zero observed allocations |

The two wheel-test exclusions inspect repository-only simulator launcher files; those tests pass
in the checkout suite. The wheel checks clear PYTHONPATH and library-path overrides, use Python
isolated mode, and verify that imports resolve inside the fresh virtual environment.

Coverage includes all 19 newly exposed blocks, both scalar dtypes, shared and per-batch
parameters, CPU/CUDA execution, caller-stream CUDA graph capture/replay, analytic examples for
all 45 checked equations, alias rejection, row-local numerical faults, status normalization,
and bounded hold-last behavior. Relocated C++ consumers query all 36 public lifecycle types and
compile canonical equations in both host code and an application-owned CUDA kernel.

SM89 has no committed deployment tuning profile in this checkout. CUDA builds used
`-DNVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF` for functional qualification. The repaired
private qualification executable was also built and exercised with `--dimension 7`. No new
performance profile was committed. These results do not establish deployment timing or physical
robot behavior.

Reproduce the installed-wheel checks with:

```sh
python -m pip wheel . --no-deps -w /tmp/controllers-wheels
python tools/verify_python_wheel.py \
  --wheel /tmp/controllers-wheels/<wheel-file> \
  --work-dir /tmp/controllers-wheel-consumer-new
```

For CUDA, build with `CONTROLLERS_CMAKE_ARGS` specifying the device SM and add `--cuda` to the
consumer check. Each consumer work directory must be new. See the
[public API guide](public-controller-api.md) for configuration, lifecycle and ownership contracts.

## Franka point-field examples, 2026-10-02

Validated local uncommitted changes on `feat/franka-virtual-walls-and-apf`, based on
`dfc60683d42c7301b50e4737d4d2f7d18360619d`. The new examples share canonical wall/sphere equations,
prepared orientation/posture control, and bounded translation forces. The existing `InverseAPF`
now reports invalid parameter domains during preparation/application on both backends.

| Check | Result |
|---|---|
| Fresh CPU build and CTest | 47 passed |
| CUDA SM89 build and CTest | 55 passed |
| FER/Panda libfranka 0.9.2 build and CTest | Nine binaries built; 57 passed |
| FR3 libfranka 0.21.3 build and CTest | Nine binaries built; 57 passed |
| Focused equations, legacy APF, public blocks, and inventory on CPU/CUDA | 626 passed |
| Optional Torch field composition, CPU/CUDA and both dtypes | 8 passed |
| Complete Python suite with real CUDA | 807 passed; 16 pre-existing force-admittance sign assertions failed |
| `make check` | Generators, license, cpplint, Black, flake8 and mypy passed |

The unrelated failing parametrized test is
`test_force_admittance_integral_bounds_are_optional`. An untouched archive of the baseline SHA,
built separately with CPU/Python support, reproduces all eight CPU cases: expected and observed
integral signs are reversed (without limits, +100 expected and -100 observed for positive external
wrench). The candidate has the same
eight CUDA failures. This work does not change force-admittance behavior or those expectations.

Bounded Isaac Lab checks use two environments per model, a 1 ms period, and one deliberately
injected non-finite row. APF validation additionally requires observed repulsion, clearance above
5 mm, final goal error at most 5 mm, and speed at most 0.01 m/s for at least 0.5 s in every row.

| Model | APF duration | Minimum surface clearance | Maximum final goal error | Minimum settled time |
|---|---|---|---|---|
| FR3 | 10 s | 15.708 mm | 0.091 mm | 6.229 s |
| Panda | 10 s | 16.663 mm | 0.014 mm | 6.558 s |

Both APF runs passed fault isolation/recovery and persistent-buffer checks, with zero aborted rows
and zero actuator saturation. The virtual-wall hold smoke runs use 2 s per model; independent
numerical probes exercise all six faces, corner superposition, unilateral damping, and the force
cap. Virtual shapes have no physical collider. The simulator hand-body origin and its Jacobian
are the task point; the hardware uses its configured `O_T_EE` and matching EE Jacobian.

Reproduce focused checks with the built checkout extension:

```sh
CONTROLLERS_TEST_CUDA=1 PYTHONPATH=python/src python -m pytest -q \
  python/tests/controllers/cartesian_fields_test.py \
  python/tests/controllers/equations_test.py \
  python/tests/controllers/public_blocks_test.py \
  python/tests/controllers/public_inventory_test.py
CONTROLLERS_TEST_FIELD_TORCH=1 PYTHONPATH=python/src "$ISAACLAB_PYTHON" -m pytest -q \
  python/optional_tests/cartesian_fields_torch_test.py
PYTHONPATH=python/src "$ISAACLAB_PYTHON" \
  examples/isaaclab/franka_artificial_potential_field_controller_parallel.py \
  --headless --num_envs 2 --duration 10 --validate --robot-model fr3
```

Use `--robot-model panda` for Panda; replace the launcher with
`franka_virtual_walls_controller_parallel.py` and duration with 2 for the wall smoke.
Build the extension for the selected interpreter, as described in the example guides.
These are local mathematical/software/simulator checks, without physical motion or hard-RT qualification.

## Franka point-field rebase integration, 2026-10-06

Validated the conflict-resolution working tree while replaying the branch ending at
`b95ec08e7f9dc3883f389b9f209796c8b989d58d` onto the fetched `origin/main` commit
`2ef76906905c149b3a14eacdef84220a60010b5a`. The combined registry retains the original
20-entry C++ accessor and exposes 24 entries through the full view/Python registry.
The tank and point-field companions use the shared hardware profiles; point fields retain their
static orientation and mass-weighted nullspace composition. Physical runners include both
startup confirmation gates. Simulator faults latch stopped rows; APF goal/settled diagnostics
exclude stopped rows, while clearance and guard-abort checks still include every row.

| Check | Result |
|---|---|
| CPU build and complete CTest | 137 passed |
| Candidate CUDA SM89 build and complete CTest | 146 passed |
| Focused CPU/CUDA equations, APF, tank, public API, hardware profiles and documentation | 784 passed |
| Optional Torch fields, CPU/CUDA and float32/float64 | 12 passed |
| FER/Panda 0.9.2 and FR3 0.21.3 SDK builds | All nine binaries built for each; 14 focused CTest checks passed per build |
| Isaac interpreter native profile/documentation tests | 21 passed |
| Generator freshness, source audit, license, formatting, lint and type checks | Passed |

The candidate uses Python 3.12 for native tests and the ABI-matching Python 3.11/Isaac Sim 5.1
interpreter for Torch/simulator checks on an RTX 4090. The simulator's float64 controller buffers
remain separate from float32 PhysX/marker buffers. A marker dtype mismatch exposed by the first
post-rebase startup was corrected at that visualization boundary.

Local evidence is retained under `/tmp/controllers-rebase-fields-`: `cpu-ctest.log`,
`cuda-ctest.log`, `cuda-pytest.log`, `torch-pytest.log`, `fer-build.log`, `fr3-build.log`,
`fer-ctest.log`, `fr3-ctest.log`, `isaac-profiles.log` and `check.log`. This focused integration
validation does not replace the historical full-suite failures above or establish physical
hardware, hard-RT or deployment qualification.

The post-rebase APF checks ran two environments per model for 10 simulated seconds
with one deliberately injected fault. Both reported stable buffers, exactly one latched fault,
verified healthy-row isolation, zero clearance aborts and zero actuator saturation.
Minimum clearance includes every row; goal/settled values describe healthy rows.

| Model | Minimum clearance (mm) | Final maximum goal error (mm) | Minimum settled time (s) |
|---|---|---|---|
| fr3 | 9.629 | 0.083 | 6.235 |
| panda | 12.937 | 0.023 | 6.563 |

Simulator evidence: `/tmp/controllers-rebase-fields-fr3-apf.log` and
`/tmp/controllers-rebase-fields-panda-apf.log`.

## Dual Rizon companion alignment, 2026-10-15

Validated the uncommitted candidate based on
`745f5d12e1d8e9d288099002e0f2e360ce4d19f0`. Hardware source, immutable profiles, gains and
limits remain unchanged. Both companions use the complete public bimanual torque controller,
expose all four hardware main-frame selections, and capture the Cartesian/SO(3) midpoint and
relative displacement. Quaternion-based reference capture handles relative rotations up to
180 degrees. Circular pose references remain in arm-1-base axes; desired velocity and
acceleration rotate into the configured main axes. Metrics report the frame configuration and
inter-wrist distance error.

Actual simulator startup exposed a native CUDA defect: the independent-arm nullspace kernel
allocated compact scratch matrices but accessed them through helpers with a fixed 32-column
stride. Compute Sanitizer reported five errors before the fix. The corrected helpers accept an
explicit stride, and this kernel consistently uses seven-column scratch, including its 6x6
solve. The complete controller now also establishes the caller execution context before its
velocity-error stage. No allocation, logging or synchronization was added to the step path.

| Check | Result |
|---|---|
| Fresh CPU build and complete CTest | 141 passed |
| Fresh CUDA build for the Isaac interpreter | Passed |
| CPU/CUDA API, profiles, public blocks/inventory, adapter and documentation tests | 410 passed, 2 optional helper-oracle checks skipped |
| Complete CUDA-build CTest | 149 passed; 1 package-consumer failure from the host driver mismatch |
| Full Python suite, repository Python 3.12 / NumPy 2.5.2 | 581 passed, 2 optional helper-oracle checks skipped |
| Focused Torch CPU/CUDA reference, frame, posture, fault and graph suite | 20 passed |
| Compute Sanitizer, both complete-controller CUDA profiles | 2 passed, zero errors |
| Isaac nominal matrix, 1 and 16 pairs, both profiles, 10 s each | 4 cases passed |
| Isaac injected-fault matrix, same batches/profiles/duration | 4 cases passed |
| Actual circle launcher, rotated base, explicit-arm2 main frame, 2 pairs / 3 s | Passed |
| Generator freshness, license, lint, formatting and type checks | Passed |

Nominal results over 10,000 steps per case:

| Profile | Pairs | Maximum task tracking error (mm) | Maximum wrist-distance error (mm) |
|---|---|---|---|
| Circle | 1 | 0.013569 | 0.000566 |
| Circle | 16 | 0.013568 | 0.000685 |
| Compliant hold | 1 | 0 | 0 |
| Compliant hold | 16 | 0 | 0 |

Both circle cases reach approximately 200 mm displacement. Every nominal case has zero unsafe
rows and zero saturation, with stable caller-owned buffers. Fault checks inject non-finite
velocity in row 0, require exactly one latched fault and verify healthy-row isolation.
The single-pair fault cases have no healthy comparison row; the 16-pair cases provide that check.
All four fault runs end with a zero native command for row 0 and no saturation. The largest
healthy-pair trajectory spread is 0.0023 mm.

An additional actual CLI circle run used a 90-degree base rotation about z and an explicit
main frame referenced to arm 2 with a 90-degree rotation about x. Both pairs completed 3,000
steps and moved approximately 190 mm. Maximum task tracking error was 0.010781 mm and
wrist-distance error was 0.002696 mm, with no unsafe rows or saturation and stable buffers.

The CUDA candidate uses GCC 13.3, CUDA 13.1, SM89 with
`NVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF`, Python 3.11.15,
Torch 2.7.0+cu128, Isaac Sim 5.1 and Isaac Lab 0.54.3 on an RTX 4090. Assets are the official
Flexiv checkout at `2d07d67887aff718ea93250e68e9e8eff4b0a62d`; simulator pairs use Rizon4s.
Torque limits are nominal model fixtures, not measurements from physical robots.

The host has kernel driver 580.173.02 and installed userspace libraries 580.178.04. Test processes
route NVIDIA library loads to existing matching 580.173.02 libraries under `/tmp` using
process-local `LD_AUDIT`, `LD_LIBRARY_PATH` and a Vulkan ICD. No installed driver, module or service
was changed. The Python 3.11 source-extension link points at the fresh candidate build; its
previous target is recorded in `/tmp/controllers-dual-isaac-oct15-previous-extension-link.txt`.

The formal CUDA package-relocation test explicitly clears `LD_AUDIT`, `LD_LIBRARY_PATH` and
`LD_PRELOAD` before executing its consumer. It therefore encounters the host driver mismatch.
The exact installed/relocated CUDA executable succeeds when run with matching process-local
driver routing. This manual success does not replace the failed clean-environment CTest;
that test remains unqualified until the host driver installation is consistent.

An additional full Python run in Isaac's Python 3.11 / NumPy 1.26 environment returned
580 passed, 2 skipped and one existing scalar-bitwise assertion failure at
`python/tests/controllers/composition_test.py:61`. The repository Python 3.12 suite passes that
same test. This interpreter-specific failure is separate from the focused controller and
simulator checks.

Evidence is under `/tmp/controllers-dual-isaac-oct15-`: `native-fix-build.log`, `cpu-build.log`,
`cpu-ctest.log`, `cuda-ctest.log`, `cuda-pytest.log`, `cuda-relocated-routed.log`,
`final-cpu-pytest.log`, `fixed-torch.log`, `fixed-memcheck.log`, `matrix.log`,
`matrix/` child logs, `matrix-summary.json`, `rotated-frame.log`, `rotated-frame-metrics.json`,
`final-check.log`, and `provenance.json`.
The original `memcheck.log` retains the failing pre-fix evidence. Approved cleanup removed the
temporary probe scripts and debug ledger while retaining logs. These are local software and
simulator results; physical execution, workcell safety and hard real-time qualification remain
separate.

## 2026-10-29 licensing-only source and package checks

Base: `be14efb82bdd2de3e0736fbb6324708f43737a15` on `codex/controllers-release`.
The owner selected the NVIDIA License dated 22 March 2022 for NVIDIA-owned material.
The complete primary license SHA-256 is
`0799fbb323e03399571dcc8279d7cd7433f2374e33502b2c723961635ea9fea2`.
The Franka helper, UR model licenses and historical qualification artifacts are unchanged.

The local checks used macOS arm64, Apple Clang, CMake/CPack, Python 3.12.14,
setuptools 84.2.1 and the available Black/isort/clang-format tools. Runtime Python ASTs,
public API inventory and executable C++/CUDA source are unchanged; generator templates,
license validation and packaging checks were updated. The 42 current hardware-file hashes
that changed were checked against exact SPDX-identifier-only edits before rebinding them.

| Local check | Result |
|---|---|
| License text/headers, negative fixtures and generator freshness | Passed |
| Retained third-party bytes, historical artifacts and dependency pins | Unchanged |
| Native CPU library build and installed external package consumer | Passed |
| Real source distribution, CMake installation and CPack TGZ legal files | Passed |
| Python distribution license metadata | Passed |
| Native Python wheel | Blocked: required local pybind11 package unavailable |

The new `tools/check_packaged_licenses.py` validates complete legal-file groups in
source/wheel/CPack archives and native installation directories. Its rejection tests also
run with Python assertions disabled. The wheel's metadata was prepared, but no wheel
build or runtime-wheel pass is claimed. CUDA, hardware and other platform packages were
not tested for this licensing-only change. Evidence is retained in the private release
workspace under `controllers-license-20220322-20261029`. No new security scan, human
attestation, upstream approval, push or publication is established by these checks.
