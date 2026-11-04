# Controllers

A C++17 controller library with one array-based API for CPU and CUDA execution. The
same controller types, lifecycle, parameter names, status flags, safety policy, and caller-owned
storage contract are used for a batch of one on a physical robot and for large simulation batches.

Project website: <https://srl.gitlab-master-pages.nvidia.com/cpp/controllers/>

## One API, optional CUDA

Every public C++ controller follows this lifecycle:

1. Create a `ControllerConfig` with device, dtype, batch size, joint dimension, frame, and safety
   policy.
2. Query `requirements(config)` when storage shapes need to be discovered.
3. Allocate parameters, state, scratch, output, status, and safety history with the array provider
   appropriate to the application.
4. Call `prepare(...)` once and `reset(...)` before execution.
5. Call `step(...)` with an `ExecutionContext` on every control tick.

CPU and CUDA are device choices, not different controller APIs. A normal CPU build has no CUDA
toolkit, driver, GPU, Torch, or CuPy requirement. A CUDA-enabled build adds device kernels behind the
same public classes. The library never migrates memory between devices.

```text
ControllerConfig + caller-owned arrays
                  |
       prepare / reset / step
                  |
        +---------+---------+
        |                   |
      CPU kernels         CUDA kernels
   no GPU dependency   caller-owned stream
```

The public C++ headers are:

- `nvidia/srl/controllers/controllers.hpp`
- `nvidia/srl/controllers/equations.hpp`
- `nvidia/srl/controllers/composition.hpp`
- `nvidia/srl/controllers/example_specs.hpp`
- `nvidia/srl/controllers/types.hpp`
- the generated `nvidia/srl/controllers/version.hpp`

The retired Eigen `init`/`compute` classes, `Prepared*` names, and free-function tensor API have
been removed. These headers are the complete public controller surface; there is no second
"prepared" tier. Eigen is used only by optional
hardware-example adapters and is not a library or runtime dependency.

## Custom controllers

All 36 controller and block types are public in C++ and Python. Users can combine them with
custom equations, state and control flow. The canonical equations are also installed for use in
user-written C++ and CUDA kernels, with checked batched Python adapters.

See the [complete public API and composition contract](docs/public-controller-api.md),
[reviewed machine-readable inventory](docs/public_api_inventory.json), and
[synthetic composition examples](examples/composition/README.md).

## Supported controllers

| Public type | Output | Additional caller-owned state/scratch |
|---|---|---|
| `TorqueFilter` | magnitude- and slew-limited joint torque | previous torque command |
| `JointImpedanceController` | joint torque, Nm | safety history |
| `NullspaceController` | joint torque, Nm | safety history |
| `OperationalSpaceMass` | 6x6 Cartesian mass | factorization scratch |
| `CartesianImpedanceController` | Cartesian wrench, N/Nm | inertial history |
| `CartesianAdmittancePoseController` | acceleration, twist, pose offset | integrator history |
| `OperationalSpaceInverseDynamicsController` (`OperationalSpaceController` remains available) | joint torque, Nm | safety history |
| `CartesianForceController` | Cartesian wrench, N/Nm | force-loop history |
| `CartesianUnifiedForceImpedanceController` | Cartesian wrench, N/Nm | force/inertial state and scratch |
| `VariableCartesianImpedanceController` | Cartesian wrench, N/Nm | stiffness/inertial state and scratch |
| `TaskWrenchToJointTorqueController` | filtered joint torque, Nm | unfiltered torque and filter state |
| `CartesianImpedanceTorqueController` | filtered joint torque, Nm | mass, impedance, nullspace, and filter storage |
| `VariableCartesianImpedanceTorqueController` | filtered joint torque, Nm | mass, variable impedance, nullspace, and filter storage |
| `CartesianUnifiedForceImpedanceTorqueController` | filtered joint torque, Nm | mass, force/impedance, nullspace, and filter storage |
| `BimanualCartesianImpedanceTorqueController` | filtered joint torque, Nm | bimanual kinematics, mass, task, nullspace, and filter storage |

Six-axis Cartesian configurations require an explicit `CartesianFrame::kRobotBase` or
`CartesianFrame::kEndEffector`. Joint positions and rotations are radians, translations are meters,
joint torques are Nm, and Cartesian wrenches are ordered force then torque in N/Nm. Pose-error
rotation uses a three-element rotation-vector representation. Vendor pose conversion remains the
consumer's responsibility.

`BimanualTaskConfig::coordination_mode` selects decoupled cooperative coordinates (the default
Cartesian/SO(3) midpoint plus shared-base TCP displacement and relative orientation) or a coupled
moving-frame rigid transform. The library builds the current 12D task and `12x14` Jacobian from the
two robot-provided `6x7` TCP Jacobians. Complete bimanual controllers accept desired task velocity,
form measured task velocity internally, and project nullspace feedback independently per arm.

The operational-space inverse-dynamics controller is a torque-level component. For
each batch row it computes
`tau = joint_bias_torque + J^T Lambda (desired_acceleration + Kp*pose_error + Kv*velocity_error - Jdot*qdot)`.
Inputs use the configured Cartesian frame; `Jdot*qdot` and Cartesian acceleration are ordered
linear then angular acceleration, and the bias/output arrays are joint torques in Nm. Nullspace
feedback remains a separate component and uses the dynamically consistent torque projector
`N^T = I - J^T Lambda J M^-1`.

## C++ example

This batch-one example uses ordinary CPU arrays. The complete argument-free version is
[`examples/synthetic/joint_impedance_step.cpp`](examples/synthetic/joint_impedance_step.cpp).

```cpp
#include <array>
#include "nvidia/srl/controllers/controllers.hpp"

namespace controllers = nvidia::srl::controllers;

constexpr std::size_t joints = 2;
std::array<double, 4> stiffness{100.0, 0.0, 0.0, 200.0};
std::array<double, 2> damping{1.0, 1.0};
std::array<double, 2> position_error{0.1, -0.1};
std::array<double, 2> velocity_error{};
std::array<double, 2> desired_acceleration{};
std::array<double, 4> mass{1.0, 0.0, 0.0, 1.0};
std::array<double, 2> torque{};
std::array<double, 2> last_valid_torque{};
std::array<std::uint8_t, 1> valid{};
std::array<std::uint64_t, 1> hold_count{}, parameter_status{}, status{};

constexpr auto device = controllers::Device::kCpu;
const controllers::JointImpedanceController::Inputs inputs{
    controllers::read_view(position_error, device, 1, joints),
    controllers::read_view(velocity_error, device, 1, joints),
    controllers::read_view(desired_acceleration, device, 1, joints),
    controllers::read_view(mass, device, 1, joints, joints)};
const controllers::JointImpedanceController::Parameters parameters{
    controllers::shared(controllers::read_view(stiffness, device, joints, joints)),
    controllers::shared(controllers::read_view(damping, device, joints))};
const controllers::JointImpedanceController::Storage storage{
    controllers::write_view(torque, device, 1, joints),
    controllers::write_status(parameter_status, device),
    controllers::write_status(status, device),
    controllers::make_safety_state(
        controllers::write_view(last_valid_torque, device, 1, joints),
        controllers::write_view(valid, device, 1),
        controllers::write_view(hold_count, device, 1))};

controllers::JointImpedanceController controller;
const controllers::ControllerConfig config{
    device, controllers::DType::kFloat64, 1, joints, {}};
controllers::Result lifecycle = controller.prepare(config, parameters, storage);
if (lifecycle.ok()) {
  lifecycle = controller.reset();
}
if (lifecycle.ok()) {
  lifecycle = controller.step(inputs);
}
// Use torque only when lifecycle.ok() and the row's status has no unsafe flag;
// otherwise report lifecycle.message and enter the robot's safe state.
```

Views are non-owning and row-major. `shared(...)` and `per_batch(...)` make parameter batching
explicit; it is never guessed. Container overloads verify that the requested extents match
`size()` and fail closed with an empty view. Raw-pointer overloads cannot verify allocation length,
so that remains the caller's responsibility. Controllers retain views but never own their memory,
so all backing allocations must outlive them.

Preparation checks retained writable storage for overlapping byte ranges, including partial overlap.
Command, status, and persistent history must be independent when their lifetimes overlap. The torque
filter permits an exact input/output view alias, but its previous-command and safety history must
remain separate. Complete torque pipelines also permit intermediate scratch reuse across disjoint
stages. Padded views are checked conservatively using their full byte span; these checks cannot prove
that a raw pointer actually owns the claimed allocation or that concurrent access is safe.

Complete torque-controller `requirements()` queries describe every nested storage role. Intermediate
parameter statuses are persistent state; intermediate commands, runtime statuses, and factors are
scratch. Qualified names such as `mass.joint_mass_factor` and `task.previous_mass_matrix` distinguish
stages, while final-stage names remain unchanged.

| Complete torque controller | State arrays | Scratch arrays | Output/status arrays |
|---|---:|---:|---:|
| Cartesian impedance | 17 | 10 | 3 |
| Variable Cartesian impedance | 20 | 16 | 3 |
| Unified force/impedance | 22 | 14 | 3 |
| Bimanual Cartesian impedance | 17 | 13 | 3 |

**ABI change:** `Requirements` now has capacity for 32 state and 32 scratch descriptors. Rebuild the
library, all C++ consumers, and the Python extension together. Existing binaries compiled against the
smaller structure must not load this library without a matching rebuild.

`prepare`, parameter updates, resets, and steps return a non-throwing `Result`. Parameter and runtime
status use per-row `uint64` flags. Unsafe rows receive the configured zero or bounded hold-last-valid
policy without invalidating other batch rows.

## Python example

Python uses the same configuration, storage roles, operations, and class names. Construction is its
idiomatic equivalent of C++ `prepare`; `reset`, `apply_parameters`, and `step` have the same meaning.
Arrays may come from NumPy on CPU or an object exposing `__cuda_array_interface__` on CUDA; the
package does not import Torch or CuPy.

```python
import numpy as np
from nvidia.srl.controllers import (
    ControllerConfig, DType, Device, JointImpedanceParameters, OutputStorage,
    JointImpedanceController, SafetyState, shared,
)

batch, joints = 2, 2
zeros = lambda shape: np.zeros(shape, dtype=np.float64)
status = lambda: np.zeros(batch, dtype=np.uint64)
storage = OutputStorage(
    command=zeros((batch, joints)),
    parameter_status=status(),
    status=status(),
    safety=SafetyState(
        last_valid_output=zeros((batch, joints)),
        valid_mask=np.zeros(batch, dtype=np.bool_),
        hold_count=status(),
    ),
)
controller = JointImpedanceController(
    ControllerConfig(Device.CPU, DType.FLOAT64, batch, joints),
    JointImpedanceParameters(
        stiffness_matrix=shared(np.eye(joints) * 100.0),
        damping_factors=shared(np.ones(joints)),
    ),
    storage,
)
controller.reset()
controller.step(
    joint_position_error=zeros((batch, joints)),
    joint_velocity_error=zeros((batch, joints)),
    desired_acceleration=zeros((batch, joints)),
    joint_mass_matrix=np.repeat(np.eye(joints)[None, :, :], batch, axis=0),
)
```

Check `storage.parameter_status` after preparation and `storage.status` after each CPU step.
CUDA status reads require an explicitly ordered host observation boundary. A structural exception
means the tick did not produce a new command; stop dependent execution. Numerical faults are
row-local flags and must be handled by the application. See the
[selection, lifecycle and migration guide](docs/public-controller-api.md).

Python strongly retains parameter and storage objects. Array providers still own their allocation
lifetime and stream-ordering semantics. Arrays must use native byte order and correctly aligned
storage; non-native-endian arrays are rejected rather than copied or byteswapped. Writable storage
and status arrays must be contiguous. Read-only arrays are accepted for read-only inputs and
parameters, not for controller-owned writes.

For CUDA, keep all arrays alive and correctly ordered until their queued uses finish, including
transient step inputs and arrays replaced by parameter updates. Returning from `step`, replacing
parameters, or destroying a controller does not synchronize pending GPU work.

## Execution and real-time contract

- CPU calls ignore `ExecutionContext.cuda_stream`.
- CUDA calls enqueue on the caller-owned stream passed to `prepare`, `apply_parameters`, `reset`,
  and `step`; they do not create a stream or synchronize it.
- A caller that produces CUDA arrays on another stream must establish the appropriate dependency
  before calling the controller. `__cuda_array_interface__` does not make cross-stream access
  automatically safe.
- No steady-state C++ controller `step` performs controller-owned heap allocation.
- Input views are still validated on each call. Preparation validates static configuration,
  parameters, and caller-owned storage; it is not a license to pass changing shapes or devices.
- CPU and CUDA share canonical controller equations, but floating-point results are not promised to
  be bitwise identical.
- The Python binding is intended for simulation and orchestration; Python object handling is not a
  hard-real-time execution boundary.

These properties are necessary for a real-time deployment, not sufficient qualification. Kernel,
CPU/IRQ isolation, scheduler permissions, page locking, vendor SDK behavior, workcell hazards, and
physical tuning must be validated separately.

## Build

The supported release baseline is Ubuntu 24.04 and Python 3.12. A C++-only CPU build is:

```bash
cmake -S . -B _build \
  -DENABLE_TESTS=ON \
  -DENABLE_PYBIND=OFF
cmake --build _build --parallel
ctest --test-dir _build --output-on-failure
```

Build an ordinary native Python wheel with `python -m pip wheel . --no-deps -w dist`.
It includes the extension and controller library with relative library lookup. For CUDA, set
`CONTROLLERS_CMAKE_ARGS="-DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<SM>-real"` and satisfy the
existing tuning-profile requirements. A wheel is specific to its Python ABI, platform and selected
CPU/CUDA backend. No editable checkout or build-tree symlink is required to use it.

For Python development, prepare the editable environment before configuring CMake:

```bash
python3.12 -m venv /tmp/controllers-python
source /tmp/controllers-python/bin/activate
python -m pip install -e ".[build,ci]"
cmake -S . -B _build -DENABLE_TESTS=ON -DENABLE_PYBIND=ON
cmake --build _build --parallel
PYTHONPATH="$PWD/python/src" pytest -q python/tests
```

`pip install -e .` does not compile the pybind extension. The CMake build symlinks the extension into
`python/src/nvidia/srl/_controllers/`.

A CUDA build requires exactly one native architecture. This SM89 recipe is a candidate functional
build because only the committed matching deployment profiles satisfy the default profile gate:

```bash
cmake -S . -B _build_cuda \
  -DENABLE_CUDA=ON \
  -DENABLE_PYBIND=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89-real \
  -DNVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF
cmake --build _build_cuda --parallel
```

There is no separate tensor build option. SM89 currently has no committed deployment profile.
Do not disable the profile gate for deployment. See [validation](docs/public-api-validation.md)
and the [Isaac interpreter/ABI instructions](examples/isaaclab/README.md).

### Installation

An explicit `-DCMAKE_INSTALL_PREFIX=/chosen/prefix` is honored across reconfiguration. Without an
explicit prefix, a new top-level development build uses
`<build>/devel/controllers-<full-version>`; `DEVEL=OFF` uses
`/opt/nvidia/srl/controllers-<full-version>`. `cmake --install <build> --prefix <stage>` and `DESTDIR`
remain supported. Development convenience links are skipped for redirected/staged installs and are
never written back into the original build tree by those installs.

Installed Linux library/tool runtime paths are relative to their installation, respecting configured
binary/library directories. CUDA runtime libraries remain external dependencies. Consumer executables
choose their own runtime-path policy; package tests install and relocate a real controller consumer
with loader overrides cleared to verify that the intended library is loaded.

## Examples

- [`examples/synthetic/`](examples/synthetic/) — argument-free CPU lifecycle and 1 kHz reference
  loop; no robot connection.
- [`examples/franka/`](examples/franka/) — nine batch-one CPU examples built against a locally
  checked-out and installed libfranka. Each accepts
  `ROBOT_HOST [--duration-seconds=SECONDS] [--non-rt]` and requires confirmation by pressing Enter
  on an empty line before client construction and a synchronized move to its printed initial
  joint configuration, then a second Enter confirmation before controller startup.
- [`examples/rizon/`](examples/rizon/) — six batch-one CPU examples for selectable, locally
  bundled Flexiv RDK versions 1.9.1-1.9.4 using controllers with an Eigen vendor adapter.
  Each accepts `ROBOT_SERIAL_NUMBER [--duration-seconds=SECONDS] [--non-rt]`, requires confirmation
  by pressing Enter on an empty line before client construction, moves to the fixed
  `[0, -15, 0, 75, 0, 0, 0]` degree joint target through the RDK non-real-time position interface,
  and requires a second confirmation before controller startup.
- [`examples/flexiv_dual/`](examples/flexiv_dual/) — one dual-arm Flexiv RDK example with an
  explicit base-frame transform, configurable duration, the same initial motion for both arms,
  two-stage Enter confirmation, and optional `--non-rt` host scheduling mode.
- [`examples/ur/`](examples/ur/) — batch-one joint/Cartesian impedance and Cartesian
  admittance/force-admittance CPU/UR RTDE examples. Each requires an empty-line confirmation
  before client construction, moves to its documented initial joint position, and then captures
  its controller target after a second empty Enter confirmation before controller startup.
- [`examples/composition/`](examples/composition/) — CPU Python custom controller and standalone
  installed-package CUDA composition examples; no robot connection.
- [`examples/isaaclab/`](examples/isaaclab/) — one registered, batched Torch CUDA companion for
  each of the 21 hardware example targets, using the same public controller classes and
  caller-owned stream context. `--num_envs` is configurable and defaults to 16.

The UR joint and Cartesian impedance **Isaac companions are non-functional** on UR5e and UR10e
and retained for diagnosis. Their earlier equilibrium-only passes do not establish disturbance
stability. See [the current status and evidence](docs/public-api-validation.md#non-functional-ur-isaac-impedance-companions).

The [complete target catalog](examples/README.md) and [parity contract](docs/example-parity.md)
identify the native controller boundary and simulator-specific adapters.

Every folder README documents frames, units, fixed behavior, safety boundaries, invocation, and what
its result does not prove. A build, synthetic timing run, or simulator rollout is not physical-robot
evidence.

For every physical-robot example, `--non-rt` skips host/process real-time prerequisites such as
PREEMPT_RT enforcement, CPU affinity, locked memory, and explicit FIFO scheduling where the vendor
API exposes those controls. It does not disable command freshness, deadline/timeliness monitoring,
mode checks, torque limits, controller safety policy, operator confirmation, or vendor safety
behavior. Some vendor clients may still attempt real-time scheduling internally; see the family
README for the exact mapping and caveats.

## Contributing

This project is currently not accepting contributions. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the repository contribution policy.

## Development

- C++ tests: `ctest --test-dir _build --output-on-failure`
- Python tests: `PYTHONPATH="$PWD/python/src" pytest -q python/tests`
- Formatting: `clang-format -i <files>` and the repository Python formatter
- Static checks: `make check`
- CUDA qualification tooling remains under `tools/`; it exercises internal backend kernels and
  emits deployment evidence, not another public controller API.

On Linux/GCC, `ENABLE_ALLOCATION_GUARD_TEST=ON` adds calibrated allocation tests against a test-only
CPU archive. The tests cover warmed-up normal and safe-output/recovery paths, not whole-process
allocation or hard-real-time qualification. Use a separate sanitizer build with
`-DENABLE_ALLOCATION_GUARD_TEST=OFF` and matching compiler/linker sanitizer flags. Signal-dispatch
tests use isolated non-RT processes; the bootstrap waiter shuts down only on `SIGINT`/`SIGTERM`,
not child-process or pipe notifications.

### CUDA verification levels

The CUDA CI checks are independently enabled and default to off:

| CI job | Enable variable | Purpose |
|---|---|---|
| `cuda_compile` | `CUDA_COMPILE_ENABLED=true` | Compile library, bindings, tests, and private benchmark/qualification tools without GPU execution |
| `cuda_correctness` | `CUDA_CORRECTNESS_ENABLED=true` | Execute required CUDA lifecycle, graph, numerical, safety, and package tests without autotuning |
| `cuda_tensor_qualification` | `CUDA_TENSOR_QUALIFICATION_ENABLED=true` | Run candidate/deployment performance qualification and collect evidence |

All requested jobs require an externally supplied, approved `CUDA_CI_IMAGE` and `CUDA_SM=110`.
Runtime jobs additionally need a matching `gpu-cuda` runner. Missing requested configuration fails
an explicit prerequisite job. Empty selections and skipped CUDA tests do not count as correctness
or qualification evidence. The existing CPU pipeline does not require CUDA infrastructure.

With the toolkit on `PATH` and the Python build environment active, the same entry point supports
`compile`, `correctness`, and `qualification` modes:

```bash
CUDA_SM=110 ./ci/cuda_tensor_qualification.sh compile
```

Local runs use fresh temporary build/artifact directories. `CUDA_OUTPUT_ROOT` may select an absolute,
nonexistent directory whose parent already exists. CI reports use job-specific directories below
`artifacts/cuda/`; reviewed source tables and historical evidence are not overwritten. Generated
candidate table differences require separate review. A successful compile is not GPU correctness or
performance qualification, and none of these checks constitutes physical-robot evidence.

## External downloads for optional examples

The published source includes build instructions for optional robot examples. Enabling these
examples downloads third-party material into the build directory; the retrieved repositories and
Flexiv release binaries are not included in this source distribution.

| Integration | Retrieved material | Local installation |
|---|---|---|
| Universal Robots | UR Client Library 2.14.1 and Pinocchio 3.9.0 at the revisions recorded in the build files | Builds and installs the libraries and matching URScript resources with the examples |
| Franka | The selected libfranka release (default 0.21.3), plus Pinocchio 3.9.0 for the FR3 configurations and a fmt fallback download if its host package is unavailable | Builds and installs the libraries with the examples |
| Flexiv Rizon / dual arm | Selected RDK 1.9.1–1.9.4 source and a checksum-verified prebuilt library; dual arm requires 1.9.1; static configurations also fetch the pinned dependency stack | Installs the RDK and applicable dependency libraries with the examples |

These downloaded components retain their own terms; the NVIDIA license below does not replace
them. See [third-party notices](THIRD_PARTY_NOTICES.md) for versions, sources and licenses.
Redistributing an installed example or a package built with these options includes additional
third-party material and requires the corresponding license texts, copyright notices and
applicable attribution notices. The source-only release description does not describe such a
binary package.

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

This notice follows NVIDIA's [external-download guidance](https://nvidia.atlassian.net/wiki/spaces/OSS/pages/2529034731/Example+OSRB+Approval+Requirements).

## License

NVIDIA-owned source is available under the NVIDIA License dated 22 March 2022,
identified as `LicenseRef-NVIDIA-2022-03-22` in SPDX metadata. This is a source-available license with
non-commercial research and evaluation restrictions for recipients other than NVIDIA
and its affiliates; it is not an open-source license. See [`LICENSE`](LICENSE) for the terms.

The Franka-derived helper remains Apache-2.0, and the UR model data remains BSD-3-Clause.
Their notices and license texts, and the licenses of external dependencies, are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The primary NVIDIA License does not replace those terms.
This project is currently not accepting contributions.

See [`CHANGELOG.md`](CHANGELOG.md) for version history.
