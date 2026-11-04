# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

This is a CMake project (C++17).

```bash
cmake -S . -B _build -DENABLE_TESTS=ON -DENABLE_PYBIND=ON
cmake --build _build --parallel

# Run all tests
ctest --test-dir _build --output-on-failure

# Run a single test binary (after build)
./_build/tests/nvidia/srl/controllers/<test_name>
```

## Formatting

Google C++ style, 2-space indent, 100 column limit (see `.clang-format`):

```bash
clang-format -i <file>
```

## Architecture

The public C++ API consists only of `controllers.hpp`, `composition.hpp`, `equations.hpp`, `example_specs.hpp`, `types.hpp`, and the
generated `version.hpp`. Every controller uses the same `requirements` / `prepare` / `reset` /
`step` lifecycle with caller-owned arrays. `ControllerConfig::device` selects CPU or CUDA without
changing controller types or parameter names.

`src/nvidia/srl/controllers/controllers.cpp` implements the public lifecycle. The
public allocation-free equation primitives in `include/nvidia/srl/controllers/equations.hpp` are shared by the
private CPU and CUDA kernels in `src/nvidia/srl/controllers/tensor/`; that private directory is not
installed or exported.

Python exposes the same classes through `_api.py` and `_blocks.py`, plus checked array equations
in `equations.py`. The inventory is `docs/public_api_inventory.json`. Regenerate new adapters with
`tools/generate_public_blocks.py` and `tools/generate_equation_api.py`; use `--check` to verify freshness. The pybind
entry point is `src/nvidia/srl/controllers/controllers_python.cpp`, and its `_controller_api`
backend module is private.
Do not add raw kernel bindings or per-controller compatibility wrappers.

Eigen is not a controller-library dependency. Optional hardware adapters and hardware-free
helper/oracle tests use Eigen to convert vendor/model data to public array views.


## Examples and documentation

Hardware source is ground truth for simulator alignment. Follow [docs/example-parity.md](docs/example-parity.md):
use identical public native components/configurations, preserve hardware files during a companion
change, keep CUDA batching configurable, and separate vendor/Isaac transport from native output.
There are 22 canonical hardware/Isaac pairs; keep registry, launchers and documentation in sync.
Do not launch physical hardware programs during ordinary software validation.

Use the ABI-matching interpreter to build/test Python; see [README.md](README.md) for editable
prerequisites and the [Isaac guide](examples/isaaclab/README.md) for simulator ABI/build flags.
SM89 candidate CUDA builds require an explicit tuning-profile override; passing tests does not
qualify deployment timing. Source/configuration and native CPU/CUDA tests precede simulator and
attended physical qualification. Update [validation](docs/public-api-validation.md) with the exact
base SHA, dirty-state context, build/runtime versions, evidence paths, failures and unrun checks.

`hardware_profiles_test.py` audits immutable source hashes and branch parameters and compares
native batches 1/16/256/1024. With `CONTROLLERS_ORACLE_BUILD` it also compares the actual unchanged
Franka helper's traces. `documentation_test.py` checks local links and registry/catalog coverage.
Doxygen and installed guides include docs/examples; verify generated API adapters with their
canonical `--check` commands. Never hand-edit generated adapters or the reviewed inventory merely
to suppress a contract failure.
