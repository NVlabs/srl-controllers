# Third-Party Notices

This document records the complete direct and bundled third-party inventory referenced by the
tracked source and configuration for distributable Controllers paths. Components fetched only when
an optional example is enabled are not stored in this repository, but they are listed because their
code, libraries, or assets can become part of an installed example artifact.

Developer-only linters, documentation generators, and test tools that are not redistributed are
outside this inventory. NVIDIA CUDA components are also outside this third-party inventory and
remain subject to their own NVIDIA terms.

## Source incorporated into this repository

### libfranka example utilities

- Project: [libfranka](https://github.com/frankaemika/libfranka)
- License: Apache-2.0
- Copyright: 2017 Franka Emika GmbH; the current upstream NOTICE also identifies Franka Robotics
  GmbH.
- Incorporated file: `examples/franka/franka_utils.hpp`
- Provenance: its initial-motion generator is adapted from libfranka's `examples_common` source.
  This repository supports the pinned `0.9.2` package for Panda/FER and exact `0.10.0` or newer
  packages for FR3; the default is `0.21.3`.

The original Franka copyright and NVIDIA modification copyright are retained in the incorporated
file. The file remains Apache-2.0, including NVIDIA's modifications. The applicable license text is
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt); the primary NVIDIA License in `LICENSE` does not replace it.

### Universal Robots model data

`examples/ur/models/ur5e.urdf` and `examples/ur/models/ur10e.urdf` retain the BSD-3-Clause
terms from Universal Robots ROS2 Description at revision
`710505166d0399b506838d55663f634880d235fe`. Their license text remains in
[`examples/ur/ur_description_LICENSE`](examples/ur/ur_description_LICENSE). These models
and the license text are included together in Python and native documentation packages.

## Historical qualification artifacts

The recorded `artifacts/cuda/joint_impedance_cuda_tuning_profiles.inc` and
`artifacts/cuda/cuda_component_tuning_profiles.inc` retain their original Apache-2.0
headers and contents as historical qualification evidence. Other recorded results and
source hashes are historical evidence, not validation of a later source candidate.

## Direct runtime and build dependencies

| Component | Repository constraint | Role | License |
|---|---|---|---|
| [Python](https://www.python.org/) | 3.12 or newer | Python API and build tooling | PSF-2.0 |
| [CMake](https://gitlab.kitware.com/cmake/cmake) | `>=3.28,<4` in the release environment | C++ build and packaging | BSD-3-Clause |
| [Eigen](https://gitlab.com/libeigen/eigen) | `>=3.4,<4` package; CMake API accepts 3.3+ for selected examples | Linear algebra headers | MPL-2.0 and component-specific notices distributed by Eigen |
| [NumPy](https://github.com/numpy/numpy) | `>=2.0,<3.0` | Python array interface | BSD-3-Clause |
| [Inflection](https://github.com/jpvanhal/inflection) | Listed in NVBug 6666725; no tracked use or version constraint | Release-request dependency record only | MIT |
| [pybind11](https://github.com/pybind/pybind11) | `>=3.1,<4.0`; fallback installer tag `v3.1.0` | Optional Python binding | BSD-3-Clause |
| [setuptools](https://github.com/pypa/setuptools) | `>=84,<85` | Python build backend | MIT |
| [setuptools-scm](https://github.com/pypa/setuptools-scm) | `>=10.2,<11` | Package version derivation | MIT |

## Optional robot and simulator integrations

| Component | Repository pin or constraint | Distribution relationship | License |
|---|---|---|---|
| [libfranka](https://github.com/frankaemika/libfranka) | exact `0.9.2` or exact `>=0.10.0`; default `0.21.3` | Fetched, built and linked from a private example prefix; example-derived source is identified above | Apache-2.0; upstream NOTICE copyright Franka Robotics GmbH |
| [Universal Robots Client Library](https://github.com/UniversalRobots/Universal_Robots_Client_Library) | Example build pin `2.14.1` (`a854de920d602300ce7c0fe3bd79655cb85bd678`) | Optional example build dependency fetched into a private build prefix; its matching URScript is installed with the example | Apache-2.0; bundled subcomponents retain their upstream licenses |
| [Flexiv RDK](https://github.com/flexivrobotics/flexiv_rdk) | Rizon `1.9.1`-`1.9.4` at repository-pinned commits; dual-arm requires `1.9.1` at `5707121ffadc5a1577c5e0eabf2afa20b9842acb` | Downloaded and copied into optional installed example prefixes | Apache-2.0 |
| [{fmt}](https://github.com/fmtlib/fmt) | Supplied by the host, or libfranka 0.21.3 fallback tag `11.0.2` | Optional libfranka build dependency; fetched only when its package lookup fails | MIT with the upstream optional embedded-object exception |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | `3.9.0` at `ed3bb75ce96cf26e84aebd8b73785407950a0f1f` | Built and copied into optional UR and FR3 example prefixes | BSD-2-Clause |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and `isaaclab_assets` | supplied by the operator; no revision is pinned here | External simulator runtime; not redistributed | BSD-3-Clause |
| [PyTorch](https://github.com/pytorch/pytorch) | supplied by the Isaac Lab environment | External tensor runtime; not redistributed | BSD-3-Clause |
| [Universal Robots ROS2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) | `710505166d0399b506838d55663f634880d235fe` | Source model data used to generate the bundled UR5e/UR10e URDFs; upstream geometry remains separately provided | [BSD-3-Clause](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/710505166d0399b506838d55663f634880d235fe/LICENSE) |
| [Flexiv Isaac Sim workspace](https://github.com/flexivrobotics/isaac_sim_ws) | `2d07d67887aff718ea93250e68e9e8eff4b0a62d` | External robot assets; not redistributed by this repository | Apache-2.0 |

## Optional bundled dependency stacks

The active static Flexiv integrations use RDK 1.9.1-1.9.3 with the following immutable dependency
pins. The former RDK 2.0 selections are retained below as historical attribution only; the dual
example now requires 1.9.1. When these dependencies are copied into an installed
example prefix, their source license files are copied alongside the artifact.

| Component | RDK 1.9.x pin | RDK 2.0 selection | License |
|---|---|---|---|
| Eigen | `3147391d946bb4b6c68edd901f2add6ac1f31f8c` | `3.4.0` | MPL-2.0 plus Eigen's component-specific notices |
| spdlog | `eb3220622e73a4889eee355ffa37972b3cac3df5` | `v1.9.2` | MIT |
| tinyxml2 | `1dee28e51f9175a31955b9791c74c430fe13dc82` | `9.0.0` | Zlib |
| yaml-cpp | `0579ae3d976091d7d664aa9d2527e0d0cff25763` | `yaml-cpp-0.7.0` | MIT |
| foonathan_memory_vendor | `e91681a25711c1811b2eaf2ba1e6996e0d9ecc84` | `v1.4.1` | Apache-2.0 |
| foonathan_memory | `0f0775770fd1c506fa9c5ad566bd6ba59659db66` | version selected by the vendor package | Zlib |
| Fast CDR | `7ebc45e03e1dd010681f5594135edbdcf3556c50` | `v2.3.5` | Apache-2.0 |
| Fast DDS | `33e1d9cd5bf9c142c95d545c6ce45727ff599076` | `v3.5.0` | Apache-2.0; bundled Asio is BSL-1.0 |
| SpaceVecAlg | `0e61723c22f7f7385be865d477f7b30b86ff3cab` | `v1.2.8` | BSD-2-Clause |
| RBDyn | `9ee3a7273a1d59e5aa076c87808993ee76be0f36` | `v1.9.3` | BSD-2-Clause |
| Boost | archive SHA-256 `6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e` | `1.83.0` | BSL-1.0 |

For optional UR/Franka installations, the build also retains upstream `LICENSE*`,
`COPYING*` and `NOTICE*` files from the retrieved repository trees. URCL's vendored
cpp-httplib and nlohmann/json MIT license files are included with its installation;
its queue and endian headers retain their own upstream notices. The NVIDIA primary
license does not replace those terms.

Pinocchio's host-side Eigen,
Boost, urdfdom, console_bridge, and TinyXML requirements remain externally provided and retain
their respective upstream license terms.

`inflection` is retained above because it is named in the release request, although it is not
referenced by tracked source, build metadata, or runtime configuration.
