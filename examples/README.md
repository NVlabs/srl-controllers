# Examples

The examples are grouped by execution environment so that hardware boundaries and dependencies are
visible before anything is run.

| Folder | Backend | Commands hardware? | Interface |
|---|---|---:|---|
| [`franka/`](franka/) | Franka FER/Panda or FR3 through libfranka | Yes | Controllers; Eigen vendor adapter |
| [`rizon/`](rizon/) | Flexiv Rizon through locally bundled Flexiv RDK 1.9.1-1.9.4 | Yes | Controllers; Eigen vendor adapter |
| [`flexiv_dual/`](flexiv_dual/) | Flexiv dual-arm system through locally bundled Flexiv RDK 1.9.1 | Yes | Symmetric absolute/relative bimanual impedance; Eigen vendor adapter |
| [`ur/`](ur/) | Universal Robots through `ur_client_library` | Yes | Joint/Cartesian impedance and admittance/force-admittance controllers; Pinocchio/Eigen adapter |
| [`isaaclab/`](isaaclab/) | Replicated Franka, Rizon, UR5e, and UR10e Isaac Lab simulation on CUDA | No | Controllers |
| [`composition/`](composition/) | Synthetic Python and custom CUDA kernels | No | Public lifecycle/equations and final policy |
| [`synthetic/`](synthetic/) | CPU-only synthetic data | No | Controllers |

Hardware examples print their fixed behavior before constructing a vendor client. Some detailed
configuration logging differs by executable; the Franka admittance example intentionally omits
controller-matrix diagnostics. The
single-arm examples accept one positional robot identifier. The Flexiv dual-arm example also
requires two model link names, a calibrated base transform, and optionally a main-frame selection. Every physical
example accepts `--non-rt` after its required arguments. This skips host/process RT prerequisites
where the vendor API permits it, but does not disable robot-state validation, command freshness,
timing aborts, torque limits, mode checks, or other runtime safety gates. Every hardware controller
example requires two separate empty-line confirmations (press Enter): one before connection and
initial joint motion, and another after that motion completes, before controller startup. Text or
EOF at either prompt cancels execution. Franka moves to the predefined joint configuration printed
at the prompt. Franka, Rizon, and UR accept an optional positive `--duration-seconds=SECONDS`,
defaulting to 10 seconds. Rizon and Flexiv dual-arm move through the RDK non-real-time joint-position
interface to `[0, -15, 0, 75, 0, 0, 0]` degrees. UR moves to `[0, -90, -90, -90, 90, 0]` degrees (joint 5 is `135` degrees for the
EE-frame force example) before capturing its controller target. Its direct-torque examples also
validate the connected PolyScope version before motion.
Franka examples set the joint and Cartesian collision thresholds documented in
[`franka/`](franka/) before the initial motion. The other hardware examples do not change collision
thresholds. None of these examples write automatic log files.

`--non-rt` maps to libfranka's ignore-RT-errors setting, to priority 0 with no CPU affinity for both
Flexiv RDK integrations. The current `ur_client_library` interface has no disable switch, so the UR example accepts
the flag as a documented no-op and the upstream client may still acquire FIFO scheduling. Likewise,
libfranka may still acquire real-time scheduling when the host permits it. Each executable prints
`non_rt_requested=true` before its final connection confirmation when the flag is active.

Compilation, simulation, and synthetic timing evidence do not establish safe behavior on a physical
robot or qualify a host for hard real-time control.


## Complete controller catalog

Each target below has exactly one canonical Isaac launcher named `<target>_parallel.py` in
[`isaaclab/`](isaaclab/). The C++ full registry and Python registry enumerate all **22** pairs.
A registry entry identifies a hardware controller target, not a promise that every hardware model
has a simulator asset. [Runtime support and parity](../docs/example-parity.md) cover that distinction.

| Family | Targets | Hardware output / reference |
|---|---|---|
| Franka (9) | `franka_joint_impedance_controller`; `franka_cartesian_impedance_controller`; `franka_variable_cartesian_impedance_controller`; `franka_cartesian_admittance_controller`; `franka_cartesian_force_impedance_controller`; `franka_cartesian_force_impedance_controller_in_ee`; `franka_cartesian_force_impedance_controller_with_tank`; `franka_virtual_walls_controller`; `franka_artificial_potential_field_controller` | Torque; admittance is pose on Panda/FER and torque on FR3 |
| Rizon (6) | `rizon_joint_impedance_controller`; `rizon_cartesian_impedance_controller`; `rizon_variable_cartesian_impedance_controller`; `rizon_cartesian_admittance_controller`; `rizon_cartesian_force_impedance_controller`; `rizon_cartesian_force_impedance_controller_in_ee` | Torque except vendor Cartesian pose admittance |
| Dual Flexiv (2) | `flexiv_dual_bimanual_cartesian_impedance`; `flexiv_dual_bimanual_compliant_hold` | Symmetric midpoint/relative torque control |
| UR (5) | `joint_impedance_controller`; `cartesian_impedance_controller`; `cartesian_admittance_controller`; `cartesian_unified_force_admittance_controller`; `cartesian_unified_force_admittance_controller_in_ee` | Torque for impedance, base pose for admittance |

The synthetic and
composition folders demonstrate API use and have no physical counterpart. Historical Isaac aliases
select existing canonical targets and do not increase the catalog count.

**Non-functional Isaac companions:** the UR `joint_impedance_controller` and
`cartesian_impedance_controller` entries have non-functional simulation companions on both
UR5e and UR10e. They remain in the registry for diagnosis, with a visible CLI status notice.
This does not reclassify the physical examples or the three UR admittance companions. See
[the current status and evidence](../docs/public-api-validation.md#non-functional-ur-isaac-impedance-companions).

Hardware programs report stop reasons, timing or command/force summaries according to their family;
a zero exit code reports the program's configured run completion, not observed assembly/contact
success. Preserve stdout/stderr and the exact source/SDK/model configuration for a run. No controller
example performs motion planning or additional collision checking. See each family README for
startup prerequisites, cancellation, stop behavior and hardware acceptance limits.
