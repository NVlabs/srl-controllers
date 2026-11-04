# Synthetic examples

These argument-free CPU examples use the same controller API as CUDA and hardware
consumers. All arrays, status words, state, scratch space, and safety history are caller-owned.
They do not connect to or command a robot.

```bash
cmake -S . -B _build -DENABLE_EXAMPLES=ON -DENABLE_PYBIND=OFF
cmake --build _build --parallel
./_build/examples/synthetic/joint_impedance_step
```

`joint_impedance_step` runs one batch-one joint-impedance lifecycle and prints the resulting torque
and status. It is the smallest complete example of `prepare`, `reset`, and `step`.

`rt_reference_loop` runs a fixed five-second, 1 kHz CPU torque-filter loop at source-configured
`SCHED_FIFO` priority 80 on CPU 1. It accepts no arguments and requires suitable Linux limits,
capabilities, and memory-lock permission:

```bash
MALLOC_ARENA_MAX=1 LD_BIND_NOW=1 ./_build/examples/synthetic/rt_reference_loop
```

The timing report is synthetic evidence only. It does not establish PREEMPT_RT, CPU/IRQ isolation,
deadline compliance, or hardware safety.

Run commands above from the repository root. A C++-only build can explicitly set
`-DENABLE_PYBIND=OFF -DENABLE_CUDA=OFF`; neither Python nor a GPU is required. These two examples
have no hardware companion or Enter gate. `joint_impedance_step` prints the command and status;
check both the lifecycle result and numerical status before treating a command as usable.
The RT loop reports synthetic loop timing and exits when its five-second loop ends; setup or
runtime rejection exits nonzero. Do not infer robot execution or deadline acceptance from it.

For controller selection, ownership, reset and status handling see the
[public API guide](../../docs/public-controller-api.md). For a custom equation and explicit final
policy, use the [composition examples](../composition/README.md).
