# Synthetic custom controllers

These examples use public installed interfaces and synthetic values. They do not connect to a robot.

`custom_controller.py` composes proportional, integral and derivative blocks with a custom `tanh`
control law and an energy tank. It allocates storage from public requirements, merges all upstream
statuses and applies one final output policy using separate history.

```sh
PYTHONPATH="$PWD/python/src" python examples/composition/custom_controller.py
```

`custom_state.cu` defines application-owned state and a nonlinear control law in a custom CUDA
kernel. It calls the installed canonical equations and then the public final-output policy on its
own stream. Build against an installed CUDA-enabled package (replace the prefix and SM):

```sh
cmake -S examples/composition -B /tmp/custom-controller \
  -DCMAKE_PREFIX_PATH=/path/to/controllers-prefix \
  -DCMAKE_CUDA_ARCHITECTURES=89-real
cmake --build /tmp/custom-controller
/tmp/custom-controller/custom_state
```

Array shape, units, frames, state and status ownership belong to the application. For the complete
composition contract and the library's dimension limits, see
[the public API guide](../../docs/public-controller-api.md).


Run these commands from the repository root. Python requires a built extension with the matching
interpreter ABI and NumPy; the first example uses CPU arrays and does not require Torch. The CUDA
example requires a CUDA-enabled installed package built for the actual GPU architecture. This
consumer build does not create a deployment tuning profile for that package.

Each application owns state, array ordering and the final status policy. A numerical safe-mode
result is not a successful custom command; a structural exception prevents that tick from being
used. Console output and the process exit code describe synthetic execution only. No hardware
connection, initial-motion/Enter workflow or physical task-completion claim is involved.
