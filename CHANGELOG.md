<!-- Template version: v1.11.0 -->
# Changelog

## Unreleased

- Apply the NVIDIA License dated 22 March 2022 to NVIDIA-owned source, with retained
  Apache-2.0 and BSD-3-Clause material, packaged license texts, and matching generated
  headers and metadata. Refresh the active source-integrity hashes for verified license-comment
  edits while preserving historical validation results. Release authorization is tracked separately.

- Mark both UR joint/Cartesian impedance Isaac companions non-functional on UR5e/UR10e,
  with CLI status notices and documented perturbation, torque-encoding and solver evidence.
  Preserve hardware examples, shared controller configurations and official asset parameters.
- Align Isaac companions with physical source configurations and native CPU/CUDA components,
  including exact gains, damping branches, force signs, filters, float64 state and output modes.
- Correct dual Flexiv to two independent six-axis controller stacks while preserving the
  fixed 20-entry C++ accessor.
- Require supplied Rizon robot torque limits, keep configurable CUDA batches, and latch faulted
  rows until rollout completion; retain pre-conversion pose/torque evidence in JSON.
- Add independent source/configuration audits, actual Franka-helper trace comparisons and broad
  native batch coverage. Document two Enter gates, prerequisites, stop behavior, simulator limits,
  package/ABI requirements and current qualification gaps throughout the guides.
- Keep physical example source/header/model/CMake files unchanged during companion alignment.
- Reject unsafe partial storage overlaps while preserving exact in-place torque filtering and
  scratch reuse across disjoint composite stages.
- Reject non-native-endian, malformed, and unaligned Python array metadata without implicit copies.
- Describe all composite controller storage, with checked descriptor capacities and persistent
  intermediate parameter statuses. **ABI change:** rebuild the library, every C++ consumer, and the
  Python extension together after the `Requirements` layout expansion.
- Honor explicit install prefixes and staged installations; verify relocated shared-library loading
  without loader-path overrides.
- Restrict RT bootstrap shutdown dispatch to `SIGINT` and `SIGTERM` and handle signal-wait errors.
- Add mixed-batch safety, numerical, descriptor-allocation, signal, and calibrated CPU allocation
  regression tests, plus caller-stream CUDA graph coverage.
- Separate CUDA compilation, GPU correctness, and performance qualification gates; repair private
  benchmark/qualification linkage without exporting backend symbols. CI activation still requires
  the approved CUDA image and matching runner.

## Version 0.0.0

- Correct shared EnergyTank work accounting and pre-command budget limits. Additional
  power is now independent nonnegative recharge; negative raw power is debited through
  the diagnostic deadband. Public tank signatures/storage are unchanged.
- Add the Franka base-frame force/impedance energy-tank hardware and Isaac Lab pair.
  The example registry grows from a 20-element to a 21-element C++ array; rebuild consumers.

- Remove the retired Eigen controller classes, raw tensor functions, schema catalog, compatibility
  headers, per-controller Python wrappers, and their regression tests.
- Make the array-based controller lifecycle the only public C++ and Python API for CPU and CUDA;
  remove the confusing `Prepared*` type and module names.
- Add a typed registry and canonical Isaac Lab companion launcher for every hardware example.
- Keep CPU/CUDA kernels private and remove Eigen from the library and runtime dependencies.
- Change bimanual task kinematics to construct the midpoint and combined Jacobian from the two TCP
  poses and robot-provided Jacobians. Decoupled cooperative coordinates are now the default, with
  coupled rigid-transform coordinates available explicitly.
- Change the complete bimanual torque controller input from a caller-computed velocity error to
  desired task velocity; measured task velocity is formed internally. Nullspace feedback is now
  projected independently for each arm and rejects nonzero cross-arm gain blocks.
