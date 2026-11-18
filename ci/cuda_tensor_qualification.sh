#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

set -eu

mode="${1:-qualification}"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [compile|correctness|qualification]" >&2
  exit 2
fi
case "$mode" in
  compile|correctness|qualification) ;;
  *) echo "Unknown CUDA check mode: $mode" >&2; exit 2 ;;
esac

if [ "${CUDA_SM:-}" != "110" ]; then
  echo "CUDA_SM=110 is required; runtime modes need a matching gpu-cuda runner." >&2
  exit 2
fi
case "${CUDA_REQUIRE_COMMITTED_TUNING_PROFILE:-1}" in
  0|1) ;;
  *) echo "CUDA_REQUIRE_COMMITTED_TUNING_PROFILE must be 0 or 1." >&2; exit 2 ;;
esac

nvcc --version
supported_gpu_code=$(nvcc --list-gpu-code)
if ! printf '%s\n' "$supported_gpu_code" | grep -Eq '^[[:space:]]*sm_110[[:space:]]*$'; then
  echo "The CUDA toolkit must support native sm_110 compilation." >&2
  exit 2
fi

project_dir=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
if [ -n "${CUDA_OUTPUT_ROOT:-}" ]; then
  case "$CUDA_OUTPUT_ROOT" in
    /*) ;;
    *) echo "CUDA_OUTPUT_ROOT must be an absolute, nonexistent directory." >&2; exit 2 ;;
  esac
  mkdir "$CUDA_OUTPUT_ROOT"
  output_root=$(CDPATH= cd "$CUDA_OUTPUT_ROOT" && pwd)
elif [ "${CI:-false}" = "true" ]; then
  case "${CI_JOB_ID:-}" in
    ''|*[!0-9]*) echo "CI_JOB_ID must be numeric for isolated CUDA outputs." >&2; exit 2 ;;
  esac
  output_root="$project_dir/_build_cuda_${mode}_${CI_JOB_ID}"
  mkdir "$output_root"
else
  output_root=$(mktemp -d "${TMPDIR:-/tmp}/controllers-cuda-${mode}.XXXXXX")
fi
build_root="$output_root/build"
artifact_dir="$output_root/artifacts/cuda"
if [ "${CI:-false}" = "true" ] && [ -z "${CUDA_OUTPUT_ROOT:-}" ]; then
  artifact_dir="$project_dir/artifacts/cuda/${CI_JOB_ID}-${mode}"
fi
mkdir -p "$build_root" "$(dirname "$artifact_dir")"
mkdir "$artifact_dir"
printf 'CUDA mode: %s\nBuild root: %s\nArtifacts: %s\n' "$mode" "$build_root" "$artifact_dir"

profile_source="$project_dir/src/nvidia/srl/controllers/tensor/joint_impedance_cuda_tuning_profiles.inc"
component_profile_source="$project_dir/src/nvidia/srl/controllers/tensor/cuda_component_tuning_profiles.inc"

configure_cuda_build() {
  build_dir=$1
  shift
  env LD_LIBRARY_PATH= cmake -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEVEL=OFF \
    -DENABLE_COVERAGE=OFF \
    -DENABLE_DOCS=OFF \
    -DENABLE_CUDA=ON \
    -DENABLE_PYBIND=ON \
    -DENABLE_EXAMPLES=OFF \
    -DENABLE_TESTS=ON \
    -DENABLE_TOOLS=ON \
    -DENABLE_TENSOR_JOINT_IMPEDANCE_CUDA_BENCHMARK=ON \
    -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST \
    "-DCMAKE_CUDA_COMPILER=$(command -v nvcc)" \
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_SM}-real" \
    -DNVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=ON \
    "-DNVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE=$profile_source" \
    "-DNVIDIASRLCONTROLLERS_CUDA_COMPONENT_TUNING_PROFILE_FILE=$component_profile_source" \
    "$@"
}

run_cuda_correctness() {
  test_build_dir=$1
  cuda_test_regex='[Cc][Uu][Dd][Aa]'
  selected_tests="$artifact_dir/selected-tests.json"
  env LD_LIBRARY_PATH= ctest --test-dir "$test_build_dir" --show-only=json-v1 \
    -R "$cuda_test_regex" > "$selected_tests"
  python3 - "$selected_tests" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    tests = json.load(source)["tests"]
names = [test["name"] for test in tests]
required = {
    "ControllersTest.CudaUsesTheSameLifecycleOnACallerStream",
    "ControllersTest.CudaGraphReplaysOnCallerStream",
    "ControllersTest.CudaOperationalSpaceControllerComputesJointTorqueEquation",
    "ControllersTest.CudaNullspaceTorqueUsesTransposeProjector",
    "ControllersTest.CudaOperationalSpaceMassAppliesRegularization",
    "BimanualTaskKinematicsTest.RunsFrameTransformAndLeverArmOnCuda",
    "BimanualNullspaceTest.RunsTheFullTwelveRowTaskOnCuda",
    "BimanualImpedanceTest.MatchesTheFullImpedanceLawOnCuda",
    "nvidia_srl_controllers_cuda_tensor_execution_ownership",
    "nvidia_srl_controllers_cuda_package_consumer",
}
missing = required.difference(names)
if not names or len(names) != len(set(names)) or missing or any(not test.get("command") for test in tests):
    raise SystemExit(f"Invalid CUDA test selection; missing required tests: {sorted(missing)}")
print(f"Selected {len(names)} CUDA tests, including all {len(required)} required tests.")
PY
  env LD_LIBRARY_PATH= ctest --test-dir "$test_build_dir" --output-on-failure \
    --no-tests=error -R "$cuda_test_regex" --output-junit "$artifact_dir/cuda-tests.xml"
  python3 - "$selected_tests" "$artifact_dir/cuda-tests.xml" "$project_dir/tools/verify_cuda_python_tests.py" <<'PY'
import json
from pathlib import Path
import re
import runpy
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    selected = {test["name"]: test for test in json.load(source)["tests"]}
read_report = runpy.run_path(sys.argv[3])["read_report"]
cases = read_report(Path(sys.argv[2])).findall(".//testcase")
if not selected or len(cases) != len(selected) or {case.get("name") for case in cases} != set(selected):
    raise SystemExit("CUDA test execution did not match the nonempty selection.")
for case in cases:
    name = case.attrib["name"]
    output = case.findtext("system-out", "")
    if (case.get("status") not in ("run", "passed")
            or any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))
            or re.search(r"\[\s*SKIPPED\s*\]", output)):
        raise SystemExit(f"CUDA test did not execute successfully: {name}")
    filters = [arg.split("=", 1)[1] for arg in selected[name].get("command", [])
               if arg.startswith("--gtest_filter=")]
    for test_filter in filters:
        for marker in ("RUN", "OK"):
            if not re.search(r"\[\s*" + marker + r"\s*\]\s+" + re.escape(test_filter) + r"(?=\s|$)", output):
                raise SystemExit(f"Missing GTest {marker} evidence for CUDA test: {name}")
print(f"Verified actual execution of {len(cases)} CUDA tests with no skips.")
PY
  env LD_LIBRARY_PATH= ctest --test-dir "$test_build_dir" --output-on-failure \
    --no-tests=error -E "$cuda_test_regex" --output-junit "$artifact_dir/regression-tests.xml"
  env LD_LIBRARY_PATH= CONTROLLERS_TEST_CUDA=1 PYTHONPATH="$project_dir/python/src" \
    python3 -m pytest -q -c "$project_dir/pyproject.toml" \
    --junitxml="$artifact_dir/python-cuda-tests.xml" \
    "$project_dir/python/tests/controllers/controller_api_test.py" \
    "$project_dir/python/tests/controllers/public_blocks_test.py" \
    "$project_dir/python/tests/controllers/equations_test.py" \
    "$project_dir/python/tests/controllers/composition_test.py"
  python3 "$project_dir/tools/verify_cuda_python_tests.py" "$artifact_dir/python-cuda-tests.xml"
}

if [ "$mode" = "compile" ] || [ "$mode" = "correctness" ]; then
  if [ "$mode" = "compile" ]; then
    configure_cuda_build "$build_root/$mode" -DNVIDIASRLCONTROLLERS_PYTHON_SOURCE_SYMLINK=OFF
  else
    configure_cuda_build "$build_root/$mode"
  fi
  env LD_LIBRARY_PATH= cmake --build "$build_root/$mode" --parallel 4
  if [ "$mode" = "correctness" ]; then
    run_cuda_correctness "$build_root/$mode"
  fi
  exit 0
fi

selected_profiles="$artifact_dir/joint_impedance_cuda_tuning_profiles.inc"
candidate_build_dir="$build_root/candidates"
deployment_build_dir="$build_root/deployment"

configure_cuda_build "$candidate_build_dir" \
  -DNVIDIASRLCONTROLLERS_CUDA_REQUIRE_TUNING_PROFILE=OFF
env LD_LIBRARY_PATH= cmake --build "$candidate_build_dir" --target nvidia_srl_tensor_joint_impedance_cuda_qualification --parallel

candidate_qualification="$candidate_build_dir/tools/tensor_joint_impedance_cuda_qualification"
if [ ! -x "$candidate_qualification" ]; then
  echo "The CUDA qualification tool was not built." >&2
  exit 1
fi
env LD_LIBRARY_PATH= "$candidate_qualification" --qualify --write-dispatch-table "$selected_profiles" \
  > "$artifact_dir/qualification-candidates.jsonl"

if ! cmp -s "$profile_source" "$selected_profiles"; then
  diff -u "$profile_source" "$selected_profiles" > "$artifact_dir/tuning-profile-update.patch" || true
  echo "CUDA qualification selected a different dispatch table; the reviewable update is in $artifact_dir/tuning-profile-update.patch." >&2
  profile_update_required=1
else
  profile_update_required=0
fi

configure_cuda_build "$deployment_build_dir" \
  "-DNVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE_FILE=$selected_profiles"
env LD_LIBRARY_PATH= cmake --build "$deployment_build_dir" --parallel
library_path=$(find "$deployment_build_dir" -name 'libnvidia_srl_controllers.so' -print -quit)
if [ -z "$library_path" ]; then
  echo "The CUDA Controllers shared library was not built." >&2
  exit 1
fi
run_cuda_correctness "$deployment_build_dir"

cuobjdump --dump-sass "$library_path" > "$artifact_dir/controllers.sass.txt"
if ! grep -q "code for sm_${CUDA_SM}" "$artifact_dir/controllers.sass.txt"; then
  echo "Expected native SASS for sm_${CUDA_SM} was not found in $library_path." >&2
  exit 1
fi
cuobjdump --list-elf "$library_path" > "$artifact_dir/controllers.elf.txt"

qualification="$deployment_build_dir/tools/tensor_joint_impedance_cuda_qualification"
if [ ! -x "$qualification" ]; then
  echo "The CUDA qualification tool was not built." >&2
  exit 1
fi
env LD_LIBRARY_PATH= "$qualification" --all-dimensions > "$artifact_dir/qualification.jsonl"
python3 "$project_dir/tools/compare_cuda_qualification.py" \
  --baseline "$artifact_dir/qualification.jsonl" \
  --measurement "$artifact_dir/qualification.jsonl" \
  --profile-table "$selected_profiles" \
  --warning-only > "$artifact_dir/dispatch-profile-verification.txt"

dense_benchmark="$deployment_build_dir/tools/tensor_joint_impedance_cuda_benchmark"
if [ ! -x "$dense_benchmark" ]; then
  echo "The dense CUDA workload benchmark was not built." >&2
  exit 1
fi
env LD_LIBRARY_PATH= "$dense_benchmark" > "$artifact_dir/dense-workload.csv"

git -C "$project_dir" archive --format=tar HEAD | gzip -n | sha256sum > "$artifact_dir/source.tar.gz.sha256"
toolkit_version=$(nvcc --version | tail -n 1)
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
profile_checksum=$(sha256sum "$selected_profiles" | awk '{print $1}')
component_profile_checksum=$(sha256sum "$component_profile_source" | awk '{print $1}')
cp "$component_profile_source" "$artifact_dir/cuda_component_tuning_profiles.inc"
printf '{"commit":"%s","cuda_sm":%s,"toolkit":"%s","driver":"%s","joint_impedance_tuning_profile_sha256":"%s","component_tuning_profile_sha256":"%s"}\n' \
  "$(git -C "$project_dir" rev-parse HEAD)" "$CUDA_SM" "$toolkit_version" "$driver_version" "$profile_checksum" "$component_profile_checksum" \
  > "$artifact_dir/provenance.json"

if [ -n "${CUDA_PERFORMANCE_BASELINE:-}" ] && [ -f "$CUDA_PERFORMANCE_BASELINE" ]; then
  python3 "$project_dir/tools/compare_cuda_qualification.py" \
    --baseline "$CUDA_PERFORMANCE_BASELINE" \
    --measurement "$artifact_dir/qualification.jsonl" \
    --warning-only | tee "$artifact_dir/performance-comparison.txt"
else
  echo "No per-SM performance baseline was supplied; metrics are report-only." \
    > "$artifact_dir/performance-comparison.txt"
fi

if [ "$profile_update_required" = "1" ] && [ "${CUDA_REQUIRE_COMMITTED_TUNING_PROFILE:-1}" = "1" ]; then
  echo "CUDA_REQUIRE_COMMITTED_TUNING_PROFILE=1 requires the selected table to be committed." >&2
  exit 1
fi
