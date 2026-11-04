/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
 *
 * @file tensor_joint_impedance_cuda_qualification.cpp
 * @brief Emits versioned raw-kernel and CUDA-graph timing records for CUDA tuning.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "nvidia/srl/controllers/tensor/joint_blocks.hpp"
#include "tensor/joint_blocks_cuda.hpp"

namespace {

namespace tensor = nvidia::srl::controllers::detail::tensor;

constexpr int kSchemaVersion = 1;
constexpr int kWarmupIterations = 20;
constexpr int kMeasuredIterations = 200;
constexpr std::size_t kBatches[] = {1U, 64U, 256U, 1024U};
constexpr std::size_t kDimensions[] = {3U,  5U,  6U,  7U,  8U,  12U, 14U, 16U, 17U, 18U, 19U, 20U,
                                       21U, 22U, 23U, 24U, 25U, 26U, 27U, 28U, 29U, 30U, 31U, 32U};
constexpr int kCandidateWarps[] = {1, 2, 4, 8};
constexpr double kBatchWeights[] = {0.10, 0.20, 0.30, 0.40};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    error_ = cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  bool ok() const { return error_ == cudaSuccess; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return count_; }

  bool copy_from_host(const std::vector<T>& values, const cudaStream_t stream) const {
    return values.size() == count_ &&
           cudaMemcpyAsync(
               data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
  }

 private:
  T* data_{nullptr};
  std::size_t count_{0};
  cudaError_t error_{cudaSuccess};
};

class CudaGraph {
 public:
  ~CudaGraph() {
    if (executable_ != nullptr) {
      cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
    }
  }

  cudaGraph_t* graph_address() { return &graph_; }
  bool instantiate() {
    return cudaGraphInstantiateWithFlags(&executable_, graph_, 0) == cudaSuccess;
  }
  bool launch(const cudaStream_t stream) const {
    return cudaGraphLaunch(executable_, stream) == cudaSuccess;
  }

 private:
  cudaGraph_t graph_{nullptr};
  cudaGraphExec_t executable_{nullptr};
};

struct TimingSummary {
  double mean_us{0.0};
  double p50_us{0.0};
  double p95_us{0.0};
  double p99_us{0.0};
  double max_us{0.0};
};

struct CandidateMeasurement {
  int warps_per_block{0};
  std::array<TimingSummary, 4> batch_timings{};
  std::array<bool, 4> valid{};
};

struct SelectedProfile {
  int sm{0};
  const char* dtype_token{nullptr};
  std::size_t dimension{0};
  std::size_t capacity{0};
  int warps_per_block{0};
};

bool cuda_ok(const char* const action, const cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }
  std::cerr << action << ": " << cudaGetErrorString(error) << '\n';
  return false;
}

std::array<std::int64_t, tensor::kMaxTensorRank> shape1(const std::size_t count) {
  return {static_cast<std::int64_t>(count), 0, 0, 0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> shape2(const std::size_t batches,
                                                        const std::size_t dimension) {
  return {static_cast<std::int64_t>(batches), static_cast<std::int64_t>(dimension), 0, 0};
}

std::array<std::int64_t, tensor::kMaxTensorRank> shape3(const std::size_t batches,
                                                        const std::size_t rows,
                                                        const std::size_t columns) {
  return {static_cast<std::int64_t>(batches),
          static_cast<std::int64_t>(rows),
          static_cast<std::int64_t>(columns),
          0};
}

template <typename Scalar>
tensor::TensorView vector_view(const DeviceBuffer<Scalar>& values) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  1,
                                  shape1(values.size()),
                                  {1, 0, 0, 0});
}

template <typename Scalar>
tensor::TensorView matrix_view(const DeviceBuffer<Scalar>& values,
                               const std::size_t batches,
                               const std::size_t dimension) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  2,
                                  shape2(batches, dimension),
                                  {static_cast<std::int64_t>(dimension), 1, 0, 0});
}

template <typename Scalar>
tensor::TensorView tensor3_view(const DeviceBuffer<Scalar>& values,
                                const std::size_t batches,
                                const std::size_t dimension) {
  return tensor::make_tensor_view(values.data(),
                                  tensor::Device::kCuda,
                                  tensor::Layout::kRowMajor,
                                  3,
                                  shape3(batches, dimension, dimension),
                                  {static_cast<std::int64_t>(dimension * dimension),
                                   static_cast<std::int64_t>(dimension),
                                   1,
                                   0});
}

template <typename Scalar>
tensor::MutableTensorView mutable_matrix_view(DeviceBuffer<Scalar>& values,
                                              const std::size_t batches,
                                              const std::size_t dimension) {
  return tensor::make_mutable_tensor_view(values.data(),
                                          tensor::Device::kCuda,
                                          tensor::Layout::kRowMajor,
                                          2,
                                          shape2(batches, dimension),
                                          {static_cast<std::int64_t>(dimension), 1, 0, 0});
}

template <typename Scalar>
std::vector<Scalar> coupled_batched_matrix(const std::size_t batches,
                                           const std::size_t dimension,
                                           const Scalar diagonal,
                                           const Scalar coupling) {
  std::vector<Scalar> values(batches * dimension * dimension, static_cast<Scalar>(0));
  for (std::size_t batch = 0; batch < batches; ++batch) {
    const Scalar batch_offset = static_cast<Scalar>(batch % 7U) * static_cast<Scalar>(0.005);
    for (std::size_t row = 0; row < dimension; ++row) {
      for (std::size_t column = 0; column < dimension; ++column) {
        const std::size_t distance = row > column ? row - column : column - row;
        values[(batch * dimension + row) * dimension + column] =
            row == column
                ? diagonal + static_cast<Scalar>(row) * static_cast<Scalar>(0.1) + batch_offset
                : coupling / static_cast<Scalar>(distance + 1U);
      }
    }
  }
  return values;
}

std::size_t candidate_capacity(const std::size_t dimension) {
  switch (dimension) {
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
    case 12:
    case 14:
    case 16:
      return dimension;
    default:
      break;
  }
  if (dimension >= 17U && dimension <= 20U) {
    return 20U;
  }
  if (dimension >= 21U && dimension <= 24U) {
    return 24U;
  }
  if (dimension >= 25U && dimension <= 28U) {
    return 28U;
  }
  if (dimension >= 29U && dimension <= 32U) {
    return 32U;
  }
  return 0U;
}

bool write_dispatch_table(const std::string& path, const std::vector<SelectedProfile>& profiles) {
  const std::string temporary_path = path + ".tmp";
  std::ofstream output(temporary_path);
  if (!output.is_open()) {
    std::cerr << "Unable to open CUDA dispatch table for writing: " << temporary_path << '\n';
    return false;
  }
  output << "// Generated by tensor_joint_impedance_cuda_qualification --qualify.\n"
         << "// Review this target-GPU result and commit it with its performance baseline.\n";
  for (const SelectedProfile& profile : profiles) {
    output << "NVIDIASRLCONTROLLERS_CUDA_TUNING_PROFILE(" << profile.sm << ", "
           << profile.dtype_token << ", " << profile.dimension << ", " << profile.capacity << ", "
           << profile.warps_per_block << ")\n";
  }
  output.close();
  if (!output) {
    std::cerr << "Unable to write CUDA dispatch table: " << temporary_path << '\n';
    std::remove(temporary_path.c_str());
    return false;
  }
  if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
    std::cerr << "Unable to publish CUDA dispatch table: " << path << '\n';
    std::remove(temporary_path.c_str());
    return false;
  }
  return true;
}

double percentile(std::vector<double> samples, const double percentile_value) {
  std::sort(samples.begin(), samples.end());
  const double index = percentile_value * static_cast<double>(samples.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  const double interpolation = index - static_cast<double>(lower);
  return samples[lower] * (1.0 - interpolation) + samples[upper] * interpolation;
}

TimingSummary summarize(const std::vector<double>& samples) {
  const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
  return TimingSummary{total / static_cast<double>(samples.size()),
                       percentile(samples, 0.50),
                       percentile(samples, 0.95),
                       percentile(samples, 0.99),
                       *std::max_element(samples.begin(), samples.end())};
}

void print_timing_record(const char* const record,
                         const int sm,
                         const char* const scalar,
                         const std::size_t dimension,
                         const std::size_t batch,
                         const char* const dispatch,
                         const int warps_per_block,
                         const TimingSummary& timing) {
  std::cout << std::fixed << std::setprecision(4) << "{\"schema_version\":" << kSchemaVersion
            << ",\"record\":\"" << record << "\",\"sm\":" << sm << ",\"scalar\":\"" << scalar
            << "\",\"joint_dim\":" << dimension << ",\"batch\":" << batch << ",\"dispatch\":\""
            << dispatch << "\",\"warps_per_block\":" << warps_per_block
            << ",\"mean_us\":" << timing.mean_us << ",\"p50_us\":" << timing.p50_us
            << ",\"p95_us\":" << timing.p95_us << ",\"p99_us\":" << timing.p99_us
            << ",\"max_us\":" << timing.max_us << "}\n";
}

template <typename Scalar>
bool measure_case(const std::size_t batches,
                  const std::size_t dimension,
                  const char* const dispatch,
                  const tensor::internal::JointImpedanceCudaLaunchPlan* const requested_plan,
                  const bool measure_graph,
                  TimingSummary* const timing) {
  DeviceBuffer<Scalar> position_error(batches * dimension);
  DeviceBuffer<Scalar> velocity_error(batches * dimension);
  DeviceBuffer<Scalar> desired_acceleration(batches * dimension);
  DeviceBuffer<Scalar> mass(batches * dimension * dimension);
  DeviceBuffer<Scalar> stiffness(batches * dimension * dimension);
  DeviceBuffer<Scalar> damping(dimension);
  DeviceBuffer<Scalar> torque(batches * dimension);
  DeviceBuffer<std::uint64_t> parameter_status(batches);
  DeviceBuffer<std::uint64_t> status(batches);
  if (!position_error.ok() || !velocity_error.ok() || !desired_acceleration.ok() || !mass.ok() ||
      !stiffness.ok() || !damping.ok() || !torque.ok() || !parameter_status.ok() || !status.ok()) {
    std::cerr << "cudaMalloc failed for batch=" << batches << " dimension=" << dimension << '\n';
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok("cudaStreamCreateWithFlags",
               cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
    return false;
  }
  const std::vector<Scalar> position(batches * dimension, static_cast<Scalar>(0.1));
  const std::vector<Scalar> zeros(batches * dimension, static_cast<Scalar>(0));
  const std::vector<Scalar> mass_values =
      coupled_batched_matrix(batches, dimension, static_cast<Scalar>(2), static_cast<Scalar>(0.01));
  const std::vector<Scalar> stiffness_values =
      coupled_batched_matrix(batches, dimension, static_cast<Scalar>(8), static_cast<Scalar>(0.02));
  const std::vector<Scalar> damping_values(dimension, static_cast<Scalar>(1));
  if (!position_error.copy_from_host(position, stream) ||
      !velocity_error.copy_from_host(zeros, stream) ||
      !desired_acceleration.copy_from_host(zeros, stream) ||
      !mass.copy_from_host(mass_values, stream) ||
      !stiffness.copy_from_host(stiffness_values, stream) ||
      !damping.copy_from_host(damping_values, stream) ||
      !cuda_ok(
          "cudaMemsetAsync(parameter_status)",
          cudaMemsetAsync(parameter_status.data(), 0, batches * sizeof(std::uint64_t), stream)) ||
      !cuda_ok("cudaMemsetAsync(status)",
               cudaMemsetAsync(status.data(), 0, batches * sizeof(std::uint64_t), stream))) {
    cudaStreamDestroy(stream);
    return false;
  }

  tensor::ParameterView entries[] = {
      {"stiffness_matrix", tensor3_view(stiffness, batches, dimension), false},
      {"damping_factors", vector_view(damping), true},
  };
  const tensor::ParameterSetView parameters{entries, 2};
  const tensor::MutableStatusView mutable_parameter_status{
      parameter_status.data(), batches, tensor::Device::kCuda};
  const tensor::StatusView parameter_status_view{
      parameter_status.data(), batches, tensor::Device::kCuda};
  const tensor::MutableStatusView mutable_status{status.data(), batches, tensor::Device::kCuda};

  bool success = true;
  {
    tensor::ScopedTensorExecutionContext context{tensor::TensorExecutionContext{stream}};
    const tensor::Result apply = tensor::apply_joint_impedance_controller_parameters(
        parameters, batches, dimension, mutable_parameter_status);
    if (!apply.ok()) {
      std::cerr << "parameter apply failed: " << apply.message << '\n';
      success = false;
    }
    const auto step = [&]() {
      if (std::string(dispatch) == "generic") {
        return tensor::internal::step_joint_impedance_controller_cuda_with_route(
            matrix_view(position_error, batches, dimension),
            matrix_view(velocity_error, batches, dimension),
            matrix_view(desired_acceleration, batches, dimension),
            tensor3_view(mass, batches, dimension),
            parameters,
            parameter_status_view,
            mutable_matrix_view(torque, batches, dimension),
            mutable_status,
            tensor::internal::JointImpedanceCudaRoute::kGeneric);
      }
      if (requested_plan != nullptr) {
        return tensor::internal::step_joint_impedance_controller_cuda_with_launch_plan(
            matrix_view(position_error, batches, dimension),
            matrix_view(velocity_error, batches, dimension),
            matrix_view(desired_acceleration, batches, dimension),
            tensor3_view(mass, batches, dimension),
            parameters,
            parameter_status_view,
            mutable_matrix_view(torque, batches, dimension),
            mutable_status,
            *requested_plan);
      }
      return tensor::step_joint_impedance_controller(
          matrix_view(position_error, batches, dimension),
          matrix_view(velocity_error, batches, dimension),
          matrix_view(desired_acceleration, batches, dimension),
          tensor3_view(mass, batches, dimension),
          parameters,
          parameter_status_view,
          mutable_matrix_view(torque, batches, dimension),
          mutable_status);
    };
    if (success) {
      for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        const tensor::Result result = step();
        if (!result.ok()) {
          std::cerr << "warmup failed: " << result.message << '\n';
          success = false;
          break;
        }
      }
    }
    if (success && !cuda_ok("warmup synchronization", cudaStreamSynchronize(stream))) {
      success = false;
    }
    CudaGraph graph;
    if (success && measure_graph) {
      if (!cuda_ok("cudaStreamBeginCapture",
                   cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal))) {
        success = false;
      } else {
        const tensor::Result result = step();
        const bool end_capture_ok =
            cuda_ok("cudaStreamEndCapture", cudaStreamEndCapture(stream, graph.graph_address()));
        if (!result.ok() || !end_capture_ok || !graph.instantiate()) {
          std::cerr << "CUDA graph construction failed.\n";
          success = false;
        }
      }
    }
    cudaEvent_t begin{};
    cudaEvent_t end{};
    if (success && (!cuda_ok("cudaEventCreate(begin)", cudaEventCreate(&begin)) ||
                    !cuda_ok("cudaEventCreate(end)", cudaEventCreate(&end)))) {
      success = false;
    }
    std::vector<double> samples;
    samples.reserve(kMeasuredIterations);
    for (int iteration = 0; success && iteration < kMeasuredIterations; ++iteration) {
      success = cuda_ok("cudaEventRecord(begin)", cudaEventRecord(begin, stream));
      const tensor::Result result =
          measure_graph
              ? (graph.launch(stream)
                     ? tensor::Result{}
                     : tensor::Result{tensor::ResultCode::kUnsupported, "cudaGraphLaunch failed"})
              : step();
      if (!result.ok()) {
        std::cerr << "measurement failed: " << result.message << '\n';
        success = false;
        break;
      }
      success = cuda_ok("cudaEventRecord(end)", cudaEventRecord(end, stream)) &&
                cuda_ok("cudaEventSynchronize", cudaEventSynchronize(end));
      float elapsed_ms = 0.0F;
      if (success &&
          !cuda_ok("cudaEventElapsedTime", cudaEventElapsedTime(&elapsed_ms, begin, end))) {
        success = false;
      }
      if (success) {
        samples.push_back(static_cast<double>(elapsed_ms) * 1000.0);
      }
    }
    if (begin != nullptr) {
      cudaEventDestroy(begin);
    }
    if (end != nullptr) {
      cudaEventDestroy(end);
    }
    if (success) {
      *timing = summarize(samples);
    }
  }
  cudaStreamDestroy(stream);
  return success;
}

bool select_profile(const std::vector<CandidateMeasurement>& measurements,
                    CandidateMeasurement* const selected) {
  double best_score = std::numeric_limits<double>::infinity();
  bool found = false;
  for (const CandidateMeasurement& candidate : measurements) {
    double log_sum = 0.0;
    bool complete = true;
    for (std::size_t index = 0; index < std::size(kBatches); ++index) {
      if (!candidate.valid[index] || candidate.batch_timings[index].p50_us <= 0.0) {
        complete = false;
        break;
      }
      log_sum += kBatchWeights[index] * std::log(candidate.batch_timings[index].p50_us);
    }
    if (!complete) {
      continue;
    }
    const double score = std::exp(log_sum);
    const bool within_two_percent = found && score <= best_score * 1.02;
    const bool better_tail = within_two_percent && candidate.batch_timings.back().p99_us <
                                                       selected->batch_timings.back().p99_us;
    if (!found || score < best_score * 0.98 ||
        (within_two_percent &&
         (better_tail ||
          (candidate.batch_timings.back().p99_us == selected->batch_timings.back().p99_us &&
           candidate.warps_per_block < selected->warps_per_block)))) {
      *selected = candidate;
      best_score = score;
      found = true;
    }
  }
  return found;
}

template <typename Scalar>
bool run_dimension(const int sm,
                   const std::size_t dimension,
                   const bool qualify,
                   std::vector<SelectedProfile>* const selected_profiles) {
  const char* const scalar = sizeof(Scalar) == sizeof(float) ? "float32" : "float64";
  const char* const dtype_token = sizeof(Scalar) == sizeof(float) ? "kFloat32" : "kFloat64";
  const tensor::DType dtype =
      sizeof(Scalar) == sizeof(float) ? tensor::DType::kFloat32 : tensor::DType::kFloat64;
  const tensor::internal::JointImpedanceCudaLaunchPlan automatic_plan =
      tensor::internal::joint_impedance_cuda_launch_plan(dtype, dimension);
  const std::size_t capacity =
      automatic_plan.route == tensor::internal::JointImpedanceCudaRoute::kOptimized
          ? automatic_plan.capacity
          : candidate_capacity(dimension);
  if (capacity == 0U) {
    std::cerr << "No legal specialized capacity for dimension=" << dimension << '\n';
    return false;
  }

  for (std::size_t batch_index = 0; batch_index < std::size(kBatches); ++batch_index) {
    TimingSummary generic{};
    TimingSummary automatic{};
    TimingSummary graph{};
    if (!measure_case<Scalar>(
            kBatches[batch_index], dimension, "generic", nullptr, false, &generic) ||
        !measure_case<Scalar>(
            kBatches[batch_index], dimension, "auto", nullptr, false, &automatic) ||
        !measure_case<Scalar>(kBatches[batch_index], dimension, "auto", nullptr, true, &graph)) {
      return false;
    }
    print_timing_record(
        "raw_controller", sm, scalar, dimension, kBatches[batch_index], "generic", 0, generic);
    print_timing_record(
        "raw_controller",
        sm,
        scalar,
        dimension,
        kBatches[batch_index],
        "auto",
        automatic_plan.route == tensor::internal::JointImpedanceCudaRoute::kOptimized
            ? automatic_plan.warps_per_block
            : 0,
        automatic);
    print_timing_record(
        "captured_graph",
        sm,
        scalar,
        dimension,
        kBatches[batch_index],
        "auto",
        automatic_plan.route == tensor::internal::JointImpedanceCudaRoute::kOptimized
            ? automatic_plan.warps_per_block
            : 0,
        graph);
  }
  if (!qualify) {
    return true;
  }

  std::vector<CandidateMeasurement> candidates;
  for (const int candidate_warps : kCandidateWarps) {
    CandidateMeasurement candidate{};
    candidate.warps_per_block = candidate_warps;
    const tensor::internal::JointImpedanceCudaLaunchPlan candidate_plan{
        tensor::internal::JointImpedanceCudaRoute::kOptimized, capacity, candidate_warps};
    for (std::size_t batch_index = 0; batch_index < std::size(kBatches); ++batch_index) {
      TimingSummary timing{};
      candidate.valid[batch_index] = measure_case<Scalar>(
          kBatches[batch_index], dimension, "candidate", &candidate_plan, false, &timing);
      if (candidate.valid[batch_index]) {
        candidate.batch_timings[batch_index] = timing;
        print_timing_record("candidate",
                            sm,
                            scalar,
                            dimension,
                            kBatches[batch_index],
                            "candidate",
                            candidate_warps,
                            timing);
      }
    }
    candidates.push_back(candidate);
  }
  CandidateMeasurement selected{};
  if (!select_profile(candidates, &selected)) {
    std::cerr << "No valid geometry candidate for dimension=" << dimension << '\n';
    return false;
  }
  if (selected_profiles != nullptr) {
    selected_profiles->push_back(
        SelectedProfile{sm, dtype_token, dimension, capacity, selected.warps_per_block});
  }
  std::cout << "{\"schema_version\":" << kSchemaVersion
            << ",\"record\":\"selected_profile\",\"sm\":" << sm << ",\"scalar\":\"" << scalar
            << "\",\"joint_dim\":" << dimension
            << ",\"warps_per_block\":" << selected.warps_per_block
            << ",\"selection\":\"weighted_geomean_p50_10_20_30_40\"}\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool qualify = false;
  bool all_dimensions = false;
  std::size_t requested_dimension = 0U;
  std::string dispatch_table_path;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string value{argv[argument]};
    if (value == "--qualify") {
      qualify = true;
      all_dimensions = true;
    } else if (value == "--all-dimensions") {
      all_dimensions = true;
    } else if (value == "--dimension" && argument + 1 < argc) {
      char* end = nullptr;
      const char* const text = argv[++argument];
      const std::uint64_t parsed = std::strtoull(text, &end, 10);
      if (text[0] == '\0' || end == nullptr || *end != '\0' || parsed == 0U ||
          parsed > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "--dimension requires a positive integer\n";
        return EXIT_FAILURE;
      }
      requested_dimension = static_cast<std::size_t>(parsed);
    } else if (value == "--write-dispatch-table" && argument + 1 < argc) {
      if (!dispatch_table_path.empty()) {
        std::cerr << "--write-dispatch-table may be specified only once\n";
        return EXIT_FAILURE;
      }
      dispatch_table_path = argv[++argument];
    } else {
      std::cerr << "Usage: tensor_joint_impedance_cuda_qualification [--dimension N | "
                   "--all-dimensions] [--qualify] [--write-dispatch-table PATH]\n";
      return EXIT_FAILURE;
    }
  }
  if (requested_dimension != 0U && (all_dimensions || qualify)) {
    std::cerr << "--dimension cannot be combined with --all-dimensions or --qualify\n";
    return EXIT_FAILURE;
  }
  if (requested_dimension != 0U &&
      std::find(std::begin(kDimensions), std::end(kDimensions), requested_dimension) ==
          std::end(kDimensions)) {
    std::cerr << "--dimension is not a supported qualification dimension\n";
    return EXIT_FAILURE;
  }
  if (!dispatch_table_path.empty() && !qualify) {
    std::cerr << "--write-dispatch-table requires --qualify\n";
    return EXIT_FAILURE;
  }
  if (!dispatch_table_path.empty()) {
    all_dimensions = true;
  }
  int device_count = 0;
  cudaDeviceProp device{};
  if (!cuda_ok("cudaGetDeviceCount", cudaGetDeviceCount(&device_count)) || device_count == 0 ||
      !cuda_ok("cudaGetDeviceProperties", cudaGetDeviceProperties(&device, 0))) {
    return EXIT_FAILURE;
  }
  const int sm = device.major * 10 + device.minor;
  if (sm != tensor::internal::joint_impedance_cuda_build_sm()) {
    std::cerr << "The binary was compiled for SM "
              << tensor::internal::joint_impedance_cuda_build_sm() << " but runs on SM " << sm
              << ".\n";
    return EXIT_FAILURE;
  }
  const std::size_t* const dimensions = requested_dimension != 0U
                                            ? &requested_dimension
                                            : (all_dimensions ? kDimensions : kDimensions + 3);
  const std::size_t dimension_count = all_dimensions ? std::size(kDimensions) : 1U;
  std::vector<SelectedProfile> selected_profiles;
  for (std::size_t index = 0; index < dimension_count; ++index) {
    if (!run_dimension<float>(sm, dimensions[index], qualify, &selected_profiles) ||
        !run_dimension<double>(sm, dimensions[index], qualify, &selected_profiles)) {
      return EXIT_FAILURE;
    }
  }
  if (!dispatch_table_path.empty() &&
      !write_dispatch_table(dispatch_table_path, selected_profiles)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
