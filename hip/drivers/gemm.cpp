#include "gemm.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace gemm_driver {
namespace {

// Measured: clocks take about a second of sustained load to reach steady state
constexpr double warmup_milliseconds = 1000;
constexpr double measurement_milliseconds = 500;
constexpr double candidate_milliseconds = 50;
constexpr int candidate_count = 16;
constexpr size_t workspace_limit = 32 << 20;

[[noreturn]] void fail(
	const char* library,
	const char* operation,
	const char* message
) {
	std::fprintf(
		stderr,
		"%s failed during %s: %s\n",
		library,
		operation,
		message
	);
	std::exit(1);
}

// Multiples of 1/32 up to 6/32 keep every product and partial sum exact in FP32
float pattern(
	size_t index
) {
	return float(int(index % 13) - 6) / 32;
}

std::vector<input_type> filled(
	size_t count,
	size_t offset
) {
	std::vector<input_type> values(count);
	for(size_t index = 0; index < count; ++index) {
		values[index] = input_type(pattern(index + offset));
	}
	return values;
}

template <typename value>
value* allocate(
	size_t count,
	const char* name
) {
	value* pointer = nullptr;
	check(hipMalloc(&pointer, count * sizeof(value)), name);
	return pointer;
}

double time_launches(
	const std::function<void()>& launch,
	hipStream_t stream,
	int count
) {
	hipEvent_t start = nullptr;
	hipEvent_t stop = nullptr;
	check(hipEventCreate(&start), "create start event");
	check(hipEventCreate(&stop), "create stop event");
	check(hipEventRecord(start, stream), "record start event");
	for(int index = 0; index < count; ++index) {
		launch();
	}
	check(hipEventRecord(stop, stream), "record stop event");
	check(hipEventSynchronize(stop), "wait for stop event");
	float milliseconds = 0;
	check(hipEventElapsedTime(&milliseconds, start, stop), "read timer");
	check(hipEventDestroy(start), "destroy start event");
	check(hipEventDestroy(stop), "destroy stop event");
	return milliseconds / count;
}

// Mean duration over as many launches as fit in roughly the given time
double measure(
	const std::function<void()>& launch,
	hipStream_t stream,
	double milliseconds
) {
	double estimate = time_launches(launch, stream, 1);
	int count = std::max(1, int(milliseconds / estimate));
	return time_launches(launch, stream, count);
}

// Wall-clock time, since a first launch may include loading its code object
void warm_up(
	const std::function<void()>& launch,
	hipStream_t stream
) {
	auto end = std::chrono::steady_clock::now()
			   + std::chrono::duration<double, std::milli>(warmup_milliseconds);
	while(std::chrono::steady_clock::now() < end) {
		launch();
		check(hipStreamSynchronize(stream), "warm up");
	}
}

// Tuned solution indices, one "version m n k index" line per shape
std::filesystem::path solution_cache() {
	auto executable = std::filesystem::read_symlink("/proc/self/exe");
	return executable.parent_path() / "hipblaslt-solutions.txt";
}

int cached_solution(
	int version,
	const shape& s
) {
	std::ifstream file(solution_cache());
	int line_version, m, n, k, index;
	int found = -1;
	while(file >> line_version >> m >> n >> k >> index) {
		if(line_version == version && m == s.m && n == s.n && k == s.k) {
			found = index;
		}
	}
	return found;
}

void cache_solution(
	int version,
	const shape& s,
	int index
) {
	std::ofstream file(solution_cache(), std::ios::app);
	file << version << ' ' << s.m << ' ' << s.n << ' ' << s.k << ' ' << index
		 << '\n';
}

}

void check(
	hipError_t result,
	const char* operation
) {
	if(result != hipSuccess) {
		fail("HIP", operation, hipGetErrorString(result));
	}
}

void check(
	hipblasStatus_t result,
	const char* operation
) {
	if(result != HIPBLAS_STATUS_SUCCESS) {
		std::string code = "status " + std::to_string(int(result));
		fail("hipBLASLt", operation, code.c_str());
	}
}

problem::problem(
	const shape& s
)
	: dimensions(s) {
	size_t a_count = size_t(s.m) * s.k;
	size_t b_count = size_t(s.k) * s.n;
	size_t c_count = size_t(s.m) * s.n;
	a = allocate<input_type>(a_count, "allocate A");
	b = allocate<input_type>(b_count, "allocate B");
	c_custom = allocate<output_type>(c_count, "allocate custom C");
	c_reference = allocate<output_type>(c_count, "allocate reference C");
	auto a_values = filled(a_count, 0);
	auto b_values = filled(b_count, 17);
	check(
		hipMemcpy(
			a,
			a_values.data(),
			a_count * sizeof(input_type),
			hipMemcpyHostToDevice
		),
		"upload A"
	);
	check(
		hipMemcpy(
			b,
			b_values.data(),
			b_count * sizeof(input_type),
			hipMemcpyHostToDevice
		),
		"upload B"
	);
}

problem::~problem() {
	check(hipFree(a), "free A");
	check(hipFree(b), "free B");
	check(hipFree(c_custom), "free custom C");
	check(hipFree(c_reference), "free reference C");
}

std::vector<output_type> problem::download(
	const output_type* c
) const {
	std::vector<output_type> values(size_t(dimensions.m) * dimensions.n);
	check(
		hipMemcpy(
			values.data(),
			c,
			values.size() * sizeof(output_type),
			hipMemcpyDeviceToHost
		),
		"download C"
	);
	return values;
}

reference::reference(
	const problem& p,
	hipStream_t stream
)
	: p(p), stream(stream) {
	const shape& s = p.dimensions;
	check(hipblasLtCreate(&handle), "create handle");
	check(
		hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F),
		"create matmul"
	);
	// Column-major hipBLASLt computes row-major C = A B as C^T = B^T A^T
	check(
		hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16BF, s.k, s.m, s.k),
		"describe A"
	);
	check(
		hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16BF, s.n, s.k, s.n),
		"describe B"
	);
	check(
		hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_32F, s.n, s.m, s.n),
		"describe C"
	);
	check(hipMalloc(&workspace, workspace_limit), "allocate workspace");

	int version = 0;
	check(hipblasLtGetVersion(handle, &version), "query version");
	int cached = cached_solution(version, s);
	if(cached >= 0 && select(cached)) {
		return;
	}
	if(!select(fastest_candidate())) {
		fail("hipBLASLt", "select tuned solution", "solution is not supported");
	}
	cache_solution(version, s, index);
}

// Solution indices are stable within a hipBLASLt version
bool reference::select(
	int solution
) {
	std::vector<int> indices{solution};
	std::vector<hipblasLtMatmulHeuristicResult_t> results;
	if(hipblaslt_ext::getAlgosFromIndex(handle, indices, results)
		   != HIPBLAS_STATUS_SUCCESS
	   || results.empty()) {
		return false;
	}
	size_t bytes = 0;
	if(hipblaslt_ext::matmulIsAlgoSupported(
		   handle,
		   operation,
		   &alpha,
		   layout_b,
		   layout_a,
		   &beta,
		   layout_c,
		   layout_c,
		   results[0].algo,
		   bytes
	   ) != HIPBLAS_STATUS_SUCCESS
	   || bytes > workspace_limit) {
		return false;
	}
	algorithm = results[0].algo;
	workspace_bytes = bytes;
	index = solution;
	return true;
}

int reference::fastest_candidate() {
	hipblasLtMatmulPreference_t preference = nullptr;
	check(hipblasLtMatmulPreferenceCreate(&preference), "create preference");
	uint64_t limit = workspace_limit;
	check(
		hipblasLtMatmulPreferenceSetAttribute(
			preference,
			HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
			&limit,
			sizeof(limit)
		),
		"limit workspace"
	);
	std::vector<hipblasLtMatmulHeuristicResult_t> candidates(candidate_count);
	int count = 0;
	check(
		hipblasLtMatmulAlgoGetHeuristic(
			handle,
			operation,
			layout_b,
			layout_a,
			layout_c,
			layout_c,
			preference,
			candidate_count,
			candidates.data(),
			&count
		),
		"query heuristic"
	);
	check(hipblasLtMatmulPreferenceDestroy(preference), "destroy preference");
	if(count == 0) {
		fail("hipBLASLt", "query heuristic", "no solution for this shape");
	}

	auto run = [&] { launch(); };
	int fastest = -1;
	double fastest_milliseconds = 0;
	for(int candidate = 0; candidate < count; ++candidate) {
		algorithm = candidates[candidate].algo;
		workspace_bytes = candidates[candidate].workspaceSize;
		if(candidate == 0) {
			warm_up(run, stream);
		}
		double milliseconds = measure(run, stream, candidate_milliseconds);
		if(fastest < 0 || milliseconds < fastest_milliseconds) {
			fastest_milliseconds = milliseconds;
			fastest = hipblaslt_ext::getIndexFromAlgo(
				candidates[candidate].algo
			);
		}
	}
	return fastest;
}

reference::~reference() {
	check(hipFree(workspace), "free workspace");
	check(hipblasLtMatrixLayoutDestroy(layout_a), "destroy A layout");
	check(hipblasLtMatrixLayoutDestroy(layout_b), "destroy B layout");
	check(hipblasLtMatrixLayoutDestroy(layout_c), "destroy C layout");
	check(hipblasLtMatmulDescDestroy(operation), "destroy matmul");
	check(hipblasLtDestroy(handle), "destroy handle");
}

void reference::launch() {
	check(
		hipblasLtMatmul(
			handle,
			operation,
			&alpha,
			p.b,
			layout_b,
			p.a,
			layout_a,
			&beta,
			p.c_reference,
			layout_c,
			p.c_reference,
			layout_c,
			&algorithm,
			workspace,
			workspace_bytes,
			stream
		),
		"matmul"
	);
}

int reference::solution() const {
	return index;
}

double milliseconds_per_launch(
	const std::function<void()>& launch,
	hipStream_t stream
) {
	warm_up(launch, stream);
	return measure(launch, stream, measurement_milliseconds);
}

}
