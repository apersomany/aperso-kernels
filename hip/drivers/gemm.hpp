#pragma once

#include "kernels/gemm.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <functional>
#include <vector>

namespace gemm_driver {

using gemm_kernel::input_type;
using gemm_kernel::output_type;
using gemm_kernel::shape;

void check(hipError_t result, const char* operation);
void check(hipblasStatus_t result, const char* operation);

// Device buffers with inputs filled from a deterministic pattern
struct problem {
	explicit problem(const shape& s);
	problem(const problem&) = delete;
	problem& operator=(const problem&) = delete;
	~problem();

	std::vector<output_type> download(const output_type* c) const;

	const shape dimensions;
	input_type* a = nullptr;
	input_type* b = nullptr;
	output_type* c_custom = nullptr;
	output_type* c_reference = nullptr;
};

// hipBLASLt matmul with the fastest heuristic candidate, cached per shape
class reference {
  public:
	reference(const problem& p, hipStream_t stream);
	reference(const reference&) = delete;
	reference& operator=(const reference&) = delete;
	~reference();

	void launch();
	int solution() const;

  private:
	bool select(int solution);
	int fastest_candidate();

	const problem& p;
	hipStream_t stream;
	const float alpha = 1;
	const float beta = 0;
	hipblasLtHandle_t handle = nullptr;
	hipblasLtMatmulDesc_t operation = nullptr;
	hipblasLtMatrixLayout_t layout_a = nullptr;
	hipblasLtMatrixLayout_t layout_b = nullptr;
	hipblasLtMatrixLayout_t layout_c = nullptr;
	hipblasLtMatmulAlgo_t algorithm = {};
	int index = -1;
	void* workspace = nullptr;
	size_t workspace_bytes = 0;
};

// Warms clocks for a fixed time, then returns the mean duration of one launch
double milliseconds_per_launch(
	const std::function<void()>& launch,
	hipStream_t stream
);

}
