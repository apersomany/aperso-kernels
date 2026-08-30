#pragma once

#include "../kernels/gemm.hpp"

#include <rocblas/rocblas.h>

#include <vector>

namespace gemm_driver {

struct shape {
	int m;
	int n;
	int k;
};

struct buffers {
	gemm_kernel::input_type* a = nullptr;
	gemm_kernel::input_type* b = nullptr;
	float* custom = nullptr;
	float* reference = nullptr;
};

void check(hipError_t result, const char* operation);
void check(rocblas_status result, const char* operation);

buffers allocate(const shape& dimensions);
void initialize(const shape& dimensions, buffers& values, hipStream_t stream);
void launch_custom(
	const shape& dimensions,
	const buffers& values,
	hipStream_t stream
);
void launch_reference(
	const shape& dimensions,
	const buffers& values,
	rocblas_handle handle
);
std::vector<float> copy_output(const shape& dimensions, const float* values);
void release(buffers& values);
rocblas_handle create_reference_handle(hipStream_t stream);
void destroy_reference_handle(rocblas_handle handle);

}
