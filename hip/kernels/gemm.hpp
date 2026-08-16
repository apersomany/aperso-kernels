#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace gemm_kernel
{

using input_type = hip_bfloat16;

void launch(
	const input_type* a,
	const input_type* b,
	float* c,
	int m,
	int n,
	int k,
	hipStream_t stream
);

}
