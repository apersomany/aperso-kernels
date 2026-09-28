#pragma once

#include <hip/hip_runtime.h>

namespace gemm_kernel {

// Row-major BF16 [M, K] x [K, N] -> FP32 [M, N]
using input_type = __bf16;
using output_type = float;

struct shape {
	int m;
	int n;
	int k;
};

// Shapes must be positive multiples of this tile
extern const shape tile;

bool supports(const shape& s);

// Returns hipErrorInvalidValue for unsupported shapes
hipError_t launch(
	const input_type* a,
	const input_type* b,
	output_type* c,
	const shape& s,
	hipStream_t stream
);

}
