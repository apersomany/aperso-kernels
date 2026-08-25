#include "gemm.hpp"

#include <rocwmma/rocwmma.hpp>

namespace
{

constexpr int tile_size = 16;

__global__ void tiled_gemm_kernel(
	const gemm_kernel::input_type* a,
	const gemm_kernel::input_type* b,
	float* c,
	int m,
	int n,
	int k
)
{
	__shared__ gemm_kernel::input_type shared_a[tile_size][tile_size];
	__shared__ gemm_kernel::input_type shared_b[tile_size][tile_size];

	int row = int(blockIdx.y) * tile_size + int(threadIdx.y);
	int column = int(blockIdx.x) * tile_size + int(threadIdx.x);
	float accumulator = 0.0f;

	for(int tile_start = 0; tile_start < k; tile_start += tile_size)
	{
		int a_column = tile_start + int(threadIdx.x);
		int b_row = tile_start + int(threadIdx.y);
		shared_a[threadIdx.y][threadIdx.x] =
			row < m && a_column < k ? a[row * k + a_column]
									: gemm_kernel::input_type(0.0f);
		shared_b[threadIdx.y][threadIdx.x] =
			b_row < k && column < n ? b[b_row * n + column]
									: gemm_kernel::input_type(0.0f);
		__syncthreads();

		for(int inner = 0; inner < tile_size; ++inner)
			accumulator += static_cast<float>(shared_a[threadIdx.y][inner])
						   * static_cast<float>(shared_b[inner][threadIdx.x]);
		__syncthreads();
	}

	if(row < m && column < n)
		c[row * n + column] = accumulator;
}

}

namespace gemm_kernel
{

void launch(
	const input_type* a,
	const input_type* b,
	float* c,
	int m,
	int n,
	int k,
	hipStream_t stream
)
{
	constexpr int block_dimension = tile_size;
	dim3 block(block_dimension, block_dimension, 1);
	dim3 grid(
		(unsigned(n) + block_dimension - 1) / block_dimension,
		(unsigned(m) + block_dimension - 1) / block_dimension,
		1
	);
	hipLaunchKernelGGL(
		tiled_gemm_kernel,
		grid,
		block,
		0,
		stream,
		a,
		b,
		c,
		m,
		n,
		k
	);
}

}
