#include "gemm.hpp"

namespace {

constexpr int tile_size = 16;
constexpr int wave_size = 32;
constexpr int frag_size = tile_size * tile_size / wave_size;

static_assert(tile_size * tile_size % wave_size == 0);

using ab_frag_type = short __attribute__((ext_vector_type(frag_size)));
using cd_frag_type = float __attribute__((ext_vector_type(frag_size)));

__device__ ab_frag_type load_ab_frag(
	const gemm_kernel::input_type* source
) {
	return *reinterpret_cast<const ab_frag_type*>(source);
}

__device__ ab_frag_type transpose_ab_frag(
	const ab_frag_type& source_frag
) {
	int lane = int(threadIdx.x) % tile_size;
	int lane_group = int(threadIdx.x) / tile_size;
	ab_frag_type identity_frag = {};
	ab_frag_type zero_frag = {};
	if(lane_group == lane / frag_size) {
		identity_frag[lane % frag_size] = __builtin_bit_cast(
			short,
			gemm_kernel::input_type(1.0f)
		);
	}
	return __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
		source_frag,
		identity_frag,
		zero_frag
	);
}

__global__ void wmma_gemm_kernel(
	const gemm_kernel::input_type* a,
	const gemm_kernel::input_type* b,
	float* c,
	int n,
	int k
) {
	int lane = int(threadIdx.x) % tile_size;
	int lane_group = int(threadIdx.x) / tile_size;
	int tile_row = int(blockIdx.y) * tile_size;
	int tile_col = int(blockIdx.x) * tile_size;
	int frag_col = lane_group * frag_size;
	cd_frag_type c_frag = {};

	for(int tile_start = 0; tile_start < k; tile_start += tile_size) {
		ab_frag_type a_frag = load_ab_frag(
			a + (tile_row + lane) * k + tile_start + frag_col
		);
		ab_frag_type b_frag = load_ab_frag(
			b + (tile_start + lane) * n + tile_col + frag_col
		);
		b_frag = transpose_ab_frag(b_frag);
		c_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
			b_frag,
			a_frag,
			c_frag
		);
	}

	for(int element = 0; element < frag_size; ++element) {
		int row = tile_row + lane;
		int col = tile_col + frag_col + element;
		c[row * n + col] = c_frag[element];
	}
}

}

namespace gemm_kernel {

void launch(
	const input_type* a,
	const input_type* b,
	float* c,
	int m,
	int n,
	int k,
	hipStream_t stream
) {
	if(m % tile_size != 0 || n % tile_size != 0 || k % tile_size != 0) {
		return;
	}

	dim3 block(wave_size, 1, 1);
	dim3 grid(unsigned(n) / tile_size, unsigned(m) / tile_size, 1);
	hipLaunchKernelGGL(wmma_gemm_kernel, grid, block, 0, stream, a, b, c, n, k);
}

}
