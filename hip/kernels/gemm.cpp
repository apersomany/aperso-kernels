#include "gemm.hpp"

#include <cstddef>

namespace gemm_kernel {
namespace {

// RDNA 4 wave32 hardware
constexpr int wave_size = 32;
constexpr int simds_per_wgp = 4;
constexpr int vgprs_per_lane = 256;
constexpr int vgpr_bytes = 4;
constexpr int lds_bytes_per_clock = 128;
// v_wmma_f32_16x16x16_bf16 multiplies 16 x 16 blocks
constexpr int wmma_size = 16;

// Every lane holds 8 elements of each WMMA operand and accumulator
constexpr int vector_size = wmma_size * wmma_size / wave_size;
using vector = input_type __attribute__((ext_vector_type(vector_size)));
using accumulator = output_type __attribute__((ext_vector_type(vector_size)));
constexpr int operand_vgprs = sizeof(vector) / vgpr_bytes;
constexpr int accumulator_vgprs = sizeof(accumulator) / vgpr_bytes;
// Lanes whose 16-byte LDS accesses complete in one clock
constexpr int lds_lanes = lds_bytes_per_clock / sizeof(vector);

// The squarest split of an area moves the fewest operands per multiply
constexpr int square_side(
	int area
) {
	int side = 1;
	while((side + 1) * (side + 1) <= area) {
		++side;
	}
	return side;
}

// One wave per SIMD
constexpr int waves_m = square_side(simds_per_wgp);
constexpr int waves_n = simds_per_wgp / waves_m;
constexpr int threads = simds_per_wgp * wave_size;
// Accumulators take half the registers, the rest holds operands in flight
constexpr int wmmas = vgprs_per_lane / 2 / accumulator_vgprs;
constexpr int wmmas_m = square_side(wmmas);
constexpr int wmmas_n = wmmas / wmmas_m;
constexpr int wave_m = wmmas_m * wmma_size;
constexpr int wave_n = wmmas_n * wmma_size;
constexpr int tile_m = waves_m * wave_m;
constexpr int tile_n = waves_n * wave_n;
// Two WMMA steps per slice let one step's LDS reads overlap the other's WMMAs
constexpr int steps = 2;
constexpr int tile_k = steps * wmma_size;
// Measured: bands of two tile rows balance A and B reuse in L2
constexpr int band = 2;

constexpr int k_vectors = tile_k / vector_size;
constexpr int a_loads = tile_m * k_vectors / threads;
constexpr int b_loads = tile_n * k_vectors / threads;

static_assert(a_loads * threads == tile_m * k_vectors);
static_assert(b_loads * threads == tile_n * k_vectors);
static_assert(
	wmmas * accumulator_vgprs + 2 * (wmmas_m + wmmas_n) * operand_vgprs
		+ (a_loads + b_loads) * operand_vgprs
	< vgprs_per_lane
);

// LDS copy of a slice; each entry holds 8 k values of an A row or B column
struct stage {
	vector a[k_vectors][tile_m];
	vector b[k_vectors][tile_n];
};

// A slice in flight from global memory to LDS
struct slice {
	vector a[a_loads];
	vector b[b_loads];
};

// WMMA operands of one step
struct fragments {
	vector a[wmmas_m];
	vector b[wmmas_n];
};

struct position {
	int k_vector;
	int row;
};

// Each lane group stores consecutive rows of one k vector in a single LDS clock
__device__ position a_position(
	int load
) {
	int lane = threadIdx.x % lds_lanes;
	int group = int(threadIdx.x) / lds_lanes + load * threads / lds_lanes;
	return {group % k_vectors, group / k_vectors * lds_lanes + lane};
}

// Each lane group owns one 8 x 8 transposed block of B
__device__ position b_position(
	int load
) {
	constexpr int blocks_n = tile_n / vector_size;
	int lane = threadIdx.x % vector_size;
	int group = int(threadIdx.x) / vector_size + load * threads / vector_size;
	return {group / blocks_n, group % blocks_n * vector_size + lane};
}

__device__ slice load(
	const input_type* a,
	const input_type* b,
	int n,
	int k
) {
	slice s;
#pragma unroll
	for(int i = 0; i < a_loads; ++i) {
		auto [k_vector, row] = a_position(i);
		s.a[i] = *reinterpret_cast<const vector*>(
			a + size_t(row) * k + k_vector * vector_size
		);
	}
	// In each 8 x 8 block, lane l reads k row l and receives column l
	int lane = threadIdx.x % vector_size;
#pragma unroll
	for(int i = 0; i < b_loads; ++i) {
		auto [k_vector, column] = b_position(i);
		int k_row = k_vector * vector_size + lane;
		auto* address = b + size_t(k_row) * n + column - lane;
		s.b[i] = __builtin_amdgcn_global_load_tr_b128_v8bf16(
			reinterpret_cast<vector*>(const_cast<input_type*>(address))
		);
	}
	return s;
}

__device__ void store(
	stage& destination,
	const slice& s
) {
#pragma unroll
	for(int i = 0; i < a_loads; ++i) {
		auto [k_vector, row] = a_position(i);
		destination.a[k_vector][row] = s.a[i];
	}
#pragma unroll
	for(int i = 0; i < b_loads; ++i) {
		auto [k_vector, column] = b_position(i);
		destination.b[k_vector][column] = s.b[i];
	}
}

// Lane l supplies row or column l % 16 of k half l / 16
__device__ fragments read(
	const stage& source,
	int step,
	int row,
	int column
) {
	int lane = threadIdx.x % wave_size;
	int k_vector = step * (wmma_size / vector_size) + lane / wmma_size;
	fragments f;
#pragma unroll
	for(int i = 0; i < wmmas_m; ++i) {
		f.a[i] = source.a[k_vector][row + i * wmma_size + lane % wmma_size];
	}
#pragma unroll
	for(int j = 0; j < wmmas_n; ++j) {
		f.b[j] = source.b[k_vector][column + j * wmma_size + lane % wmma_size];
	}
	return f;
}

// Stops the compiler from sinking loads next to their first use
__device__ void keep_order() {
	__builtin_amdgcn_sched_barrier(0);
}

__device__ void accumulate(
	accumulator (&sums)[wmmas_m][wmmas_n],
	const stage& source,
	int row,
	int column
) {
	fragments current = read(source, 0, row, column);
#pragma unroll
	for(int step = 0; step < steps; ++step) {
		fragments next;
		if(step + 1 < steps) {
			next = read(source, step + 1, row, column);
		}
		keep_order();
#pragma unroll
		for(int i = 0; i < wmmas_m; ++i) {
#pragma unroll
			for(int j = 0; j < wmmas_n; ++j) {
				sums[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
					current.a[i],
					current.b[j],
					sums[i][j]
				);
			}
		}
		current = next;
	}
}

// Fences only LDS, so in-flight global loads are not waited on
__device__ void barrier() {
	__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
	__builtin_amdgcn_s_barrier();
	__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

struct tile_position {
	int row;
	int column;
};

// Consecutive workgroups walk down a band of tile rows, then move right
__device__ tile_position tile_of_workgroup() {
	int columns = gridDim.x;
	int linear = blockIdx.y * columns + blockIdx.x;
	int band_start = linear / (band * columns) * band;
	int band_rows = min(band, int(gridDim.y) - band_start);
	int in_band = linear - band_start * columns;
	return {band_start + in_band % band_rows, in_band / band_rows};
}

__global__ __launch_bounds__(threads) void gemm(
	const input_type* a,
	const input_type* b,
	output_type* c,
	int n,
	int k
) {
	__shared__ stage stages[2];
	int wave = threadIdx.x / wave_size;
	int lane = threadIdx.x % wave_size;
	int wave_row = wave / waves_n * wave_m;
	int wave_column = wave % waves_n * wave_n;
	auto [tile_row, tile_column] = tile_of_workgroup();
	a += size_t(tile_row) * tile_m * k;
	b += tile_column * tile_n;

	accumulator sums[wmmas_m][wmmas_n]{};
	int slices = k / tile_k;
	store(stages[0], load(a, b, n, k));
	barrier();
	for(int index = 1; index < slices; ++index) {
		a += tile_k;
		b += size_t(tile_k) * n;
		slice next = load(a, b, n, k);
		keep_order();
		accumulate(sums, stages[(index - 1) % 2], wave_row, wave_column);
		store(stages[index % 2], next);
		barrier();
	}
	accumulate(sums, stages[(slices - 1) % 2], wave_row, wave_column);

	// Lane l holds column l % 16 of rows 8 * (l / 16) + e
	c += size_t(tile_row * tile_m + wave_row) * n + tile_column * tile_n
		 + wave_column;
#pragma unroll
	for(int i = 0; i < wmmas_m; ++i) {
#pragma unroll
		for(int j = 0; j < wmmas_n; ++j) {
#pragma unroll
			for(int e = 0; e < vector_size; ++e) {
				int row = i * wmma_size + lane / wmma_size * vector_size + e;
				int column = j * wmma_size + lane % wmma_size;
				c[size_t(row) * n + column] = sums[i][j][e];
			}
		}
	}
}

}

const shape tile{tile_m, tile_n, tile_k};

bool supports(
	const shape& s
) {
	return s.m > 0 && s.n > 0 && s.k > 0 && s.m % tile.m == 0
		   && s.n % tile.n == 0 && s.k % tile.k == 0;
}

hipError_t launch(
	const input_type* a,
	const input_type* b,
	output_type* c,
	const shape& s,
	hipStream_t stream
) {
	if(!supports(s)) {
		return hipErrorInvalidValue;
	}
	gemm<<<dim3(s.n / tile_n, s.m / tile_m), threads, 0, stream>>>(
		a,
		b,
		c,
		s.n,
		s.k
	);
	return hipGetLastError();
}

}
