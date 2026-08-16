#include "gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gemm_driver
{
namespace
{

void stop_on_failure(
	const char* library,
	const char* operation,
	const char* message
)
{
	std::fprintf(
		stderr,
		"error: %s failed during %s\n  %s\n",
		library,
		operation,
		message
	);
	std::exit(1);
}

float input_value(
	size_t index
)
{
	return float(int(index % 13) - 6) * 0.03125f;
}

}

void check(
	hipError_t result,
	const char* operation
)
{
	if(result != hipSuccess)
		stop_on_failure("HIP", operation, hipGetErrorString(result));
}

void check(
	rocblas_status result,
	const char* operation
)
{
	if(result != rocblas_status_success)
		stop_on_failure("rocBLAS", operation, rocblas_status_to_string(result));
}

buffers allocate(
	const shape& dimensions
)
{
	buffers values;
	check(
		hipMalloc(
			&values.a,
			size_t(dimensions.m) * dimensions.k * sizeof(*values.a)
		),
		"hipMalloc A"
	);
	check(
		hipMalloc(
			&values.b,
			size_t(dimensions.k) * dimensions.n * sizeof(*values.b)
		),
		"hipMalloc B"
	);
	check(
		hipMalloc(
			&values.custom,
			size_t(dimensions.m) * dimensions.n * sizeof(*values.custom)
		),
		"hipMalloc custom output"
	);
	check(
		hipMalloc(
			&values.reference,
			size_t(dimensions.m) * dimensions.n * sizeof(*values.reference)
		),
		"hipMalloc reference output"
	);
	return values;
}

void initialize(
	const shape& dimensions,
	buffers& values,
	hipStream_t stream
)
{
	size_t a_count = size_t(dimensions.m) * dimensions.k;
	size_t b_count = size_t(dimensions.k) * dimensions.n;
	std::vector<gemm_kernel::input_type> host_a(a_count);
	std::vector<gemm_kernel::input_type> host_b(b_count);
	for(size_t index = 0; index < a_count; ++index)
		host_a[index] = gemm_kernel::input_type(input_value(index));
	for(size_t index = 0; index < b_count; ++index)
		host_b[index] = gemm_kernel::input_type(input_value(index + 17));

	check(
		hipMemcpyAsync(
			values.a,
			host_a.data(),
			a_count * sizeof(*values.a),
			hipMemcpyHostToDevice,
			stream
		),
		"copy A"
	);
	check(
		hipMemcpyAsync(
			values.b,
			host_b.data(),
			b_count * sizeof(*values.b),
			hipMemcpyHostToDevice,
			stream
		),
		"copy B"
	);
	size_t output_bytes =
		size_t(dimensions.m) * dimensions.n * sizeof(*values.custom);
	check(
		hipMemsetAsync(values.custom, 0, output_bytes, stream),
		"clear custom output"
	);
	check(
		hipMemsetAsync(values.reference, 0, output_bytes, stream),
		"clear reference output"
	);
	check(hipStreamSynchronize(stream), "initialize stream");
}

void launch_custom(
	const shape& dimensions,
	const buffers& values,
	hipStream_t stream
)
{
	gemm_kernel::launch(
		values.a,
		values.b,
		values.custom,
		dimensions.m,
		dimensions.n,
		dimensions.k,
		stream
	);
}

void launch_reference(
	const shape& dimensions,
	const buffers& values,
	rocblas_handle handle
)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;
	check(
		rocblas_gemm_ex(
			handle,
			rocblas_operation_none,
			rocblas_operation_none,
			dimensions.n,
			dimensions.m,
			dimensions.k,
			&alpha,
			values.b,
			rocblas_datatype_bf16_r,
			dimensions.n,
			values.a,
			rocblas_datatype_bf16_r,
			dimensions.k,
			&beta,
			values.reference,
			rocblas_datatype_f32_r,
			dimensions.n,
			values.reference,
			rocblas_datatype_f32_r,
			dimensions.n,
			rocblas_datatype_f32_r,
			rocblas_gemm_algo_standard,
			0,
			0
		),
		"rocblas_gemm_ex"
	);
}

std::vector<float> copy_output(
	const shape& dimensions,
	const float* values
)
{
	std::vector<float> host_values(size_t(dimensions.m) * dimensions.n);
	check(
		hipMemcpy(
			host_values.data(),
			values,
			host_values.size() * sizeof(*values),
			hipMemcpyDeviceToHost
		),
		"copy output"
	);
	return host_values;
}

void release(
	buffers& values
)
{
	if(values.a != nullptr)
		check(hipFree(values.a), "hipFree A");
	if(values.b != nullptr)
		check(hipFree(values.b), "hipFree B");
	if(values.custom != nullptr)
		check(hipFree(values.custom), "hipFree custom output");
	if(values.reference != nullptr)
		check(hipFree(values.reference), "hipFree reference output");
	values = {};
}

rocblas_handle create_reference_handle(
	hipStream_t stream
)
{
	rocblas_handle handle = nullptr;
	check(rocblas_create_handle(&handle), "rocblas_create_handle");
	check(rocblas_set_stream(handle, stream), "rocblas_set_stream");
	return handle;
}

void destroy_reference_handle(
	rocblas_handle handle
)
{
	if(handle != nullptr)
		check(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
}

}
