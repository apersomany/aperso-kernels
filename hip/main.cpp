#include "drivers/gemm.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

namespace
{

constexpr int warmup_iteration_count = 5;
constexpr int measured_iteration_count = 30;
constexpr double absolute_tolerance = 0.05;
constexpr double relative_tolerance = 0.02;
constexpr size_t profiler_buffer_size_bytes = 64 * 1024;
constexpr size_t profiler_buffer_watermark_bytes =
	profiler_buffer_size_bytes * 7 / 8;

struct kernel_record
{
	uint64_t dispatch_id;
	uint64_t kernel_id;
	uint64_t start_timestamp;
	uint64_t end_timestamp;
	uint32_t private_segment_size;
	uint32_t group_segment_size;
	uint32_t workgroup_x;
	uint32_t workgroup_y;
	uint32_t workgroup_z;
	uint32_t grid_x;
	uint32_t grid_y;
	uint32_t grid_z;
};

struct profiler_state
{
	rocprofiler_context_id_t context = {};
	rocprofiler_buffer_id_t buffer = {};
	rocprofiler_callback_thread_t callback_thread = {};
	rocprofiler_client_finalize_t finalize = nullptr;
	std::mutex mutex;
	std::vector<kernel_record> records;
	uint64_t dropped_records = 0;
	bool initialized = false;
	bool active = false;
	bool finalized = false;
};

profiler_state profiler_data;
rocprofiler_client_id_t* profiler_client = nullptr;

bool profiler_check(
	rocprofiler_status_t result,
	const char* operation
)
{
	if(result == ROCPROFILER_STATUS_SUCCESS)
		return true;
	const char* message = rocprofiler_get_status_string(result);
	std::fprintf(
		stderr,
		"ROCprofiler-SDK failure in %s: %s\n",
		operation,
		message == nullptr ? "unknown error" : message
	);
	return false;
}

void profiler_buffer_callback(
	rocprofiler_context_id_t,
	rocprofiler_buffer_id_t,
	rocprofiler_record_header_t** headers,
	size_t header_count,
	void* callback_data,
	uint64_t dropped_records
)
{
	auto& state = *static_cast<profiler_state*>(callback_data);
	std::lock_guard<std::mutex> lock(state.mutex);
	state.dropped_records += dropped_records;
	for(size_t index = 0; index < header_count; ++index)
	{
		rocprofiler_record_header_t* header = headers[index];
		if(header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING
		   || header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
			continue;
		auto* record =
			static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
				header->payload
			);
		kernel_record value{
			record->dispatch_info.dispatch_id,
			record->dispatch_info.kernel_id,
			record->start_timestamp,
			record->end_timestamp,
			record->dispatch_info.private_segment_size,
			record->dispatch_info.group_segment_size,
			record->dispatch_info.workgroup_size.x,
			record->dispatch_info.workgroup_size.y,
			record->dispatch_info.workgroup_size.z,
			record->dispatch_info.grid_size.x,
			record->dispatch_info.grid_size.y,
			record->dispatch_info.grid_size.z};
		state.records.push_back(value);
	}
}

int profiler_tool_initialize(
	rocprofiler_client_finalize_t finalize,
	void* callback_data
)
{
	auto& state = *static_cast<profiler_state*>(callback_data);
	state.finalize = finalize;
	if(!profiler_check(
		   rocprofiler_create_context(&state.context),
		   "create context"
	   ))
		return -1;
	if(!profiler_check(
		   rocprofiler_create_buffer(
			   state.context,
			   profiler_buffer_size_bytes,
			   profiler_buffer_watermark_bytes,
			   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
			   profiler_buffer_callback,
			   &state,
			   &state.buffer
		   ),
		   "create buffer"
	   ))
		return -1;
	if(!profiler_check(
		   rocprofiler_create_callback_thread(&state.callback_thread),
		   "create callback thread"
	   ))
		return -1;
	if(!profiler_check(
		   rocprofiler_assign_callback_thread(
			   state.buffer,
			   state.callback_thread
		   ),
		   "assign callback thread"
	   ))
		return -1;
	for(rocprofiler_buffer_tracing_kind_t kind :
		{ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
		 ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API})
	{
		if(!profiler_check(
			   rocprofiler_configure_buffer_tracing_service(
				   state.context,
				   kind,
				   nullptr,
				   0,
				   state.buffer
			   ),
			   "configure HSA tracing"
		   ))
			return -1;
	}
	if(!profiler_check(
		   rocprofiler_configure_buffer_tracing_service(
			   state.context,
			   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
			   nullptr,
			   0,
			   state.buffer
		   ),
		   "configure kernel dispatch tracing"
	   ))
		return -1;
	int valid = 0;
	if(!profiler_check(
		   rocprofiler_context_is_valid(state.context, &valid),
		   "check context"
	   ))
		return -1;
	if(valid == 0)
	{
		std::fprintf(stderr, "ROCprofiler-SDK rejected the tracing context\n");
		return -1;
	}
	state.initialized = true;
	if(!profiler_check(
		   rocprofiler_start_context(state.context),
		   "start context"
	   ))
		return -1;
	state.active = true;
	return 0;
}

void profiler_tool_finalize(
	void*
)
{
}

}

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
	uint32_t,
	const char*,
	uint32_t priority,
	rocprofiler_client_id_t* client_id
)
{
	if(priority > 0)
		return nullptr;
	client_id->name = "aperso-kernels";
	profiler_client = client_id;
	static rocprofiler_tool_configure_result_t configuration{
		sizeof(rocprofiler_tool_configure_result_t),
		profiler_tool_initialize,
		profiler_tool_finalize,
		&profiler_data};
	return &configuration;
}

namespace profiler
{

bool setup()
{
	int initialized = 0;
	if(!profiler_check(
		   rocprofiler_is_initialized(&initialized),
		   "query initialization"
	   ))
		return false;
	if(initialized == 0
	   && !profiler_check(
		   rocprofiler_force_configure(&rocprofiler_configure),
		   "force configuration"
	   ))
		return false;
	if(!profiler_data.initialized)
	{
		std::fprintf(stderr, "ROCprofiler-SDK tracing was not initialized\n");
		return false;
	}
	return true;
}

bool start()
{
	if(!profiler_data.initialized || profiler_data.finalized)
		return false;
	if(profiler_data.active)
		return true;
	if(!profiler_check(
		   rocprofiler_start_context(profiler_data.context),
		   "start context"
	   ))
		return false;
	profiler_data.active = true;
	return true;
}

void stop()
{
	if(!profiler_data.active)
		return;
	if(profiler_check(
		   rocprofiler_stop_context(profiler_data.context),
		   "stop context"
	   ))
		profiler_data.active = false;
}

void clear()
{
	std::lock_guard<std::mutex> lock(profiler_data.mutex);
	profiler_data.records.clear();
	profiler_data.dropped_records = 0;
}

bool flush()
{
	return profiler_data.initialized
		   && profiler_check(
			   rocprofiler_flush_buffer(profiler_data.buffer),
			   "flush buffer"
		   );
}

void flush_and_print()
{
	if(!flush())
		return;
	std::lock_guard<std::mutex> lock(profiler_data.mutex);
	std::printf("\nKernel dispatch trace\n");
	std::printf(
		"  %8s %8s %13s %14s %14s %10s %12s\n",
		"Dispatch",
		"Kernel",
		"Duration (us)",
		"Workgroup",
		"Grid",
		"LDS (B)",
		"Scratch (B)"
	);
	for(const kernel_record& record : profiler_data.records)
	{
		uint64_t duration = record.end_timestamp >= record.start_timestamp
								? record.end_timestamp - record.start_timestamp
								: 0;
		char workgroup[32] = {};
		char grid[32] = {};
		std::snprintf(
			workgroup,
			sizeof(workgroup),
			"%ux%ux%u",
			record.workgroup_x,
			record.workgroup_y,
			record.workgroup_z
		);
		std::snprintf(
			grid,
			sizeof(grid),
			"%ux%ux%u",
			record.grid_x,
			record.grid_y,
			record.grid_z
		);
		std::printf(
			"  %8" PRIu64 " %8" PRIu64 " %13.3f %14s %14s %10u %12u\n",
			record.dispatch_id,
			record.kernel_id,
			double(duration) / 1000.0,
			workgroup,
			grid,
			record.group_segment_size,
			record.private_segment_size
		);
	}
	if(profiler_data.records.empty())
		std::printf("  No kernel dispatches recorded\n");
	if(profiler_data.dropped_records != 0)
		std::printf(
			"  Dropped records: %" PRIu64 "\n",
			profiler_data.dropped_records
		);
}

void shutdown()
{
	stop();
	if(!profiler_data.initialized || profiler_data.finalized)
		return;
	if(profiler_client == nullptr || profiler_data.finalize == nullptr)
		return;
	profiler_data.finalize(*profiler_client);
	profiler_data.finalized = true;
}

}

namespace
{

gemm_driver::shape parse_shape(
	const char* value
)
{
	gemm_driver::shape dimensions{};
	char trailing_character = '\0';
	if(std::sscanf(
		   value,
		   "%dx%dx%d%c",
		   &dimensions.m,
		   &dimensions.k,
		   &dimensions.n,
		   &trailing_character
	   ) != 3
	   || dimensions.m <= 0 || dimensions.n <= 0 || dimensions.k <= 0)
	{
		std::fprintf(
			stderr,
			"invalid shape '%s'; expected positive MxKxN dimensions\n",
			value
		);
		std::exit(2);
	}
	return dimensions;
}

struct benchmark_result
{
	double milliseconds;
	double gigaflops;
};

benchmark_result benchmark(
	const gemm_driver::shape& dimensions,
	hipStream_t stream,
	const std::function<void()>& launch
)
{
	for(int index = 0; index < warmup_iteration_count; ++index)
		launch();
	gemm_driver::check(hipStreamSynchronize(stream), "warmup stream");

	hipEvent_t start = nullptr;
	hipEvent_t stop = nullptr;
	gemm_driver::check(hipEventCreate(&start), "create start event");
	gemm_driver::check(hipEventCreate(&stop), "create stop event");
	gemm_driver::check(hipEventRecord(start, stream), "record start event");
	for(int index = 0; index < measured_iteration_count; ++index)
		launch();
	gemm_driver::check(hipEventRecord(stop, stream), "record stop event");
	gemm_driver::check(hipEventSynchronize(stop), "synchronize stop event");
	float elapsed_milliseconds = 0.0f;
	gemm_driver::check(
		hipEventElapsedTime(&elapsed_milliseconds, start, stop),
		"measure elapsed time"
	);
	gemm_driver::check(hipEventDestroy(start), "destroy start event");
	gemm_driver::check(hipEventDestroy(stop), "destroy stop event");
	double average_milliseconds =
		double(elapsed_milliseconds) / measured_iteration_count;
	double operation_count = 2.0 * dimensions.m * dimensions.n * dimensions.k;
	return {
		average_milliseconds,
		operation_count / (average_milliseconds * 1.0e6)};
}

void print_benchmarks(
	const gemm_driver::shape& dimensions,
	const benchmark_result& custom,
	const benchmark_result& reference
)
{
	std::printf("GEMM %dx%dx%d\n", dimensions.m, dimensions.k, dimensions.n);
	std::printf("\nBenchmark\n");
	std::printf(
		"  %-16s %12s %16s %12s\n",
		"Implementation",
		"Time (ms)",
		"GFLOP/s",
		"vs rocBLAS"
	);
	std::printf(
		"  %-16s %12.4f %16.3f %11.2fx\n",
		"Custom",
		custom.milliseconds,
		custom.gigaflops,
		reference.milliseconds / custom.milliseconds
	);
	std::printf(
		"  %-16s %12.4f %16.3f %11.2fx\n",
		"rocBLAS",
		reference.milliseconds,
		reference.gigaflops,
		1.0
	);
}

struct validation_result
{
	double maximum_absolute_error = 0.0;
	double maximum_relative_error = 0.0;
	size_t maximum_error_index = 0;
	bool valid = true;
};

validation_result validate(
	const std::vector<float>& custom,
	const std::vector<float>& reference
)
{
	validation_result result;
	for(size_t index = 0; index < custom.size(); ++index)
	{
		if(!std::isfinite(custom[index]) || !std::isfinite(reference[index]))
		{
			result.valid = false;
			return result;
		}
		double absolute_error =
			std::abs(double(custom[index]) - double(reference[index]));
		double relative_error =
			absolute_error
			/ std::max(std::abs(double(reference[index])), 1.0e-12);
		if(absolute_error > result.maximum_absolute_error)
		{
			result.maximum_absolute_error = absolute_error;
			result.maximum_error_index = index;
		}
		result.maximum_relative_error =
			std::max(result.maximum_relative_error, relative_error);
		if(absolute_error
		   > absolute_tolerance
				 + relative_tolerance * std::abs(double(reference[index])))
			result.valid = false;
	}
	return result;
}

void profile_custom(
	const gemm_driver::shape& dimensions,
	const gemm_driver::buffers& values,
	hipStream_t stream
)
{
	profiler::clear();
	if(!profiler::start())
	{
		std::fprintf(stderr, "custom kernel was not profiled\n");
		return;
	}
	gemm_driver::launch_custom(dimensions, values, stream);
	gemm_driver::check(hipGetLastError(), "custom kernel launch");
	gemm_driver::check(hipStreamSynchronize(stream), "profile stream");
	profiler::stop();
	profiler::flush_and_print();
}

}

int main(
	int argument_count,
	char** arguments
)
{
	if(argument_count < 2 || argument_count > 3
	   || (argument_count == 3
		   && std::strcmp(arguments[2], "--no-profile") != 0))
	{
		std::fprintf(stderr, "usage: %s MxKxN [--no-profile]\n", arguments[0]);
		return 2;
	}
	gemm_driver::shape dimensions = parse_shape(arguments[1]);
	bool profile = argument_count == 2;
	bool profiler_ready = profile && profiler::setup();

	hipStream_t stream = nullptr;
	gemm_driver::check(hipStreamCreate(&stream), "create stream");
	gemm_driver::buffers values = gemm_driver::allocate(dimensions);
	gemm_driver::initialize(dimensions, values, stream);
	rocblas_handle reference_handle =
		gemm_driver::create_reference_handle(stream);
	gemm_driver::check(
		hipDeviceSynchronize(),
		"profiler initialization device"
	);
	if(profiler_ready)
	{
		profiler::stop();
		profiler::flush();
		profiler::clear();
	}

	benchmark_result custom_benchmark = benchmark(
		dimensions,
		stream,
		[&] { gemm_driver::launch_custom(dimensions, values, stream); }
	);
	benchmark_result reference_benchmark = benchmark(
		dimensions,
		stream,
		[&]
		{ gemm_driver::launch_reference(dimensions, values, reference_handle); }
	);
	print_benchmarks(dimensions, custom_benchmark, reference_benchmark);
	gemm_driver::check(hipStreamSynchronize(stream), "validation stream");

	std::vector<float> custom_output =
		gemm_driver::copy_output(dimensions, values.custom);
	std::vector<float> reference_output =
		gemm_driver::copy_output(dimensions, values.reference);
	validation_result validation = validate(custom_output, reference_output);
	std::printf("\nValidation\n");
	std::printf(
		"  %-8s %18s %18s\n",
		"Status",
		"Max abs error",
		"Max rel error"
	);
	std::printf(
		"  %-8s %18.6g %18.6g\n",
		validation.valid ? "PASS" : "FAIL",
		validation.maximum_absolute_error,
		validation.maximum_relative_error
	);
	if(!validation.valid)
	{
		std::fprintf(
			stderr,
			"validation failed at index %zu: custom=%.9g, reference=%.9g\n",
			validation.maximum_error_index,
			custom_output[validation.maximum_error_index],
			reference_output[validation.maximum_error_index]
		);
		gemm_driver::destroy_reference_handle(reference_handle);
		gemm_driver::release(values);
		gemm_driver::check(hipStreamDestroy(stream), "destroy stream");
		profiler::shutdown();
		return 1;
	}

	if(profiler_ready)
		profile_custom(dimensions, values, stream);

	gemm_driver::destroy_reference_handle(reference_handle);
	gemm_driver::release(values);
	gemm_driver::check(hipStreamDestroy(stream), "destroy stream");
	profiler::shutdown();
	return 0;
}
