#include "drivers/gemm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

using gemm_driver::check;
using gemm_driver::shape;

// Exact inputs make both results equal; tolerances only guard reordering
constexpr double absolute_tolerance = 0.05;
constexpr double relative_tolerance = 0.02;

[[noreturn]] void usage(
	const char* program
) {
	std::fprintf(stderr, "usage: %s MxKxN [--once]\n", program);
	std::exit(2);
}

shape parse_shape(
	std::string_view text
) {
	shape s{};
	char trailing = '\0';
	if(std::sscanf(text.data(), "%dx%dx%d%c", &s.m, &s.k, &s.n, &trailing)
	   != 3) {
		std::fprintf(
			stderr,
			"invalid shape '%s'; expected MxKxN\n",
			text.data()
		);
		std::exit(2);
	}
	if(!gemm_kernel::supports(s)) {
		const shape& tile = gemm_kernel::tile;
		std::fprintf(
			stderr,
			"unsupported shape '%s'; M, K, and N must be positive multiples of "
			"%d, %d, and %d\n",
			text.data(),
			tile.m,
			tile.k,
			tile.n
		);
		std::exit(2);
	}
	return s;
}

struct comparison {
	double maximum_absolute_error = 0;
	double maximum_relative_error = 0;
	size_t worst_index = 0;
	bool passed = true;
};

comparison compare(
	const std::vector<gemm_driver::output_type>& custom,
	const std::vector<gemm_driver::output_type>& reference
) {
	comparison result;
	for(size_t index = 0; index < custom.size(); ++index) {
		double expected = reference[index];
		double error = std::abs(custom[index] - expected);
		double tolerance = absolute_tolerance
						   + relative_tolerance * std::abs(expected);
		// Negated comparisons also flag NaN
		if(!(error <= tolerance)) {
			result.passed = false;
		}
		if(!(error <= result.maximum_absolute_error)) {
			result.maximum_absolute_error = error;
			result.worst_index = index;
		}
		if(expected != 0) {
			result.maximum_relative_error = std::max(
				result.maximum_relative_error,
				error / std::abs(expected)
			);
		}
	}
	return result;
}

void report(
	const char* name,
	double milliseconds,
	double baseline_milliseconds,
	const shape& s
) {
	double tflops = 2.0 * s.m * s.n * s.k / milliseconds / 1e9;
	std::printf(
		"%-20s %10.4f %10.2f %9.2fx\n",
		name,
		milliseconds,
		tflops,
		baseline_milliseconds / milliseconds
	);
}

int run(
	const shape& s,
	bool once,
	hipStream_t stream
) {
	gemm_driver::problem p(s);
	auto custom = [&] {
		check(
			gemm_kernel::launch(p.a, p.b, p.c_custom, s, stream),
			"launch custom GEMM"
		);
	};
	// One launch keeps profiler output focused on the custom kernel
	if(once) {
		custom();
		check(hipStreamSynchronize(stream), "finish launch");
		return 0;
	}

	gemm_driver::reference reference(p, stream);
	double custom_milliseconds = gemm_driver::milliseconds_per_launch(
		custom,
		stream
	);
	double reference_milliseconds = gemm_driver::milliseconds_per_launch(
		[&] { reference.launch(); },
		stream
	);
	char reference_name[32];
	std::snprintf(
		reference_name,
		sizeof(reference_name),
		"hipBLASLt #%d",
		reference.solution()
	);
	std::printf("GEMM %dx%dx%d\n\n", s.m, s.k, s.n);
	std::printf(
		"%-20s %10s %10s %10s\n",
		"Implementation",
		"Time (ms)",
		"TFLOP/s",
		"Speedup"
	);
	report("Custom", custom_milliseconds, reference_milliseconds, s);
	report(reference_name, reference_milliseconds, reference_milliseconds, s);

	auto custom_c = p.download(p.c_custom);
	auto reference_c = p.download(p.c_reference);
	comparison result = compare(custom_c, reference_c);
	std::printf(
		"\nValidation %s: max absolute error %g, max relative error %g\n",
		result.passed ? "PASS" : "FAIL",
		result.maximum_absolute_error,
		result.maximum_relative_error
	);
	if(!result.passed) {
		size_t worst = result.worst_index;
		std::fprintf(
			stderr,
			"worst element at row %zu, column %zu: custom %.9g, reference "
			"%.9g\n",
			worst / s.n,
			worst % s.n,
			custom_c[worst],
			reference_c[worst]
		);
		return 1;
	}
	return 0;
}

}

int main(
	int argument_count,
	char** arguments
) {
	bool once = argument_count == 3
				&& std::string_view(arguments[2]) == "--once";
	if(argument_count != 2 && !once) {
		usage(arguments[0]);
	}
	shape s = parse_shape(arguments[1]);
	hipStream_t stream = nullptr;
	check(hipStreamCreate(&stream), "create stream");
	int status = run(s, once, stream);
	check(hipStreamDestroy(stream), "destroy stream");
	return status;
}
