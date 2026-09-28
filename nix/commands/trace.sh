mkdir -p build/trace
exec rocprofv3 \
  --output-directory build/trace \
  --output-format csv \
  --kernel-trace \
  --kernel-include-regex gemm_kernel \
  -- "$build_directory/gemm" "$@" --once
