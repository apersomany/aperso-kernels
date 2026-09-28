mkdir -p build/att
exec rocprofv3 \
  --att \
  --output-directory build/att \
  --kernel-include-regex gemm_kernel \
  -- "$build_directory/gemm" "$@" --once
