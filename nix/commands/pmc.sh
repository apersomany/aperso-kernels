perf_level="$(amd-smi metric --gpu 0 --perf-level)"
if [[ "$perf_level" != *AMDSMI_DEV_PERF_LEVEL_STABLE_STD* ]]; then
  printf 'PMC values may be missing in AUTO; set GPU 0 to STABLE_STD first.\n' >&2
  exit 2
fi
read -r -a counters <<< "${GEMM_PMC_COUNTERS:-GPUBusy VALUInsts SQ_INSTS_WAVE32_VALU SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_WAIT_ANY LDSBankConflict MemUnitBusy}"
mkdir -p build/pmc
exec rocprofv3 \
  --output-directory build/pmc \
  --output-format csv \
  --kernel-include-regex gemm_kernel \
  --pmc "${counters[@]}" \
  -- "$build_directory/gemm" "$@" --once
