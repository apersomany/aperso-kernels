IFS=', ' read -r -a architectures <<< "${GPU_ARCHITECTURE:-gfx1200}"
architecture_list="$(IFS=';'; printf '%s' "${architectures[*]}")"
if [[ -z "$architecture_list" ]]; then
  printf 'GPU_ARCHITECTURE must contain at least one target\n' >&2
  exit 2
fi

build_directory="${GEMM_BUILD_DIR:-build/rocm10-sdk}"
# Explicit compilers make CMake reconfigure an existing cache when the SDK store path changes
cmake -S . -B "$build_directory" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$HIPCXX" \
  -DCMAKE_CXX_COMPILER="$HIPCXX" \
  -DCMAKE_HIP_ARCHITECTURES="$architecture_list" \
  -DGPU_TARGETS="$architecture_list" >/dev/null
cmake --build "$build_directory" --parallel
