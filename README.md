# aperso-kernels

A small hand-written HIP GEMM for learning GPU kernel development.

The interface is contiguous row-major BF16 `[M, K] x [K, N] ->` FP32 `[M, N]`. Shapes use the form `MxKxN`; `M` and `N` must be multiples of 128, and `K` must be a multiple of 32. The tile sizes follow from RDNA 4 register and SIMD counts; see `hip/kernels/gemm.cpp`.

## Commands

Enter the development shell:

```bash
nix develop
```

Configure and build with CMake:

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

The shell applications configure and build incrementally:

```bash
gemm-build
gemm-run 256x256x256
gemm-run 256x256x256 --once
gemm-trace 1024x1024x1024
gemm-att 1024x1024x1024
```

They are also available through `nix run`:

```bash
nix run .#build
nix run .#run -- 256x256x256
nix run .#trace -- 1024x1024x1024
nix run .#att -- 1024x1024x1024
nix run .#pmc -- 1024x1024x1024
```

The executable warms each implementation for a second, times it, and validates the kernel against hipBLASLt. The fastest of hipBLASLt's heuristic candidates is cached per shape and library version in `hipblaslt-solutions.txt` next to the executable. `--once` launches only the custom kernel, which the trace, ATT, and PMC commands use. Their output is written below `build/trace`, `build/att`, and `build/pmc` respectively. The flake uses AMD's pinned ROCm 10.0 TheRock `gfx120X-all` SDK tarball.

PMC collection requires a stable GPU performance level. Record the existing level, set `STABLE_STD`, and restore the recorded level afterward; the `pmc` command checks the level but does not change system GPU state. The final command below assumes the original level was `AUTO`:

```bash
amd-smi metric --gpu 0 --perf-level
sudo amd-smi set --gpu 0 --perf-level STABLE_STD
nix run .#pmc -- 1024x1024x1024
sudo amd-smi set --gpu 0 --perf-level AUTO
```

The WMMA kernel requires an RDNA 4 target. The default is `gfx1200`; select another compatible target with `GPU_ARCHITECTURE`:

```bash
GPU_ARCHITECTURE=gfx1201 gemm-build
GPU_ARCHITECTURE="gfx1200,gfx1201" gemm-build
```

CMake writes its compilation database to `build/compile_commands.json`.
