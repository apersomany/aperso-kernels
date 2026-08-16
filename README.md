# aperso-kernels

A small hand-written HIP GEMM for learning GPU kernel development.

The interface is contiguous row-major BF16 `[M, K] × [K, N] →` FP32 `[M, N]`. Shapes use the form `MxKxN`.

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
gemm-run 256x256x256 --no-profile
gemm-trace 1024x1024x1024
gemm-att 1024x1024x1024
```

They are also available through `nix run`:

```bash
nix run .#build
nix run .#run -- 256x256x256
nix run .#trace -- 1024x1024x1024
nix run .#att -- 1024x1024x1024
```

The executable validates the kernel against rocBLAS. Profiling output is written below `build/trace` or `build/att`.

The default target is `gfx1200`. Select one or more targets with `GPU_ARCHITECTURE`:

```bash
GPU_ARCHITECTURE=gfx1036 gemm-build
GPU_ARCHITECTURE="gfx1200,gfx1036" gemm-build
```

CMake writes its compilation database to `build/compile_commands.json`.
