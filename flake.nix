{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      packages = nixpkgs.legacyPackages.${system};
      library = packages.lib;
      rocm = packages.rocmPackages;
      rocm_dependencies = [
        rocm.clr
        rocm.rocm-device-libs
        rocm.rocm-comgr
        rocm.rocm-runtime
        rocm.rocblas
        rocm.rocprofiler-sdk.dev
      ];
      command_dependencies = [
        packages.cmake
        packages.ninja
        rocm.rocprofiler-sdk
      ];
      environment = {
        HIPCC = "${rocm.clr}/bin/hipcc";
        HIP_PATH = "${rocm.clr}";
        HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
        HSA_PATH = "${rocm.rocm-runtime}";
        ROCPROFILER_SDK_PATH = "${rocm.rocprofiler-sdk}";
        ROCPROFILER_SDK_INCLUDE_PATH = "${rocm.rocprofiler-sdk.dev}/include";
        CMAKE_PREFIX_PATH = library.concatStringsSep ";" rocm_dependencies;
        LD_LIBRARY_PATH = library.makeLibraryPath [
          rocm.clr
          rocm.rocm-runtime
          rocm.rocblas
          rocm.rocprofiler-sdk
        ];
      };
      configure_command = ''
        IFS=', ' read -r -a architectures <<< "''${GPU_ARCHITECTURE:-gfx1200}"
        architecture_list="$(IFS=';'; printf '%s' "''${architectures[*]}")"
        if [[ -z "$architecture_list" ]]; then
          printf 'GPU_ARCHITECTURE must contain at least one target\n' >&2
          exit 2
        fi

        compiler="$HIP_PATH/bin/amdclang++"
        cmake -S . -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_HIP_COMPILER="$compiler" \
          -DCMAKE_CXX_COMPILER="$compiler" \
          -DCMAKE_HIP_ARCHITECTURES="$architecture_list" \
          -DHIP_CXX_COMPILER="$compiler" \
          -DGPU_TARGETS="$architecture_list" \
          -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" >/dev/null
        cmake --build build --parallel
      '';
      commands = {
        build = configure_command;
        run = ''
          ${configure_command}
          exec ./build/gemm "$@"
        '';
        trace = ''
          ${configure_command}
          mkdir -p build/trace
          exec rocprofv3 \
            --output-directory build/trace \
            --output-format csv \
            --kernel-trace \
            --kernel-include-regex wmma_gemm_kernel \
            -- ./build/gemm "$@" --no-profile
        '';
        att = ''
          ${configure_command}
          mkdir -p build/att
          exec rocprofv3 \
            --att \
            -d build/att \
            --kernel-include-regex wmma_gemm_kernel \
            -- ./build/gemm "$@" --no-profile
        '';
      };
      scripts = library.mapAttrs (
        name: text:
        packages.writeShellApplication {
          name = "gemm-${name}";
          runtimeInputs = command_dependencies;
          runtimeEnv = environment;
          inherit text;
        }
      ) commands;
      applications = library.mapAttrs (name: package: {
        type = "app";
        program = library.getExe package;
        meta.description = "Run gemm-${name}";
      }) scripts;
    in
    {
      apps.${system} = applications // {
        default = applications.run;
      };

      devShells.${system}.default = packages.mkShell (
        environment
        // {
          packages = command_dependencies ++ rocm_dependencies ++ builtins.attrValues scripts;
        }
      );
    };
}
