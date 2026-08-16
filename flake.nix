{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      packages = nixpkgs.legacyPackages.${system};
      rocm_packages = packages.rocmPackages;
      runtime_library_path = packages.lib.makeLibraryPath [
        rocm_packages.clr
        rocm_packages.rocm-runtime
        rocm_packages.rocblas
        rocm_packages.rocprofiler-sdk
      ];
      cmake_prefix_path = packages.lib.concatStringsSep ";" [
        rocm_packages.clr
        rocm_packages.rocm-device-libs
        rocm_packages.rocm-comgr
        rocm_packages.rocm-runtime
        rocm_packages.rocblas
        rocm_packages.rocprofiler-sdk.dev
      ];
      shell_environment = {
        HIPCC = "${rocm_packages.clr}/bin/hipcc";
        HIP_PATH = "${rocm_packages.clr}";
        HIP_DEVICE_LIB_PATH = "${rocm_packages.rocm-device-libs}/amdgcn/bitcode";
        HSA_PATH = "${rocm_packages.rocm-runtime}";
        ROCBLAS_PATH = "${rocm_packages.rocblas}";
        ROCWMMA_PATH = "${rocm_packages.rocwmma}";
        ROCPROFILER_SDK_PATH = "${rocm_packages.rocprofiler-sdk}";
        ROCPROFILER_SDK_INCLUDE_PATH = "${rocm_packages.rocprofiler-sdk.dev}/include";
        CMAKE_PREFIX_PATH = cmake_prefix_path;
        LD_LIBRARY_PATH = runtime_library_path;
      };
      environment_script = packages.lib.concatStringsSep "\n" (
        packages.lib.mapAttrsToList (
          name: value: "export ${name}=${packages.lib.escapeShellArg value}"
        ) shell_environment
      );
      application_runtime_inputs = [
        packages.cmake
        packages.ninja
        rocm_packages.rocprofiler-sdk
      ];
      configure_command = ''
        architecture_text="''${GPU_ARCHITECTURE:-gfx1200}"
        IFS=', ' read -ra architectures <<< "$architecture_text"
        architecture_list=""
        for architecture in "''${architectures[@]}"; do
          if [[ -n "$architecture" ]]; then
            if [[ -n "$architecture_list" ]]; then
              architecture_list+=";"
            fi
            architecture_list+="$architecture"
          fi
        done
        if [[ -z "$architecture_list" ]]; then
          printf 'GPU_ARCHITECTURE must contain at least one target\n' >&2
          exit 2
        fi

        cmake -S . -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_HIP_COMPILER="$HIP_PATH/bin/amdclang++" \
          -DCMAKE_CXX_COMPILER="$HIP_PATH/bin/amdclang++" \
          -DCMAKE_HIP_ARCHITECTURES="$architecture_list" \
          -DHIP_CXX_COMPILER="$HIP_PATH/bin/amdclang++" \
          -DGPU_TARGETS="$architecture_list" \
          -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" >/dev/null
        cmake --build build --parallel
      '';
      application_commands = {
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
            --kernel-include-regex tiled_gemm_kernel \
            -- ./build/gemm "$@" --no-profile
        '';
        att = ''
          ${configure_command}
          mkdir -p build/att
          exec rocprofv3 \
            --att \
            -d build/att \
            --kernel-include-regex tiled_gemm_kernel \
            -- ./build/gemm "$@" --no-profile
        '';
      };
      create_shell_application = name: command: packages.writeShellApplication {
        name = "gemm-${name}";
        runtimeInputs = application_runtime_inputs;
        text = ''
          ${environment_script}
          ${command}
        '';
      };
      shell_applications = packages.lib.mapAttrs create_shell_application application_commands;
      create_app = name: package: {
        type = "app";
        program = packages.lib.getExe package;
        meta.description = "Run gemm-${name}";
      };
    in
    {
      apps.${system} = packages.lib.mapAttrs create_app shell_applications // {
        default = create_app "run" shell_applications.run;
      };

      devShells.${system}.default = packages.mkShell (shell_environment // {
        packages = [
          packages.cmake
          packages.ninja
          rocm_packages.clr
          rocm_packages.rocwmma
          rocm_packages.rocblas
          rocm_packages.rocprofiler-sdk
        ] ++ builtins.attrValues shell_applications;
      });
    };
}
