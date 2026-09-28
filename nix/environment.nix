{ packages, rocm }:

{
  inputs = [
    packages.cmake
    packages.ninja
    packages.python3
    rocm
  ];

  variables = {
    ROCM_PATH = "${rocm}";
    HIP_PATH = "${rocm}";
    HIP_PLATFORM = "amd";
    HIPCXX = "${rocm}/bin/therock-hip-clang++";
    CXX = "${rocm}/bin/therock-hip-clang++";
    CMAKE_PREFIX_PATH = "${rocm}";
    PYTHONPATH = "${rocm}/lib/python3/site-packages";
    # CMake omits the SDK from the executable runpath; this also overrides any ambient ROCm
    LD_LIBRARY_PATH = "${rocm}/lib";
    # Read by CMakeLists.txt
    HIP_DEVICE_LIB_PATH = "${rocm}/lib/llvm/amdgcn/bitcode";
  };
}
