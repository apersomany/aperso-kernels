cmake_minimum_required(VERSION 3.25)

project(aperso_kernels LANGUAGES HIP CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(hip CONFIG REQUIRED)
find_package(hipblaslt CONFIG REQUIRED)

set(sources
  hip/kernels/gemm.cpp
  hip/drivers/gemm.cpp
  hip/main.cpp
)

set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)

add_executable(gemm ${sources})

target_include_directories(gemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip
  $ENV{HIP_PATH}/include
)

target_compile_options(gemm PRIVATE
  "-I$ENV{HIP_PATH}/include"
  "--rocm-device-lib-path=$ENV{HIP_DEVICE_LIB_PATH}"
)

foreach(directory IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
  target_compile_options(gemm PRIVATE "-isystem${directory}")
endforeach()

target_link_libraries(gemm PRIVATE
  hip::host
  hip::device
  roc::hipblaslt
)
