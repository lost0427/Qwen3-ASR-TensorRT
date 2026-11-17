cmake_minimum_required(VERSION 3.24)
project(qwen3_asr_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps" CACHE PATH "Directory with TensorRT headers and nvinfer_11.lib")
set(PYTHON_SITE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Lib/site-packages" CACHE PATH "Python environment used for its runtime DLLs")

find_package(CUDAToolkit REQUIRED)

add_executable(qwen3-asr-native
  src/main.cpp
  src/engine.cpp
  src/assets.cpp
  src/audio.cpp
  src/decoder.cpp
  src/ops.cu
  src/mel_cpu.cpp
  src/mel_device.cpp
  src/mel_gpu.cu
src/prefill_copy.cu
  ../csrc/w4a16_kernel.cu
  ../csrc/decode_attention.cu
  ../csrc/decode_fusions.cu
)

target_include_directories(qwen3-asr-native PRIVATE include "${TENSORRT_ROOT}/include")
target_link_libraries(qwen3-asr-native PRIVATE "${TENSORRT_ROOT}/lib/nvinfer_11.lib" CUDA::cudart)

if (MSVC)
  target_compile_options(qwen3-asr-native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
endif()

# TensorRT, libsndfile and cudart have to sit next to the executable.
set(_runtime_dlls
  "${PYTHON_SITE}/tensorrt_libs/nvinfer_11.dll"
  "${PYTHON_SITE}/tensorrt_libs/nvinfer_plugin_11.dll"
  "${PYTHON_SITE}/_soundfile_data/libsndfile_x64.dll"
)
foreach(dll IN LISTS _runtime_dlls)
  add_custom_command(TARGET qwen3-asr-native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:qwen3-asr-native>")
endforeach()
add_custom_command(TARGET qwen3-asr-native POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDAToolkit_BIN_DIR}/cudart64_12.dll" "$<TARGET_FILE_DIR:qwen3-asr-native>")
