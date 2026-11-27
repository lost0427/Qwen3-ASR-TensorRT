cmake_minimum_required(VERSION 3.24)
project(qwen3_asr_native LANGUAGES CXX)

option(QASR_BUILD_RUNTIME "Build the inference runtime executable (needs CUDA)" ON)
option(QASR_BUILD_TOOLS "Build the standalone engine builder (needs TensorRT only)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps" CACHE PATH "Directory with TensorRT headers and import libraries")
set(PYTHON_SITE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Lib/site-packages" CACHE PATH "Python environment used for its runtime DLLs")
set(QASR_BUILDER_ARCH "89" CACHE STRING "Compute capability whose TensorRT builder resource is copied next to build-engine (75 80 86 89 90 100 120)")

# The engine builder only needs the TensorRT parser and builder, so it builds on a
# machine without nvcc. Configure with -DQASR_BUILD_RUNTIME=OFF for that.
if(QASR_BUILD_RUNTIME)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures compiled into the decode kernels")

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(qwen3-asr-native
    src/main.cpp
    src/engine.cpp
    src/assets.cpp
    src/audio.cpp
    src/decoder.cpp
    src/aligner.cpp
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

  if(MSVC)
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
endif()

if(QASR_BUILD_TOOLS)
  # TensorRT headers include cuda_runtime_api.h, so CUDA headers are needed even
  # though build-engine has no kernels and never calls nvcc. Prefer a full
  # toolkit; the pip runtime wheel ships the flat headers and the nvcc wheel
  # ships include/crt, so those two together also work without an installer.
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(_cuda_includes ${CUDAToolkit_INCLUDE_DIRS})
  else()
    set(_cuda_includes "")
    foreach(_cuda_dir
        "${PYTHON_SITE}/nvidia/cuda_runtime/include"
        "${PYTHON_SITE}/nvidia/cuda_nvcc/include")
      if(EXISTS "${_cuda_dir}")
        list(APPEND _cuda_includes "${_cuda_dir}")
      endif()
    endforeach()
    if(NOT _cuda_includes)
      message(FATAL_ERROR
        "CUDA headers not found; install the CUDA toolkit, or install nvidia-cuda-runtime-cu12 and nvidia-cuda-nvcc-cu12")
    endif()
  endif()

  add_executable(build-engine tools/build_engine.cpp)
  target_include_directories(build-engine PRIVATE "${TENSORRT_ROOT}/include" ${_cuda_includes})
  target_link_libraries(build-engine PRIVATE
    "${TENSORRT_ROOT}/lib/nvinfer_11.lib"
    "${TENSORRT_ROOT}/lib/nvonnxparser_11.lib"
    "${TENSORRT_ROOT}/lib/nvinfer_plugin_11.lib")

  if(MSVC)
    target_compile_options(build-engine PRIVATE /utf-8 /EHsc)
  endif()

  set(_tool_dlls
    "${PYTHON_SITE}/tensorrt_libs/nvinfer_11.dll"
    "${PYTHON_SITE}/tensorrt_libs/nvonnxparser_11.dll"
    "${PYTHON_SITE}/tensorrt_libs/nvinfer_plugin_11.dll"
  )
  foreach(dll IN LISTS _tool_dlls)
    if(EXISTS "${dll}")
      add_custom_command(TARGET build-engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:build-engine>")
    endif()
  endforeach()

  # TensorRT 11 splits the builder into one resource DLL per compute capability,
  # and the builder only loads the exact one for the current GPU.
  set(_builder_resource
    "${PYTHON_SITE}/tensorrt_libs/nvinfer_builder_resource_sm${QASR_BUILDER_ARCH}_11.dll")
  if(EXISTS "${_builder_resource}")
    add_custom_command(TARGET build-engine POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_builder_resource}"
        "$<TARGET_FILE_DIR:build-engine>")
  else()
    message(WARNING
      "builder resource for sm${QASR_BUILDER_ARCH} not found; set -DQASR_BUILDER_ARCH to the compute capability of this GPU")
  endif()

  # cudart ships with the pip torch/nvidia wheels before CUDA itself is installed.
  set(_cudart_candidates
    "${PYTHON_SITE}/nvidia/cuda_runtime/bin/cudart64_12.dll"
    "${PYTHON_SITE}/torch/lib/cudart64_12.dll")
  foreach(dll IN LISTS _cudart_candidates)
    if(EXISTS "${dll}")
      add_custom_command(TARGET build-engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:build-engine>")
      break()
    endif()
  endforeach()
endif()
