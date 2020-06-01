# TODO: remove for release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type")
endif()

# Require 3.3 and set policy CMP0057 for IN_LIST operator support
cmake_minimum_required(VERSION 3.3)
cmake_policy(SET CMP0057 NEW)
project(onnx-mlir)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(ONNX_MLIR_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNX_MLIR_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(MLIR.cmake)

if (MSVC)
  # Disable warnings that prevent building of dependencies
  # like onnx
  set(CMAKE_CXX_FLAGS "-wd4244 -wd4267 -wd4530 -wd4624")
endif()
add_subdirectory(third_party/onnx)
add_subdirectory(third_party/benchmark)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/variant)

set(CMAKE_CXX_STANDARD 14)

if ($ENV{EXCLUDE_ONNX_ML})
  set(INCLUDE_ONNX_ML FALSE)
else()
  set(INCLUDE_ONNX_ML TRUE)
endif()

message(STATUS "INCLUDE_ONNX_ML Dialect " ${INCLUDE_ONNX_ML})
message(STATUS "CMake objcopy" ${CMAKE_OBJCOPY})

add_subdirectory(utils)
add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(test)
