# SPDX-License-Identifier: Apache-2.0

# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.13.4)

project(onnx-mlir)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(ONNX_MLIR_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNX_MLIR_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Lit test suite requires at least python 3.6
set(LLVM_MINIMUM_PYTHON_VERSION 3.6)
find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)

include(CTest)
include(MLIR.cmake)

if (MSVC)
  # Disable warnings that prevent building of dependencies
  # like onnx
  set(CMAKE_CXX_FLAGS "-wd4244 -wd4267 -wd4530 -wd4624 /bigobj")
endif()

# The onnx build on windows requires the provision of either static or shared libs for protobuf
# and to achieve that there are several different flags that need to be set in protobuf, onnx
# as well as cmake (for find_package). At the very minimum ONNX_USE_PROTOBUF_SHARED_LIBS and
# Protobuf_USE_STATIC_LIBS need to be opposites of each other, but onnx does not currently set them
# up that way (onnx doesn't set Protobuf_USE_STATIC_LIBS at all and cmake interprets that as setting
# Protobuf_USE_STATIC_LIBS to OFF). If/when onnx is updated to correctly set up both variables,
# this can be removed (see https://github.com/onnx/onnx/issues/3345).
if (NOT DEFINED ONNX_USE_PROTOBUF_SHARED_LIBS AND NOT DEFINED Protobuf_USE_STATIC_LIBS)
  set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF)
  set(Protobuf_USE_STATIC_LIBS ON)
elseif (NOT DEFINED Protobuf_USE_STATIC_LIBS)
  if (ONNX_USE_PROTOBUF_SHARED_LIBS)
    set(Protobuf_USE_STATIC_LIBS OFF)
  else()
    set(Protobuf_USE_STATIC_LIBS ON)
  endif()
elseif (NOT DEFINED ONNX_USE_PROTOBUF_SHARED_LIBS)
  if (Protobuf_USE_STATIC_LIBS)
    set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF)
  else()
    set(ONNX_USE_PROTOBUF_SHARED_LIBS ON)
  endif()
elseif ((ONNX_USE_PROTOBUF_SHARED_LIBS AND Protobuf_USE_STATIC_LIBS)
         OR (NOT ONNX_USE_PROTOBUF_SHARED_LIBS AND NOT Protobuf_USE_STATIC_LIBS))
  message(FATAL_ERROR
    "ONNX_USE_PROTOBUF_SHARED_LIBS and Protobuf_USE_STATIC_LIBS must be opposites of each other.")
endif()
add_subdirectory(third_party/onnx)
# Use googletest from llvm instead.
#add_subdirectory(third_party/googletest)
SET(BENCHMARK_ENABLE_GTEST_TESTS OFF)
add_subdirectory(third_party/benchmark)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/variant)
add_subdirectory(third_party/rapidcheck)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(utils)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(test)
