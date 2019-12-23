# TODO: remove for release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type")
endif()

# Require 3.3 and set policy CMP0057 for IN_LIST operator support
cmake_minimum_required(VERSION 3.3)
cmake_policy(SET CMP0057 NEW)
project(onnf)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(ONNF_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNF_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(MLIR.cmake)

add_subdirectory(third_party/onnx)
add_subdirectory(third_party/benchmark)
add_subdirectory(third_party/pybind11)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(src)

add_subdirectory(test)

