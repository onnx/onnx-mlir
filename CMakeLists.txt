# SPDX-License-Identifier: Apache-2.0

# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.13.4)

project(onnx-mlir)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/bin)

set(ONNX_MLIR_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(ONNX_MLIR_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR})

set(ONNX_MLIR_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(ONNX_MLIR_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

include(CTest)
include(MLIR.cmake)

# MLIR.cmake calls find_package(MLIR) which sets LLVM_MINIMUM_PYTHON_VERSION
find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)

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
# We are setting CMP0077 to NEW here, so that onnx does not override our regular variables for
# protobuf when the project is included below.
if (POLICY CMP0077)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif (POLICY CMP0077)
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

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/variant)
add_subdirectory(third_party/rapidcheck)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(utils)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(docs)
add_subdirectory(test)
