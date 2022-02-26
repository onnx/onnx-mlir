# SPDX-License-Identifier: Apache-2.0

set(DLC_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DLC_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(DLC_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(DLC_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(DLC_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(DLC_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

# The path to find the zDNN library.
if (DEFINED ENV{ZDNN_LIBRARY_DIR} AND EXISTS $ENV{ZDNN_LIBRARY_DIR})
  message(DEBUG "ZDNN_LIBRARY_DIR        : " $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_DIR $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_ENABLED 1)
else()
  message(DEBUG "ZDNN_LIBRARY_DIR not found. zDNN-related tests will be turned off.")
  set(ZDNN_LIBRARY_ENABLED 0)
endif()

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Support)
add_subdirectory(Transform)
add_subdirectory(Compiler)

add_onnx_mlir_library(NNPAAccel
  NNPAAccelerator.cpp
  ../Accelerator.cpp

  DEPENDS
  DLCompilerUtils
  onnx
  
  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  # /workdir/onnx-mlir/third_party/onnx

  LINK_LIBS PUBLIC
  onnx
  DLCompilerUtils
  OMZHighOps
  OMZLowOps
  )

