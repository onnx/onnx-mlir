# SPDX-License-Identifier: Apache-2.0

set(NNPA_ENABLED 1 CACHE BOOL "flag indicating NNPA enabled")

set(NNPA_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NNPA_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(NNPA_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(NNPA_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(NNPA_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set(NNPA_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(NNPA_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

# The path to find the zDNN library.
if (DEFINED ENV{ZDNN_LIBRARY_DIR} AND EXISTS $ENV{ZDNN_LIBRARY_DIR})
  message(DEBUG "ZDNN_LIBRARY_DIR        : " $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_DIR $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_ENABLED 1)
else()
  message(DEBUG "ZDNN_LIBRARY_DIR not found. zDNN-related tests will be turned off.")
  set(ZDNN_LIBRARY_ENABLED 0)
endif()

include(zdnn.cmake)
setup_zdnn(v1.0.0)

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Support)
add_subdirectory(Transform)
add_subdirectory(Compiler)

add_onnx_mlir_library(NNPAAccel
  NNPAAccelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_ONNX_MLIR_SRC_ROOT}
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  
  LINK_LIBS PUBLIC
  onnx
  NNPACompilerUtils
  Accelerator
  OMZHighOps
  OMZHighToZLow
  OMZLowOps
  OMZLowToLLVM
  )
