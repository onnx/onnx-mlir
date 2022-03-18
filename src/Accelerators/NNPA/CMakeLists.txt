# SPDX-License-Identifier: Apache-2.0

set(NNPA_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NNPA_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(NNPA_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(NNPA_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

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

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Support)
add_subdirectory(Transform)
add_subdirectory(Compiler)

add_onnx_mlir_library(NNPAAccelerator STATIC
  NNPAAccelerator.cpp

  DEPENDS
  Accelerator
  NNPACompilerUtils
  onnx
  
  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_ONNX_MLIR_SRC_ROOT}
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  
  LINK_LIBS PRIVATE
  onnx
  Accelerator
  NNPACompilerUtils
  OMZHighOps
  OMZLowOps
  )

 # Force the system linker to keep the all definitions in the NNPAAccelerator library.
 # This is necessary in order to overwrite weak definitions in the base Accelerator library 
 # with strong definitions provided by the NNPAAccelerator library when it is built.
 target_link_options(NNPAAccelerator INTERFACE 
 "$<$<CXX_COMPILER_ID:AppleClang>:SHELL:LINKER:-force_load $<TARGET_LINKER_FILE:NNPAAccelerator>>"
 "$<$<CXX_COMPILER_ID:GNU>:SHELL:LINKER:--whole-archive $<TARGET_LINKER_FILE:NNPAAccelerator> LINKER:--no-whole-archive>"
 "$<$<CXX_COMPILER_ID:MSVC>:SHELL:LINKER:-WHOLEARCHIVE:$<TARGET_LINKER_FILE:NNPAAccelerator>>"
)
