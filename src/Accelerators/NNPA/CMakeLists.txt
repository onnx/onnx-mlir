# SPDX-License-Identifier: Apache-2.0

# When an accelerator is specified to be built in ONNX_MLIR_ACCELERATORS,
# if for whatever reason we do not want to build it on a specific arch,
# XX_ENABLED variable can be used to control that. For example, if we
# only want to compile NNPA on s390x, we can do:
#
# if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
#   set(NNPA_ENABLED 1 BOOL PARENT_SCOPE)
#   ...
# endif()

set(NNPA_ENABLED 1 BOOL PARENT_SCOPE)

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
setup_zdnn(v1.1.2)

# Note: consider to use a compile flag instead.
option(ZDNNX_DEBUG "Enable ZDNNX debug information" OFF)
if (ZDNNX_DEBUG)
  add_compile_definitions(ZDNNX_DEBUG)
endif()

# Note: consider to use a compile flag instead.
option(ZDNNX_WITH_OMP "Enable ZDNNX with OMP" OFF)
if (ZDNNX_WITH_OMP)
  add_compile_definitions(ZDNNX_WITH_OMP)
  # If LLVM is built with OpenMP enabled, use LLVM's OpenMP.
  # Otherwise, use GCC's OpenMP.
  if (OMP_SUPPORTED)
    set(ZDNNX_WITH_OMP_LLVM 1)
    add_compile_definitions(ZDNNX_WITH_OMP_LLVM)
    message(STATUS "zdnnx with OpenMP        : ON (LLVM's OpenMP)")
  else()
    message(STATUS "zdnnx with OpenMP        : ON (GCC's OpenMP)")
  endif()
else()
  message(STATUS "zdnnx with OpenMP        : OFF")
endif()

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Support)
add_subdirectory(Transform)
add_subdirectory(Compiler)
add_subdirectory(Runtime)

add_onnx_mlir_library(OMNNPAAccel
  NNPAAccelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  libzdnn
  RuntimeNNPA

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_ONNX_MLIR_SRC_ROOT}
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  ${NNPA_INCLUDE_PATH}

  LINK_LIBS PUBLIC
  onnx
  OMNNPACompilerUtils
  OMAccelerator
  OMZHighOps
  OMZHighToZLow
  OMZLowOps
  OMZLowToLLVM
  )
