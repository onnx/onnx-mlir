# SPDX-License-Identifier: Apache-2.0

include_directories(${PROJECT_BINARY_DIR}/src)
add_onnx_mlir_library(OnnxMlirCompiler SHARED
  OnnxMlirCompiler.cpp

  EXCLUDE_FROM_OM_LIBS
  
  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  MainUtils
  )
