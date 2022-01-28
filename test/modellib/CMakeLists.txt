# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(ModelLib
  ConvModel.cpp

  LINK_LIBS PRIVATE 
  CompilerUtils
  ExecutionSession
  )
