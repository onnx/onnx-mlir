# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(ModelLib
  ConvModel.cpp
  GemmModel.cpp
  MatMulModel.cpp
  ModelLib.hpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PRIVATE 
  CompilerUtils
  ExecutionSession
  )
