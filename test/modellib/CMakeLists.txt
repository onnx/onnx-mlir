# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(ModelLib
  ConvModel.cpp
  GemmModel.cpp
  GRUModel.cpp
  MatMulModel.cpp
  ModelLib.hpp
  ModelHelper.cpp
  ModelHelper.hpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PRIVATE 
  CompilerUtils
  ExecutionSession
  )
