# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(ModelLib
  ConvModel.cpp
  GRUModel.cpp
  GemmModel.cpp
  LSTMModel.cpp
  MatMulModel.cpp
  ModelHelper.cpp
  ModelHelper.hpp
  ModelLib.hpp
  RNNModel.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PRIVATE 
  CompilerUtils
  ExecutionSession
  )
