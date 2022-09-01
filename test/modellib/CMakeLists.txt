# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(ModelLib
  CategoryMapperModel.cpp
  ConvModel.cpp
  GRUModel.cpp
  GemmModel.cpp
  LSTMModel.cpp
  LeakyReluModel.cpp
  MatMulModel.cpp
  ModelLib.cpp
  RNNModel.cpp
  ScanModel.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OnnxMlirCompiler
  CompilerUtils
  ExecutionSession
  ${CMAKE_DL_LIBS}
  )
