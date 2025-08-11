# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_ENABLE_PYBIND)
  return()
endif()

add_onnx_mlir_library(ModelLib
  CategoryMapperModel.cpp
  ConvModel.cpp
  ElementwiseModel.cpp
  GRUModel.cpp
  GemmModel.cpp
  LSTMModel.cpp
  LeakyReluModel.cpp
  MatMulModel.cpp
  ModelLib.cpp
  RNNModel.cpp
  ScanModel.cpp
  SoftplusModel.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMCompiler
  OMCompilerUtils
  OMExecutionSession
  ${CMAKE_DL_LIBS}
  )
