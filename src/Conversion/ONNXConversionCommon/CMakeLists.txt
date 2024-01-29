# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXConversionCommon
  RNN/RNNBase.cpp
  RNN/LSTM.cpp

  LINK_LIBS PUBLIC

  OMONNXOps
  OMSupport
  MLIRTransforms
)
