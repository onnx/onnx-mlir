# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  DialectBuilder.cpp
  ONNXToTOSALegalizeUtils.cpp

  Math/Elementwise.cpp
  Math/Softmax.cpp
  NN/MaxPoolSingleOut.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
