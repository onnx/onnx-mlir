# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  ONNXToTOSALegalizeUtils.cpp

  Math/Elementwise.cpp
  Tensor/Constant.cpp
  Math/Gemm.cpp
  NN/MaxPoolSingleOut.cpp


  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
