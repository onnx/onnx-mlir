# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  Math/Elementwise.cpp
  Tensor/Constant.cpp
  NN/MaxPoolSingleOut.cpp
  Math/Gemm.cpp


  ConvertONNXToTOSA.cpp
  ONNXToTosaUtils.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
