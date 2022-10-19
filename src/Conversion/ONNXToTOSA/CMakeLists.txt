# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  ONNXToTOSALegalizeUtils.cpp

  Math/Elementwise.cpp
  Tensor/Constant.cpp
  Tensor/PaddingOp.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
