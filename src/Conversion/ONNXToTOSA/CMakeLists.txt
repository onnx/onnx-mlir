# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  ONNXToTOSALegalizeUtils.cpp
  ONNXToTOSACommon.cpp

  Math/Elementwise.cpp
  Math/Conv2D.cpp
  Math/Softmax.cpp
  Math/Gemm.cpp
  Math/ReduceMean.cpp
  Tensor/Reshape.cpp
  Tensor/Constant.cpp
  Tensor/PaddingOp.cpp
  Tensor/Flatten.cpp
  NN/MaxPoolSingleOut.cpp


  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
