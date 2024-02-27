# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  DialectBuilder.cpp
  ONNXToTOSALegalizeUtils.cpp

  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/Softmax.cpp
  Math/ReduceMean.cpp
  Math/Conv2D.cpp
  NN/MaxPoolSingleOut.cpp
  NN/AveragePool.cpp
  Tensor/Constant.cpp
  Tensor/Reshape.cpp
  Tensor/Resize.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
