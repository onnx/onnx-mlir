# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  ONNXToTOSALegalizeUtils.cpp
  ONNXToTOSACommon.cpp
  DialectBuilder.cpp

  Math/Elementwise.cpp
  Math/Conv2D.cpp
  Math/Softmax.cpp
  Math/Gemm.cpp
  Math/ReduceMean.cpp
  Tensor/Concat.cpp
  Tensor/Gather.cpp
  Tensor/Reshape.cpp
  Tensor/Resize.cpp
  Tensor/Constant.cpp
  Tensor/PaddingOp.cpp
  Tensor/Flatten.cpp
  Tensor/Slice.cpp
  Tensor/Transpose.cpp
  Tensor/Unsqueeze.cpp
  NN/MaxPoolSingleOut.cpp
  NN/AveragePool.cpp
  NN/QuantizeLinear.cpp
  NN/DequantizeLinear.cpp
  Flow/EntryPoint.cpp


  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
