# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTOSA
  ConvertONNXToTOSA.cpp
  DialectBuilder.cpp
  ONNXToTOSALegalizeUtils.cpp
  ONNXToTOSACommon.cpp

  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/Softmax.cpp
  Math/ReduceMean.cpp
  Math/Conv2D.cpp
  Math/Softmax.cpp
  Math/Gemm.cpp
  Math/ReduceMean.cpp
  NN/MaxPoolSingleOut.cpp
  NN/AveragePool.cpp
  NN/QuantizeLinear.cpp
  NN/DequantizeLinear.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/Flatten.cpp
  Tensor/Gather.cpp
  Tensor/PaddingOp.cpp
  Tensor/Reshape.cpp
  Tensor/Resize.cpp
  Tensor/Slice.cpp
  Tensor/Squeeze.cpp
  Tensor/Tile.cpp
  Tensor/Transpose.cpp
  Flow/EntryPoint.cpp


  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTosaDialect
  MLIRTransforms
  )
