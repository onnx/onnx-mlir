# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

set(STABLEHLO_LIBS
  ChloOps
  InterpreterOps
  StablehloAssemblyFormat
  StablehloBase
  StablehloBroadcastUtils
  StablehloOps
  StablehloPasses
  StablehloReferenceNumPy
  StablehloReferenceScope
  StablehloRegister
  StablehloTypeInference
  VhloOps
  VhloTypes
  Version
  )

install(TARGETS
  ${STABLEHLO_LIBS}
  DESTINATION lib
  )

add_onnx_mlir_library(OMONNXToStableHlo
  ConvertONNXToStableHlo.cpp
  ONNXToStableHloCommon.cpp
  DialectBuilder.cpp

  Math/Clip.cpp
  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/MatMul.cpp
  Math/Reduction.cpp
  NN/Conv.cpp
  NN/ConvTranspose.cpp
  NN/Normalization.cpp
  NN/Pooling.cpp
  Tensor/ArgMax.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/Expand.cpp
  Tensor/Flatten.cpp
  Tensor/Gather.cpp
  Tensor/GatherElements.cpp
  Tensor/Identity.cpp
  Tensor/Pad.cpp
  Tensor/Reshape.cpp
  Tensor/Shape.cpp
  Tensor/Slice.cpp
  Tensor/Split.cpp
  Tensor/Squeeze.cpp
  Tensor/Tile.cpp
  Tensor/Transpose.cpp
  Tensor/Unsqueeze.cpp

  LINK_LIBS PUBLIC
  MLIRShapeDialect
  MLIRQuantDialect
  MLIRTransforms
  OMAccelerator
  OMONNXRewrite
  OMONNXOps
  OMSupport
  ${STABLEHLO_LIBS}
  )
