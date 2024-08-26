# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

set(STABLEHLO_LIBS
  ChloOps
  InterpreterOps
  StablehloAssemblyFormat
  StablehloBase
  StablehloBroadcastUtils
  StablehloLinalgTransforms
  StablehloOps
  StablehloPasses
  StablehloReferenceAxes
  StablehloReferenceConfiguration
  StablehloReferenceElement
  StablehloReferenceErrors
  StablehloReferenceIndex
  StablehloReferenceNumPy
  StablehloReferenceOps
  StablehloReferenceProcess
  StablehloReferenceProcessGrid
  StablehloReferenceScope
  StablehloReferenceTensor
  StablehloReferenceToken
  StablehloReferenceTypes
  StablehloReferenceValue
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

add_onnx_mlir_library(OMONNXToStablehlo
  ConvertONNXToStablehlo.cpp
  ONNXToStablehloCommon.cpp
  DialectBuilder.cpp

  Math/Clip.cpp
  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/MatMul.cpp
  Math/Reduction.cpp
  Math/Softmax.cpp
  NN/Conv.cpp
  NN/ConvTranspose.cpp
  NN/Normalization.cpp
  NN/Pooling.cpp
  RNN/LSTM.cpp
  RNN/RNNBase.cpp
  Tensor/ArgMax.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/DepthToSpace.cpp
  Tensor/Dim.cpp
  Tensor/Expand.cpp
  Tensor/Flatten.cpp
  Tensor/Gather.cpp
  Tensor/GatherElements.cpp
  Tensor/Identity.cpp
  Tensor/OneHot.cpp
  Tensor/Pad.cpp
  Tensor/Reshape.cpp
  Tensor/ScatterND.cpp
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
  OMONNXConversionCommon
  OMONNXRewrite
  OMONNXOps
  OMSupport
  ${STABLEHLO_LIBS}
  )
