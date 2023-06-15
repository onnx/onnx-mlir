# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

set(MHLO_LIBS
  ChloOps
  HloOpsCommon
  MLIRMhloUtils
  MhloDialect
  MhloRegisterDialects
  StablehloAssemblyFormat
  StablehloBase
  StablehloBroadcastUtils
  StablehloOps
  StablehloRegister
  StablehloTypeInference
  VhloOps
  VhloTypes
  Version
  )

install(TARGETS
  ${MHLO_LIBS}
  DESTINATION lib
  )

add_onnx_mlir_library(OMONNXToMhlo
  ConvertONNXToMhlo.cpp
  ONNXToMhloCommon.cpp
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
  Tensor/Identity.cpp
  Tensor/Reshape.cpp
  Tensor/Shape.cpp
  Tensor/Slice.cpp
  Tensor/Split.cpp
  Tensor/Squeeze.cpp
  Tensor/Tile.cpp
  Tensor/Transpose.cpp
  Tensor/Unsqueeze.cpp

  LINK_LIBS PUBLIC
  OMAccelerator
  ${MHLO_LIBS}
  MLIRShapeDialect
  MLIRQuantDialect
  MLIRTransforms
  OMONNXRewrite
  OMONNXOps
  OMSupport
  )

# This is a workaround because MhloDialect has a dependency that is needed for
# the shared libs build, but is currently missing.
target_link_libraries(MhloDialect PUBLIC
  StablehloAssemblyFormat
  StablehloTypeInference
  )

target_link_libraries(StablehloTypeInference PUBLIC
  StablehloBase
  )

target_link_libraries(StablehloTypeInference PUBLIC
  StablehloBase
  )
