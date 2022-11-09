# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

install(TARGETS 
  MhloDialect
  MLIRMhloUtils
  HloOpsCommon
  )

add_onnx_mlir_library(OMONNXToMhlo
  ConvertONNXToMhlo.cpp
  ONNXToMhloCommon.cpp

  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/MatMul.cpp
  Math/Reduction.cpp
  NN/Conv.cpp
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
  Accelerator
  MhloDialect
  MLIRShapeDialect
  MLIRQuantDialect
  MLIRTransforms
  OMConstPropHelper
  OMONNXRewrite
  OMONNXOps
  OMSupport
  )

# This is a workaround because MhloDialect has a dependency that is needed for
# the shared libs build, but is currently missing.
target_link_libraries(MhloDialect PUBLIC
  MLIRSparseTensorDialect
  )
