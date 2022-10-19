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
  Math/Reduction.cpp
  NN/Normalization.cpp
  NN/Pooling.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/Reshape.cpp

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
