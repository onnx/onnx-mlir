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

# This is a workaround until this change is commited into mlir-hlo and it is merged into onnx-mlir:
# https://github.com/tensorflow/tensorflow/pull/56844
# In short, the MhloDialect has some dependencies that are needed for the shared libs build, but are
# currently missing.
target_link_libraries(MhloDialect PUBLIC
  MLIRQuantDialect
  )
target_include_directories(MhloDialect PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/third_party/mlir-hlo/include
  ${ONNX_MLIR_BIN_ROOT}/third_party/mlir-hlo/include
  )
