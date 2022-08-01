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
