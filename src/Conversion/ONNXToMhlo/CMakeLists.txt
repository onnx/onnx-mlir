# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

install(TARGETS 
  MhloDialect 
  MhloInferShapeEqualityOpInterface
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

  DEPENDS
  MhloDialect

  LINK_LIBS PUBLIC
  Accelerator
  MhloDialect
  MLIRShape
  MLIRTransforms
  OMConstPropHelper
  OMONNXRewrite
  OMONNXOps
  OMSupport
  )

target_include_directories(OMONNXToMhlo PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/third_party/mlir-hlo/include
  ${ONNX_MLIR_BIN_ROOT}/third_party/mlir-hlo/include
  )
