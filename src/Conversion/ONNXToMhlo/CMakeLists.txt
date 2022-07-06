# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.
add_onnx_mlir_library(OMONNXToMhlo
  ConvertONNXToMhlo.cpp

  Math/Elementwise.cpp
  Math/Gemm.cpp
  Math/Softmax.cpp
  NN/Pooling.cpp
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
  OMONNXOps
  OMSupport
  )

  target_include_directories(OMONNXToMhlo PUBLIC 
  ${CMAKE_SOURCE_DIR}/third_party/mlir-hlo/include
  ${CMAKE_BINARY_DIR}/third_party/mlir-hlo/include
  )
