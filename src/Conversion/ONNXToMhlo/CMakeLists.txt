# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.
add_onnx_mlir_library(OMONNXToMhlo
  ConvertONNXToMhlo.cpp

  Math/Elementwise.cpp

  DEPENDS
  MhloDialect

  LINK_LIBS PUBLIC
  Accelerator
  OMConstPropHelper
  OMONNXOps
  OMSupport
  MhloDialect
  MLIRTransforms
  )

  target_include_directories(OMONNXToMhlo PUBLIC 
  ${CMAKE_SOURCE_DIR}/third_party/mlir-hlo/include
  ${CMAKE_BINARY_DIR}/third_party/mlir-hlo/include
  )
