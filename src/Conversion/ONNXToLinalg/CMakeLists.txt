# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.
add_onnx_mlir_library(OMONNXToLinalg
  ConvertONNXToLinagl.cpp
  ONNXToLinaglCommon.cpp
  Math/MatMul.cpp

  LINK_LIBS PUBLIC
  Accelerator
  OMConstPropHelper
  OMONNXOps
  OMSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  )
