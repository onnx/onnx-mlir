# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToLinalg
  ConvertONNXToLinalg.cpp
  Math/MatMul.cpp

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  OMONNXOps
  OMShapeInferenceOpInterface
  )

