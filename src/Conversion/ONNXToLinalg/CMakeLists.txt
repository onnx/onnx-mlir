# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ONNXToLinalg)
add_public_tablegen_target(OMONNXToLinalgPassIncGen)

add_onnx_mlir_library(OMONNXToLinalg
  ConvertONNXToLinalg.cpp
  Math/MatMul.cpp
  NN/Conv.cpp
  ONNXBufferizableOpInterface.cpp

  DEPENDS
  OMONNXToLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  OMONNXOps
  OMShapeInferenceOpInterface
  OMOptionUtils
  )

