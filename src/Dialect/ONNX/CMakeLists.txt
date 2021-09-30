# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the ONNXOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(ONNXOps)
add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

add_onnx_mlir_rewriter(Rewrite)

add_onnx_mlir_library(OMONNXOps
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  ONNXShapeHelper.cpp
  Rewrite.cpp
  IndexExpr.cpp
  IndexExprDetail.cpp
  MLIRDialectBuilder.cpp

  DEPENDS
  OMHasOnnxSubgraphOpInterfaceIncGen
  OMONNXOpsIncGen
  OMONNXRewriteIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRMath
  MLIRAffine
  )
