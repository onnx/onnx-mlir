# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMMlirDialects
  AttributesHelper.cpp
  DialectBuilder.cpp
  IndexExpr.cpp
  IndexExprDetail.cpp
  ResourcePool.cpp

  DEPENDS
  OMKrnlIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  MLIRMathDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  )
