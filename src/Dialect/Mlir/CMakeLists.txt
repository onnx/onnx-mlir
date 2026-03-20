# SPDX-License-Identifier: Apache-2.0

set(OMMlirDialects_EXTRA_DEPS)
if (ONNX_MLIR_ENABLE_KRNL)
  list(APPEND OMMlirDialects_EXTRA_DEPS OMKrnlIncGen)
endif()

add_onnx_mlir_library(OMMlirDialects
  IndexExpr.cpp
  IndexExprDetail.cpp
  IndexExprBuilder.cpp
  DialectBuilder.cpp
  VectorMachineSupport.cpp

  DEPENDS
  ${OMMlirDialects_EXTRA_DEPS}
  OMSpecializedKernelOpInterface
  OMCompilerOptions

  LINK_LIBS PUBLIC
  # OMCompilerOptions
  MLIRMathDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRLLVMIRTransforms
  MLIRShapeDialect
  )
