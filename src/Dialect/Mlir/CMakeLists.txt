# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMMlirDialects
  IndexExpr.cpp
  IndexExprDetail.cpp
  DialectBuilder.cpp

  LINK_LIBS PUBLIC
  MLIRMath
  MLIRAffine
  MLIRSCF
  MLIRVector
  )
