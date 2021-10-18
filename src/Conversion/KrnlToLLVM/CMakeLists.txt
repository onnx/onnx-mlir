# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToLLVM
  KrnlToLLVM.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToStandard
  MLIRShapeToStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  onnx
  )
