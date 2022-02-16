# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToLLVM
  KrnlToLLVM.cpp
  RuntimeAPI.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToStandard
  MLIRShapeToStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  onnx
  )
