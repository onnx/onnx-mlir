# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  MLIRViewLikeInterface
  OMZLowOps
  )

add_onnx_mlir_library(OMZLowDummyOpForMultiDeref
  ZLowDummyOpForMultiDerefPass.cpp

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZLowOps
  )
