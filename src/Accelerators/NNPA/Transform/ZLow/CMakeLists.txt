# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  MLIRViewLikeInterface
  OMZLowOps
  )
