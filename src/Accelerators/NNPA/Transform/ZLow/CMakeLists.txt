# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  MLIRViewLikeInterface
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZLowDummyOpForMultiDeref
  ZLowDummyOpForMultiDerefPass.cpp

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZLowOps
  )
