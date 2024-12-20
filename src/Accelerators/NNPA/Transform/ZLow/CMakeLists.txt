# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp
  ZLowStickExpansion.cpp

  DEPENDS
  OMLayoutHelper

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  MLIRViewLikeInterface
  OMONNXToKrnl
  OMZHighToZLow
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

