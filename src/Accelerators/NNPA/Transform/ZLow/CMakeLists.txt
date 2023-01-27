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

add_onnx_mlir_library(OMZLowInsertDLF16Conversion
  ZLowInsertDLF16Conversion.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  OMKrnlOps
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
