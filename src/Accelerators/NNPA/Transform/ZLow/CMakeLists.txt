# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp

  DEPENDS
  OMLayoutHelper

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

add_onnx_mlir_library(OMZLowInsertDLF16Conversion
  ZLowInsertDLF16Conversion.cpp

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZLowLoopSinking
  ZLowLoopSinking.cpp

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZLowFusion
  ZLowFusion.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransformUtils
  OMZLowOps
  OMONNXToKrnl # this is due to calling insertAllocAndDeallocSimple

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
