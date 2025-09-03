# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_rewriter(ZHighConstPropagation)
add_onnx_mlir_library(OMZHighConstPropagation
  ZHighConstPropagation.cpp

  DEPENDS
  OMLayoutHelper
  OMONNXZHighConstPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMLayoutHelper
  OMStickify
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(ZHighLayoutPropagation)
add_onnx_mlir_library(OMZHighLayoutPropagation
  ZHighLayoutPropagation.cpp

  DEPENDS
  OMLayoutHelper
  OMONNXZHighLayoutPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(ZHighDecomposeStickUnstick)
add_onnx_mlir_library(OMZHighDecomposeStickUnstick
  ZHighDecomposeStickUnstick.cpp

  DEPENDS
  OMLayoutHelper
  OMONNXZHighDecomposeStickUnstickIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(ZHighRecomposeToStickUnstick)
add_onnx_mlir_library(OMZHighRecomposeToStickUnstick
  ZHighRecomposeToStickUnstick.cpp

  DEPENDS
  OMLayoutHelper
  OMONNXZHighRecomposeToStickUnstickIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZHighScrubDisposable
  ZHighScrubDisposablePass.cpp

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
