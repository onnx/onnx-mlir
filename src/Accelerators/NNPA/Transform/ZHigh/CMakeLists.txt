# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Stickify)

add_onnx_mlir_rewriter(ZHighConstPropagation)
add_onnx_mlir_library(OMZHighConstPropagation
  ZHighConstPropagation.cpp

  DEPENDS
  OMONNXZHighConstPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMLayoutHelper
  OMStickify
  OMZHighOps
  OMONNXOps
  )

add_onnx_mlir_rewriter(ZHighLayoutPropagation)
add_onnx_mlir_library(OMZHighLayoutPropagation
  ZHighLayoutPropagation.cpp

  DEPENDS
  OMONNXZHighLayoutPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps
  )

add_onnx_mlir_library(OMInstrumentZHigh
  InstrumentZHighPass.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
  MLIRPass
  OMNNPASupport
  )
