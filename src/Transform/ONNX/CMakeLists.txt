# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMElideConstants
  ElideConstants.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMConstPropHelper
  ConstPropHelper.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  )

add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)

add_onnx_mlir_library(OMONNXRewrite
  Decompose.cpp
  ConstProp.cpp

  DEPENDS
  OMONNXDecomposeIncGen
  OMONNXConstPropIncGen

  LINK_LIBS PUBLIC
  OMConstPropHelper
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMShapeInference
  ShapeInferencePass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  MLIRPass
  )

add_onnx_mlir_library(OMInstrumentONNX
  InstrumentONNXPass.cpp

  LINK_LIBS PUBLIC
  CompilerOptions
  OMONNXOps
  OMKrnlOps
  MLIRPass
  )

add_onnx_mlir_library(OMOpTransform
  ONNXOpTransformPass.cpp

  LINK_LIBS PUBLIC
  CompilerOptions
  OMONNXOps
  MLIRPass
  OMONNXRewrite
  OMShapeInference
  MLIRTransforms
  )

add_onnx_mlir_library(OMONNXPreKrnlVerifyONNX
  ONNXPreKrnlVerifyPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  OMSupport
  )
