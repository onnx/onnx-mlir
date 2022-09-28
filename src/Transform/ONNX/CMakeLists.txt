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
  ConstProp.cpp
  ConvOpt.cpp
  Decompose.cpp
  DecomposeEinsum.cpp

  DEPENDS
  OMONNXOps
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
  MLIRFuncDialect
  MLIRPass
  )

add_onnx_mlir_library(OMInstrumentONNX
  InstrumentONNXPass.cpp
  InstrumentONNXSignaturePass.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  OMKrnlOps
  MLIRPass
  )

add_onnx_mlir_library(OMOpTransform
  ONNXOpTransformPass.cpp

  LINK_LIBS PUBLIC
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

add_onnx_mlir_library(OMONNXSimplifyShapeRelatedOps
  SimplifyShapeRelatedOps.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXRewrite
  OMShapeInference
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  )
