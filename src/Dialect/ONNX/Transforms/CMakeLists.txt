# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(ConstProp)
add_onnx_mlir_rewriter(ConvOpt)

add_onnx_mlir_library(OMShapeInference
  ShapeInference.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  MLIRFuncDialect
  )

add_onnx_mlir_library(OMShapeInferencePass
  ShapeInferencePass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  MLIRFuncDialect
  MLIRPass
  OMShapeInference
  )

add_onnx_mlir_library(OMInstrumentONNX
  InstrumentONNXSignaturePass.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  OMKrnlOps
  MLIRPass
  OMOptionUtils
  )

add_onnx_mlir_library(OMONNXRewrite
  ConstProp.cpp
  ConvOpt.cpp
  Decompose.cpp
  DecomposeEinsum.cpp
  ScrubDisposablePass.cpp
  SetONNXNodeName.cpp
  Recompose.cpp

  DEPENDS
  OMONNXDecomposeIncGen
  OMONNXConstPropIncGen
  OMONNXConvOptIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  OMONNXOps
  )

add_onnx_mlir_library(OMOpTransform
  ONNXOpTransformPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  OMONNXRewrite
  OMShapeInferencePass
  MLIRTransforms
  )

add_onnx_mlir_library(OMHybridTransform
  ONNXHybridTransformPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMShapeInferenceOpInterface
  OMONNXRewrite
  MLIRPass
  MLIRTransforms
  OMShapeInference
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
  OMShapeInferencePass
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  )

add_onnx_mlir_library(OMONNXStandardFuncReturnPass
  StandardFuncReturnPass.cpp

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  MLIRFuncDialect
  OMONNXOps
  OMShapeInference
  )
