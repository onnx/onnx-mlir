# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_rewriter(Decompose)
add_onnx_mlir_rewriter(DecomposeConvTranspose)
add_onnx_mlir_rewriter(DecomposeConvTransposePhased)
add_onnx_mlir_rewriter(DecomposeConvTranspose1dPhased)

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
  MLIRPass
  OMOptionUtils
  )

add_onnx_mlir_library(OMONNXRewrite
  ConstProp.cpp
  QDQAroundOpOpt.cpp    
  DQBinaryQOpt.cpp  
  QDQCanonicalize.cpp
  ConvOpt.cpp
  Decompose.cpp
  DecomposeEinsum.cpp
  ScrubDisposablePass.cpp
  SetONNXNodeName.cpp
  Recompose.cpp
  LegalizeQuarkQuantizedOps.cpp
  QuantTypes.cpp
  ConvertToChannelLast.cpp
  ResultNamesUpdater.cpp
  xmc/MergeSliceConcatPass.cpp
  xmc/CombineTransposePair.cpp
  xmc/MergeContinuousStridedSlice.cpp
  xmc/RemoveContinuousTransposeWithReshape.cpp
  xmc/RemoveUselessQLinearPoolPass.cpp
  xmc/ReplaceHsigmoidAndHswishPass.cpp
  xmc/MergeStridedSliceConcatConvPass.cpp
  xmc/ConvertPoolLikeOpToPool2dPass.cpp
  xmc/InstanceNormToGroupNormPass.cpp
  xmc/RemoveDilationConvPass.cpp
  xmc/RemoveSemanticallyUselessOps.cpp
  xmc/Resize_to_conv.cpp
  xmc/StandardizeStridedSlice.cpp
  xmc/TransferConvSliceToConvPass.cpp
  xmc/TransferReduceMeanSumToConvPass.cpp
  xmc/ConvertMulToDepthwiseConv2dPass.cpp
  xmc/TransferOp3dToOp2dPass.cpp
  xmc/TransferPoolFixToDownsampleFixPass.cpp
  xmc/TransferDepthwiseConv2dWithChannelMultiplierPass.cpp
  xmc/TransformReshapelikeOpToReshapePass.cpp
  xmc/Transform5DTransposeTo4DPass.cpp
  xmc/EliminateReshapeAroundSlicePass.cpp
  xmc/OptimizeSliceReshapeTransposeBlockPass.cpp
  xmc/Transfer5dBlockTo4dPass.cpp
  xmc/Transfer5dStridedSliceTo4d.cpp
  xmc/TransferSpaceToDepthToConv2dPass.cpp
  xmc/MergeBatchnormToConvPass.cpp
  xmc/BatchReductionToReshapeReductionPass.cpp
  xmc/RemoveRedundantReluPass.cpp
  xmc/ReplaceNDimTransposePass.cpp
  xmc/ONNXTransposeOptimizationPass.cpp
  xmc/ONNXTransposeOptimizationAxisChangePatterns.cpp

  DEPENDS
  OMONNXDecomposeIncGen
  OMONNXDecomposeConvTransposeIncGen
  OMONNXDecomposeConvTransposePhasedIncGen
  OMONNXDecomposeConvTranspose1dPhasedIncGen
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
