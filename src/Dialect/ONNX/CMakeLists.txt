# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the ONNXOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(ONNX onnx)
add_onnx_mlir_dialect_doc(onnx ONNX.td)

add_onnx_mlir_rewriter(Rewrite)

add_onnx_mlir_library(OMONNXOps
  # Top files for ONNX dialect
  DialectBuilder.cpp
  ONNXDialect.cpp
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  ONNXShapeHelper.cpp  
  ONNXTypes.cpp
  Rewrite.cpp
  
  # Support for shape inference and verifiers
  Additional/CallOp.cpp
  Additional/Custom.cpp
  Additional/Dim.cpp
  Additional/EntryPoint.cpp
  Additional/LayoutTransform.cpp
  Additional/None.cpp
  ControlFlow/If.cpp
  ControlFlow/Loop.cpp
  ControlFlow/Scan.cpp
  ML/CategoryMapper.cpp
  ML/OneHotEncoder.cpp
  Math/Bernoulli.cpp
  Math/Clip.cpp
  Math/Einsum.cpp
  Math/EinsumHelper.cpp
  Math/ElementwiseBroadcast.cpp
  Math/ElementwiseUnary.cpp
  Math/Gemm.cpp
  Math/IsNan.cpp
  Math/LRN.cpp
  Math/MatMul.cpp
  Math/RandomNormal.cpp
  Math/RandomNormalLike.cpp
  Math/Reduction.cpp
  Math/TopK.cpp
  NN/Conv.cpp
  NN/Dropout.cpp
  NN/Normalization.cpp
  NN/Pooling.cpp
  NN/RoiAlign.cpp
  ObjectDetection/NonMaxSuppression.cpp
  Quantize/DequantizeLinear.cpp
  Quantize/DynamicQuantizeLinear.cpp
  Quantize/QuantizeLinear.cpp
  RNN/RNN.cpp
  Sequence/Sequence.cpp
  Sequence/SplitToSequence.cpp
  Tensor/ArgMinMax.cpp
  Tensor/Compress.cpp
  Tensor/Concat.cpp
  Tensor/ConcatFromSequence.cpp
  Tensor/Constant.cpp
  Tensor/ConstantOfShape.cpp
  Tensor/DepthToSpace.cpp
  Tensor/Expand.cpp
  Tensor/EyeLike.cpp
  Tensor/Flatten.cpp
  Tensor/Gather.cpp
  Tensor/GatherElements.cpp
  Tensor/GatherND.cpp
  Tensor/Identity.cpp
  Tensor/NonZero.cpp
  Tensor/OneHot.cpp
  Tensor/Optional.cpp
  Tensor/Pad.cpp
  Tensor/Range.cpp
  Tensor/Reshape.cpp
  Tensor/Resize.cpp
  Tensor/ReverseSequence.cpp
  Tensor/Shape.cpp
  Tensor/Size.cpp
  Tensor/Slice.cpp
  Tensor/SpaceToDepth.cpp
  Tensor/Split.cpp
  Tensor/Squeeze.cpp
  Tensor/Tile.cpp
  Tensor/Transpose.cpp
  Tensor/Unique.cpp
  Tensor/Unsqueeze.cpp
  Tensor/Upsample.cpp

  DEPENDS
  OMHasOnnxSubgraphOpInterfaceIncGen
  OMONNXIncGen
  OMONNXRewriteIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PRIVATE
  Diagnostic
    
  LINK_LIBS PUBLIC
  OMMlirDialects
  OMMlirUtilities
  onnx
  MLIRFuncDialect
  )

configure_file(ONNXOps.td.inc.dc.in
  ${CMAKE_CURRENT_BINARY_DIR}/ONNXOps.td.inc.dc
  @ONLY
  )
