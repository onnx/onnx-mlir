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

  # start of new support
  ControlFlow/If.cpp
  ControlFlow/Loop.cpp
  ControlFlow/Scan.cpp
  Math/Clip.cpp
  Math/ElementwiseBroadcast.cpp
  Math/ElementwiseUnary.cpp
  Math/Gemm.cpp
  Math/LRN.cpp
  Math/MatMul.cpp
  Math/RandomNormal.cpp
  Math/RandomNormalLike.cpp
  Math/Reduction.cpp
  Math/TopK.cpp
  ML/CategoryMapper.cpp
  NN/Conv.cpp
  NN/Normalization.cpp
  NN/Pooling.cpp
  ObjectDetection/NonMaxSuppression.cpp
  RNN/RNN.cpp
  Sequence/Sequence.cpp
  Tensor/ArgMinMax.cpp
  Tensor/Compress.cpp
  Tensor/Concat.cpp
  Tensor/ConcatFromSequence.cpp
  Tensor/Constant.cpp
  Tensor/ConstantOfShape.cpp
  Tensor/DepthToSpace.cpp
  Tensor/Dim.cpp
  Tensor/Expand.cpp
  Tensor/Flatten.cpp
  Tensor/Gather.cpp
  Tensor/GatherElements.cpp
  Tensor/GatherND.cpp
  Tensor/Identity.cpp
  Tensor/NonZero.cpp
  Tensor/OneHot.cpp
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
  Tensor/Unsqueeze.cpp

  DialectBuilder.cpp
  ONNXDialect.cpp
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  ONNXEinsumOpHelper.cpp
  ONNXTypes.cpp
  Rewrite.cpp

  ShapeInference/ArgMinMax.cpp
  ShapeInference/AveragePool.cpp
  ShapeInference/CategoryMapper.cpp  
  ShapeInference/Clip.cpp  
  ShapeInference/Compress.cpp
  ShapeInference/Concat.cpp
  ShapeInference/Conv.cpp
  ShapeInference/DepthToSpace.cpp
  ShapeInference/Expand.cpp
  ShapeInference/Flatten.cpp
  ShapeInference/Gather.cpp
  ShapeInference/GatherElements.cpp  
  ShapeInference/GatherND.cpp  
  ShapeInference/Gemm.cpp
  ShapeInference/LRN.cpp
  ShapeInference/GenericMatMul.cpp
  ShapeInference/MaxPool.cpp
  ShapeInference/ONNXShapeHelper.cpp  
  ShapeInference/OneHot.cpp
  ShapeInference/Pad.cpp
  ShapeInference/Reshape.cpp
  ShapeInference/ReverseSequence.cpp
  ShapeInference/RoiAlign.cpp
  ShapeInference/Shape.cpp
  ShapeInference/Slice.cpp
  ShapeInference/SpaceToDepth.cpp
  ShapeInference/Split.cpp
  ShapeInference/Squeeze.cpp
  ShapeInference/Tile.cpp
  ShapeInference/TopK.cpp
  ShapeInference/Transpose.cpp
  ShapeInference/Unsqueeze.cpp

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
