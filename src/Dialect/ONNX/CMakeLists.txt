# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the ONNXOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(ONNXOps)
add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

add_onnx_mlir_rewriter(Rewrite)

add_onnx_mlir_library(OMONNXOps
  ONNXOps.cpp
  ONNXOpsHelper.cpp
  IndexExpr.cpp
  IndexExprDetail.cpp
  MLIRDialectBuilder.cpp
  Rewrite.cpp

  ShapeInference/ArgMax.cpp
  ShapeInference/AveragePool.cpp
  ShapeInference/CategoryMapper.cpp  
  ShapeInference/Compress.cpp
  ShapeInference/Concat.cpp
  ShapeInference/Conv.cpp
  ShapeInference/DepthToSpace.cpp
  ShapeInference/Expand.cpp
  ShapeInference/Gather.cpp
  ShapeInference/Gemm.cpp
  ShapeInference/LRN.cpp
  ShapeInference/MatMul.cpp
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
  OMONNXOpsIncGen
  OMONNXRewriteIncGen
  OMResultTypeInferenceOpInterfaceIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRMath
  MLIRAffine
  MLIRSCF
  )
