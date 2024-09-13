add_onnx_mlir_rewriter(ONNXToZHigh)
add_onnx_mlir_rewriter(ZHighToONNX)

add_onnx_mlir_library(OMONNXToZHigh
  ONNXLegalityCheck.cpp
  ONNXToZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXONNXToZHighIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMNNPACompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMShapeInferencePass
  OMZHighOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(RewriteONNXForZHigh)

add_onnx_mlir_library(OMRewriteONNXForZHigh
  ONNXLegalityCheck.cpp
  RewriteONNXForZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXRewriteONNXForZHighIncGen
  libzdnn
  
  LINK_LIBS PUBLIC
  OMNNPACompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMZHighOps
  OMLayoutHelper


  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZHighToONNX
  ZHighToONNX.cpp

  DEPENDS
  OMONNXZHighToONNXIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXOps
  OMZHighOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMDevicePlacement
  DevicePlacement.cpp
  DevicePlacementHeuristic.cpp
  PerfModel.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXToZHigh
  OMRewriteONNXForZHigh

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
