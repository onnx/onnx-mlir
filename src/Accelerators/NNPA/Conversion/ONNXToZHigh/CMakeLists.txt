add_onnx_mlir_rewriter(ONNXToZHigh)

add_onnx_mlir_library(OMONNXToZHigh
  ONNXLegalityCheck.cpp
  ONNXToZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXONNXToZHighIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXToKrnl
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
  OMONNXToKrnl
  OMZHighOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
