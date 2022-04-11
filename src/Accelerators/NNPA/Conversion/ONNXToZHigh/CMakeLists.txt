add_onnx_mlir_rewriter(ONNXToZHigh)

add_onnx_mlir_library(OMONNXToZHigh
  ONNXLegalityCheck.cpp
  ONNXToZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXONNXToZHighIncGen

  LINK_LIBS PUBLIC
  OMONNXToKrnl
  OMZHighOps
  )

add_onnx_mlir_rewriter(RewriteONNXForZHigh)

add_onnx_mlir_library(OMRewriteONNXForZHigh
  ONNXLegalityCheck.cpp
  RewriteONNXForZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXRewriteONNXForZHighIncGen
  
  LINK_LIBS PUBLIC
  OMONNXToKrnl
  OMZHighOps
  )
