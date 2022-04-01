set(LLVM_TARGET_DEFINITIONS ONNXToZHigh.td)
mlir_tablegen(ONNXToZHigh.inc -gen-rewriters "-I${NNPA_ONNX_MLIR_SRC_ROOT}" "-I${NNPA_SRC_ROOT}")
add_public_tablegen_target(OMONNXToZHighIncGen)

add_onnx_mlir_library(OMONNXToZHigh
  ONNXLegalityCheck.cpp
  ONNXToZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXToZHighIncGen

  LINK_LIBS PUBLIC
  OMONNXToKrnl
  OMZHighOps
  )

set(LLVM_TARGET_DEFINITIONS RewriteONNXForZHigh.td)
mlir_tablegen(RewriteONNXForZHigh.inc -gen-rewriters "-I${NNPA_ONNX_MLIR_SRC_ROOT}" "-I${NNPA_SRC_ROOT}")
add_public_tablegen_target(OMRewriteONNXForZHighIncGen)

add_onnx_mlir_library(OMRewriteONNXForZHigh
  ONNXLegalityCheck.cpp
  RewriteONNXForZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMRewriteONNXForZHighIncGen
  
  LINK_LIBS PUBLIC
  OMONNXToKrnl
  OMZHighOps
  )
