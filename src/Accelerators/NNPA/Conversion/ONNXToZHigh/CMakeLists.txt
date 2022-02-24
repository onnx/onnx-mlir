set(LLVM_TARGET_DEFINITIONS ONNXToZHigh.td)
mlir_tablegen(ONNXToZHigh.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMONNXToZHighIncGen)

add_dlc_library(OMONNXToZHigh
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
mlir_tablegen(RewriteONNXForZHigh.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMRewriteONNXForZHighIncGen)

add_dlc_library(OMRewriteONNXForZHigh
  ONNXLegalityCheck.cpp
  RewriteONNXForZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMRewriteONNXForZHighIncGen
  
  LINK_LIBS PUBLIC
  OMONNXToKrnl
  OMZHighOps
  )
