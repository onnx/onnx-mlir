set(LLVM_TARGET_DEFINITIONS ZHighToZLow.td)
mlir_tablegen(ZHighToZLow.inc -gen-rewriters "-I${NNPA_ONNX_MLIR_SRC_ROOT}" "-I${NNPA_SRC_ROOT}")
add_public_tablegen_target(OMZHighToZLowIncGen)

add_onnx_mlir_library(OMZHighToZLow
  ZHighToZLow.cpp

  DEPENDS
  OMZHighToZLowIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMONNXToKrnl
  OMZHighOps
  OMZLowOps
  )

