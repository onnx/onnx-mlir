set(LLVM_TARGET_DEFINITIONS ZHighToZLow.td)
mlir_tablegen(ZHighToZLow.inc -gen-rewriters "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighToZLowIncGen)

add_dlc_library(OMZHighToZLow
  ZHighToZLow.cpp

  DEPENDS
  OMZHighToZLowIncGen

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMONNXToKrnl
  OMZHighOps
  OMZLowOps
  )

