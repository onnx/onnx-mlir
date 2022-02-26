set(LLVM_TARGET_DEFINITIONS ZHighToZLow.td)
mlir_tablegen(ZHighToZLow.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
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

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  )

