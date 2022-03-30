set(LLVM_TARGET_DEFINITIONS ZLowOps.td)
mlir_tablegen(ZLowOps.hpp.inc -gen-op-decls)
mlir_tablegen(ZLowOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(OMZLowOpsIncGen)

add_onnx_mlir_library(OMZLowOps
  ZLowOps.cpp

  DEPENDS
  OMZLowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
 ${NNPA_SRC_ROOT}
 ${NNPA_BIN_ROOT}
  )
