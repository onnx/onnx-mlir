set(LLVM_TARGET_DEFINITIONS ZHighOps.td)
mlir_tablegen(ZHighOps.hpp.inc -gen-op-decls "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighOps.cpp.inc -gen-op-defs "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighAttrs.hpp.inc -gen-attrdef-decls "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighAttrs.cpp.inc -gen-attrdef-defs "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ZHighCombine.td)
mlir_tablegen(ZHighCombine.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighCombineIncGen)

add_dlc_library(OMZHighOps
  ZHighHelper.cpp
  ZHighOps.cpp
  ZHighCombine.cpp
  ZHighShapeHelper.cpp

  DEPENDS
  OMLayoutHelper
  OMZHighOpsIncGen
  OMZHighCombineIncGen
  OMShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  OMLayoutHelper
  MLIRIR
  )
