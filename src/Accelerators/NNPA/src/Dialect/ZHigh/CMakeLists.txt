set(LLVM_TARGET_DEFINITIONS ZHighOps.td)
mlir_tablegen(ZHighOps.hpp.inc -gen-op-decls "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighOps.cpp.inc -gen-op-defs "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighAttrs.hpp.inc -gen-attrdef-decls "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
mlir_tablegen(ZHighAttrs.cpp.inc -gen-attrdef-defs "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ZHighCombine.td)
mlir_tablegen(ZHighCombine.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighCombineIncGen)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

message(STATUS "printing variables")
#print_all_variables()
message(STATUS "DLC_ONNX_MLIR_SRC_ROOT=${DLC_ONNX_MLIR_SRC_ROOT}") 
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
