get_property(DLCOMLibs GLOBAL PROPERTY DLC_LIBS)


add_onnx_mlir_library(DLCompilerUtils
  EXCLUDE_FROM_OM_LIBS
  DLCompilerUtils.cpp

  INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PUBLIC
  CompilerUtils
  ${DLCOMLibs}

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_BIN_ROOT}
  )

# add_dependencies(CompilerUtils NNPAAccel) 