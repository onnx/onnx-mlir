add_dlc_executable(dlc-wmlz
  DLC-WMLz.cpp

  INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PRIVATE
  CompilerUtils
  )
