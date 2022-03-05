get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_dlc_executable(dlc-opt
  DLCOpt.cpp

  INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_BIN_ROOT}

  LINK_LIBS PRIVATE
  ${OMLibs}
  MLIROptLib
  MLIRAffineTransforms
  MLIRLinalgTransforms
  )
