add_onnx_mlir_library(OMFoldStdAlloc
  FoldStdAlloc.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRAffineToStandard

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
   )

add_subdirectory(ZHigh)
