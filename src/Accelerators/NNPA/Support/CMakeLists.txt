add_onnx_mlir_library(OMLayoutHelper
  LayoutHelper.cpp

  LINK_LIBS PUBLIC
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
)

add_onnx_mlir_library(OMDLCSupport
  OMDLCOptions.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
)
