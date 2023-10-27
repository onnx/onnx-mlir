add_subdirectory(Stickify)

add_onnx_mlir_library(OMLayoutHelper
  LayoutHelper.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  ${NNPA_INCLUDE_PATH}
)
