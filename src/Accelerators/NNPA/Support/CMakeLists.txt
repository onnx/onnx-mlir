add_subdirectory(Stickify)

add_onnx_mlir_library(OMLayoutHelper
  LayoutHelper.cpp
  NNPALimit.cpp

  DEPENDS
  libzdnn
  OMCompilerOptions

  LINK_LIBS PUBLIC
  MLIRIR
  OMCompilerOptions

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  ${NNPA_INCLUDE_PATH}
)
