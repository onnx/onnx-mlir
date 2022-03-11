# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMFoldStdAlloc
  FoldStdAlloc.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRAffineToStandard

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
   )

add_subdirectory(ZHigh)
add_subdirectory(ZLow)
