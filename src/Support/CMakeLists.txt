# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSupport
  KrnlSupport.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMKrnlOps
  )
