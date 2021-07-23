# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSupport
  KrnlSupport.cpp
  OptimizeOptions.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
  )
