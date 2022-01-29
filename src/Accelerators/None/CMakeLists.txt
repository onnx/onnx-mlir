# SPDX-License-Identifier: Apache-2.0
add_onnx_mlir_library(OMNoAccel
  PrepareAccelerator.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  )