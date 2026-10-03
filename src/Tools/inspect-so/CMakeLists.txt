# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(inspect-so
  InspectSo.cpp

  LINK_LIBS PRIVATE
  ${CMAKE_DL_LIBS}
  )
