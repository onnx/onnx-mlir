# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(omtest
  OMTest.cpp

  NO_INSTALL

  INCLUDE_DIRS PRIVATE
  ${CMAKE_BINARY_DIR}/src

  LINK_LIBS PRIVATE
  om
  )
