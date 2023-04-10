# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestBType
  TestBType.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMONNXOps
  )

add_test(NAME TestBType COMMAND TestBType)
