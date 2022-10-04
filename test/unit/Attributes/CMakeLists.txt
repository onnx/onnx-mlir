# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestONNXAttributes
  TestONNXAttributes.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMONNXOps
  )

add_test(NAME TestONNXAttributes COMMAND TestONNXAttributes)
