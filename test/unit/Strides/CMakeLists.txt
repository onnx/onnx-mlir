# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestStrides
  TestStrides.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMONNXOps
  )

add_test(NAME TestStrides COMMAND TestStrides)
