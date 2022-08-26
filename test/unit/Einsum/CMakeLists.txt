# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestONNXEinsumOp
  TestONNXEinsumOp.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMONNXOps
  )

add_test(NAME TestONNXEinsumOp COMMAND TestONNXEinsumOp)
