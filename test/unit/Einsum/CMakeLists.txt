# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_unittest(TestONNXEinsumOp
  TestONNXEinsumOp.cpp

  LINK_LIBS PRIVATE
  OMONNXOps
  )
