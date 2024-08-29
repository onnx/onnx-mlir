# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_unittest(TestBType
  TestBType.cpp

  LINK_LIBS PRIVATE
  OMONNXOps
  )
