# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestDisposableElementsAttr
  TestDisposableElementsAttr.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMONNXOps
  )

add_test(NAME TestDisposableElementsAttr COMMAND TestDisposableElementsAttr)
