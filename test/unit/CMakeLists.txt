# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestInstrumentation
  TestInstrumentation.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  cruntime
  )

add_test(NAME TestInstrumentation COMMAND TestInstrumentation)

add_subdirectory(Runtime)
