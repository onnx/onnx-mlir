# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestFloatingPoint16
  TestFloatingPoint16.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMMlirUtilities
  benchmark
  )

add_test(NAME TestFloatingPoint16 COMMAND TestFloatingPoint16)
