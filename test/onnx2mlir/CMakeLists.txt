# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(CustomFnTest
  CustomFnTest.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  ${OMLibs}
  )

add_test(NAME CustomFnTest COMMAND CustomFnTest)
