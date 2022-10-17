# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(OMTensorTest
  OMTensorTest.c

  NO_INSTALL

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  cruntime
  )

add_test(NAME OMTensorTest COMMAND OMTensorTest)
