# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_ENABLE_PYBIND)
  return()
endif()

add_onnx_mlir_unittest(OMTensorTest
  OMTensorTest.c

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  cruntime
  )
