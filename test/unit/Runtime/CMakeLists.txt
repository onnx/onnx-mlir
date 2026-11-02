# SPDX-License-Identifier: Apache-2.0

add_unittest(OMTensorTest
  OMTensorTest.c

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  cruntime
  )

add_unittest(OMTensorListHelperTest
  OMTensorListHelperTest.cpp

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  OMDebugRuntime
  )

add_unittest(OMResizeTest
  OMResizeTest.c

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  cruntime
  m
  )
