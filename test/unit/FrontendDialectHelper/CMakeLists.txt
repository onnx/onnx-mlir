# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_unittest(TestFrontendDialectHelper
  TestFrontendDialectHelper.cpp

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_SRC_ROOT}/src/Builder
    ${ONNX_MLIR_BIN_ROOT}
    ${ONNX_MLIR_SRC_ROOT}

  LINK_LIBS PRIVATE
    OMBuilder
    OMONNXOps
    ${MLIRLibs}
    ${ONNX_PROTO_LIB}
)
