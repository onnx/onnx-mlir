# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_onnx_mlir_executable(CompilerLibTest
  CompilerLibTest.cpp

  NO_INSTALL

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  OMCompiler
  )
