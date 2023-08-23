# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(binary-decoder
  BinaryDecoder.cpp

  LINK_LIBS PRIVATE
  LLVMSupport
  onnx
  )
