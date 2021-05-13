# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(binary-decoder
  NO_INSTALL
  BinaryDecoder.cpp
  )

target_link_libraries(binary-decoder
  PRIVATE
  LLVMSupport
  onnx
  )
