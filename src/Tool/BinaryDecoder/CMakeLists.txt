# SPDX-License-Identifier: Apache-2.0

add_executable(binary-decoder BinaryDecoder.cpp)
target_link_libraries(binary-decoder
  PRIVATE
  LLVMSupport
  onnx
  )
