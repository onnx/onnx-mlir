# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(onnx-mlir-opt
  NO_INSTALL
  ONNXMLIROpt.cpp
  )

target_link_libraries(onnx-mlir-opt
  ${OMLibs}
  ${MLIRLibs}
  MLIROptLib
  onnx
  )
