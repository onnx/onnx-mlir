# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(onnx-mlir-opt
  ONNXMLIROpt.cpp
  )

target_link_libraries(onnx-mlir-opt
  ${OMLibs}
  MainUtils
  ${MLIRLibs}
  MLIROptLib
  onnx
  )
