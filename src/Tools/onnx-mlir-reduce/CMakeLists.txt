# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(onnx-mlir-reduce
  onnx-mlir-reduce.cpp

  LINK_LIBS PRIVATE
  ${OMLibs}
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRReduceLib
  )
