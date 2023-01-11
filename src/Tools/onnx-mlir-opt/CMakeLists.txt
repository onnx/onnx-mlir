# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(onnx-mlir-opt
  onnx-mlir-opt.cpp

  LINK_LIBS PRIVATE
  ${OMLibs}
  OMCompilerOptions
  OMCompilerUtils
  OMAccelerator
  OMInitAccelerators
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROptLib
  )
