# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)
get_property(DialectLibs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(ConversionLibs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_onnx_mlir_executable(onnx-mlir-opt
  onnx-mlir-opt.cpp

  LINK_LIBS PRIVATE
  ${DialectLibs}
  ${ConversionLibs}
  ${OMLibs}
  CompilerOptions
  Accelerator
  InitAccelerators
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROptLib
  )
