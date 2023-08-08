# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_executable(onnx-mlir-opt
  onnx-mlir-opt.cpp
  RegisterPasses.cpp

  LINK_LIBS PRIVATE
  ${OMLibs}
  OMCompilerDialects
  OMCompilerOptions
  OMCompilerPasses
  OMAccelerator
  OMVersion
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIROptLib
  MLIRSCFToOpenMP
  )
