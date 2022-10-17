# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

if (ONNX_MLIR_VENDOR)
  list(APPEND DEFINITIONS "ONNX_MLIR_VENDOR=\"${ONNX_MLIR_VENDOR}\"")
endif()

add_onnx_mlir_executable(onnx-mlir-opt
  onnx-mlir-opt.cpp

  DEFINE PUBLIC ${DEFINITIONS}
  LINK_LIBS PRIVATE
  ${OMLibs}
  CompilerOptions
  Accelerator
  InitAccelerators
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROptLib
  )
