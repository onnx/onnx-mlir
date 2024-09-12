# SPDX-License-Identifier: Apache-2.0

if(ONNX_MLIR_INSTALL_HEADERS)
  install(FILES OMCompilerTypes.h DESTINATION include/onnx-mlir/Compiler)
  install(FILES OMCompilerMacros.h DESTINATION include/onnx-mlir/Compiler)
endif()
