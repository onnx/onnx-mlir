# SPDX-License-Identifier: Apache-2.0

add_subdirectory(onnx-mlir)

if(ONNX_MLIR_INSTALL_HEADERS)
  install(FILES OnnxMlirCompiler.h DESTINATION include)
  install(FILES OnnxMlirRuntime.h DESTINATION include)
endif()
