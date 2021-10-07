# SPDX-License-Identifier: Apache-2.0

add_subdirectory(onnx-mlir)

install(FILES OnnxMlirCompiler.h DESTINATION include)
install(FILES OnnxMlirRuntime.h DESTINATION include)
