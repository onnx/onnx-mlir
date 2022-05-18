# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Accelerators)
add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tools)
add_subdirectory(Compiler)

set_property(SOURCE onnx-mlir.cpp APPEND PROPERTY COMPILE_DEFINITIONS "LLVM_PACKAGE_VERSION=\"${LLVM_PACKAGE_VERSION}\"")

add_onnx_mlir_executable(onnx-mlir
  onnx-mlir.cpp

  LINK_LIBS PRIVATE
  CompilerUtils
  )
  
