# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Mlir)
if (ONNX_MLIR_ENABLE_KRNL)
  add_subdirectory(Krnl)
endif()
add_subdirectory(ONNX)
