# SPDX-License-Identifier: Apache-2.0

if (ONNX_MLIR_ACCELERATORS)
  foreach(t ${ONNX_MLIR_ACCELERATORS})
    add_subdirectory(${t})
  endforeach(t)
endif(ONNX_MLIR_ACCELERATORS)
