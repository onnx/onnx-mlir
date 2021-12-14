// RUN: (onnx-mlir input.afo || exit 0) 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:     Invalid input file 'input.afo': Either ONNX model or MLIR file needs to be provided.
