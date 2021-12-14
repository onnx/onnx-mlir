// RUN: onnx-mlir ../../../../docs/doc_example/add.onnx 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:     ../../../../docs/doc_example/add.so has been compiled.
