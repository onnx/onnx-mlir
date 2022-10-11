// RUN: onnx-mlir --version | FileCheck %s

// CHECK:   onnx-mlir version {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}, onnx version {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}
// CHECK:   LLVM version {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}
// CHECK:   {{[DEBUG|Optimized]}} build{{[ with assertions]*}}
// CHECK:   Default target:
// CHECK:   Host CPU:
