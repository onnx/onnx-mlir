// RUN: onnx-mlir-opt %s | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NOT: attributes
module {
  // CHECK: tvp.module @kernels {
  tvp.module @kernels {
  }
}
