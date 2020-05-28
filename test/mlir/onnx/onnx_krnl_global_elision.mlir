// RUN: onnx-mlir-opt --elide-krnl-constants %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_elide_krnl_global_constant(%arg0: memref<1xf32>) -> memref<1x70xf32>
func @test_elide_krnl_global_constant(%arg0: memref<1xf32>) -> memref<1x70xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 10], value = dense<[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]> : tensor<1x70xf32>} : () -> memref<1x70xf32>
  return %0 : memref<1x70xf32>

  // CHECK: %0 = "krnl.global"() {name = "constant_0", shape = [1, 10]} : () -> memref<1x70xf32>
  // CHECK: return %0 : memref<1x70xf32>
}