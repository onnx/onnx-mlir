// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// COM: Lower a krnl constant with alignment.
func @test_krnl_aligned_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_aligned_const
}

// -----

// COM: Lower a krnl constant with a opaque attribute
func @test_krnl_opaque_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = opaque<"krnl", "0x68656C6C6F"> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_opaque_const
}

