// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=false' %s -split-input-file | FileCheck %s

// CHECK: [[CONST_PACK:%.+]] = "krnl.packed_const"() {is_le = true, size_in_bytes = 32 : i64, value = dense<[0, 0, 0, 0, 0, 0, -128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0, -128, 63, 0, 0, 0, 64, 0, 0, 64, 64]> : tensor<32xi8>} : () -> i64
// CHECK-LABEL: func @test_krnl_const_packing() -> memref<1x4xf32> {
// CHECK-NEXT: [[CONST0:%.+]] = "krnl.global"() {name = "constant_0", offset = 0 : i64, shape = [1, 4]} : () -> memref<1x4xf32>
// CHECK-NEXT: [[CONST1:%.+]] = "krnl.global"() {name = "constant_1", offset = 16 : i64, shape = [1, 4]} : () -> memref<1x4xf32>
func @test_krnl_const_packing() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %1 = "krnl.global"() {name = "constant_1", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
}