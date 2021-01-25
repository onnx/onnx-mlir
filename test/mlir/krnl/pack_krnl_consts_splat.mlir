// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=true filename=test-pack-consts-to-file-splat.bin' %s -split-input-file | FileCheck %s

// CHECK:         [[VAR_0_:%.+]] = "krnl.packed_const"() {file_name = "test-pack-consts-to-file-splat.bin", is_le = false, size_in_bytes = 0 : i64} : () -> i64
// CHECK-LABEL:  func @test_krnl_const_packing_file_splat() -> memref<1x4xf32> {
// CHECK-NEXT:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_0", offset = 0 : i64, shape = [1, 4], value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> memref<1x4xf32>
// CHECK-NEXT:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_1", offset = 0 : i64, shape = [1, 4], value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> memref<1x4xf32>
func @test_krnl_const_packing_file_splat() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %1 = "krnl.global"() {name = "constant_1", shape = [1, 4], value = dense<0.000000e+00> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  return %1 : memref<1x4xf32>
}
