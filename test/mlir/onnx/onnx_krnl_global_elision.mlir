// RUN: onnx-mlir-opt --elide-krnl-constants %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_elide_krnl_global_constant(%arg0: memref<1xf32>) -> memref<1x70xf32>
func @test_elide_krnl_global_constant(%arg0: memref<1xf32>) -> memref<1x70xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 70], value = dense<[[0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]> : tensor<1x70xf32>} : () -> memref<1x70xf32>
  return %0 : memref<1x70xf32>

  // CHECK: {{.*}} = "krnl.global"() {name = "constant_00", shape = [1, 70]} : () -> memref<1x70xf32>
  // CHECK: return {{.*}} : memref<1x70xf32>
}

// -----

func @test_elide_krnl_global_constant() -> memref<1x80xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 80], value = opaque<"krnl", "0x3F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC00000400000003F0000003F8000003FC0000040000000"> : tensor<1x80xf32>} : () -> memref<1x80xf32>
  return %0 : memref<1x80xf32>

// CHECK: {{.*}} = "krnl.global"() {name = "constant_01", shape = [1, 80]} : () -> memref<1x80xf32>
// CHECK: return {{.*}} : memref<1x80xf32>
}
