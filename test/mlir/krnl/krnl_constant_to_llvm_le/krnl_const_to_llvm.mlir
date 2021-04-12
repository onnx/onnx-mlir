// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// COM: Check lowering a krnl constant with alignment.
func @test_krnl_aligned_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_aligned_const
  // CHECK: [[ALLOC:%.+]] = llvm.alloca {{.*}} x !llvm.array<1 x array<4 x f32>> {alignment = 1024 : i64} : (i64) -> !llvm.ptr<array<1 x array<4 x f32>>>
  // CHECK: [[ALLOC_PTR:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr<array<1 x array<4 x f32>>> to !llvm.ptr<i8>
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[ALLOC_PTR]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
}

// -----

// COM: Check lowering a krnl constant with an opaque attribute
// COM: Data for the opaque attribute: [0.5, 1., 1.5, 2.]
// CHECK-DAG: llvm.mlir.global internal constant @constant_0("?\00\00\00?\80\00\00?\C0\00\00@\00\00\00")
func @test_krnl_opaque_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = opaque<"krnl", "0x3F0000003F8000003FC0000040000000"> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_opaque_const
  // CHECK: [[CONSTANT:%.+]] = llvm.mlir.addressof @constant_0 : !llvm.ptr<array<16 x i8>>
  // CHECK: [[CONSTANT_PTR:%.+]] = llvm.bitcast [[CONSTANT]] : !llvm.ptr<array<16 x i8>> to !llvm.ptr<f32>
}

