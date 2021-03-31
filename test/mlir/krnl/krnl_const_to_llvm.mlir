// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// COM: Check lowering a krnl constant with alignment.
func @test_krnl_aligned_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_aligned_const
  // CHECK: {{.*}} = llvm.alloca {{.*}} x !llvm.array<1 x array<4 x f32>> {alignment = 1024 : i64} : (i64) -> !llvm.ptr<array<1 x array<4 x f32>>> 
}

// -----

// COM: Check lowering a krnl constant with a opaque attribute
// CHECK-DAG: llvm.mlir.global internal constant @constant_0(opaque<"krnl", "0x68656C6C6F"> : tensor<1x4xf32>) : !llvm.array<1 x array<4 x f32>>
func @test_krnl_opaque_const() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = opaque<"krnl", "0x68656C6C6F"> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
  // CHECK-LABEL: test_krnl_opaque_const
  // CHECK: [[ALLOC:%.+]] = llvm.alloca %0 x !llvm.array<1 x array<4 x f32>> {alignment = 1024 : i64} : (i64) -> !llvm.ptr<array<1 x array<4 x f32>>>
  // CHECK: [[ALLOC_PTR:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr<array<1 x array<4 x f32>>> to !llvm.ptr<i8>
  // CHECK: [[CONSTANT:%.+]] = llvm.mlir.addressof @constant_0 : !llvm.ptr<array<1 x array<4 x f32>>>
  // CHECK: [[CONSTANT_PTR:%.+]] = llvm.bitcast [[CONSTANT]] : !llvm.ptr<array<1 x array<4 x f32>>> to !llvm.ptr<i8>
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[ALLOC_PTR]], [[CONSTANT_PTR]], %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
}

