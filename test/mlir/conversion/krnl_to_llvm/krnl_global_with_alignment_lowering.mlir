// RUN: onnx-mlir-opt -O3 --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// Test that the global constant is aligned as specified by the explicit alignment.
func.func @test_krnl_global_constant_alignment() -> memref<3xf32> {
  %0 = "krnl.global"() {name = "constant", alignment = 1024 : i64, shape = [3], value = dense<[0.0, 0.1, 0.2]> : tensor<3xf32>} : () -> memref<3xf32>
  return %0 : memref<3xf32>

// CHECK:         llvm.mlir.global internal constant @constant(dense<[0.000000e+00, 1.000000e-01, 2.000000e-01]> : tensor<3xf32>) {addr_space = 0 : i32, alignment = 1024 : i64} : !llvm.array<3 x f32>
// CHECK-LABEL:   llvm.func @test_krnl_global_constant_alignment() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.addressof @constant : !llvm.ptr
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.bitcast [[VAR_0_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_2_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_3_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_6_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_8_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return [[VAR_10_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
}

// -----

// Test that the global constant is aligned on a 16 bytes boundary when an explicit alignment is not specified. 
func.func @test_krnl_global_constant_no_alignment() -> memref<2xi64> {
  %0 = "krnl.global"() {name = "constant", shape = [2], value = dense<[0, 1]> : tensor<2xi64>} : () -> memref<2xi64>
  return %0 : memref<2xi64>

// CHECK:         llvm.mlir.global internal constant @constant(dense<[0, 1]> : tensor<2xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<2 x i64>
// CHECK-LABEL:   llvm.func @test_krnl_global_constant_no_alignment() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.addressof @constant : !llvm.ptr
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.bitcast [[VAR_0_]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_2_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_3_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_6_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_8_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return [[VAR_10_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
}
