// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func @test_krnl_global_constant_alignment() -> memref<3xf32> {
  %0 = "krnl.global"() {name = "constant", alignment = 1024 : i64, shape = [3], value = dense<[0.0, 0.1, 0.2]> : tensor<3xf32>} : () -> memref<3xf32>
  return %0 : memref<3xf32>

// CHECK:         llvm.mlir.global internal constant @constant(dense<[0.000000e+00, 1.000000e-01, 2.000000e-01]> : tensor<3xf32>) {alignment = 1024 : i64} : !llvm.array<3 x f32>
// CHECK:         llvm.func @test_krnl_global_constant_alignment() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.addressof @constant : !llvm.ptr<array<3 x f32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.bitcast [[VAR_0_]] : !llvm.ptr<array<3 x f32>> to !llvm.ptr<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           [[VAR_3_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_2_]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_1_]], [[VAR_3_]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_6_]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           [[VAR_10_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_8_]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return [[VAR_10_]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }
}
