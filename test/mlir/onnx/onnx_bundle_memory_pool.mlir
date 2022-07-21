// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --enable-memory-pool --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

func.func @test_bundle_memory_pool(%arg0: tensor<10x10xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<10x10xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %2 = "onnx.Add"(%1, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %3 = "onnx.Add"(%0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %4 = "onnx.MatMul"(%3, %arg1) : (tensor<10x10xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %5 = "onnx.Add"(%4, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %5 : tensor<10x20xf32>

// CHECK-LABEL: test_bundle_memory_pool
// CHECK-DAG:       [[CST_800_:%.+]] = arith.constant 800 : i64
// CHECK-DAG:       [[CST_1200_:%.+]] = arith.constant 1200 : i64
// CHECK-DAG:       [[CST_2000_:%.+]] = arith.constant 2000 : i64
// CHECK-DAG:       [[CST_2800_:%.+]] = arith.constant 2800 : i64
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<3200xi8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_2800_]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_2000_]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_1200_]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_800_]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_0_1_]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK:           memref.dealloc [[VAR_1_]] : memref<3200xi8>
// CHECK:           return [[VAR_0_]] : memref<10x20xf32>
}
