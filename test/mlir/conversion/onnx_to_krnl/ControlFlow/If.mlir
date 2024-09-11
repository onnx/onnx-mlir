// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// COM: Check simple if lowering.
func.func @test_if_simple(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.If"(%arg0) ({
    onnx.Yield %arg1 : tensor<i64>
  }, {
    onnx.Yield %arg2 : tensor<i64>
  }) : (tensor<i1>) -> tensor<i64>
  return %0 : tensor<i64>
// CHECK-LABEL:  func.func @test_if_simple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i1>, [[PARAM_1_:%.+]]: memref<i64>, [[PARAM_2_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<i64> to tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<i64> to tensor<i64>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i1>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = scf.if [[LOAD_PARAM_0_MEM_]] -> (memref<i64>) {
// CHECK-DAG:         [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_1_]] : tensor<i64> to memref<i64>
// CHECK:             scf.yield [[VAR_4_]] : memref<i64>
// CHECK:           } else {
// CHECK:             [[VAR_4_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_0_]] : tensor<i64> to memref<i64>
// CHECK:             scf.yield [[VAR_4_1_]] : memref<i64>
// CHECK:           }
// CHECK:           return [[VAR_3_]] : memref<i64>
// CHECK:         }
}

