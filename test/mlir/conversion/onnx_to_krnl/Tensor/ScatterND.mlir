// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_scatter_nd1(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
// CHECK-LABEL:  @test_scatter_nd1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<4x4x4xf32>, [[PARAM_1:%.+]]: memref<2x1xi64>, [[PARAM_2:%.+]]: memref<2x4x4xf32>) -> memref<4x4x4xf32> {
// CHECK-DAG:       [[CST_4:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<4x4x4xf32>
// CHECK-DAG:       [[CST_64:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES]], %arg0, [[CST_64]], [[CST_0]], [[CST_0]]) : (memref<4x4x4xf32>, memref<4x4x4xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with
// CHECK-SAME:      ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 4, [[LOOP_0]]#2 -> [[I_2:%.+]] = 0 to 4){
// CHECK-DAG:         [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[INDEX:%.+]] = krnl.load [[PARAM_1]]{{.}}[[IV]]#0, [[CST_0]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[UPDATE:%.+]] = krnl.load [[PARAM_2]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<2x4x4xf32>
// CHECK-DAG:         [[CAST_INDEX:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             krnl.store [[UPDATE]], [[RES]]{{.}}[[CAST_INDEX]], [[IV]]#1, [[IV]]#2{{.}} : memref<4x4x4xf32>
// CHECK-NEXT:      }
// CHECK:           return [[RES]] : memref<4x4x4xf32>
}

// -----

func.func @test_scatter_with_dynamic_indices(%arg0: tensor<2x1xi64>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>) -> tensor<2x1xi64> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "none"} : (tensor<2x1xi64>, tensor<?x2xi64>, tensor<2xi64>) -> tensor<2x1xi64>]
  return %0 : tensor<2x1xi64>
}
