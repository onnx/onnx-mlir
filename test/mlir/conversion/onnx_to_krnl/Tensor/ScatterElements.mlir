// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_scatter_elements1(%arg0: tensor<3x3xf32>, %arg1: tensor<3x2xi64>, %arg2: tensor<3x2xf32>) -> (tensor<*xf32>,tensor<*xf32>) {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<*xf32>
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<*xf32>
  return %0, %1 : tensor<*xf32>, tensor<*xf32>
// CHECK-LABEL:  @test_scatter_elements1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x3xf32>, [[PARAM_1:%.+]]: memref<3x2xi64>, [[PARAM_2:%.+]]: memref<3x2xf32>) -> (memref<3x3xf32>, memref<3x3xf32>) {
// CHECK-DAG:       [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES1:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x3xf32>
// CHECK-DAG:       [[CST_9:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES1]], %arg0, [[CST_9]], [[CST_0]], [[CST_0]]) : (memref<3x3xf32>, memref<3x3xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 3, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[INDEX:%.+]] = krnl.load [[PARAM_1]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<3x2xi64>
// CHECK-DAG:         [[UPDATE_VAL:%.+]] = krnl.load [[PARAM_2]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<3x2xf32>
// CHECK:             [[CAST_INDEX:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK-DAG:         [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[THREE:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[CAST_INDEX]], [[ZERO]] : index
// CHECK-DAG:         [[ADDI:%.+]] = arith.addi [[CAST_INDEX]], [[THREE]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[ADDI]], [[CAST_INDEX]] : index
// CHECK:             krnl.store [[UPDATE_VAL]], [[RES1]]{{.}}[[SEL]], [[IV]]#1{{.}} : memref<3x3xf32>
// CHECK-NEXT:      }
//
// CHECK-DAG:       [[RES2:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x3xf32>
// CHECK-DAG:       [[CST_9_1:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[CST_0_1:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES2]], %arg0, [[CST_9_1]], [[CST_0_1]], [[CST_0_1]]) : (memref<3x3xf32>, memref<3x3xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_1:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1) with ([[LOOP_1]]#0 -> [[I_0:%.+]] = 0 to 3, [[LOOP_1]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[INDEX:%.+]] = krnl.load [[PARAM_1]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<3x2xi64>
// CHECK-DAG:         [[UPDATE_VAL:%.+]] = krnl.load [[PARAM_2]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<3x2xf32>
// CHECK:             [[CAST_INDEX:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK-DAG:         [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[THREE:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[CAST_INDEX]], [[ZERO]] : index
// CHECK-DAG:         [[ADDI:%.+]] = arith.addi [[CAST_INDEX]], [[THREE]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[ADDI]], [[CAST_INDEX]] : index
// CHECK:             krnl.store [[UPDATE_VAL]], [[RES2]]{{.}}[[IV]]#0, [[SEL]]{{.}} : memref<3x3xf32>
// CHECK-NEXT:      }
// CHECK:           return [[RES1]], [[RES2]] : memref<3x3xf32>, memref<3x3xf32>
}
