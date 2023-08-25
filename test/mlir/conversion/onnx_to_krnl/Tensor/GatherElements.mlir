// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_gather_elements(%arg0: tensor<4xi64>, %arg1: tensor<2xi64>) -> tensor<2xi64> {
  %0 = "onnx.GatherElements"(%arg0, %arg1) : (tensor<4xi64>, tensor<2xi64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
// CHECK-LABEL:  @test_gather_elements
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<4xi64>, [[PARAM_1:%.+]]: memref<2xi64>) -> memref<2xi64> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0]]) with ([[LOOP_0]] -> [[I_0:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_INDEX:%.+]] = krnl.load [[PARAM_1]]{{.*}}[[IV]]]{{.*}} : memref<2xi64>
// CHECK-DAG:         [[INDEX:%.+]] = arith.index_cast [[LOAD_INDEX]] : i64 to index
// CHECK-DAG:         [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_4:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[INDEX]], [[CST_0]] : index
// CHECK-DAG:         [[VAR_1:%.+]] = arith.addi [[INDEX]], [[CST_4]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[VAR_1]], [[INDEX]] : index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]]{{.}}[[SEL]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]]{{.}}[[IV]]{{.}} : memref<2xi64>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2xi64>
}

