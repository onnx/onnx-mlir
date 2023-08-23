// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_gather_scalar(%arg0: tensor<4xi64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gather_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i64>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK-DAG:         [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK:             [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_4_2_]] : index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]], [[VAR_1_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][] : memref<i64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<i64>
// CHECK:         }
}
