// RUN: onnx-mlir-opt --enable-safe-code-gen --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Checking of print has to be manual modified.
func.func @test_gather_scalar(%arg0: tensor<4xi64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
// CHECK-LABEL:  func.func @test_gather_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_minus_4_:%.+]] = arith.constant -4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i64>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK:             [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_minus_4_]] : index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sge, [[VAR_1_]], [[CST_4_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.ori [[VAR_2_]], [[VAR_3_]] : i1
// CHECK:             scf.if [[VAR_4_]] {
// CHECK:               "krnl.print"
// CHECK:               "krnl.print"
// CHECK:             }
// CHECK:           }
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK:             [[VAR_1_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_2_1_:%.+]] = arith.cmpi slt, [[VAR_1_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_3_1_:%.+]] = arith.addi [[VAR_1_1_]], [[CST_4_]] : index
// CHECK:             [[VAR_4_1_:%.+]] = arith.select [[VAR_2_1_]], [[VAR_3_1_]], [[VAR_1_1_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_1_]], [[CST_0_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[VAR_4_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi sge, [[VAR_6_]], [[CST_4_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_3_]], [[VAR_6_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][] : memref<i64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<i64>
// CHECK:         }

}
