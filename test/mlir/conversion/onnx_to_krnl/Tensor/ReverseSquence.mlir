// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_reversesequence_1(%arg0: tensor<10x?xf32>, %arg1: tensor<10xi64>) -> tensor<*xf32> {
  %0 = "onnx.ReverseSequence"(%arg0, %arg1) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x?xf32>, tensor<10xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reversesequence_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?xf32>, [[PARAM_1_:%.+]]: memref<10xi64>) -> memref<10x?xf32> {
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<10x?xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#1] : memref<10xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_1_]]#0, [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.subi [[VAR_3_]], [[VAR_1_]]#0 : index
// CHECK:             [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[CST_1_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_4_]], [[VAR_6_]], [[VAR_1_]]#0 : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_1_]]#1] : memref<10x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<10x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x?xf32>
// CHECK:         }
}
